# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stability benchmark: ControllerPD vs ControllerStablePD on UR10 (MuJoCo).

For each (controller, dt) cell we drive the UR10 from HOME_POSE under a
constant step setpoint (HOME_POSE + perturbation) and measure:

    - diverged:     non-finite or |q| > 20 rad or max|qd| > 1e3 rad/s anywhere
    - rms_err:      RMS joint position error over the whole run (rad)
    - ss_err:       steady-state error — RMS over the last 0.5 s window (rad)
    - max_qd:       peak |qd| across all joints / time (rad/s)
    - rms_tau:      RMS commanded torque (N·m), summed over joints

Plain PD with implicit (semi-explicit MuJoCo) integration becomes unstable
above some critical dt that depends on Kp/Kd; Tan 2011 stable-PD pushes that
limit out by absorbing the Kp/Kd error projection through the implicit
``(M + diag(Kd)·dt)·qddot`` solve. This script measures that gap.

Run::

    uv run python scripts/benchmark_ur10_pd_vs_stable_pd.py
    uv run python scripts/benchmark_ur10_pd_vs_stable_pd.py --duration 5.0 \\
        --dts 0.002 0.004 0.008 0.016 0.033 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode
from newton.actuators import Actuator as _NewtonActuator
from newton.actuators import ClampingMaxEffort, ControllerPD, ControllerStablePD
from newton.examples.uipc.example_uipc_ur10_force import (
    _compute_coriolis_from_mass_derivatives,
)
from newton.selection import ArticulationView

# UR10 home pose, identical to example_uipc_ur10_force.Example.HOME_POSE.
HOME_POSE = np.array(
    [0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, np.pi / 2, 0.0],
    dtype=np.float32,
)
# Step disturbance applied to the target so the controllers must do work.
TARGET_OFFSET = np.array(
    [0.3, 0.2, -0.25, 0.4, -0.3, 0.5],
    dtype=np.float32,
)
KP = np.array([300.0, 300.0, 200.0, 100.0, 60.0, 30.0], dtype=np.float32)
KD = np.array([40.0, 40.0, 30.0, 15.0, 10.0, 5.0], dtype=np.float32)
MAX_TORQUE = np.array([330.0, 330.0, 150.0, 54.0, 54.0, 54.0], dtype=np.float32)


@dataclass
class TrialResult:
    controller: str
    dt: float
    steps: int
    wall_time_s: float
    diverged: bool
    diverge_step: int | None
    rms_err: float
    ss_err: float
    max_qd: float
    rms_tau: float


def _build_ur10_model(stable_pd: bool) -> newton.Model:
    """Build a single-world UR10 anchored to a pedestal, EFFORT mode on every DOF."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    asset_path = newton.utils.download_asset("universal_robots_ur10")
    asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
    height = 1.2
    builder.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0.0, 0.0, height)),
        floating=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    builder.add_shape_cylinder(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, height / 2)),
        half_height=height / 2,
        radius=0.08,
    )
    for i in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[i] = 0.0
        builder.joint_target_kd[i] = 0.0
        builder.joint_target_mode[i] = int(JointTargetMode.EFFORT)
        if builder.joint_type[i] == newton.JointType.REVOLUTE:
            builder.joint_armature[i] = 1e-2

    controller_cls = ControllerStablePD if stable_pd else ControllerPD
    for dof_idx in range(len(KP)):
        extra = {"num_worlds": 1} if stable_pd else {}
        builder.add_actuator(
            controller_cls,
            index=dof_idx,
            kp=float(KP[dof_idx]),
            kd=float(KD[dof_idx]),
            clamping=[(ClampingMaxEffort, {"max_effort": float(MAX_TORQUE[dof_idx])})],
            **extra,
        )

    builder.joint_q = HOME_POSE.tolist()
    builder.add_ground_plane()
    return builder.finalize()


def _gravity_torque(
    model: newton.Model,
    state: newton.State,
    J_buf: wp.array | None,
    dof: int,
) -> tuple[np.ndarray, wp.array]:
    """Jacobian-transpose gravity compensation torque (length ``dof``)."""
    J_buf = newton.eval_jacobian(model, state, J_buf)
    if J_buf is None:
        raise RuntimeError("eval_jacobian returned None")
    J_np = J_buf.numpy()[0]
    g = model.gravity.numpy()[0]
    body_mass = model.body_mass.numpy()
    tau_g = np.zeros(dof, dtype=np.float32)
    for b in range(model.body_count):
        m = float(body_mass[b])
        if m <= 0.0:
            continue
        J_lin = J_np[6 * b : 6 * b + 3, :dof]
        tau_g -= J_lin.T @ (m * g)
    return tau_g, J_buf


def _coriolis_bias(
    model: newton.Model,
    state: newton.State,
    H_fd_buf: wp.array | None,
    dof: int,
    eps: float = 1e-3,
) -> tuple[np.ndarray, wp.array]:
    """Finite-difference Christoffel Coriolis term — same recipe as the example."""
    j_q = state.joint_q
    j_qd = state.joint_qd
    if j_q is None or j_qd is None:
        raise RuntimeError("state must expose joint_q/joint_qd")
    q0 = j_q.numpy().astype(np.float32, copy=True)
    qd = j_qd.numpy().astype(np.float32, copy=True)[:dof]
    dH_dq = np.empty((dof, dof, dof), dtype=np.float32)
    try:
        for p in range(dof):
            qp = q0.copy()
            qm = q0.copy()
            qp[p] += eps
            qm[p] -= eps
            j_q.assign(qp)
            newton.eval_fk(model, j_q, j_qd, state)
            res = newton.eval_mass_matrix(model, state, H=H_fd_buf)
            if res is None:
                raise RuntimeError("eval_mass_matrix returned None")
            H_fd_buf = res
            H_plus = res.numpy()[0, :dof, :dof].astype(np.float32, copy=True)
            j_q.assign(qm)
            newton.eval_fk(model, j_q, j_qd, state)
            res = newton.eval_mass_matrix(model, state, H=H_fd_buf)
            if res is None:
                raise RuntimeError("eval_mass_matrix returned None")
            H_fd_buf = res
            H_minus = res.numpy()[0, :dof, :dof].astype(np.float32, copy=True)
            dH_dq[p] = (H_plus - H_minus) / (2.0 * eps)
    finally:
        j_q.assign(q0)
        newton.eval_fk(model, j_q, j_qd, state)
    return _compute_coriolis_from_mass_derivatives(dH_dq, qd), H_fd_buf


def run_trial(*, controller: str, dt: float, duration: float, verbose: bool) -> TrialResult:
    """Run one (controller, dt) cell and return the metrics."""
    stable_pd = controller == "stable_pd"
    model = _build_ur10_model(stable_pd=stable_pd)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    solver = newton.solvers.SolverMuJoCo(model)

    arti = ArticulationView(
        model,
        "*ur10*",
        exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
    )
    dof = arti.joint_dof_count
    assert dof == 6, f"expected 6 UR10 DOFs, got {dof}"

    target_pose = (HOME_POSE + TARGET_OFFSET).reshape(1, 1, dof).astype(np.float32)
    qd_target = np.zeros((1, 1, dof), dtype=np.float32)
    arti.set_attribute("joint_target_pos", control, target_pose)
    arti.set_attribute("joint_target_vel", control, qd_target)

    expected_cls = ControllerStablePD if stable_pd else ControllerPD
    pd_actuator = next(
        a for a in model.actuators if isinstance(a, _NewtonActuator) and isinstance(a.controller, expected_cls)
    )
    act_state = pd_actuator.state() if stable_pd else None

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    H_buf: wp.array | None = None
    H_fd_buf: wp.array | None = None
    J_buf: wp.array | None = None

    n_steps = max(1, int(round(duration / dt)))
    ss_window_steps = max(1, int(round(0.5 / dt)))

    err_sq_acc = 0.0
    tau_sq_acc = 0.0
    ss_err_sq_acc = 0.0
    ss_count = 0
    max_qd = 0.0
    diverged = False
    diverge_step: int | None = None

    target_q = (HOME_POSE + TARGET_OFFSET).astype(np.float32)
    t0 = time.perf_counter()
    for k in range(n_steps):
        # Stable-PD needs M and bias every substep.
        if stable_pd:
            H_res = newton.eval_mass_matrix(model, state_0, H=H_buf)
            if H_res is None:
                raise RuntimeError("eval_mass_matrix returned None")
            H_buf = H_res
            tau_g, J_buf = _gravity_torque(model, state_0, J_buf, dof)
            tau_c, H_fd_buf = _coriolis_bias(model, state_0, H_fd_buf, dof)
            ctrl_state = act_state.controller_state  # type: ignore[union-attr]
            ctrl_state.mass_matrix.assign(H_buf)
            ctrl_state.bias_forces.assign((-tau_g + tau_c).reshape(1, dof).astype(np.float32))

        control.joint_f.zero_()
        pd_actuator.step(
            sim_state=state_0,
            sim_control=control,
            current_act_state=act_state,
            next_act_state=act_state,
            dt=dt,
        )

        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        q = arti.get_attribute("joint_q", state_0).numpy()[0, 0]
        qd = arti.get_attribute("joint_qd", state_0).numpy()[0, 0]
        f = arti.get_attribute("joint_f", control).numpy()[0, 0]

        if (
            not np.all(np.isfinite(q))
            or not np.all(np.isfinite(qd))
            or np.any(np.abs(q) > 20.0)
            or np.any(np.abs(qd) > 1.0e3)
        ):
            diverged = True
            diverge_step = k
            if verbose:
                print(f"  ! diverged at step {k} (t={k * dt:.4f}s)")
            break

        err = q - target_q
        err_sq_acc += float(np.sum(err * err))
        tau_sq_acc += float(np.sum(f * f))
        max_qd = max(max_qd, float(np.max(np.abs(qd))))

        if k >= n_steps - ss_window_steps:
            ss_err_sq_acc += float(np.sum(err * err))
            ss_count += 1

    wall = time.perf_counter() - t0

    if diverged:
        rms_err = float("inf")
        ss_err = float("inf")
        rms_tau = float("nan")
    else:
        rms_err = math.sqrt(err_sq_acc / max(1, n_steps * dof))
        ss_err = math.sqrt(ss_err_sq_acc / max(1, ss_count * dof))
        rms_tau = math.sqrt(tau_sq_acc / max(1, n_steps * dof))

    return TrialResult(
        controller=controller,
        dt=dt,
        steps=n_steps if not diverged else (diverge_step or 0),
        wall_time_s=wall,
        diverged=diverged,
        diverge_step=diverge_step,
        rms_err=rms_err,
        ss_err=ss_err,
        max_qd=max_qd,
        rms_tau=rms_tau,
    )


def _format_table(results: list[TrialResult]) -> str:
    by_dt: dict[float, dict[str, TrialResult]] = {}
    for r in results:
        by_dt.setdefault(r.dt, {})[r.controller] = r
    dts = sorted(by_dt.keys())

    header = (
        f"{'dt (ms)':>9}  {'hz':>6}  "
        f"{'PD diverge':>11}  {'PD rms_err':>10}  {'PD ss_err':>10}  {'PD max_qd':>10}  "
        f"{'SPD diverge':>12}  {'SPD rms_err':>11}  {'SPD ss_err':>11}  {'SPD max_qd':>11}"
    )
    lines = [header, "-" * len(header)]
    for dt in dts:
        cell = by_dt[dt]
        pd = cell.get("pd")
        spd = cell.get("stable_pd")

        def _fmt(r: TrialResult | None) -> tuple[str, str, str, str]:
            if r is None:
                return ("-", "-", "-", "-")
            div = "yes" if r.diverged else "no"
            re_ = "inf" if math.isinf(r.rms_err) else f"{r.rms_err:.4f}"
            se = "inf" if math.isinf(r.ss_err) else f"{r.ss_err:.4f}"
            mq = f"{r.max_qd:.2f}"
            return (div, re_, se, mq)

        pd_d, pd_re, pd_se, pd_mq = _fmt(pd)
        sp_d, sp_re, sp_se, sp_mq = _fmt(spd)
        lines.append(
            f"{dt * 1000:>9.3f}  {1.0 / dt:>6.0f}  "
            f"{pd_d:>11}  {pd_re:>10}  {pd_se:>10}  {pd_mq:>10}  "
            f"{sp_d:>12}  {sp_re:>11}  {sp_se:>11}  {sp_mq:>11}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dts",
        type=float,
        nargs="+",
        default=[1.0 / 480, 1.0 / 240, 1.0 / 120, 1.0 / 60, 1.0 / 30, 1.0 / 15],
        help="Time steps to scan, seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Simulated duration per trial, seconds.",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=("pd", "stable_pd"),
        default=("pd", "stable_pd"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results: list[TrialResult] = []
    for dt in args.dts:
        for ctrl in args.controllers:
            label = "ControllerPD" if ctrl == "pd" else "ControllerStablePD"
            print(f"[run] {label}  dt={dt * 1000:.3f}ms  duration={args.duration:.2f}s")
            r = run_trial(controller=ctrl, dt=dt, duration=args.duration, verbose=args.verbose)
            tag = (
                f"DIVERGED@step{r.diverge_step}"
                if r.diverged
                else f"rms_err={r.rms_err:.4f}rad ss_err={r.ss_err:.4f}rad max_qd={r.max_qd:.2f}rad/s"
            )
            print(f"       -> {tag}  wall={r.wall_time_s:.2f}s")
            results.append(r)

    print()
    print("=" * 80)
    print("UR10 stability sweep - MuJoCo solver, ControllerPD vs ControllerStablePD")
    print(f"  duration={args.duration}s, target = HOME + {TARGET_OFFSET.tolist()}")
    print("=" * 80)
    print(_format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
