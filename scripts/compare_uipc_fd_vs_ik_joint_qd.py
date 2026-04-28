# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare UIPC's finite-difference joint_qd readback against ``newton.eval_ik``.

After each :meth:`SolverUIPC.step`, two things are true about ``state_out``:

1. ``state_out.joint_q`` / ``state_out.joint_qd`` carry the values written by
   the FD readback in :class:`Articulation` (read straight off UIPC edge
   ``angle`` / ``distance`` attributes, with ``qd = (q_t - q_{t-1}) / dt``).
2. ``state_out.body_q`` / ``state_out.body_qd`` carry the body-space state
   synced from UIPC's affine bodies.

We snapshot (1), then re-derive (1) from (2) via :func:`newton.eval_ik`, and
print the per-DOF max/RMS differences.

Usage::

    python scripts/compare_uipc_fd_vs_ik_joint_qd.py --steps 60
"""

from __future__ import annotations

import argparse

import numpy as np
import uipc
import warp as wp
from uipc import view

import newton
import newton.utils
from newton import JointTargetMode


def build_ur10_model(device, world_count: int = 1):
    ur10 = newton.ModelBuilder()
    asset_path = newton.utils.download_asset("universal_robots_ur10")
    asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
    height = 1.2
    ur10.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0.0, 0.0, height)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    ur10.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0.0, 0.0, height / 2)), half_height=height / 2, radius=0.08)
    # EFFORT (force/torque) control — match example_uipc_ur10_force.py.
    for i in range(len(ur10.joint_target_ke)):
        ur10.joint_target_ke[i] = 0.0
        ur10.joint_target_kd[i] = 0.0
        ur10.joint_target_mode[i] = int(JointTargetMode.EFFORT)
        if ur10.joint_type[i] == newton.JointType.REVOLUTE:
            ur10.joint_armature[i] = 1e-2

    if world_count > 1:
        builder = newton.ModelBuilder()
        builder.replicate(ur10, world_count, spacing=(2.0, 2.0, 0.0))
    else:
        builder = ur10

    rng = np.random.default_rng(42)
    builder.joint_q = rng.uniform(-wp.pi, wp.pi, builder.joint_dof_count).tolist()
    builder.add_ground_plane()
    return builder.finalize(device=device)


def probe_uipc_angles(solver, model) -> np.ndarray:
    """Read UIPC's per-joint ``angle`` / ``distance`` edge attribute directly.

    Returns a flat ``np.ndarray`` of length ``model.joint_dof_count`` indexed
    by the joint ``qd_start`` so it can be compared against ``state.joint_q``
    /``state.joint_qd``.
    """
    out = np.full(model.joint_dof_count, np.nan, dtype=np.float64)
    builder = solver._articulation_builder
    for art in builder.articulations.values():
        for newton_j in art.active_joint_indices:
            if newton_j not in art._joint_edge_idx:
                continue
            edge_idx = art._joint_edge_idx[newton_j]
            geo = art.joint_geo_slots[newton_j].geometry()
            attr_name = "angle" if art._joint_is_revolute[newton_j] else "distance"
            val = float(view(geo.edges().find(attr_name))[edge_idx])
            qd_start = art._joint_qd_start[newton_j]
            out[qd_start] = val
    return out


def sinusoidal_torques(control, model, world_count: int, t: float, amp: float):
    """Apply a sinusoidal torque profile to every revolute DOF (EFFORT mode)."""
    joint_f = control.joint_f.numpy()
    dof_per_world = model.joint_dof_count // world_count if world_count > 0 else 0
    for w in range(world_count):
        dof_start = w * dof_per_world
        for i in range(dof_per_world):
            di = dof_start + i
            joint_f[di] = float(amp * np.sin(t * 1.5 + i * 0.8 + w * 0.3))
    control.joint_f.assign(joint_f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30, help="Number of solver steps to run.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--world-count", type=int, default=1)
    parser.add_argument("--device", default=None, help="Warp device (default: auto).")
    parser.add_argument("--print-every", type=int, default=1, help="Print comparison every N steps (1 = every step).")
    parser.add_argument("--torque-amp", type=float, default=20.0, help="Sinusoidal torque amplitude per DOF [N·m].")
    args = parser.parse_args()

    if args.device is not None:
        wp.set_device(args.device)
    device = wp.get_device()
    print(f"[compare] device={device}")

    dt = 1.0 / args.fps
    model = build_ur10_model(device, world_count=args.world_count)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    solver = newton.solvers.SolverUIPC(model, dt=dt, logger_level=uipc.Logger.Warn)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    solver.initialize(state_0)

    # Diagnostic: read UIPC's `angle` edge attribute IMMEDIATELY after init,
    # before any world.advance(). If init_angle seeding worked, this should
    # equal state_0.joint_q on the active DOFs.
    angle_after_init = probe_uipc_angles(solver, model)
    q0_np = state_0.joint_q.numpy()
    with np.printoptions(precision=6, suppress=True, linewidth=200):
        print("[probe] post-initialize:")
        print(f"    state_0.joint_q  = {q0_np}")
        print(f"    UIPC angle attr  = {angle_after_init}")
        print(f"    diff             = {angle_after_init - q0_np}")

    # Buffers for the IK re-derivation. Allocate fresh so the FD-written
    # values in state_1 stay intact while we recompute.
    q_ik = wp.zeros_like(state_1.joint_q)
    qd_ik = wp.zeros_like(state_1.joint_qd)

    abs_q_peaks: list[float] = []
    abs_qd_peaks: list[float] = []
    rms_q_peaks: list[float] = []
    rms_qd_peaks: list[float] = []

    sim_time = 0.0
    for step in range(args.steps):
        state_0.clear_forces()
        sinusoidal_torques(control, model, args.world_count, sim_time, args.torque_amp)
        # Snapshot the pre-step joint_q the solver should be FD-ing against,
        # plus the UIPC `angle` attribute right now (= what read_pre_advance
        # will see).
        q_pre_external = state_0.joint_q.numpy().copy()
        angle_pre = probe_uipc_angles(solver, model)
        solver.step(state_0, state_1, control, contacts, dt)

        q_fd = state_1.joint_q.numpy().copy()
        qd_fd = state_1.joint_qd.numpy().copy()
        # FD that an outside observer would compute from state_0/state_1
        # (no wrapping). If qd_fd matches this, the solver-internal FD
        # readback is self-consistent — any gap to qd_ik is then purely
        # a (un)wrap-around artefact of how each side measures the angle.
        qd_fd_external = (q_fd - q_pre_external) / dt

        q_ik.zero_()
        qd_ik.zero_()
        newton.eval_ik(model, state_1, q_ik, qd_ik)
        q_ik_np = q_ik.numpy()
        qd_ik_np = qd_ik.numpy()

        dq = q_fd - q_ik_np
        dqd = qd_fd - qd_ik_np
        abs_q = float(np.max(np.abs(dq))) if dq.size else 0.0
        abs_qd = float(np.max(np.abs(dqd))) if dqd.size else 0.0
        rms_q = float(np.sqrt(np.mean(dq * dq))) if dq.size else 0.0
        rms_qd = float(np.sqrt(np.mean(dqd * dqd))) if dqd.size else 0.0
        abs_q_peaks.append(abs_q)
        abs_qd_peaks.append(abs_qd)
        rms_q_peaks.append(rms_q)
        rms_qd_peaks.append(rms_qd)

        if (step % args.print_every) == 0:
            argmax_q = int(np.argmax(np.abs(dq))) if dq.size else -1
            argmax_qd = int(np.argmax(np.abs(dqd))) if dqd.size else -1
            print(
                f"[step {step:04d} t={sim_time:7.4f}s] "
                f"|Δq|_max={abs_q:.3e}  rms={rms_q:.3e}   "
                f"|Δqd|_max={abs_qd:.3e}  rms={rms_qd:.3e}   "
                f"argmax_q={argmax_q}  argmax_qd={argmax_qd}"
            )
            # Wrap Δq into (-π, π] so we can see if the disagreement is
            # purely a 2π revolution offset.
            dq_wrapped = (dq + np.pi) % (2.0 * np.pi) - np.pi
            with np.printoptions(precision=6, suppress=True, linewidth=200):
                print(f"    q_pre  (state_0)        = {q_pre_external}")
                print(f"    q_pre  (UIPC angle pre) = {angle_pre}")
                print(f"    q_fd  = {q_fd}")
                print(f"    q_ik  = {q_ik_np}")
                print(f"    Δq    = {dq}")
                print(f"    Δq%2π = {dq_wrapped}")
                print(f"    qd_fd          = {qd_fd}")
                print(f"    qd_fd_ext      = {qd_fd_external}   (= (q_fd - q_pre[state_0]) / dt, no wrap)")
                print(f"    qd_fd_uipc     = {(q_fd - angle_pre) / dt}   (= (q_fd - UIPC angle pre) / dt)")
                print(f"    qd_ik          = {qd_ik_np}")
                print(f"    Δqd            = {dqd}")

        state_0, state_1 = state_1, state_0
        sim_time += dt

    print()
    print("=" * 72)
    print(f"summary over {args.steps} steps:")
    print(f"  |Δq|  : max={max(abs_q_peaks):.3e}  mean_rms={float(np.mean(rms_q_peaks)):.3e}")
    print(f"  |Δqd| : max={max(abs_qd_peaks):.3e}  mean_rms={float(np.mean(rms_qd_peaks)):.3e}")
    print("=" * 72)


if __name__ == "__main__":
    main()
