# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for example_uipc_ur10_force (Coriolis helper + M/bias parity across builder paths)."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton.examples.uipc.example_uipc_ur10_force import _compute_coriolis_from_mass_derivatives
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_test_devices

# Matches ``example_uipc_ur10_force.Example.HOME_POSE`` (6 revolute DOFs).
_UR10_HOME = np.array(
    [0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, np.pi / 2, 0.0],
    dtype=np.float32,
)


def _build_ur10_force_style_model(register_mujoco_attrs: bool, device):
    """UR10 + pedestal from ``uipc_ur10_force`` with optional MuJoCo USD custom attrs."""
    builder = newton.ModelBuilder()
    if register_mujoco_attrs:
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
        xform=wp.transform(wp.vec3(0, 0, height / 2)),
        half_height=height / 2,
        radius=0.08,
    )
    for i in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[i] = 0.0
        builder.joint_target_kd[i] = 0.0
        builder.joint_target_mode[i] = int(JointTargetMode.EFFORT)
        if builder.joint_type[i] == newton.JointType.REVOLUTE:
            builder.joint_armature[i] = 1e-2
    builder.joint_q = _UR10_HOME.tolist()
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _h_and_bias_stable_pd_style(
    model: newton.Model,
    state: newton.State,
    q_full: np.ndarray,
    qd_full: np.ndarray,
    *,
    H_buf: wp.array | None,
    J_buf: wp.array | None,
    H_fd_buf: wp.array | None,
    coriolis_eps: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Same recipe as example ``_apply_feedback`` (stable-PD): H, bias = -τ_g + τ_c."""
    dof = model.joint_dof_count
    j_q = state.joint_q
    j_qd = state.joint_qd
    if j_q is None or j_qd is None:
        raise RuntimeError("state must expose joint_q and joint_qd")
    j_q.assign(q_full)
    j_qd.assign(qd_full)
    newton.eval_fk(model, j_q, j_qd, state)
    H_buf = newton.eval_mass_matrix(model, state, H=H_buf)
    if H_buf is None:
        raise RuntimeError("eval_mass_matrix returned None")
    H = H_buf.numpy()[0, :dof, :dof].astype(np.float32, copy=True)

    J_buf = newton.eval_jacobian(model, state, J_buf)
    if J_buf is None:
        raise RuntimeError("eval_jacobian returned None")
    J_np = J_buf.numpy()[0]
    gravity = model.gravity.numpy()
    body_mass = model.body_mass.numpy()
    g_vec = gravity[0]
    tau_g = np.zeros(dof, dtype=np.float32)
    for b in range(model.body_count):
        m = float(body_mass[b])
        if m <= 0.0:
            continue
        J_lin = J_np[6 * b : 6 * b + 3, :dof]
        tau_g -= J_lin.T @ (m * g_vec)

    q0 = j_q.numpy().astype(np.float32, copy=True)
    qd_slice = j_qd.numpy().astype(np.float32, copy=True)[:dof]
    dH_dq = np.empty((dof, dof, dof), dtype=np.float32)
    eps = np.float32(coriolis_eps)
    try:
        for p in range(dof):
            q_plus = q0.copy()
            q_minus = q0.copy()
            q_plus[p] += eps
            q_minus[p] -= eps
            j_q.assign(q_plus)
            newton.eval_fk(model, j_q, j_qd, state)
            H_fd_res = newton.eval_mass_matrix(model, state, H=H_fd_buf)
            if H_fd_res is None:
                raise RuntimeError("eval_mass_matrix (perturbed) returned None")
            H_fd_buf = H_fd_res
            H_plus = H_fd_res.numpy()[0, :dof, :dof].astype(np.float32, copy=True)
            j_q.assign(q_minus)
            newton.eval_fk(model, j_q, j_qd, state)
            H_fd_res = newton.eval_mass_matrix(model, state, H=H_fd_buf)
            if H_fd_res is None:
                raise RuntimeError("eval_mass_matrix (perturbed) returned None")
            H_fd_buf = H_fd_res
            H_minus = H_fd_res.numpy()[0, :dof, :dof].astype(np.float32, copy=True)
            dH_dq[p] = (H_plus - H_minus) / (2.0 * eps)
    finally:
        j_q.assign(q0)
        j_qd.assign(qd_full)
        newton.eval_fk(model, j_q, j_qd, state)

    tau_c = _compute_coriolis_from_mass_derivatives(dH_dq, qd_slice)
    bias = -tau_g + tau_c
    return H, bias


def test_ur10_force_mass_and_bias_match_with_without_mujoco_custom_attrs(self, device):
    """UIPC vs MuJoCo *builder* paths: same ``q``/``qd`` => same H and stable-PD bias.

    The dynamics quantities use :func:`newton.eval_mass_matrix` / :func:`newton.eval_jacobian`
    on the :class:`newton.Model` + :class:`newton.State` only; they do not read the simulation
    solver. This test still proves that the optional MuJoCo USD custom-attribute registration
    does not change the final rigid-body model used by those evals.
    """
    if not USD_AVAILABLE:
        self.skipTest("USD / UR10 asset not available in this environment.")
    wp.set_device(device)
    model_a = _build_ur10_force_style_model(register_mujoco_attrs=False, device=device)
    model_b = _build_ur10_force_style_model(register_mujoco_attrs=True, device=device)

    self.assertEqual(model_a.joint_dof_count, model_b.joint_dof_count)
    dof = model_a.joint_dof_count
    n_q = int(model_a.joint_q.numpy().size)
    n_qd = int(model_a.joint_qd.numpy().size)
    self.assertEqual(n_q, n_qd)

    q = model_a.joint_q.numpy().astype(np.float32, copy=True)
    q[: min(dof, len(_UR10_HOME))] = _UR10_HOME[: min(dof, len(_UR10_HOME))]
    qd = np.zeros(n_qd, dtype=np.float32)
    qd[:dof] = np.linspace(0.1, 0.35, num=dof, dtype=np.float32)

    H_a, bias_a = _h_and_bias_stable_pd_style(model_a, model_a.state(), q, q, H_buf=None, J_buf=None, H_fd_buf=None)
    H_b, bias_b = _h_and_bias_stable_pd_style(model_b, model_b.state(), q, q, H_buf=None, J_buf=None, H_fd_buf=None)

    np.testing.assert_allclose(H_a, H_b, rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(bias_a, bias_b, rtol=0.0, atol=5.0e-4)


class TestUipcUr10Force(unittest.TestCase):
    def test_compute_coriolis_from_mass_derivatives(self):
        """Christoffel contraction matches a hand-computed 2-DOF example."""
        dH_dq = np.zeros((2, 2, 2), dtype=np.float32)
        # H(q) = [[1 + 2 q0, q0],
        #         [q0,       3 + q1]]
        dH_dq[0] = np.array([[2.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        dH_dq[1] = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        qd = np.array([2.0, 3.0], dtype=np.float32)

        coriolis = _compute_coriolis_from_mass_derivatives(dH_dq, qd)

        np.testing.assert_allclose(coriolis, np.array([4.0, 8.5], dtype=np.float32), rtol=1.0e-6)

    def test_compute_coriolis_is_zero_at_zero_velocity(self):
        dH_dq = np.ones((3, 3, 3), dtype=np.float32)
        qd = np.zeros(3, dtype=np.float32)

        coriolis = _compute_coriolis_from_mass_derivatives(dH_dq, qd)

        np.testing.assert_array_equal(coriolis, np.zeros(3, dtype=np.float32))


add_function_test(
    TestUipcUr10Force,
    "test_ur10_force_mass_and_bias_match_with_without_mujoco_custom_attrs",
    test_ur10_force_mass_and_bias_match_with_without_mujoco_custom_attrs,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
