# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval_inverse_dynamics()."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Planar 2R point-mass arm parameters (revolute about +Z, links along +X,
# gravity along -Y). Point masses -> exact, independently derivable Lagrangian.
_M1, _M2 = 1.3, 0.7
_R1, _R2 = 0.6, 0.4
_L1 = 1.0
_G = 9.81


def _build_planar_2r(device):
    """Build the planar 2R point-mass arm and return (model, state)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)  # default gravity -9.81 along -Y

    link0 = builder.add_link(com=wp.vec3(_R1, 0.0, 0.0), mass=_M1)
    link1 = builder.add_link(com=wp.vec3(_R2, 0.0, 0.0), mass=_M2)
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link0,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(),
        child_xform=wp.transform(),
    )
    j1 = builder.add_joint_revolute(
        parent=link0,
        child=link1,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(_L1, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j0, j1], label="arm2r")
    model = builder.finalize(device=device)
    return model, model.state()


def _bias_lagrangian(q, qd):
    """Independent Euler-Lagrange bias C(q,q̇)q̇ + g(q) for the planar 2R arm.

    Nested finite differences of the point-mass Lagrangian -- shares no code with
    the Featherstone kernels, so agreement validates the kernel-driving.
    """

    def rot(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s], [s, c]])

    def positions(cfg):
        q1, q2 = cfg
        p1 = rot(q1) @ np.array([_R1, 0.0])
        joint2 = rot(q1) @ np.array([_L1, 0.0])
        p2 = joint2 + rot(q1 + q2) @ np.array([_R2, 0.0])
        return p1, p2

    def jac(pfun, cfg, eps=1e-6):
        out = np.zeros((2, 2))
        for i in range(2):
            d = np.zeros(2)
            d[i] = eps
            out[:, i] = (pfun(cfg + d) - pfun(cfg - d)) / (2 * eps)
        return out

    def mass(cfg):
        j1 = jac(lambda x: positions(x)[0], cfg)
        j2 = jac(lambda x: positions(x)[1], cfg)
        return _M1 * j1.T @ j1 + _M2 * j2.T @ j2

    def potential(cfg):
        p1, p2 = positions(cfg)
        return _M1 * _G * p1[1] + _M2 * _G * p2[1]

    eps = 1e-5
    g = np.zeros(2)
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps
        g[k] = (potential(q + d) - potential(q - d)) / (2 * eps)
    dM = np.zeros((2, 2, 2))
    for a in range(2):
        d = np.zeros(2)
        d[a] = eps
        dM[a] = (mass(q + d) - mass(q - d)) / (2 * eps)
    c = np.zeros(2)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                c[k] += (dM[i][k, j] - 0.5 * dM[k][i, j]) * qd[i] * qd[j]
    return g + c


def test_inverse_dynamics_bias_matches_lagrangian(test, device):
    """Bias forces (joint_acc=None) must equal the independent Lagrangian C q̇ + g."""
    model, state = _build_planar_2r(device)

    q = np.array([0.3, -0.5], dtype=np.float32)
    qd = np.array([1.1, -0.7], dtype=np.float32)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    tau = newton.eval_inverse_dynamics(model, state)
    tau_np = tau.numpy()[0, : model.joint_dof_count]

    expected = _bias_lagrangian(q.astype(np.float64), qd.astype(np.float64))
    np.testing.assert_allclose(tau_np, expected, rtol=1e-3, atol=1e-4)


def test_inverse_dynamics_full_id_adds_mass_matrix_acc(test, device):
    """tau(q̈) - tau(0) must equal H·q̈ (the M q̈ term reuses eval_mass_matrix)."""
    model, state = _build_planar_2r(device)

    q = np.array([0.2, 0.4], dtype=np.float32)
    qd = np.array([-0.5, 0.9], dtype=np.float32)
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    qdd = np.array([0.7, -1.3], dtype=np.float32)
    joint_acc = wp.array(qdd, dtype=float, device=device)

    tau_bias = newton.eval_inverse_dynamics(model, state).numpy()[0, : model.joint_dof_count]
    tau_full = newton.eval_inverse_dynamics(model, state, joint_acc=joint_acc).numpy()[0, : model.joint_dof_count]

    H = newton.eval_mass_matrix(model, state).numpy()[0, : model.joint_dof_count, : model.joint_dof_count]
    expected = H @ qdd.astype(np.float64)

    np.testing.assert_allclose(tau_full - tau_bias, expected, rtol=1e-4, atol=1e-4)


def test_inverse_dynamics_empty_model(test, device):
    """An articulation-free model returns None."""
    builder = newton.ModelBuilder()
    model = builder.finalize(device=device)
    state = model.state()
    test.assertIsNone(newton.eval_inverse_dynamics(model, state))


class TestInverseDynamics(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(
    TestInverseDynamics,
    "test_inverse_dynamics_bias_matches_lagrangian",
    test_inverse_dynamics_bias_matches_lagrangian,
    devices=devices,
)
add_function_test(
    TestInverseDynamics,
    "test_inverse_dynamics_full_id_adds_mass_matrix_acc",
    test_inverse_dynamics_full_id_adds_mass_matrix_acc,
    devices=devices,
)
add_function_test(
    TestInverseDynamics,
    "test_inverse_dynamics_empty_model",
    test_inverse_dynamics_empty_model,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
