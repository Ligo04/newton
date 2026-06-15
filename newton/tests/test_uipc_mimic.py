# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverUIPC mimic joint coupling (follower = coef0 + coef1 * leader)."""

from __future__ import annotations

import importlib.util
import unittest

import warp as wp

import newton
from newton import JointTargetMode
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")

_Q_ID = wp.quat_identity(dtype=wp.float64)  # pyright: ignore[reportArgumentType]


def _add_revolute_pair(builder: newton.ModelBuilder, y: float):
    """A fixed-to-world anchor with two independently hinged child links.

    Returns ``(leader_joint, follower_joint)`` — two revolute joints that
    share no kinematic coupling in the model, so any tracking between them
    must come from a mimic constraint.
    """
    hx = hy = 0.1
    upper_hz = 0.2
    link_hz = 0.5
    drop_z = 2.0

    anchor = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=_Q_ID))
    builder.add_shape_box(anchor, hx=hx, hy=hy, hz=upper_hz)

    leader_link = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z - link_hz), q=_Q_ID))
    builder.add_shape_box(leader_link, hx=hx, hy=hy, hz=link_hz)

    follower_link = builder.add_link(xform=wp.transform(p=wp.vec3(0.5, y, drop_z - link_hz), q=_Q_ID))
    builder.add_shape_box(follower_link, hx=hx, hy=hy, hz=link_hz)

    j_fixed = builder.add_joint_fixed(
        parent=-1,
        child=anchor,
        parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=_Q_ID),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=_Q_ID),
        label="anchor",
    )
    j_leader = builder.add_joint_revolute(
        parent=anchor,
        child=leader_link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=_Q_ID),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +link_hz), q=_Q_ID),
        label="leader",
    )
    j_follower = builder.add_joint_revolute(
        parent=anchor,
        child=follower_link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, -upper_hz), q=_Q_ID),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +link_hz), q=_Q_ID),
        label="follower",
    )
    builder.add_articulation([j_fixed, j_leader, j_follower], label="mimic_articulation")
    return j_leader, j_follower


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCMimicResolution(unittest.TestCase):
    """Mimic resolution — runs with ``backend="none"`` (no GPU required)."""

    def test_mimic_resolved_into_constraints(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        j_leader, j_follower = _add_revolute_pair(builder, y=0.0)
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=0.1, coef1=-1.0)

        model = builder.finalize()
        self.assertEqual(model.constraint_mimic_count, 1)

        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        constraints = solver._articulation_builder._mimic_constraints
        self.assertEqual(len(constraints), 1)
        follower_art, follower_local, leader_art, leader_local, coef0, coef1 = constraints[0]
        self.assertAlmostEqual(coef0, 0.1, places=6)
        self.assertAlmostEqual(coef1, -1.0, places=6)
        # Both joints are active revolute joints in the same articulation.
        self.assertIs(follower_art, leader_art)
        self.assertNotEqual(follower_local, leader_local)

    def test_disabled_mimic_skipped(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        j_leader, j_follower = _add_revolute_pair(builder, y=0.0)
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, enabled=False)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        self.assertEqual(len(solver._articulation_builder._mimic_constraints), 0)

    def test_mimic_with_inactive_joint_skipped_with_warning(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        _, j_follower = _add_revolute_pair(builder, y=0.0)
        # The fixed anchor joint (index 0) is not an active UIPC joint.
        builder.add_constraint_mimic(joint0=j_follower, joint1=0, coef1=1.0)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        with self.assertWarns(UserWarning):
            solver.initialize(model.state())

        self.assertEqual(len(solver._articulation_builder._mimic_constraints), 0)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCMimicTracking(unittest.TestCase):
    """Follower physically tracks ``coef0 + coef1 * leader`` after stepping."""

    def _run(self, device, coef0: float, coef1: float):
        wp.set_device(device)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        j_leader, j_follower = _add_revolute_pair(builder, y=0.0)
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)

        # Position-drive the leader; stiff enough to track its target.
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 1.0e4
            builder.joint_target_kd[i] = 1.0e2
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

        model = builder.finalize(device=device)
        solver = newton.solvers.SolverUIPC(model, backend="cuda", dt=1.0 / 60.0)
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        leader_qd_start = int(model.joint_qd_start.numpy()[j_leader])
        leader_target = 0.4
        target_q = control.joint_target_q.numpy()
        target_q[leader_qd_start] = leader_target
        control.joint_target_q.assign(target_q)

        for _ in range(120):
            solver.step(state_0, state_1, control, None, dt=1.0 / 60.0)
            state_0, state_1 = state_1, state_0

        q = state_0.joint_q.numpy()
        leader_q_start = int(model.joint_q_start.numpy()[j_leader])
        follower_q_start = int(model.joint_q_start.numpy()[j_follower])
        q_leader = float(q[leader_q_start])
        q_follower = float(q[follower_q_start])

        # Leader reached its commanded target, follower tracks the coupling.
        self.assertAlmostEqual(q_leader, leader_target, delta=0.05)
        self.assertAlmostEqual(q_follower, coef0 + coef1 * q_leader, delta=0.05)

    def test_mimic_tracking_inverse(self):
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                self._run(device, coef0=0.0, coef1=-1.0)

    def test_mimic_tracking_scaled_offset(self):
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                self._run(device, coef0=0.1, coef1=0.5)


if __name__ == "__main__":
    unittest.main()
