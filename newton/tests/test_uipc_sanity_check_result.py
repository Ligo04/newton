# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC sanity-check result handling."""

from __future__ import annotations

import importlib.util
import unittest
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

_HAS_UIPC = importlib.util.find_spec("uipc") is not None

if _HAS_UIPC:
    import uipc
    from uipc.core import SanityCheckResult

    from newton._src.solvers.uipc import articulation_builder as articulation_builder_module
    from newton._src.solvers.uipc.converter import UIpcMappingInfo
    from newton._src.solvers.uipc.solver_uipc import SolverUIPC


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCSanityCheckResult(unittest.TestCase):
    def test_sanity_check_result_is_truthy_for_all_statuses(self):
        self.assertTrue(bool(SanityCheckResult.Success))
        self.assertTrue(bool(SanityCheckResult.Warning))
        self.assertTrue(bool(SanityCheckResult.Error))

    def test_sanity_check_warning_raises_runtime_error(self):
        class _Checker:
            def check(self):
                return SanityCheckResult.Warning

            def report(self):
                return "mock sanity-check warning"

        class _World:
            def sanity_checker(self):
                return _Checker()

        solver = object.__new__(SolverUIPC)
        solver.world = _World()

        with self.assertRaisesRegex(RuntimeError, "Warning.*mock sanity-check warning"):
            SolverUIPC._raise_if_sanity_check_failed(solver)

    def test_sanity_check_success_is_silent(self):
        class _Checker:
            def check(self):
                return SanityCheckResult.Success

            def report(self):
                raise AssertionError("report should not be requested on success")

        class _World:
            def sanity_checker(self):
                return _Checker()

        solver = object.__new__(SolverUIPC)
        solver.world = _World()

        SolverUIPC._raise_if_sanity_check_failed(solver)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCFEMSyncToBackend(unittest.TestCase):
    def test_particle_state_sync_to_uipc_round_trips(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=1.0,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="cuda", logger_level=uipc.Logger.Warn)
        solver.initialize()

        particle_q = model.particle_q.numpy()
        particle_q[:, 0] += 1.0
        model.particle_q.assign(particle_q)
        particle_qd = model.particle_qd.numpy()
        particle_qd[:, 1] = 2.0
        model.particle_qd.assign(particle_qd)

        solver._sync_particle_state_to_uipc()
        roundtrip_state = model.state()
        solver._sync_particle_state_from_uipc(roundtrip_state)

        np.testing.assert_allclose(roundtrip_state.particle_q.numpy(), particle_q, atol=1.0e-6)
        np.testing.assert_allclose(roundtrip_state.particle_qd.numpy(), particle_qd, atol=1.0e-6)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCFreeJointSoftTransformConstraint(unittest.TestCase):
    def test_free_joint_applies_soft_transform_once_per_shared_geometry(self):
        class _Geometry:
            pass

        class _Slot:
            def __init__(self, geometry):
                self._geometry = geometry

            def geometry(self):
                return self._geometry

        class _SoftTransformConstraint:
            applied_geometries: ClassVar[list] = []

            def apply_to(self, geometry):
                self.applied_geometries.append(geometry)

        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_link()
        body_b = builder.add_link()
        joint_a = builder.add_joint_free(child=body_a)
        joint_b = builder.add_joint_free(child=body_b)
        builder.add_articulation([joint_a])
        builder.add_articulation([joint_b])
        model = builder.finalize()

        shared_geometry = _Geometry()
        shared_slot = _Slot(shared_geometry)
        mapping = UIpcMappingInfo()
        mapping.body_geo_slots[body_a] = shared_slot
        mapping.body_geo_slots[body_b] = shared_slot

        articulation_builder = articulation_builder_module.ArticulationBuilder(
            model=model,
            scene=None,
            mapping=mapping,
            dt=1.0 / 60.0,
        )
        with patch.object(articulation_builder_module, "SoftTransformConstraint", _SoftTransformConstraint):
            articulation_builder.build_joints(contact_elem=None, joint_range=(0, model.joint_count))

        self.assertEqual(_SoftTransformConstraint.applied_geometries, [shared_geometry])

    def test_free_joint_enables_dynamic_aim_target(self):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_link(is_kinematic=False)
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = 1000.0
        builder.add_shape_sphere(body, radius=0.02, cfg=cfg)
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="cuda", logger_level=uipc.Logger.Error)
        solver.initialize()

        self.assertEqual(solver.mapping.body_instance_ids[body], 0)
        geo = solver.mapping.body_geo_slots[body].geometry()
        aim_attr = geo.instances().find("aim_transform")
        constrained_attr = geo.instances().find("is_constrained")
        fixed_attr = geo.instances().find("is_fixed")
        strength_attr = geo.instances().find("strength_ratio")
        self.assertIsNotNone(aim_attr)
        self.assertIsNotNone(constrained_attr)
        self.assertIsNotNone(fixed_attr)
        self.assertIsNotNone(strength_attr)

        self.assertEqual(int(uipc.view(constrained_attr)[0]), 0)
        self.assertEqual(int(uipc.view(fixed_attr)[0]), 0)
        np.testing.assert_allclose(uipc.view(strength_attr)[0].reshape(2), [100.0, 100.0])

        uipc.view(constrained_attr)[0] = 1
        target = np.eye(4, dtype=np.float64)
        target[0, 3] = 0.1
        uipc.view(aim_attr)[0] = target

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        for _ in range(5):
            solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0

        self.assertGreater(float(state_0.body_q.numpy()[body, 0]), 0.05)


if __name__ == "__main__":
    unittest.main()
