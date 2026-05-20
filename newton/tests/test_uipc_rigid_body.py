# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC rigid-body construction."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
if _HAS_UIPC:
    import uipc


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCRigidBodyKappa(unittest.TestCase):
    @staticmethod
    def _add_box_body(builder: newton.ModelBuilder, x: float, kappa: float | None = None) -> int:
        custom_attributes = {"uipc:abd_kappa": kappa} if kappa is not None else None
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.2), wp.quat_identity()),
            custom_attributes=custom_attributes,
            label=f"box_{x}",
        )
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        return body

    def test_register_custom_attributes_adds_body_abd_kappa(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        body_a = self._add_box_body(builder, 0.0, 1.0e8)
        body_b = self._add_box_body(builder, 0.3)

        model = builder.finalize()

        self.assertTrue(hasattr(model, "uipc"))
        self.assertTrue(hasattr(model.uipc, "abd_kappa"))
        values = model.uipc.abd_kappa.numpy()
        self.assertEqual(values.shape[0], model.body_count)
        self.assertAlmostEqual(float(values[body_a]), 1.0e8)
        self.assertAlmostEqual(float(values[body_b]), -1.0)

    def test_body_abd_kappa_splits_instancing_and_sets_uipc_kappa(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        body_soft = self._add_box_body(builder, 0.0, 1.0e8)
        body_stiff = self._add_box_body(builder, 0.3, 1.0e10)
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model,
            backend="none",
            logger_level=uipc.Logger.Error,
            auto_sync_inertia=False,
        )
        solver.initialize(model.state())

        soft_slot = solver.mapping.body_geo_slots[body_soft]
        stiff_slot = solver.mapping.body_geo_slots[body_stiff]
        self.assertIsNot(soft_slot, stiff_slot)
        self.assertEqual(solver.mapping.body_instance_ids[body_soft], 0)
        self.assertEqual(solver.mapping.body_instance_ids[body_stiff], 0)

        np.testing.assert_allclose(
            uipc.view(soft_slot.geometry().instances().find("kappa")),
            np.array([1.0e8], dtype=np.float32),
        )
        np.testing.assert_allclose(
            uipc.view(stiff_slot.geometry().instances().find("kappa")),
            np.array([1.0e10], dtype=np.float32),
        )

    def test_negative_body_abd_kappa_uses_solver_default_and_preserves_instancing(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        body_a = self._add_box_body(builder, 0.0)
        body_b = self._add_box_body(builder, 0.3)
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model,
            backend="none",
            kappa=3.0e8,
            logger_level=uipc.Logger.Error,
            auto_sync_inertia=False,
        )
        solver.initialize(model.state())

        shared_slot = solver.mapping.body_geo_slots[body_a]
        self.assertIs(shared_slot, solver.mapping.body_geo_slots[body_b])
        np.testing.assert_allclose(
            uipc.view(shared_slot.geometry().instances().find("kappa")),
            np.array([3.0e8, 3.0e8], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
