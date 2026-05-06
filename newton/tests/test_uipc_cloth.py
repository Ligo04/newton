# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC cloth material and soft-position controls."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.uipc.cloth import ClothBuilder

_HAS_UIPC = importlib.util.find_spec("uipc") is not None

if _HAS_UIPC:
    from uipc import view


class TestUIPCClothConfiguration(unittest.TestCase):
    def test_cloth_model_aliases(self):
        self.assertEqual(
            ClothBuilder._normalize_cloth_model("strain_limiting"),
            ClothBuilder.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
        )
        self.assertEqual(
            ClothBuilder._normalize_cloth_model("strain-limiting-baraff-witkin-shell"),
            ClothBuilder.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
        )
        self.assertEqual(
            ClothBuilder._normalize_cloth_model("neo_hookean_shell"),
            ClothBuilder.CLOTH_MODEL_NEO_HOOKEAN,
        )
        with self.assertRaises(ValueError):
            ClothBuilder._normalize_cloth_model("unknown")


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCClothSoftPosition(unittest.TestCase):
    def test_particle_custom_cloth_thick_sets_uipc_thickness(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="cloth_thick",
                dtype=wp.float32,
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                default=0.001,
            )
        )
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=0.0,
            custom_attributes_particles={"cloth_thick": 2.5e-4},
        )
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        geo = solver.mapping.cloth_geo_slots[0].geometry()
        thickness = view(geo.vertices().find("thickness"))
        np.testing.assert_allclose(thickness, np.full(model.particle_count, 2.5e-4), rtol=1.0e-6)

    def test_closed_cloth_mesh_is_rejected(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.1, 0.0, 0.0),
                wp.vec3(0.0, 0.1, 0.0),
                wp.vec3(0.0, 0.0, 0.1),
            ],
            indices=[
                0,
                2,
                1,
                0,
                1,
                3,
                1,
                2,
                3,
                2,
                0,
                3,
            ],
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=0.0,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)

        with self.assertRaisesRegex(RuntimeError, "closed"):
            solver.initialize(model.state())

    def test_soft_position_constraint_attributes_are_writable(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=0.0,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        state = model.state()
        solver.initialize(state)

        target = np.array([[0.0, 0.0, 0.2]], dtype=np.float64)
        solver.set_cloth_soft_position_constraints([0], target, strength_ratio=42.0)

        geo = solver.mapping.cloth_geo_slots[0].geometry()
        constrained = view(geo.vertices().find("is_constrained"))
        aim = view(geo.vertices().find("aim_position"))
        strength = view(geo.vertices().find("strength_ratio"))
        thickness = view(geo.vertices().find("thickness"))

        self.assertEqual(int(constrained[0]), 1)
        np.testing.assert_allclose(aim[0].reshape(3), target[0])
        self.assertAlmostEqual(float(strength[0]), 42.0)
        np.testing.assert_allclose(thickness, np.full(model.particle_count, 0.001), rtol=1.0e-6)

        solver.clear_cloth_soft_position_constraints([0])
        self.assertEqual(int(constrained[0]), 0)


if __name__ == "__main__":
    unittest.main()
