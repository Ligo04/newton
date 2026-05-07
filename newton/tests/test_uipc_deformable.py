# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC deformable-body construction."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
if _HAS_UIPC:
    import uipc
    import uipc.builtin as uipc_builtin


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCDeformableBuilder(unittest.TestCase):
    @staticmethod
    def _add_disconnected_tet_soft_mesh(
        builder: newton.ModelBuilder,
        k_mu: float | list[float] = 5.0e4,
        k_lambda: float | list[float] = 5.0e4,
    ) -> None:
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.1, 0.0, 0.0),
                wp.vec3(0.0, 0.1, 0.0),
                wp.vec3(0.0, 0.0, 0.1),
                wp.vec3(0.4, 0.0, 0.0),
                wp.vec3(0.5, 0.0, 0.0),
                wp.vec3(0.4, 0.1, 0.0),
                wp.vec3(0.4, 0.0, 0.1),
            ],
            indices=[0, 1, 2, 3, 4, 5, 6, 7],
            density=1.0e3,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=1.0e-6,
            label="disconnected_tets",
        )

    def test_disconnected_soft_grids_use_authored_ranges(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        stiffness_values = [2.0e4, 5.0e4, 1.0e5, 2.0e5]
        for i, stiffness in enumerate(stiffness_values):
            builder.add_soft_grid(
                pos=wp.vec3(0.0, i * 0.4, 0.6),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.1,
                cell_y=0.1,
                cell_z=0.1,
                density=1.0e3,
                k_mu=stiffness,
                k_lambda=stiffness,
                k_damp=1.0e-6,
                fix_left=True,
            )

        model = builder.finalize()
        state = model.state()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
        )

        solver.initialize(state)

        self.assertEqual(len(solver.mapping.deformable_geo_slots), len(stiffness_values))
        self.assertEqual(len(solver.mapping.deformable_rest_geo_slots), len(stiffness_values))
        self.assertEqual(len(solver.mapping.deformable_particle_indices), len(stiffness_values))

    def test_disconnected_soft_grids_with_same_material_use_authored_ranges(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for i in range(2):
            builder.add_soft_grid(
                pos=wp.vec3(0.0, i * 0.4, 0.6),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.1,
                cell_y=0.1,
                cell_z=0.1,
                density=1.0e3,
                k_mu=5.0e4,
                k_lambda=5.0e4,
                k_damp=1.0e-6,
                fix_left=True,
                label=f"soft_{i}",
            )

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
        )

        solver.initialize(model.state())

        self.assertEqual(len(model.soft_body_ranges), 2)
        self.assertEqual(len(solver.mapping.deformable_geo_slots), 2)
        self.assertEqual(len(solver.mapping.deformable_rest_geo_slots), 2)
        self.assertEqual(len(solver.mapping.deformable_particle_indices), 2)

    def test_single_soft_range_with_disconnected_tets_builds_one_deformable(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self._add_disconnected_tet_soft_mesh(builder)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
        )

        solver.initialize(model.state())

        self.assertEqual(len(model.soft_body_ranges), 1)
        self.assertEqual(len(solver.mapping.deformable_geo_slots), 1)
        self.assertEqual(len(solver.mapping.deformable_rest_geo_slots), 1)
        self.assertEqual(len(solver.mapping.deformable_particle_indices), 1)
        self.assertEqual(len(solver.mapping.deformable_particle_indices[0]), 8)

    def test_disconnected_tet_materials_are_copied_per_tet(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        k_mu = [2.0e4, 8.0e4]
        k_lambda = [3.0e4, 9.0e4]
        self._add_disconnected_tet_soft_mesh(builder, k_mu=k_mu, k_lambda=k_lambda)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
        )

        solver.initialize(model.state())

        geometry = solver.mapping.deformable_geo_slots[0].geometry()
        mu_attr = geometry.tetrahedra().find("mu")
        lambda_attr = geometry.tetrahedra().find("lambda")
        self.assertIsNotNone(mu_attr)
        self.assertIsNotNone(lambda_attr)

        np_mu = uipc.view(mu_attr)
        np_lambda = uipc.view(lambda_attr)
        expected_mu = np.array([4.0 * value / 3.0 for value in k_mu], dtype=np.float32)
        expected_lambda = np.array(
            [lam + 5.0 * mu / 6.0 for mu, lam in zip(k_mu, k_lambda, strict=True)],
            dtype=np.float32,
        )
        self.assertEqual(np_mu.shape[0], 2)
        self.assertEqual(np_lambda.shape[0], 2)
        np.testing.assert_allclose(np_mu, expected_mu, rtol=0.0, atol=2.0e-2)
        np.testing.assert_allclose(np_lambda, expected_lambda, rtol=0.0, atol=2.0e-2)

    def test_soft_position_constraint_can_be_disabled(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self._add_disconnected_tet_soft_mesh(builder)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
            enable_soft_position_constraint=False,
        )

        solver.initialize(model.state())

        geometry = solver.mapping.deformable_geo_slots[0].geometry()
        self.assertIsNone(geometry.vertices().find("is_constrained"))
        self.assertIsNone(geometry.vertices().find("aim_position"))
        self.assertIsNone(geometry.vertices().find("strength_ratio"))

    def test_soft_position_constraint_attributes_are_writable(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self._add_disconnected_tet_soft_mesh(builder)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
        )
        solver.initialize(model.state())

        target = np.array([[0.0, 0.0, 0.2]], dtype=np.float64)
        solver.set_deformable_soft_position_constraints([0], target, strength_ratio=42.0)

        geometry = solver.mapping.deformable_geo_slots[0].geometry()
        constrained = uipc.view(geometry.vertices().find("is_constrained"))
        aim = uipc.view(geometry.vertices().find("aim_position"))
        strength = uipc.view(geometry.vertices().find("strength_ratio"))

        self.assertEqual(int(constrained[0]), 1)
        np.testing.assert_allclose(aim[0].reshape(3), target[0])
        self.assertAlmostEqual(float(strength[0]), 42.0)

        solver.clear_deformable_soft_position_constraints([0])
        self.assertEqual(int(constrained[0]), 0)

    def test_deformable_fallback_without_ranges_builds_one_deformable(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self._add_disconnected_tet_soft_mesh(builder)

        model = builder.finalize()
        model.soft_body_ranges.clear()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
        )

        solver.initialize(model.state())

        self.assertEqual(len(solver.mapping.deformable_geo_slots), 1)
        self.assertEqual(len(solver.mapping.deformable_rest_geo_slots), 1)
        self.assertEqual(len(solver.mapping.deformable_particle_indices), 1)
        self.assertEqual(len(solver.mapping.deformable_particle_indices[0]), 8)

    def test_replicated_soft_grids_build_one_deformable_per_world(self):
        base_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        base_builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.6),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=5.0e4,
            k_lambda=5.0e4,
            k_damp=1.0e-6,
            fix_left=True,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(base_builder, 2, spacing=(0.0, 0.8, 0.0))
        builder.add_ground_plane()

        model = builder.finalize()
        state = model.state()
        solver = newton.solvers.SolverUIPC(
            model=model,
            backend="none",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Warn,
        )

        solver.initialize(state)

        self.assertEqual(model.world_count, 2)
        self.assertEqual(len(solver.mapping.deformable_geo_slots), 2)
        self.assertEqual(len(solver.mapping.deformable_rest_geo_slots), 2)
        self.assertEqual(len(solver.mapping.deformable_particle_indices), 2)
        self.assertTrue(all(len(indices) > 0 for indices in solver.mapping.deformable_particle_indices))
        fixed_attr = solver.mapping.deformable_geo_slots[0].geometry().vertices().find(uipc_builtin.is_fixed)
        self.assertIsNotNone(fixed_attr)
        self.assertGreater(int(uipc.view(fixed_attr).sum()), 0)


if __name__ == "__main__":
    unittest.main()
