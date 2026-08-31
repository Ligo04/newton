# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC cloth material and soft-position controls."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")

if _HAS_UIPC:
    import uipc.builtin as uipc_builtin
    from uipc import view
    from uipc.constitution import NeoHookeanShell, StrainLimitingBaraffWitkinShell

    import newton._src.solvers.uipc.cloth as cloth_module
    from newton._src.solvers.uipc.cloth import ClothBuilder


class TestUIPCClothConfiguration(unittest.TestCase):
    @unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
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
    def test_vertex_thickness_scales_volume_from_applied_default(self):
        class _Vertices:
            def __init__(self):
                self.attrs = {
                    "thickness": np.zeros(3, dtype=np.float64),
                    "volume": np.full(3, 2.0, dtype=np.float64),
                }

            def find(self, name):
                return self.attrs.get(name)

        class _Geometry:
            def __init__(self):
                self._vertices = _Vertices()

            def vertices(self):
                return self._vertices

        geo = _Geometry()
        thickness_values = np.array([0.0005, 0.001, 0.002], dtype=np.float64)
        original_view_attr = cloth_module._view_attr
        try:
            cloth_module._view_attr = lambda attr: attr
            ClothBuilder._write_vertex_thickness(geo, thickness_values, applied_thickness=0.001)
        finally:
            cloth_module._view_attr = original_view_attr

        np.testing.assert_allclose(geo.vertices().find("thickness"), thickness_values)
        np.testing.assert_allclose(geo.vertices().find("volume"), np.array([1.0, 2.0, 4.0]))


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCClothSoftPosition(unittest.TestCase):
    def test_disconnected_cloth_grids_use_authored_ranges(self):
        """Build one UIPC cloth geometry per authored custom-frequency row."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        for i in range(2):
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, i * 0.4, 0.0),
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
                label=f"cloth_{i}",
            )

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        self.assertEqual(model.custom_frequency_counts["uipc:cloth"], 2)
        np.testing.assert_array_equal(model.uipc.cloth_particle_first.numpy(), [0, 9])
        np.testing.assert_array_equal(model.uipc.cloth_particle_last.numpy(), [8, 17])
        self.assertEqual(model.uipc.cloth_label, ["cloth_0", "cloth_1"])
        self.assertEqual(len(solver.mapping.cloth_geo_slots), 2)
        self.assertEqual(len(solver.mapping.cloth_particle_indices), 2)

    @staticmethod
    def _add_minimal_cloth(builder: newton.ModelBuilder) -> None:
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=0.0,
        )

    def test_register_custom_attributes_adds_default_cloth_model(self):
        """Materialize default constitutions and range metadata for cloth rows."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        self._add_minimal_cloth(builder)
        self._add_minimal_cloth(builder)

        model = builder.finalize()

        self.assertTrue(hasattr(model, "uipc"))
        self.assertTrue(hasattr(model.uipc, "cloth_model"))
        self.assertEqual(model.custom_frequency_counts["uipc:cloth"], 2)
        self.assertEqual(model.uipc.cloth_model, ["strain_limiting_baraff_witkin", "strain_limiting_baraff_witkin"])
        np.testing.assert_array_equal(model.uipc.cloth_triangle_first.numpy(), [0, 2])
        np.testing.assert_array_equal(model.uipc.cloth_triangle_last.numpy(), [1, 3])

    def test_late_registration_materializes_cloth_group_rows(self):
        """Resolve cloth rows when SolverUIPC metadata is registered after authoring."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        self._add_minimal_cloth(builder)
        self._add_minimal_cloth(builder)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)

        model = builder.finalize()

        self.assertEqual(model.custom_frequency_counts["uipc:cloth"], 2)
        np.testing.assert_array_equal(model.uipc.cloth_particle_first.numpy(), [0, 4])
        np.testing.assert_array_equal(model.uipc.cloth_particle_last.numpy(), [3, 7])

    def test_replicated_cloth_group_rows_are_offset_and_prefixed(self):
        """Preserve cloth ranges, worlds, and labels through batched replication."""
        source = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        source.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            label="sheet",
        )
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        builder.replicate(source, 2, label_prefixes=["left", "right"])

        model = builder.finalize()

        self.assertEqual(model.uipc.cloth_label, ["left/sheet", "right/sheet"])
        np.testing.assert_array_equal(model.uipc.cloth_world.numpy(), [0, 1])
        np.testing.assert_array_equal(model.uipc.cloth_particle_first.numpy(), [0, 4])
        np.testing.assert_array_equal(model.uipc.cloth_particle_last.numpy(), [3, 7])

    def test_cloth_model_custom_attribute_selects_per_range_constitution(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        self._add_minimal_cloth(builder)
        self._add_minimal_cloth(builder)

        model = builder.finalize()
        model.uipc.cloth_model[1] = "neo_hookean"
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        geo_0 = solver.mapping.cloth_geo_slots[0].geometry()
        geo_1 = solver.mapping.cloth_geo_slots[1].geometry()
        self.assertEqual(
            int(view(geo_0.meta().find("constitution_uid"))[0]),
            StrainLimitingBaraffWitkinShell().uid(),
        )
        self.assertEqual(int(view(geo_1.meta().find("constitution_uid"))[0]), NeoHookeanShell().uid())

    def test_fixed_cloth_grid_edges_mark_uipc_vertices_fixed(self):
        dim_x = 4
        dim_y = 3
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=0.0,
            fix_left=True,
            fix_right=True,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(
            model,
            backend="none",
            dt=1.0 / 60.0,
            enable_soft_position_constraint=False,
        )
        solver.initialize(model.state())

        geo = solver.mapping.cloth_geo_slots[0].geometry()
        fixed_attr = geo.vertices().find(uipc_builtin.is_fixed)
        self.assertIsNotNone(fixed_attr)
        fixed = np.asarray(view(fixed_attr), dtype=np.int32).reshape(-1)

        expected_fixed = np.zeros((dim_y + 1, dim_x + 1), dtype=np.int32)
        expected_fixed[:, 0] = 1
        expected_fixed[:, dim_x] = 1
        np.testing.assert_array_equal(fixed, expected_fixed.reshape(-1))

    def test_fixed_cloth_grid_edges_remain_fixed_after_step(self):
        if not _CUDA_TEST_DEVICES:
            self.skipTest("Requires a CUDA test device")

        dim_x = 4
        dim_y = 3
        with wp.ScopedDevice(_CUDA_TEST_DEVICES[0]):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 0.0, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.01,
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=0.0,
                fix_left=True,
                fix_right=True,
            )
            model = builder.finalize()
            state_0 = model.state()
            state_1 = model.state()
            solver = newton.solvers.SolverUIPC(
                model,
                backend="cuda",
                dt=1.0 / 60.0,
                enable_soft_position_constraint=False,
            )
            solver.initialize(state_0)
            initial_q = state_0.particle_q.numpy()

            solver.step(state_0, state_1, model.control(), model.contacts(), 1.0 / 60.0)
            final_q = state_1.particle_q.numpy()

        fixed_mask = np.zeros((dim_y + 1, dim_x + 1), dtype=bool)
        fixed_mask[:, 0] = True
        fixed_mask[:, dim_x] = True
        fixed_mask = fixed_mask.reshape(-1)

        np.testing.assert_allclose(final_q[fixed_mask], initial_q[fixed_mask], atol=1.0e-5)
        self.assertLess(float(np.min(final_q[~fixed_mask, 2] - initial_q[~fixed_mask, 2])), -1.0e-5)

    def test_particle_radius_sets_uipc_thickness(self):
        """Use authored surface density together with per-particle thickness."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
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
            particle_radius=2.5e-4,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        geo = solver.mapping.cloth_geo_slots[0].geometry()
        mass_density_attr = geo.meta().find("mass_density")
        self.assertIsNotNone(mass_density_attr)
        self.assertAlmostEqual(float(view(mass_density_attr)[0]), 9000.0, places=3)
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

    def test_triangle_materials_are_copied_per_triangle(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        for vertex in [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.1, 0.0, 0.0),
            wp.vec3(0.1, 0.1, 0.0),
            wp.vec3(0.0, 0.1, 0.0),
        ]:
            builder.add_particle(vertex, wp.vec3(0.0, 0.0, 0.0), 0.01)

        tri_ke = [2.0e3, 8.0e3]
        tri_ka = [3.0e3, 9.0e3]
        builder.add_triangles(
            [0, 0],
            [1, 2],
            [2, 3],
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=[0.0, 0.0],
        )
        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        geo = solver.mapping.cloth_geo_slots[0].geometry()
        mu_attr = geo.triangles().find("mu")
        lambda_attr = geo.triangles().find("lambda")
        self.assertIsNotNone(mu_attr)
        self.assertIsNotNone(lambda_attr)
        np.testing.assert_allclose(view(mu_attr), np.asarray(tri_ke, dtype=np.float32), rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(view(lambda_attr), np.asarray(tri_ka, dtype=np.float32), rtol=0.0, atol=1.0e-5)

    def test_bending_stiffness_is_copied_per_edge(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        for vertex in [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.1, 0.0, 0.0),
            wp.vec3(0.2, 0.0, 0.0),
            wp.vec3(0.0, 0.1, 0.0),
            wp.vec3(0.1, 0.1, 0.0),
            wp.vec3(0.2, 0.1, 0.0),
        ]:
            builder.add_particle(vertex, wp.vec3(0.0, 0.0, 0.0), 0.01)

        builder.add_triangles(
            [0, 1, 1, 2],
            [1, 4, 2, 5],
            [3, 3, 4, 4],
            tri_ke=[1.0e3] * 4,
            tri_ka=[1.0e3] * 4,
            tri_kd=[0.0] * 4,
        )
        expected_by_edge = {
            (1, 3): 2.5e-2,
            (1, 4): 7.5e-2,
            (2, 4): 1.25e-1,
        }
        builder.add_edge(0, 4, 1, 3, edge_ke=expected_by_edge[(1, 3)], edge_kd=0.0)
        builder.add_edge(3, 2, 1, 4, edge_ke=expected_by_edge[(1, 4)], edge_kd=0.0)
        builder.add_edge(1, 5, 2, 4, edge_ke=expected_by_edge[(2, 4)], edge_kd=0.0)

        model = builder.finalize()
        solver = newton.solvers.SolverUIPC(model, backend="none", dt=1.0 / 60.0)
        solver.initialize(model.state())

        geo = solver.mapping.cloth_geo_slots[0].geometry()
        bending_attr = geo.edges().find("bending_stiffness")
        self.assertIsNotNone(bending_attr)
        edges = np.asarray(view(geo.edges().topo()), dtype=np.int32).reshape(-1, 2)
        bending_by_edge = {
            tuple(sorted((int(edge[0]), int(edge[1])))): float(value)
            for edge, value in zip(edges, view(bending_attr), strict=True)
        }

        for edge, expected in expected_by_edge.items():
            self.assertIn(edge, bending_by_edge)
            self.assertAlmostEqual(bending_by_edge[edge], expected)

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
        np.testing.assert_allclose(thickness, np.full(model.particle_count, 0.1), rtol=1.0e-6)

        solver.clear_cloth_soft_position_constraints([0])
        self.assertEqual(int(constrained[0]), 0)


if __name__ == "__main__":
    unittest.main()
