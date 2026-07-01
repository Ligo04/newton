# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

_HAS_UIPC = importlib.util.find_spec("uipc") is not None

if _HAS_UIPC:
    import uipc

    import newton._src.solvers.uipc.articulation_builder as uipc_articulation_builder
    from newton._src.solvers.uipc.articulation_builder import ArticulationBuilder


class _Array:
    def __init__(self, values):
        self._values = np.array(values)

    def numpy(self):
        return self._values


class _FakeArticulation:
    def __init__(self):
        self.joint_geo_slots = {}
        self.joint_mesh = {}
        self._joint_edge_idx = {}
        self._joint_is_revolute = {}

    def register_joint(self, _joint_idx, _q_start, _qd_start):
        pass

    def revolute_joint_anim(self, _info, _geo, _newton_j, _edge_idx):
        pass

    def prismatic_joint_anim(self, _info, _geo, _newton_j, _edge_idx):
        pass


class _FakeMapping:
    def __init__(self):
        self.joint_geo_slots = {}
        self.joint_mesh = {}


class _FakeGeometryCollection:
    def create(self, _geometry):
        return [object()]


class _FakeObject:
    def geometries(self):
        return _FakeGeometryCollection()


class _FakeObjects:
    def create(self, _name):
        return _FakeObject()


class _FakeAnimator:
    def insert(self, _object, _callback):
        pass


class _FakeScene:
    def objects(self):
        return _FakeObjects()

    def animator(self):
        return _FakeAnimator()


class _FakeJointConstitution:
    def create_geometry(self, *_args):
        return object()


class _NoOpConstitution:
    def apply_to(self, *_args):
        pass


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCArticulationBuilder(unittest.TestCase):
    def test_extract_limit_strength_uses_joint_limit_ke_dof(self):
        joint_qd_start = wp.array([0, 2], dtype=int, device="cpu")
        joint_limit_ke = wp.array([10.0, 20.0, 345.0], dtype=float, device="cpu")

        strength = ArticulationBuilder._extract_limit_strength(1, joint_qd_start, joint_limit_ke)

        self.assertEqual(strength, 345.0)

    def test_revolute_limit_strength_uses_joint_limit_ke(self):
        captured = {}

        class _LimitConstitution:
            def apply_to(self, _geometry, _lowers, _uppers, strengths):
                captured["strengths"] = strengths.copy()

        builder = ArticulationBuilder.__new__(ArticulationBuilder)
        builder._scene = _FakeScene()
        builder._mapping = _FakeMapping()

        model = self._make_single_dof_model(limit_ke=234.0)
        joints = [self._make_joint_data()]

        with (
            patch.object(uipc_articulation_builder, "AffineBodyRevoluteJoint", return_value=_FakeJointConstitution()),
            patch.object(uipc_articulation_builder, "AffineBodyDrivingRevoluteJoint", return_value=_NoOpConstitution()),
            patch.object(
                uipc_articulation_builder, "AffineBodyRevoluteJointExternalForce", return_value=_NoOpConstitution()
            ),
            patch.object(uipc_articulation_builder, "AffineBodyRevoluteJointLimit", return_value=_LimitConstitution()),
            patch.object(ArticulationBuilder, "_validate_revolute_anchors", return_value=None),
        ):
            builder._build_revolute_joints_batch(joints, model)

        np.testing.assert_array_equal(captured["strengths"], np.array([234.0], dtype=np.float64))

    def test_prismatic_limit_strength_uses_joint_limit_ke(self):
        captured = {}

        class _LimitConstitution:
            def apply_to(self, _geometry, _lowers, _uppers, strengths):
                captured["strengths"] = strengths.copy()

        builder = ArticulationBuilder.__new__(ArticulationBuilder)
        builder._scene = _FakeScene()
        builder._mapping = _FakeMapping()

        model = self._make_single_dof_model(limit_ke=567.0)
        joints = [self._make_joint_data()]

        with (
            patch.object(uipc_articulation_builder, "AffineBodyPrismaticJoint", return_value=_FakeJointConstitution()),
            patch.object(
                uipc_articulation_builder, "AffineBodyDrivingPrismaticJoint", return_value=_NoOpConstitution()
            ),
            patch.object(
                uipc_articulation_builder, "AffineBodyPrismaticJointExternalForce", return_value=_NoOpConstitution()
            ),
            patch.object(uipc_articulation_builder, "AffineBodyPrismaticJointLimit", return_value=_LimitConstitution()),
            patch.object(ArticulationBuilder, "_validate_prismatic_anchors", return_value=None),
        ):
            builder._build_prismatic_joints_batch(joints, model)

        np.testing.assert_array_equal(captured["strengths"], np.array([567.0], dtype=np.float64))

    @staticmethod
    def _make_single_dof_model(limit_ke: float):
        class _Model:
            joint_axis = _Array([[1.0, 0.0, 0.0]])
            joint_qd_start = _Array([0])
            joint_q_start = _Array([0])
            joint_q = None
            joint_target_ke = _Array([0.0])
            joint_limit_lower = _Array([-0.5])
            joint_limit_upper = _Array([0.5])
            joint_limit_ke = _Array([limit_ke])

        return _Model()

    @staticmethod
    def _make_joint_data():
        return {
            "j": 0,
            "art": _FakeArticulation(),
            "parent_pivot": np.zeros(3, dtype=np.float64),
            "parent_rot": np.eye(3, dtype=np.float64),
            "child_pivot": np.zeros(3, dtype=np.float64),
            "child_rot": np.eye(3, dtype=np.float64),
            "parent_slot": object(),
            "parent_instance_id": 0,
            "child_slot": object(),
            "child_instance_id": 0,
        }


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCShapelessProxyInertia(unittest.TestCase):
    def test_shapeless_proxy_uses_authored_mass_com_inertia(self):
        """A shapeless link's ABD proxy must carry the Newton-authored mass
        properties, not the historical hardcoded ``mass=1.0`` / zero COM."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        # link0 has a shape -> real geometry ABD body.
        link0 = builder.add_link(mass=1.0)
        builder.add_shape_box(link0, hx=0.05, hy=0.05, hz=0.05)
        # link1 has NO shape -> shapeless ABD proxy; nonzero COM offset and a
        # non-identity inertia so a hardcoded default would be clearly wrong.
        link1 = builder.add_link(
            com=wp.vec3(0.1, 0.0, 0.0),
            inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.05),
            mass=2.5,
        )
        j0 = builder.add_joint_revolute(parent=-1, child=link0, axis=newton.Axis.Z)
        j1 = builder.add_joint_fixed(
            parent=link0,
            child=link1,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1], label="arm")
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model, backend="none", logger_level=uipc.Logger.Error, auto_sync_inertia=False
        )
        solver.initialize(model.state())

        # link1 is shapeless -> mapped to an ABD proxy.
        self.assertIn(link1, solver.mapping.body_geo_slots)

        model_mass = float(model.body_mass.numpy()[link1])
        model_com = model.body_com.numpy()[link1].astype(np.float64)
        model_inertia = model.body_inertia.numpy()[link1].astype(np.float64)
        self.assertGreater(model_mass, 1.5)  # guard: authored mass must differ from the 1.0 default

        props = solver.read_uipc_body_inertia(link1)
        self.assertAlmostEqual(props["mass"], model_mass, places=5)
        np.testing.assert_allclose(props["mass_center"], model_com, atol=1e-6)
        np.testing.assert_allclose(props["inertia"], model_inertia, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
