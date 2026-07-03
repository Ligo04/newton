# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode

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

        builder = self._make_builder()

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

        builder = self._make_builder()

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

    def test_revolute_strengths_decoupled_from_target_ke(self):
        """Constraint strength must come from joint_strength_ratio; drive
        strength must come from drive_strength_ratio, not joint_target_ke."""
        captured = {}

        class _JointConstitution:
            def create_geometry(self, *args):
                captured["strengths"] = args[-1].copy()
                return object()

        class _DrivingConstitution:
            def apply_to(self, _geometry, strengths):
                captured["drive_strengths"] = strengths.copy()

        builder = self._make_builder(dt=0.1, joint_strength_ratio=42.0, drive_strength_ratio=7.5)

        model = self._make_single_dof_model(limit_ke=1.0, target_ke=720.0, body_mass=[3.0, 5.0])
        joints = [self._make_joint_data(parent_body=0, child_body=1)]

        with (
            patch.object(uipc_articulation_builder, "AffineBodyRevoluteJoint", return_value=_JointConstitution()),
            patch.object(
                uipc_articulation_builder, "AffineBodyDrivingRevoluteJoint", return_value=_DrivingConstitution()
            ),
            patch.object(
                uipc_articulation_builder, "AffineBodyRevoluteJointExternalForce", return_value=_NoOpConstitution()
            ),
            patch.object(uipc_articulation_builder, "AffineBodyRevoluteJointLimit", return_value=_NoOpConstitution()),
            patch.object(ArticulationBuilder, "_validate_revolute_anchors", return_value=None),
        ):
            builder._build_revolute_joints_batch(joints, model)

        np.testing.assert_array_equal(captured["strengths"], np.array([42.0], dtype=np.float64))
        # drive strength is the solver knob verbatim — ke=720 must not leak in
        np.testing.assert_allclose(captured["drive_strengths"], np.array([7.5], dtype=np.float64))

    def test_extract_drive_strength_per_joint_override(self):
        builder = self._make_builder(drive_strength_ratio={0: 250.0})
        model = self._make_single_dof_model(limit_ke=1.0, target_ke=720.0)

        self.assertEqual(builder._extract_drive_strength(0, model), 250.0)

    def test_extract_drive_strength_dict_falls_back_to_default(self):
        builder = self._make_builder(drive_strength_ratio={3: 250.0})
        model = self._make_single_dof_model(limit_ke=1.0)

        # joint 0 is not in the override mapping -> class default 100.0
        self.assertEqual(builder._extract_drive_strength(0, model), 100.0)

    def test_extract_drive_strength_non_position_mode_disables_drive(self):
        builder = self._make_builder(drive_strength_ratio=250.0)
        for mode in (JointTargetMode.NONE, JointTargetMode.VELOCITY, JointTargetMode.EFFORT):
            model = self._make_single_dof_model(limit_ke=1.0, target_ke=720.0, target_mode=int(mode))
            self.assertEqual(builder._extract_drive_strength(0, model), 0.0)

    def test_extract_drive_strength_position_velocity_mode_drives(self):
        builder = self._make_builder(drive_strength_ratio=250.0)
        model = self._make_single_dof_model(limit_ke=1.0, target_mode=int(JointTargetMode.POSITION_VELOCITY))

        self.assertEqual(builder._extract_drive_strength(0, model), 250.0)

    @staticmethod
    def _make_builder(
        dt: float = 1.0 / 60.0,
        joint_strength_ratio: float = 100.0,
        drive_strength_ratio: float | dict[int, float] = 100.0,
    ):
        builder = ArticulationBuilder.__new__(ArticulationBuilder)
        builder._scene = _FakeScene()
        builder._mapping = _FakeMapping()
        builder._dt = dt
        builder._joint_strength_ratio = joint_strength_ratio
        builder._drive_strength_ratio = drive_strength_ratio
        return builder

    @staticmethod
    def _make_single_dof_model(
        limit_ke: float,
        target_ke: float = 0.0,
        body_mass: list[float] | None = None,
        target_mode: int = int(JointTargetMode.POSITION),
    ):
        class _Model:
            joint_axis = _Array([[1.0, 0.0, 0.0]])
            joint_qd_start = _Array([0])
            joint_q_start = _Array([0])
            joint_q = None
            joint_target_ke = _Array([target_ke])
            joint_target_mode = _Array([target_mode])
            joint_limit_lower = _Array([-0.5])
            joint_limit_upper = _Array([0.5])
            joint_limit_ke = _Array([limit_ke])

        _Model.body_mass = _Array(body_mass) if body_mass is not None else None
        return _Model()

    @staticmethod
    def _make_joint_data(parent_body: int = -1, child_body: int = 0):
        return {
            "j": 0,
            "art": _FakeArticulation(),
            "parent_pivot": np.zeros(3, dtype=np.float64),
            "parent_rot": np.eye(3, dtype=np.float64),
            "child_pivot": np.zeros(3, dtype=np.float64),
            "child_rot": np.eye(3, dtype=np.float64),
            "parent_body": parent_body,
            "parent_slot": object(),
            "parent_instance_id": 0,
            "child_body": child_body,
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


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCRevoluteArmatureInertia(unittest.TestCase):
    """A REVOLUTE joint's ``joint_armature`` must fold into the child ABD
    body's inertia as ``a * axis ⊗ axis`` (see ``_armature_rotational_inertia``
    in ``rigid_body.py``), auto-selecting the custom-inertia path with no
    explicit ``sync_uipc_inertia_with_model`` call required."""

    def _build_and_initialize(self, armature: float | None):
        """A world-anchored revolute pendulum with a boxed child link.

        The child's mass/inertia come entirely from the box shape's default
        density (no extra point mass from ``add_link``): UIPC's default,
        non-custom-inertia path rebuilds inertia from
        ``mass_density = body_mass / mesh_volume``, so an added point mass
        at the COM would inflate ``body_mass`` without changing
        ``body_inertia`` (a point mass at the COM has no rotational
        inertia), breaking that reconstruction independently of armature.
        ``child_xform`` is left at its identity default, so the joint's
        parent-anchor axis (``newton.Axis.Z``) equals the child body-frame
        axis directly -- no extra rotation to account for when checking the
        folded inertia.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        child = builder.add_link()
        builder.add_shape_box(child, hx=0.1, hy=0.08, hz=0.06)
        joint = builder.add_joint_revolute(parent=-1, child=child, axis=newton.Axis.Z, armature=armature)
        builder.add_articulation([joint], label="armature_pendulum")
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model, backend="none", logger_level=uipc.Logger.Error, auto_sync_inertia=False
        )
        solver.initialize(model.state())
        return model, solver, child

    def test_revolute_armature_adds_axis_outer_product_to_child_inertia(self):
        """With armature set, UIPC's ABD inertia must equal the Newton-authored
        child inertia plus ``armature * axis ⊗ axis``."""
        armature = 0.4
        model, solver, child = self._build_and_initialize(armature)

        axis = np.array(newton.Axis.Z.to_vector(), dtype=np.float64)
        model_inertia = model.body_inertia.numpy()[child].astype(np.float64)
        expected = model_inertia + armature * np.outer(axis, axis)

        inertia = solver.read_uipc_body_inertia(child)["inertia"]
        np.testing.assert_allclose(inertia, expected, rtol=1e-4, atol=1e-6)

    def test_revolute_without_armature_leaves_child_inertia_unchanged(self):
        """With no armature, the child body must take the default
        density-derived ABD path and reproduce the Newton-authored inertia."""
        model, solver, child = self._build_and_initialize(armature=None)

        model_inertia = model.body_inertia.numpy()[child].astype(np.float64)
        inertia = solver.read_uipc_body_inertia(child)["inertia"]
        np.testing.assert_allclose(inertia, model_inertia, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
