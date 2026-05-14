# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

_HAS_UIPC = importlib.util.find_spec("uipc") is not None

if _HAS_UIPC:
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


if __name__ == "__main__":
    unittest.main()
