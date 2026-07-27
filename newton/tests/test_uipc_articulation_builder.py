# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode, JointType

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
    def test_extract_limit_strength_per_joint_override(self):
        builder = self._make_builder(limit_strength_ratio={0: 345.0})

        self.assertEqual(builder._extract_limit_strength(0), 345.0)
        # joint 1 is not in the override mapping -> class default 10.0
        self.assertEqual(builder._extract_limit_strength(1), 10.0)

    def test_revolute_limit_strength_uses_limit_strength_ratio(self):
        captured = {}

        class _LimitConstitution:
            def apply_to(self, _geometry, _lowers, _uppers, strengths):
                captured["strengths"] = strengths.copy()

        builder = self._make_builder(limit_strength_ratio=3.5)

        # limit_ke=234 must not leak into the UIPC limit strength
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

        np.testing.assert_array_equal(captured["strengths"], np.array([3.5], dtype=np.float64))

    def test_prismatic_limit_strength_uses_limit_strength_ratio(self):
        captured = {}

        class _LimitConstitution:
            def apply_to(self, _geometry, _lowers, _uppers, strengths):
                captured["strengths"] = strengths.copy()

        builder = self._make_builder(limit_strength_ratio=6.5)

        # limit_ke=567 must not leak into the UIPC limit strength
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

        np.testing.assert_array_equal(captured["strengths"], np.array([6.5], dtype=np.float64))

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

    def test_implicit_pd_params_maps_physical_gains(self):
        builder = self._make_builder(dt=0.1, implicit_pd=True)
        model = self._make_single_dof_model(limit_ke=1.0, target_ke=720.0, target_kd=80.0, body_mass=[3.0, 5.0])

        strength, damp_blend = builder._implicit_pd_params(0, {"parent_body": 0, "child_body": 1}, model)

        # ratio_p = ke*dt^2/mass = 720*0.01/8 = 0.9; ratio_d = kd*dt/mass = 80*0.1/8 = 1.0
        self.assertAlmostEqual(strength, 1.9)
        self.assertAlmostEqual(damp_blend, 1.0 / 1.9)

    def test_implicit_pd_params_world_parent_uses_unit_proxy_mass(self):
        builder = self._make_builder(dt=0.1, implicit_pd=True)
        model = self._make_single_dof_model(limit_ke=1.0, target_ke=720.0, body_mass=[5.0])

        strength, damp_blend = builder._implicit_pd_params(0, {"parent_body": -1, "child_body": 0}, model)

        # world anchor proxy has unit mass: ratio_p = 720*0.01/(1+5) = 1.2, no damping
        self.assertAlmostEqual(strength, 1.2)
        self.assertEqual(damp_blend, 0.0)

    def test_implicit_pd_params_zero_gains_disable_drive(self):
        builder = self._make_builder(implicit_pd=True)
        model = self._make_single_dof_model(limit_ke=1.0, target_ke=0.0, target_kd=0.0, body_mass=[3.0, 5.0])

        self.assertEqual(builder._implicit_pd_params(0, {"parent_body": 0, "child_body": 1}, model), (0.0, 0.0))

    def test_implicit_pd_params_non_position_mode_disables_drive(self):
        builder = self._make_builder(implicit_pd=True)
        model = self._make_single_dof_model(
            limit_ke=1.0, target_ke=720.0, target_kd=80.0, body_mass=[3.0, 5.0], target_mode=int(JointTargetMode.EFFORT)
        )

        self.assertEqual(builder._implicit_pd_params(0, {"parent_body": 0, "child_body": 1}, model), (0.0, 0.0))

    def test_implicit_pd_params_velocity_mode_is_pure_damping_servo(self):
        builder = self._make_builder(dt=0.1, implicit_pd=True)
        model = self._make_single_dof_model(
            limit_ke=1.0,
            target_ke=720.0,  # must be ignored in VELOCITY mode
            target_kd=80.0,
            body_mass=[3.0, 5.0],
            target_mode=int(JointTargetMode.VELOCITY),
        )

        strength, damp_blend = builder._implicit_pd_params(0, {"parent_body": 0, "child_body": 1}, model)

        # ratio_d = kd*dt/mass = 80*0.1/8 = 1.0; ke does not contribute
        self.assertAlmostEqual(strength, 1.0)
        self.assertEqual(damp_blend, 1.0)

    def test_implicit_pd_params_velocity_mode_without_kd_disables_drive(self):
        builder = self._make_builder(implicit_pd=True)
        model = self._make_single_dof_model(
            limit_ke=1.0, target_ke=720.0, body_mass=[3.0, 5.0], target_mode=int(JointTargetMode.VELOCITY)
        )

        self.assertEqual(builder._implicit_pd_params(0, {"parent_body": 0, "child_body": 1}, model), (0.0, 0.0))

    @staticmethod
    def _make_builder(
        dt: float = 1.0 / 60.0,
        joint_strength_ratio: float = 100.0,
        drive_strength_ratio: float | dict[int, float] = 100.0,
        limit_strength_ratio: float | dict[int, float] = 10.0,
        implicit_pd: bool = False,
    ):
        builder = ArticulationBuilder.__new__(ArticulationBuilder)
        builder._scene = _FakeScene()
        builder._mapping = _FakeMapping()
        builder._dt = dt
        builder._joint_strength_ratio = joint_strength_ratio
        builder._drive_strength_ratio = drive_strength_ratio
        builder._limit_strength_ratio = limit_strength_ratio
        builder._implicit_pd = implicit_pd
        return builder

    @staticmethod
    def _make_single_dof_model(
        limit_ke: float,
        target_ke: float = 0.0,
        target_kd: float = 0.0,
        body_mass: list[float] | None = None,
        target_mode: int = int(JointTargetMode.POSITION),
        joint_type: int = int(JointType.REVOLUTE),
        armature: float = 0.0,
    ):
        class _Model:
            joint_count = 1
            joint_axis = _Array([[1.0, 0.0, 0.0]])
            joint_qd_start = _Array([0])
            joint_q_start = _Array([0])
            joint_q = None
            joint_target_ke = _Array([target_ke])
            joint_target_kd = _Array([target_kd])
            joint_target_mode = _Array([target_mode])
            joint_limit_lower = _Array([-0.5])
            joint_limit_upper = _Array([0.5])
            joint_limit_ke = _Array([limit_ke])

        _Model.joint_type = _Array([joint_type])
        _Model.joint_armature = _Array([armature])
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

    def test_revolute_without_armature_leaves_child_inertia_unchanged(self):
        """With no armature, the child body must take the default
        density-derived ABD path and reproduce the Newton-authored inertia."""
        model, solver, child = self._build_and_initialize(armature=None)

        model_inertia = model.body_inertia.numpy()[child].astype(np.float64)
        inertia = solver.read_uipc_body_inertia(child)["inertia"]
        np.testing.assert_allclose(inertia, model_inertia, rtol=1e-4, atol=1e-6)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCArmatureInertiaSyncSymmetry(unittest.TestCase):
    """``sync_model_inertia_from_uipc`` must subtract back exactly the armature
    inertia that ``build_affine_bodies`` folded in.

    Regression: a revolute child with a degenerate mesh (``mesh_vol <= 1e-12``)
    skips the fold, but the read-back used to subtract it unconditionally,
    driving ``body_inertia`` negative (diagonal ``≈ -a``) → StablePD Cholesky
    NaN. Fold and unfold now share one record, so an unfolded body is untouched.
    """

    def _build_initialize(self, half_extent: float, armature: float):
        """World-anchored revolute pendulum with a cubic child of the given
        half-extent; ``half_extent`` small enough drives ``mesh_vol`` below the
        ``1e-12`` custom-inertia gate."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        child = builder.add_link()
        builder.add_shape_box(child, hx=half_extent, hy=half_extent, hz=half_extent)
        joint = builder.add_joint_revolute(parent=-1, child=child, axis=newton.Axis.Z, armature=armature)
        builder.add_articulation([joint], label="armature_sync")
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model, backend="none", logger_level=uipc.Logger.Error, auto_sync_inertia=False
        )
        solver.initialize(model.state())
        return model, solver, child

    def test_degenerate_mesh_armature_does_not_produce_negative_inertia(self):
        """Degenerate-mesh revolute child: the build side skips the armature
        fold, so the sync side must not subtract it -- inertia stays PSD."""
        # half_extent 1e-5 -> mesh volume ~8e-15, three orders below the gate.
        model, solver, child = self._build_initialize(half_extent=1e-5, armature=1.0)
        solver.sync_model_inertia_from_uipc([child])

        inertia = model.body_inertia.numpy()[child].astype(np.float64)
        min_eig = float(np.linalg.eigvalsh(0.5 * (inertia + inertia.T)).min())
        self.assertGreaterEqual(
            min_eig,
            -1e-6,
            f"synced inertia is not PSD (min eigenvalue {min_eig}); armature was over-subtracted",
        )

    def test_normal_mesh_sync_roundtrip_restores_authored_inertia(self):
        """Non-degenerate revolute child: fold-then-unfold must cancel, so the
        synced inertia reproduces the Newton-authored inertia."""
        model, solver, child = self._build_initialize(half_extent=0.1, armature=0.4)
        authored = model.body_inertia.numpy()[child].astype(np.float64).copy()

        solver.sync_model_inertia_from_uipc([child])

        synced = model.body_inertia.numpy()[child].astype(np.float64)
        np.testing.assert_allclose(synced, authored, rtol=1e-4, atol=1e-6)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCImplicitPD(unittest.TestCase):
    """Physical gain semantics of ``SolverUIPC(implicit_pd=True)``.

    A 1 m rod pendulum hinged at the origin is held horizontal
    (``target_q = 0``) against gravity: the steady state must sag by
    ``tau_g / kp`` (P-gain semantics), and ``kd`` must damp the transient
    (D-gain semantics) — the aim-drive blend has no explicit damping
    channel, so this is the property that distinguishes implicit PD from
    the plain position drive.
    """

    @staticmethod
    def _run_pendulum(kp: float, kd: float, frames: int = 180, gravity_comp: bool = False):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.5, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
        j = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Y)
        builder.add_articulation([j])
        dof = builder.joint_qd_start[j]
        builder.joint_target_ke[dof] = kp
        builder.joint_target_kd[dof] = kd
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        model = builder.finalize()

        dt = 1.0 / 60.0
        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_implicit_pd_test",
            dt=dt,
            logger_level=uipc.Logger.Error,
            implicit_pd=True,
        )
        solver.sync_uipc_inertia_with_model()
        solver.initialize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        control.joint_target_q.fill_(0.0)
        gravity_force = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        coriolis_force = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)

        traj = []
        for _ in range(frames):
            if gravity_comp:
                # Pure position-domain compensation: offset the aim by
                # tau_g/ke (q_ref = 0), no force channel. eval_inverse_dynamics_passive
                # reads state.body_q, so refresh it from joint_q first.
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                newton.eval_inverse_dynamics_passive(
                    model, state_0, gravity_force=gravity_force, coriolis_force=coriolis_force
                )
                tau_ff = float(gravity_force.numpy()[0] + coriolis_force.numpy()[0])
                control.joint_target_q.fill_(tau_ff / kp)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            traj.append(float(state_0.joint_q.numpy()[0]))
        tau_g = float(model.body_mass.numpy()[link]) * 9.81 * 0.5
        return np.asarray(traj), tau_g

    def test_stiffness_sets_physical_steady_state_sag(self):
        traj, tau_g = self._run_pendulum(kp=200.0, kd=20.0)
        sag = tau_g / 200.0
        self.assertAlmostEqual(float(traj[-1]), sag, delta=0.05 * sag)

    def test_gravity_compensation_cancels_steady_state_sag(self):
        """A per-step aim offset of tau_g/ke (pure position control, no force
        channel) must cancel the tau_g/kp sag and land the joint on q_ref."""
        traj, _tau_g = self._run_pendulum(kp=200.0, kd=20.0, gravity_comp=True)
        self.assertLess(abs(float(traj[-1])), 0.005)

    def test_damping_removes_oscillation(self):
        undamped, tau_g = self._run_pendulum(kp=50.0, kd=0.0)
        damped, _ = self._run_pendulum(kp=50.0, kd=20.0)
        sag = tau_g / 50.0

        tail = slice(len(damped) * 2 // 3, None)
        tv_undamped = float(np.abs(np.diff(undamped[tail])).sum())
        tv_damped = float(np.abs(np.diff(damped[tail])).sum())
        self.assertLess(tv_damped, 0.05 * tv_undamped)
        # kd=0 rings past the steady state; kd=20 settles without overshoot.
        self.assertGreater(float(undamped.max()), 1.5 * sag)
        self.assertLess(float(damped.max()), 1.1 * sag)

    def test_velocity_mode_tracks_target_velocity(self):
        """VELOCITY mode must act as a velocity servo: a rotor spinning about
        a gravity-parallel axis (zero gravity torque) converges to the
        commanded joint velocity."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.5, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
        j = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        dof = builder.joint_qd_start[j]
        builder.joint_target_kd[dof] = 5.0
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.VELOCITY)
        model = builder.finalize()

        dt = 1.0 / 60.0
        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_velocity_servo_test",
            dt=dt,
            logger_level=uipc.Logger.Error,
            implicit_pd=True,
        )
        solver.sync_uipc_inertia_with_model()
        solver.initialize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        control.joint_target_qd.fill_(0.5)

        qd_hist = []
        for _ in range(120):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            qd_hist.append(float(state_0.joint_qd.numpy()[0]))

        # Converged and holding the commanded rate over the final second.
        tail = np.asarray(qd_hist[-60:])
        np.testing.assert_allclose(tail, 0.5, rtol=0.05)

    def test_notify_joint_dof_properties_refreshes_drive_strength(self):
        """Runtime ke/kd edits must re-derive the ``driving/strength_ratio``
        edge attribute and the aim-blend weight via notify_model_changed."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.5, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
        j = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Y)
        dof = builder.joint_qd_start[j]
        builder.joint_target_ke[dof] = 200.0
        builder.joint_target_kd[dof] = 20.0
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        model = builder.finalize()

        dt = 0.1
        solver = newton.solvers.SolverUIPC(
            model, backend="none", dt=dt, logger_level=uipc.Logger.Error, implicit_pd=True
        )
        solver.initialize()

        art_builder = solver._articulation_builder
        art = next(iter(art_builder.articulations.values()))
        local = art._joint_to_local[j]
        mass_sum = 1.0 + float(model.body_mass.numpy()[link])  # world anchor proxy + child

        def read_strength() -> float:
            geo = art.joint_geo_slots[j].geometry()
            attr = geo.edges().find("driving/strength_ratio")
            return float(uipc_articulation_builder._view_attr(attr)[art._joint_edge_idx[j]])

        self.assertAlmostEqual(read_strength(), (200.0 * dt * dt + 20.0 * dt) / mass_sum, places=6)
        self.assertIn(local, art.aim_blend_weights)

        ke_np = model.joint_target_ke.numpy()
        ke_np[dof] = 400.0
        model.joint_target_ke.assign(ke_np)
        kd_np = model.joint_target_kd.numpy()
        kd_np[dof] = 0.0
        model.joint_target_kd.assign(kd_np)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

        self.assertAlmostEqual(read_strength(), 400.0 * dt * dt / mass_sum, places=6)
        # kd -> 0: pure position spring, blend entry must be dropped.
        self.assertNotIn(local, art.aim_blend_weights)

    def test_runtime_gain_update_changes_steady_state(self):
        """A live ke change through notify_model_changed must move the
        gravity sag to tau_g / ke_new — the MJWarp-style DR gain path."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.5, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
        j = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Y)
        dof = builder.joint_qd_start[j]
        builder.joint_target_ke[dof] = 100.0
        builder.joint_target_kd[dof] = 20.0
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        model = builder.finalize()

        dt = 1.0 / 60.0
        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_gain_update_test",
            dt=dt,
            logger_level=uipc.Logger.Error,
            implicit_pd=True,
        )
        solver.sync_uipc_inertia_with_model()
        solver.initialize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        control.joint_target_q.fill_(0.0)
        tau_g = float(model.body_mass.numpy()[link]) * 9.81 * 0.5

        def run(frames: int) -> float:
            nonlocal state_0, state_1
            for _ in range(frames):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0
            return float(state_0.joint_q.numpy()[0])

        sag_soft = run(120)
        self.assertAlmostEqual(sag_soft, tau_g / 100.0, delta=0.05 * tau_g / 100.0)

        ke_np = model.joint_target_ke.numpy()
        ke_np[dof] = 400.0
        model.joint_target_ke.assign(ke_np)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

        sag_stiff = run(120)
        self.assertAlmostEqual(sag_stiff, tau_g / 400.0, delta=0.05 * tau_g / 400.0)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCPrismaticArmature(unittest.TestCase):
    """A PRISMATIC joint's ``joint_armature`` (reflected slider inertia) is
    applied as an implicit kinetic potential on the joint coordinate via
    libuipc's ``ExternalArticulationConstraint`` (see
    ``ArticulationBuilder._build_external_articulation``): mass = armature,
    aim = ``q_prev + dt*qd_prev`` (the gravity-free inertial prediction). It
    is exact and independent of the drive channel, so it applies to driven
    and passive (NONE/EFFORT) joints alike.
    """

    @staticmethod
    def _run_slider(kp: float, kd: float, armature: float | None, frames: int = 180):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05)
        j = builder.add_joint_prismatic(parent=-1, child=link, axis=newton.Axis.Z, armature=armature)
        dof = builder.joint_qd_start[j]
        builder.joint_target_ke[dof] = kp
        builder.joint_target_kd[dof] = kd
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        model = builder.finalize()

        dt = 1.0 / 60.0
        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_prismatic_armature_test",
            dt=dt,
            logger_level=uipc.Logger.Error,
            implicit_pd=True,
        )
        solver.sync_uipc_inertia_with_model()
        solver.initialize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        control.joint_target_q.fill_(0.0)

        traj = []
        for _ in range(frames):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            traj.append(float(state_0.joint_q.numpy()[0]))
        weight = float(model.body_mass.numpy()[link]) * 9.81
        return np.asarray(traj), weight

    def test_armature_leaves_steady_state_sag_unchanged(self):
        """Armature is absorbed via a gravity-free aim (q_prev + dt*qd_prev),
        so it must not shift the static force balance kp*sag = weight."""
        kp = 200.0
        no_armature, weight = self._run_slider(kp=kp, kd=20.0, armature=None)
        with_armature, _ = self._run_slider(kp=kp, kd=20.0, armature=5.0)

        # gravity is -Z; the spring pulls the slider back toward q_ref=0, so
        # the equilibrium displacement is negative.
        sag = -weight / kp
        self.assertAlmostEqual(float(no_armature[-1]), sag, delta=0.05 * abs(sag))
        self.assertAlmostEqual(float(with_armature[-1]), sag, delta=0.05 * abs(sag))

    def test_armature_slows_the_transient(self):
        """Armature adds apparent inertia to the drive channel: released from
        rest at the (unloaded) target, the armature-loaded slider must fall
        measurably less than the bare slider over the first few frames."""
        no_armature, _ = self._run_slider(kp=200.0, kd=0.0, armature=None, frames=6)
        with_armature, _ = self._run_slider(kp=200.0, kd=0.0, armature=20.0, frames=6)

        self.assertLess(abs(float(with_armature[3])), 0.8 * abs(float(no_armature[3])))

    @staticmethod
    def _run_passive_slider(armature: float | None, frames: int = 40):
        """Free slider under gravity with no drive (``target_mode`` NONE).

        This is the case the old aim-drive fold could not carry armature for
        (it only folded into the drive channel of POSITION/PD joints); the
        ExternalArticulationConstraint applies regardless of drive mode.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05)
        j = builder.add_joint_prismatic(parent=-1, child=link, axis=newton.Axis.Z, armature=armature)
        dof = builder.joint_qd_start[j]
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.NONE)
        model = builder.finalize()

        dt = 1.0 / 60.0
        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_passive_armature_test",
            dt=dt,
            logger_level=uipc.Logger.Error,
        )
        solver.sync_uipc_inertia_with_model()
        solver.initialize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()

        traj = []
        for _ in range(frames):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            traj.append(float(state_0.joint_q.numpy()[0]))
        return np.asarray(traj), float(model.body_mass.numpy()[link])

    def test_passive_armature_reduces_gravity_acceleration(self):
        """A passive (NONE-mode) prismatic joint receives armature through the
        ExternalArticulationConstraint — impossible with the old aim-drive
        fold. The added effective mass must slow free fall to
        ``a = g·m/(m + m_a)``."""
        g = 9.81
        dt = 1.0 / 60.0

        def accel(traj):
            # constant acceleration: q(t) = q0 + v0*t - ½ a*t²; the t² coeff is
            # -½ a. Full quadratic fit absorbs the first steps' solver transient.
            t = np.arange(len(traj)) * dt
            return -2.0 * float(np.polyfit(t, traj, 2)[0])

        no_arm, m_body = self._run_passive_slider(armature=None)
        m_a = 2.0 * m_body  # armature = 2x body mass -> a ~= g/3
        with_arm, _ = self._run_passive_slider(armature=m_a)

        self.assertAlmostEqual(accel(no_arm), g, delta=0.02 * g)
        expected = g * m_body / (m_body + m_a)
        self.assertAlmostEqual(accel(with_arm), expected, delta=0.05 * expected)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCRevoluteArmature(unittest.TestCase):
    """A REVOLUTE joint's ``joint_armature`` (reflected rotational inertia
    about the joint axis) is applied as an implicit kinetic potential on the
    joint coordinate via libuipc's ``ExternalArticulationConstraint`` (see
    ``ArticulationBuilder._build_external_articulation``) -- exactly like the
    PRISMATIC case in :class:`TestUIPCPrismaticArmature`, just with a
    rotational rather than a translational coordinate. This mirrors that
    class's passive (NONE-mode) test with a horizontal single pendulum:
    released from rest under gravity, its initial angular acceleration must
    drop from ``alpha0 = m*g*L/I_axis`` to ``alpha = m*g*L/(I_axis + a)``.
    """

    @staticmethod
    def _run_passive_pendulum(armature: float | None, frames: int = 8, arm_length: float = 0.5):
        """A world-anchored revolute joint about the world X axis with a
        cubic child link whose COM is offset by ``arm_length`` along Y
        (perpendicular to the axis): released from rest (horizontal), it
        swings like a single pendulum under gravity (-Z) with no drive
        (``target_mode`` NONE), so any armature must reach it purely through
        the ExternalArticulationConstraint -- the case the old aim-drive fold
        could not carry (see ``TestUIPCPrismaticArmature._run_passive_slider``).

        ``frames`` is kept small so the swing stays within a few degrees of
        horizontal: the pendulum's true torque is ``-m*g*L*cos(theta)``, only
        constant to good approximation near ``theta=0``, which is what lets
        the same full-trajectory quadratic fit used by the prismatic
        free-fall test recover the initial angular acceleration here.
        """
        hx = 0.05
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=hx, hy=hx, hz=hx, xform=wp.transform(wp.vec3(0.0, arm_length, 0.0)))
        j = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.X, armature=armature)
        dof = builder.joint_qd_start[j]
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.NONE)
        model = builder.finalize()

        dt = 1.0 / 60.0
        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_revolute_passive_armature_test",
            dt=dt,
            logger_level=uipc.Logger.Error,
        )
        solver.sync_uipc_inertia_with_model()
        solver.initialize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()

        traj = []
        for _ in range(frames):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            traj.append(float(state_0.joint_q.numpy()[0]))

        body_mass = float(model.body_mass.numpy()[link])
        body_inertia = model.body_inertia.numpy()[link].astype(np.float64)
        return np.asarray(traj), body_mass, body_inertia

    def test_passive_armature_reduces_angular_acceleration(self):
        """A passive (NONE-mode) revolute joint receives armature through the
        ExternalArticulationConstraint -- impossible with the old aim-drive
        fold. The added effective rotational inertia must slow the pendulum's
        initial angular acceleration to ``alpha = m*g*L/(I_axis + a)``."""
        dt = 1.0 / 60.0
        arm_length = 0.5
        armature = 0.5  # kg*m^2, same order of magnitude as I_axis (~0.25 kg*m^2)

        def angular_accel(traj):
            # constant angular acceleration near theta=0: joint_q(t) = q0 +
            # qd0*t + 0.5*qdd0*t^2; the t^2 coeff is 0.5*qdd0. Full quadratic
            # fit absorbs the first steps' solver transient (same trick as
            # TestUIPCPrismaticArmature's accel()).
            t = np.arange(len(traj)) * dt
            return -2.0 * float(np.polyfit(t, traj, 2)[0])

        no_arm, m_body, inertia_body = self._run_passive_pendulum(armature=None, arm_length=arm_length)
        with_arm, _, _ = self._run_passive_pendulum(armature=armature, arm_length=arm_length)

        alpha0 = angular_accel(no_arm)
        alpha_a = angular_accel(with_arm)

        # armature adds rotational inertia -> smaller angular acceleration
        # (theory ratio I_axis/(I_axis+a) ~= 0.33 here; 0.6 leaves headroom).
        self.assertGreater(alpha0, 0.0)
        self.assertLess(alpha_a, 0.6 * alpha0)

        # self-consistent inversion (does not depend on knowing I_axis up
        # front): alpha0/alpha_a = (I_axis+a)/I_axis => I_axis = a*alpha_a/(alpha0-alpha_a)
        inferred_i_axis = armature * alpha_a / (alpha0 - alpha_a)
        self.assertGreater(inferred_i_axis, 0.0)

        # cross-check against I_axis computed directly from the model via the
        # parallel-axis theorem: the joint axis (world X through the origin)
        # is parallel to, and offset by arm_length from, the body's own axis
        # through its COM.
        axis = np.array([1.0, 0.0, 0.0])
        i_axis_theory = float(axis @ inertia_body @ axis) + m_body * arm_length**2
        self.assertAlmostEqual(inferred_i_axis, i_axis_theory, delta=0.08 * i_axis_theory)

    def test_passive_armature_inference_is_consistent_across_magnitudes(self):
        """The I_axis inferred from the self-consistent inversion must agree
        whether probed with armature=a1 or armature=a2, confirming armature
        enters as an exact additive term (I_axis + a), not just "some
        slowdown"."""
        dt = 1.0 / 60.0
        arm_length = 0.5

        def angular_accel(traj):
            t = np.arange(len(traj)) * dt
            return -2.0 * float(np.polyfit(t, traj, 2)[0])

        no_arm, _, _ = self._run_passive_pendulum(armature=None, arm_length=arm_length)
        alpha0 = angular_accel(no_arm)

        a1, a2 = 0.25, 0.5
        traj_a1, _, _ = self._run_passive_pendulum(armature=a1, arm_length=arm_length)
        traj_a2, _, _ = self._run_passive_pendulum(armature=a2, arm_length=arm_length)

        i_axis_1 = a1 * angular_accel(traj_a1) / (alpha0 - angular_accel(traj_a1))
        i_axis_2 = a2 * angular_accel(traj_a2) / (alpha0 - angular_accel(traj_a2))

        self.assertGreater(i_axis_1, 0.0)
        self.assertAlmostEqual(i_axis_1, i_axis_2, delta=0.06 * i_axis_1)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCArmatureRuntimeRefresh(unittest.TestCase):
    """``SolverUIPC.notify_model_changed(JOINT_DOF_PROPERTIES)`` re-applies
    ``model.joint_armature`` to the live ExternalArticulationConstraint mass
    diagonal (``ArticulationBuilder.refresh_armature``); libuipc re-collects
    that attribute every step, so the new reflected inertia takes effect on
    the next ``world.advance()``. Joints with no armature at build time have
    no constraint edge, so a runtime enable warns and stays inert.
    """

    _DT = 1.0 / 60.0

    @classmethod
    def _make_passive_slider(cls, armature: float | None):
        """Passive (NONE-mode) vertical slider under gravity, as in
        ``TestUIPCPrismaticArmature._run_passive_slider``, but returning the
        live model/solver so the test can edit armature mid-run."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05)
        j = builder.add_joint_prismatic(parent=-1, child=link, axis=newton.Axis.Z, armature=armature)
        builder.joint_target_mode[builder.joint_qd_start[j]] = int(newton.JointTargetMode.NONE)
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_runtime_armature_test",
            dt=cls._DT,
            logger_level=uipc.Logger.Error,
        )
        solver.sync_uipc_inertia_with_model()
        solver.initialize()
        return model, solver, link

    @classmethod
    def _step_window(cls, model, solver, frames: int):
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        traj = []
        for _ in range(frames):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, cls._DT)
            state_0, state_1 = state_1, state_0
            traj.append(float(state_0.joint_q.numpy()[0]))
        return np.asarray(traj)

    @classmethod
    def _accel(cls, traj):
        # constant acceleration: the t^2 coefficient of q(t) is -a/2; the
        # quadratic fit absorbs the window's initial velocity and the
        # solver transient (same trick as TestUIPCPrismaticArmature).
        t = np.arange(len(traj)) * cls._DT
        return -2.0 * float(np.polyfit(t, traj, 2)[0])

    def test_notify_updates_free_fall_acceleration(self):
        """Free fall at ``a = g*m/(m + m_a)`` must track a mid-run armature
        edit: 2x body mass (a ~= g/3) rewritten to 0.5x (a ~= 2g/3) via
        ``notify_model_changed(JOINT_DOF_PROPERTIES)``."""
        g = 9.81
        model, solver, link = self._make_passive_slider(armature=1.0)  # placeholder, overwritten below
        m_body = float(model.body_mass.numpy()[link])

        # First window: armature = 2x body mass, set through the same
        # runtime path so the whole test exercises refresh_armature.
        model.joint_armature.assign(np.array([2.0 * m_body], dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        first = self._step_window(model, solver, frames=40)
        self.assertAlmostEqual(self._accel(first), g / 3.0, delta=0.05 * g / 3.0)

        # Second window: rewrite to 0.5x body mass mid-run.
        model.joint_armature.assign(np.array([0.5 * m_body], dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        second = self._step_window(model, solver, frames=40)
        expected = g * m_body / (m_body + 0.5 * m_body)
        self.assertAlmostEqual(self._accel(second), expected, delta=0.05 * expected)

    def test_enabling_armature_at_runtime_warns_and_stays_inert(self):
        """With no armature at build time there is no constraint edge:
        enabling it at runtime must warn once and leave free fall at ``g``."""
        g = 9.81
        model, solver, link = self._make_passive_slider(armature=None)
        m_body = float(model.body_mass.numpy()[link])

        model.joint_armature.assign(np.array([2.0 * m_body], dtype=np.float32))
        with self.assertWarnsRegex(UserWarning, "Recreate the solver"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

        traj = self._step_window(model, solver, frames=40)
        self.assertAlmostEqual(self._accel(traj), g, delta=0.02 * g)


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCCacheControlGraphCapture(unittest.TestCase):
    """``cache_joint_control`` must stay CUDA-graph capturable.

    The kernel + D2H mirror copies may not block-synchronize; the sync
    point is owned by ``SolverUIPC.step()`` (before the host-side mimic /
    animator consumers read the CPU arrays).
    """

    def test_cache_joint_control_inside_graph_capture(self):
        if not wp.get_device().is_cuda:
            self.skipTest("CUDA device required for graph capture")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.5, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.5, 0.0, 0.0)))
        j = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Y)
        builder.add_articulation([j])
        dof = builder.joint_qd_start[j]
        builder.joint_target_ke[dof] = 100.0
        builder.joint_target_kd[dof] = 1.0
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        model = builder.finalize()

        solver = newton.solvers.SolverUIPC(
            model,
            workspace="/tmp/newton_uipc_graph_capture_test",
            dt=1.0 / 60.0,
            logger_level=uipc.Logger.Error,
        )
        solver.initialize()
        art_builder = solver._articulation_builder

        control = model.control()
        control.joint_target_q.fill_(0.25)

        # Warm-up loads the kernel module and fills the _empty_f32 cache so
        # the capture below sees no allocations.
        art_builder.cache_joint_control(control)
        wp.synchronize_device()

        with wp.ScopedCapture() as capture:
            art_builder.cache_joint_control(control)
        wp.capture_launch(capture.graph)
        wp.synchronize_device()

        checked = 0
        for art in art_builder.articulations.values():
            if art.num_active_joints > 0:
                np.testing.assert_allclose(art.target_position.numpy(), 0.25)
                np.testing.assert_array_equal(art.is_constrained.numpy(), 1)
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
