# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation (joint) builder for the UIPC solver backend.

Creates :class:`Articulation` objects from a Newton :class:`Model`, builds
UIPC joint constitutions (revolute, prismatic, fixed, free, ball), and provides
the top-level ``cache_joint_control`` / ``write_joint_readback`` methods
consumed by :class:`SolverUIPC` each simulation step.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import uipc.builtin as uipc_builtin
import warp as wp
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyDrivingPrismaticJoint,
    AffineBodyDrivingRevoluteJoint,
    AffineBodyFixedJoint,
    AffineBodyPrismaticJoint,
    AffineBodyPrismaticJointExternalForce,
    AffineBodyPrismaticJointLimit,
    AffineBodyRevoluteJoint,
    AffineBodyRevoluteJointExternalForce,
    AffineBodyRevoluteJointLimit,
    AffineBodySphericalJoint,
    SoftTransformConstraint,
)
from uipc.core import Animation, Object
from uipc.geometry import SimplicialComplex, SimplicialComplexSlot
from uipc.unit import MPa

from newton._src.solvers.uipc.utils import _view_attr

from ...math import normalize_with_norm
from ...sim import Control, JointType, Model, State
from .articulation import Articulation
from .converter import UIpcMappingInfo, newton_transform_to_mat4


class ArticulationBuilder:
    """Build UIPC joint constitutions from Newton articulation joints.

    For each Newton articulation, an :class:`Articulation` runtime object is
    created to own the per-joint state, animation callbacks, and readback
    logic.  This builder handles only the **construction** phase:

    1. Group Newton joints by articulation index.
    2. Create UIPC geometry (linemesh) for each driven joint.
    3. Register UIPC Animator callbacks that delegate to the owning
       :class:`Articulation`.

    After :meth:`build_joints`, the builder exposes three methods that
    :class:`SolverUIPC` calls every step:

    - :meth:`cache_joint_control` — extract from Newton ``Control``.
    - :meth:`write_joint_readback` — write back to Newton ``State``.
    - :meth:`increment_step` — bump all articulation frame counters.
    """

    def __init__(
        self,
        model: Model,
        scene: Any,
        mapping: UIpcMappingInfo,
        dt: float,
        kappa: float = 100 * MPa,
    ) -> None:
        self._model = model
        self._scene = scene
        self._mapping = mapping
        self._dt = dt
        self._device = model.device
        self._abd = AffineBodyConstitution()
        self._kappa = kappa

        # Per-articulation runtime objects (populated by build_joints)
        self.articulations: dict[int, Articulation] = {}

        # Cache of proxy geo slots (world anchors + shapeless body proxies)
        self._proxy_slots: dict[str, SimplicialComplexSlot] = {}

        # Transient subscene element set per build_joints call
        self._subscene_elem: Any | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_joints(
        self,
        contact_elem: Any,
        joint_range: tuple[int, int],
        subscene_elem: Any | None = None,
    ) -> None:
        """Convert Newton joints to UIPC joint constitutions.

        Creates one :class:`Articulation` per Newton articulation, builds
        the UIPC geometry for each joint, and registers Animator callbacks.

        Joint world-space pivots and axes are computed directly from
        ``model.body_q`` and ``model.joint_X_p``.

        Args:
            contact_elem: Contact element for robot link geometries.
            joint_range: ``(start, end)`` slice of joints to process, or
                ``None`` for all joints.
            subscene_elem: UIPC subscene element for anchor bodies, or ``None``.
        """
        # Store for use by _create_proxy
        self._contact_elem = contact_elem
        self._subscene_elem = subscene_elem

        model = self._model
        if model.joint_count == 0:
            return

        # Validate required model arrays. Keep local bindings so type checkers
        # can narrow the Optional array attributes after the explicit guard.
        joint_type = model.joint_type
        joint_parent = model.joint_parent
        joint_child = model.joint_child
        joint_X_p = model.joint_X_p
        joint_X_c = model.joint_X_c
        joint_axis = model.joint_axis
        joint_q_start = model.joint_q_start
        joint_qd_start = model.joint_qd_start
        if (
            joint_type is None
            or joint_parent is None
            or joint_child is None
            or joint_X_p is None
            or joint_X_c is None
            or joint_axis is None
            or joint_q_start is None
            or joint_qd_start is None
        ):
            return

        jstart, jend = joint_range[0], joint_range[1]
        # Collect articulation indices referenced by joints in range
        joint_articulation = (
            model.joint_articulation.numpy()
            if model.joint_articulation is not None
            else np.zeros(model.joint_count, dtype=np.int32)
        )
        art_indices_in_range = set()
        for j in range(jstart, jend):
            art_indices_in_range.add(int(joint_articulation[j]))

        # Create Articulation objects for referenced articulations (skip existing)
        for a in art_indices_in_range:
            if a in self.articulations:
                continue
            label = (
                model.articulation_label[a]
                if (model.articulation_label and 0 <= a < len(model.articulation_label))
                else f"articulation_{a}"
            )
            self.articulations[a] = Articulation(name=label, dt=self._dt, device=self._device)

        # Pre-fetch numpy arrays
        joint_X_p_np = joint_X_p.numpy()
        joint_X_c_np = joint_X_c.numpy()
        joint_type_np = joint_type.numpy()
        joint_parent_np = joint_parent.numpy()
        joint_child_np = joint_child.numpy()

        # -- Pre-pass: create proxy meshes for shapeless bodies ----------------
        # Shapeless bodies (e.g. URDF frame-only links ``fr3_link8``) get
        # dynamic proxies at their world pose — Newton FIXED joints in the
        # model naturally constrain them to their neighbour.
        # Body -1 (world frame) is handled per-joint in the batch builders
        # and must NOT be registered in body_geo_slots (it has no entry in
        # model.body_q, so the GPU sync kernels would index out of bounds).
        for j in range(jstart, jend):
            if JointType(joint_type_np[j]) == JointType.FREE:
                continue
            for b in (int(joint_parent_np[j]), int(joint_child_np[j])):
                if b >= 0 and b not in self._mapping.body_geo_slots:
                    self._create_shapeless_proxy(b)

        # -- Classify joints by type and collect per-joint data ----------------
        revolute_joints: list[dict] = []
        prismatic_joints: list[dict] = []
        fixed_joints: list[dict] = []
        free_joints: list[dict] = []
        ball_joints: list[dict] = []

        for j in range(jstart, jend):
            joint_type = JointType(joint_type_np[j])
            parent_body = int(joint_parent_np[j])
            child_body = int(joint_child_np[j])

            # Check that both parent and child bodies have ABD geometry.
            # Body -1 (world frame) is exempt. FREE joints are exempt
            # (they represent floating root bodies managed separately).
            joint_name = model.joint_label[j] if j < len(model.joint_label) else "?"
            missing_geo = False

            if joint_type != JointType.FREE and child_body not in self._mapping.body_geo_slots:
                child_name = model.body_label[child_body] if child_body < len(model.body_label) else "?"
                warnings.warn(
                    f"Joint {j} ({joint_name}): child body {child_body} ({child_name}) has no ABD "
                    f"geometry (joint type {joint_type.name}); "
                    f"SolverUIPC is dropping this joint.",
                    stacklevel=2,
                )
                missing_geo = True

            if joint_type != JointType.FREE and parent_body >= 0 and parent_body not in self._mapping.body_geo_slots:
                parent_name = model.body_label[parent_body] if parent_body < len(model.body_label) else "?"
                warnings.warn(
                    f"Joint {j} ({joint_name}): parent body {parent_body} ({parent_name}) has no ABD "
                    f"geometry (joint type {joint_type.name}); "
                    f"SolverUIPC is dropping this joint.",
                    stacklevel=2,
                )
                missing_geo = True

            if missing_geo:
                continue

            child_slot = self._mapping.body_geo_slots[child_body]
            child_instance_id = self._mapping.body_instance_ids.get(child_body, 0)
            parent_slot = self._mapping.body_geo_slots.get(parent_body)
            parent_instance_id = self._mapping.body_instance_ids.get(parent_body, 0)

            # Joint anchor and rotation in parent-local and child-local frames
            jp = joint_X_p_np[j]
            parent_pivot = np.array(jp[:3], dtype=np.float64)
            parent_rot = newton_transform_to_mat4(wp.transform(jp[:3], jp[3:]))[:3, :3].copy()

            jc = joint_X_c_np[j]
            child_pivot = np.array(jc[:3], dtype=np.float64)
            child_rot = newton_transform_to_mat4(wp.transform(jc[:3], jc[3:]))[:3, :3].copy()

            # Resolve owning articulation
            art_idx = int(joint_articulation[j])
            if art_idx not in self.articulations:
                self.articulations[art_idx] = Articulation(
                    name=f"articulation_{art_idx}",
                    dt=self._dt,
                    device=self._device,
                )
            art = self.articulations[art_idx]

            jdata = {
                "j": j,
                "art": art,
                "parent_pivot": parent_pivot,
                "parent_rot": parent_rot,
                "child_pivot": child_pivot,
                "child_rot": child_rot,
                "parent_body": parent_body,
                "parent_slot": parent_slot,
                "parent_instance_id": parent_instance_id,
                "child_body": child_body,
                "child_slot": child_slot,
                "child_instance_id": child_instance_id,
            }

            if joint_type == JointType.REVOLUTE:
                revolute_joints.append(jdata)
            elif joint_type == JointType.PRISMATIC:
                prismatic_joints.append(jdata)
            elif joint_type == JointType.FIXED:
                fixed_joints.append(jdata)
            elif joint_type == JointType.FREE:
                free_joints.append(jdata)
            elif joint_type == JointType.BALL:
                ball_joints.append(jdata)
            elif joint_type in (JointType.DISTANCE, JointType.D6):
                warnings.warn(
                    f"Joint {j}: JointType {joint_type.name} is not yet supported by SolverUIPC, skipping",
                    stacklevel=2,
                )

        # -- Batch build each joint type -------------------------------------
        if revolute_joints:
            self._build_revolute_joints_batch(
                revolute_joints,
                model,
            )
        if prismatic_joints:
            self._build_prismatic_joints_batch(
                prismatic_joints,
                model,
            )
        if fixed_joints:
            self._build_fixed_joints_batch(fixed_joints)
        if ball_joints:
            self._build_ball_joints_batch(ball_joints, model)
        applied_free_joint_geometry_ids: set[int] = set()
        for jdata in free_joints:
            geometry = jdata["child_slot"].geometry()
            geometry_id = id(geometry)
            if geometry_id in applied_free_joint_geometry_ids:
                continue

            applied_free_joint_geometry_ids.add(geometry_id)
            stc = SoftTransformConstraint()
            stc.apply_to(geometry)

        # Finalise all articulations that have active joints
        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.setup_state()

        # Seed initial ``target_position`` from ``model.joint_q`` so the
        # first animator callback (which may fire inside ``world.init``
        # before :meth:`SolverUIPC.step` runs ``cache_control``) sees the
        # intended rest pose. Without this the prismatic gripper fingers
        # (or any joint with non-zero initial ``joint_q``) race toward
        # ``aim = 0`` and, combined with the ``-init_q`` edge offset,
        # snap to the fully-closed/zero configuration on step 0.
        if model.joint_q is not None:
            joint_q_np = model.joint_q.numpy()
            for art in self.articulations.values():
                if art.num_active_joints > 0:
                    art.seed_initial_targets(joint_q_np)

    # ------------------------------------------------------------------
    # Joint building helpers
    # ------------------------------------------------------------------

    def _create_proxy(
        self,
        name: str,
        transform: np.ndarray,
        *,
        is_fixed: bool = False,
    ) -> SimplicialComplexSlot:
        """Create a 1-vertex ABD proxy body.

        Used for two purposes:

        * **World anchors** (``is_fixed=True``): fixed proxy at the origin
          serving as the parent side of a world-attached joint constraint.
        * **Shapeless link proxies** (``is_fixed=False``): dynamic proxy at
          a shapeless body's pose, constrained to its neighbour by the
          model's own FIXED joints.

        Args:
            name: Unique name for the UIPC object.
            transform: 4x4 world-frame transform for the proxy.
            is_fixed: If ``True`` the proxy is marked kinematic.

        Returns:
            The UIPC geometry slot for the proxy body.
        """
        if name in self._proxy_slots:
            return self._proxy_slots[name]

        mass = 1.0
        mass_center = np.zeros(3, dtype=np.float64)
        inertia = np.eye(3, dtype=np.float64) * 1e-6
        volume = 1e-9
        sc = self._abd.create_proxy(self._kappa, mass, mass_center, inertia, volume)

        _view_attr(sc.transforms())[:] = transform

        if is_fixed:
            _view_attr(sc.instances().find(uipc_builtin.is_fixed))[:] = 1

        # Apply contact / subscene so the proxy participates in the same
        # contact group and subscene as other robot bodies.
        self._contact_elem.apply_to(sc)
        if self._subscene_elem is not None:
            self._subscene_elem.apply_to(sc)

        obj: Object = self._scene.objects().create(name)
        geo_slot: SimplicialComplexSlot = obj.geometries().create(sc)[0]
        self._proxy_slots[name] = geo_slot
        return geo_slot

    def _create_shapeless_proxy(self, body_idx: int) -> SimplicialComplexSlot:
        """Create a proxy for a shapeless body and register it in the mapping."""
        model = self._model
        if model.body_q is not None:
            bq = model.body_q.numpy()[body_idx]
            tf = newton_transform_to_mat4(wp.transform(bq[:3], bq[3:]))
        else:
            tf = np.eye(4, dtype=np.float64)

        geo_slot = self._create_proxy(f"shapeless_proxy_{body_idx}", tf)
        self._mapping.body_geo_slots[body_idx] = geo_slot
        self._mapping.body_instance_ids[body_idx] = 0
        return geo_slot

    def _build_revolute_joints_batch(
        self,
        joints: list[dict],
        model: Any,
    ) -> None:
        """Create all revolute joints in a single batched linemesh."""
        l_verts: list[np.ndarray] = []  # parent-side positions
        r_verts: list[np.ndarray] = []  # child-side positions
        parent_slots: list[SimplicialComplexSlot] = []
        parent_ids: list[int] = []
        child_slots: list[SimplicialComplexSlot] = []
        child_ids: list[int] = []
        strengths: list[float] = []
        drive_strengths: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        limit_strengths: list[float] = []
        init_angles: list[float] = []
        has_any_limit = False

        joint_axis = model.joint_axis
        joint_qd_start = model.joint_qd_start
        joint_q_start = model.joint_q_start
        if joint_axis is None or joint_qd_start is None or joint_q_start is None:
            return

        joint_axis_np = joint_axis.numpy()
        joint_qd_start_np = joint_qd_start.numpy()
        joint_q_start_np = joint_q_start.numpy()
        # Revolute-only: UIPC's ``angle`` edge attribute measures rotation
        # *relative to the build-time body configuration*, so we seed
        # ``init_angle`` with the build-time Newton angle ``joint_q`` so
        # that both the readback (``angle``) and the drive target
        # (``aim_angle``) operate in Newton's absolute joint-q space.
        # NOTE: the prismatic counterpart (``distance``) is already in
        # Newton absolute units — do NOT write ``init_distance``.
        joint_q_np = model.joint_q.numpy() if model.joint_q is not None else None

        # Dispatch list for animator callback: (art, newton_joint_idx, edge_idx)
        anim_dispatch: list[tuple[Articulation, int, int]] = []
        for edge_idx, jdata in enumerate(joints):
            j: int = jdata["j"]
            art: Articulation = jdata["art"]
            parent_pivot: np.ndarray = jdata["parent_pivot"]
            parent_rot: np.ndarray = jdata["parent_rot"]
            child_pivot: np.ndarray = jdata["child_pivot"]
            child_rot: np.ndarray = jdata["child_rot"]

            p_slot: SimplicialComplexSlot | None = jdata["parent_slot"]
            p_id: int = jdata["parent_instance_id"]
            c_slot: SimplicialComplexSlot = jdata["child_slot"]
            c_id: int = jdata["child_instance_id"]

            if p_slot is None:
                p_slot = self._create_proxy("world_anchor", np.eye(4, dtype=np.float64), is_fixed=True)
                p_id = 0

            qd_start = int(joint_qd_start_np[j])
            axis_joint = joint_axis_np[qd_start]
            parent_axis = parent_rot @ axis_joint
            child_axis = child_rot @ axis_joint
            q_start = int(joint_q_start_np[j])

            lp0 = parent_pivot
            lp1 = parent_pivot + parent_axis
            rp0 = child_pivot
            rp1 = child_pivot + child_axis

            self._validate_revolute_anchors(
                j,
                p_slot,
                p_id,
                c_slot,
                c_id,
                lp0,
                lp1,
                rp0,
                rp1,
            )

            l_verts.append(lp0)
            l_verts.append(lp1)
            r_verts.append(rp0)
            r_verts.append(rp1)

            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            child_slots.append(c_slot)
            child_ids.append(c_id)
            target_ke = self._extract_target_strength(j, model.joint_qd_start, model.joint_target_ke)
            strengths.append(target_ke)
            drive_strengths.append(target_ke)

            # Limits
            lower, upper = self._extract_limits(
                j,
                model.joint_qd_start,
                model.joint_limit_lower,
                model.joint_limit_upper,
            )
            if lower is not None and upper is not None:
                lowers.append(lower)
                uppers.append(upper)
                limit_strengths.append(self._extract_limit_strength(j, model.joint_qd_start, model.joint_limit_ke))
                has_any_limit = True
            else:
                lowers.append(-1e18)
                uppers.append(1e18)
                limit_strengths.append(self._extract_limit_strength(j, model.joint_qd_start, model.joint_limit_ke))

            init_angles.append(float(joint_q_np[q_start]) if joint_q_np is not None else 0.0)
            art.register_joint(j, q_start, qd_start)
            anim_dispatch.append((art, j, edge_idx))

        # Build batched linemesh via create_geometry (4-position overload)
        l_pos0s = np.array(l_verts[0::2], dtype=np.float64)
        l_pos1s = np.array(l_verts[1::2], dtype=np.float64)
        r_pos0s = np.array(r_verts[0::2], dtype=np.float64)
        r_pos1s = np.array(r_verts[1::2], dtype=np.float64)
        jm = AffineBodyRevoluteJoint().create_geometry(
            l_pos0s,
            l_pos1s,
            r_pos0s,
            r_pos1s,
            parent_slots,
            np.array(parent_ids, dtype=np.int32),
            child_slots,
            np.array(child_ids, dtype=np.int32),
            np.array(strengths, dtype=np.float64),
        )
        AffineBodyDrivingRevoluteJoint().apply_to(
            jm,
            np.array(drive_strengths, dtype=np.float64),
        )
        AffineBodyRevoluteJointExternalForce().apply_to(jm)
        if has_any_limit:
            AffineBodyRevoluteJointLimit().apply_to(
                jm,
                np.array(lowers, dtype=np.float64),
                np.array(uppers, dtype=np.float64),
                np.array(limit_strengths, dtype=np.float64),
            )

        # Shift UIPC's delta-from-rest ``angle`` into Newton's absolute
        # joint-q space. Only applied when model.joint_q is populated;
        # otherwise UIPC's default (init_angle = 0) preserves the raw
        # delta-from-rest semantics. NOT applied to prismatic — its
        # ``distance`` is already absolute.
        if joint_q_np is not None:
            init_angles_np = np.array(init_angles, dtype=np.float64)
            init_angle_view: np.ndarray = _view_attr(jm.edges().find("init_angle"))
            init_angle_view[:] = init_angles_np

        jobj: Object = self._scene.objects().create("joints_revolute")
        jslot: SimplicialComplexSlot = jobj.geometries().create(jm)[0]

        # Record mappings for each joint
        for art, j, edge_idx in anim_dispatch:
            art.joint_geo_slots[j] = jslot
            art.joint_mesh[j] = jm
            art._joint_edge_idx[j] = edge_idx
            art._joint_is_revolute[j] = True
            self._mapping.joint_geo_slots[j] = jslot
            self._mapping.joint_mesh[j] = jm

        # Single animator callback dispatching to all revolute joints.
        # ``anim_dispatch`` is a local list that is no longer mutated
        # after this point, so the closure can capture it by reference.
        def _revolute_batch_anim(info: Animation.UpdateInfo) -> None:
            try:
                geo: SimplicialComplex = info.geo_slots()[0].geometry()
            except (TypeError, IndexError):
                return
            for art, newton_j, edge_idx in anim_dispatch:
                art.revolute_joint_anim(info, geo, newton_j, edge_idx)

        self._scene.animator().insert(jobj, _revolute_batch_anim)

    def _build_prismatic_joints_batch(
        self,
        joints: list[dict],
        model: Any,
    ) -> None:
        """Create all prismatic joints in a single batched linemesh."""
        l_verts: list[np.ndarray] = []  # parent-side positions
        r_verts: list[np.ndarray] = []  # child-side positions
        parent_slots: list[SimplicialComplexSlot] = []
        parent_ids: list[int] = []
        child_slots: list[SimplicialComplexSlot] = []
        child_ids: list[int] = []
        strengths: list[float] = []
        drive_strengths: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        limit_strengths: list[float] = []
        has_any_limit = False

        joint_axis = model.joint_axis
        joint_qd_start = model.joint_qd_start
        joint_q_start = model.joint_q_start
        if joint_axis is None or joint_qd_start is None or joint_q_start is None:
            return

        joint_axis_np = joint_axis.numpy()
        joint_qd_start_np = joint_qd_start.numpy()
        joint_q_start_np = joint_q_start.numpy()

        anim_dispatch: list[tuple[Articulation, int, int]] = []
        for edge_idx, jdata in enumerate(joints):
            j: int = jdata["j"]
            art: Articulation = jdata["art"]
            parent_pivot: np.ndarray = jdata["parent_pivot"]
            parent_rot: np.ndarray = jdata["parent_rot"]
            child_pivot: np.ndarray = jdata["child_pivot"]
            child_rot: np.ndarray = jdata["child_rot"]

            p_slot: SimplicialComplexSlot | None = jdata["parent_slot"]
            p_id: int = jdata["parent_instance_id"]
            c_slot: SimplicialComplexSlot = jdata["child_slot"]
            c_id: int = jdata["child_instance_id"]

            if p_slot is None:
                p_slot = self._create_proxy("world_anchor", np.eye(4, dtype=np.float64), is_fixed=True)
                p_id = 0

            qd_start = int(joint_qd_start_np[j])
            axis_joint = joint_axis_np[qd_start]
            parent_axis = parent_rot @ axis_joint
            child_axis = child_rot @ axis_joint
            q_start = int(joint_q_start_np[j])

            lp0 = parent_pivot
            lp1 = parent_pivot + parent_axis
            rp0 = child_pivot
            rp1 = child_pivot + child_axis
            self._validate_prismatic_anchors(
                j,
                p_slot,
                p_id,
                c_slot,
                c_id,
                lp0,
                lp1,
                rp0,
                rp1,
            )

            l_verts.append(lp0)
            l_verts.append(lp1)
            r_verts.append(rp0)
            r_verts.append(rp1)

            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            child_slots.append(c_slot)
            child_ids.append(c_id)
            target_ke = self._extract_target_strength(j, model.joint_qd_start, model.joint_target_ke)
            strengths.append(target_ke)
            drive_strengths.append(target_ke)

            # Limits
            lower, upper = self._extract_limits(
                j,
                model.joint_qd_start,
                model.joint_limit_lower,
                model.joint_limit_upper,
            )
            if lower is not None and upper is not None:
                lowers.append(lower)
                uppers.append(upper)
                limit_strengths.append(self._extract_limit_strength(j, model.joint_qd_start, model.joint_limit_ke))
                has_any_limit = True
            else:
                lowers.append(-1e18)
                uppers.append(1e18)
                limit_strengths.append(self._extract_limit_strength(j, model.joint_qd_start, model.joint_limit_ke))

            art.register_joint(j, q_start, qd_start)
            anim_dispatch.append((art, j, edge_idx))

        # Build batched linemesh via create_geometry (4-position overload)
        l_pos0s = np.array(l_verts[0::2], dtype=np.float64)
        l_pos1s = np.array(l_verts[1::2], dtype=np.float64)
        r_pos0s = np.array(r_verts[0::2], dtype=np.float64)
        r_pos1s = np.array(r_verts[1::2], dtype=np.float64)
        jm = AffineBodyPrismaticJoint().create_geometry(
            l_pos0s,
            l_pos1s,
            r_pos0s,
            r_pos1s,
            parent_slots,
            np.array(parent_ids, dtype=np.int32),
            child_slots,
            np.array(child_ids, dtype=np.int32),
            np.array(strengths, dtype=np.float64),
        )
        AffineBodyDrivingPrismaticJoint().apply_to(
            jm,
            np.array(drive_strengths, dtype=np.float64),
        )
        AffineBodyPrismaticJointExternalForce().apply_to(jm)
        if has_any_limit:
            AffineBodyPrismaticJointLimit().apply_to(
                jm,
                np.array(lowers, dtype=np.float64),
                np.array(uppers, dtype=np.float64),
                np.array(limit_strengths, dtype=np.float64),
            )

        jobj: Object = self._scene.objects().create("joints_prismatic")
        jslot: SimplicialComplexSlot = jobj.geometries().create(jm)[0]

        for art, j, edge_idx in anim_dispatch:
            art.joint_geo_slots[j] = jslot
            art.joint_mesh[j] = jm
            art._joint_edge_idx[j] = edge_idx
            art._joint_is_revolute[j] = False
            self._mapping.joint_geo_slots[j] = jslot
            self._mapping.joint_mesh[j] = jm

        def _prismatic_batch_anim(info: Animation.UpdateInfo) -> None:
            try:
                geo: SimplicialComplex = info.geo_slots()[0].geometry()
            except (TypeError, IndexError):
                return
            for art, newton_j, edge_idx in anim_dispatch:
                art.prismatic_joint_anim(info, geo, newton_j, edge_idx)

        self._scene.animator().insert(jobj, _prismatic_batch_anim)

    def _build_fixed_joints_batch(
        self,
        joints: list[dict],
    ) -> None:
        """Create all fixed joints in a single batched pointcloud."""
        # Separate world-anchored (no parent) from inter-body fixed joints
        l_positions: list[np.ndarray] = []
        r_positions: list[np.ndarray] = []
        child_slots: list[SimplicialComplexSlot] = []
        child_ids: list[int] = []
        parent_slots: list[SimplicialComplexSlot] = []
        parent_ids: list[int] = []
        strengths: list[float] = []
        joint_indices: list[int] = []

        for jdata in joints:
            j: int = jdata["j"]
            parent_body: int = jdata["parent_body"]
            parent_pivot: np.ndarray = jdata["parent_pivot"]
            child_pivot: np.ndarray = jdata["child_pivot"]
            p_slot: SimplicialComplexSlot | None = jdata["parent_slot"]
            p_id: int = jdata["parent_instance_id"]
            c_slot: SimplicialComplexSlot = jdata["child_slot"]
            c_id: int = jdata["child_instance_id"]

            if parent_body == -1:
                # World-attached FIXED joint → just pin the child directly.
                _view_attr(c_slot.geometry().instances().find(uipc_builtin.is_fixed))[c_id] = 1
                continue

            l_positions.append(parent_pivot)
            r_positions.append(child_pivot)
            child_slots.append(c_slot)
            child_ids.append(c_id)
            if p_slot is None:
                raise RuntimeError(f"Missing parent geometry slot for fixed joint {j}.")
            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            strengths.append(100.0)
            joint_indices.append(j)

        if not child_slots:
            return

        jm = AffineBodyFixedJoint().create_geometry(
            np.array(l_positions, dtype=np.float64),
            np.array(r_positions, dtype=np.float64),
            parent_slots,
            np.array(parent_ids, dtype=np.int32),
            child_slots,
            np.array(child_ids, dtype=np.int32),
            np.array(strengths, dtype=np.float64),
        )

        jobj: Object = self._scene.objects().create("joints_fixed")
        jslot: SimplicialComplexSlot = jobj.geometries().create(jm)[0]
        for j in joint_indices:
            self._mapping.joint_geo_slots[j] = jslot
            self._mapping.joint_mesh[j] = jm

    def _build_ball_joints_batch(
        self,
        joints: list[dict],
        model: Any,
    ) -> None:
        """Create all spherical (ball) joints in a single batched pointcloud."""
        parent_slots: list[SimplicialComplexSlot] = []
        parent_ids: list[int] = []
        child_slots: list[SimplicialComplexSlot] = []
        child_ids: list[int] = []
        l_positions: list[np.ndarray] = []
        r_positions: list[np.ndarray] = []
        strengths: list[float] = []
        joint_indices: list[int] = []

        joint_X_p = model.joint_X_p
        joint_X_c = model.joint_X_c
        if joint_X_p is None:
            return

        joint_X_p_np = joint_X_p.numpy()
        joint_X_c_np = joint_X_c.numpy() if joint_X_c is not None else None

        for jdata in joints:
            j: int = jdata["j"]
            p_slot: SimplicialComplexSlot | None = jdata["parent_slot"]
            p_id: int = jdata["parent_instance_id"]
            c_slot: SimplicialComplexSlot = jdata["child_slot"]
            c_id: int = jdata["child_instance_id"]

            if p_slot is None:
                p_slot = self._create_proxy("world_anchor", np.eye(4, dtype=np.float64), is_fixed=True)
                p_id = 0

            # Parent-side local anchor (joint_X_p translation)
            l_pos = np.array(joint_X_p_np[j][:3], dtype=np.float64)

            # Child-side local anchor (joint_X_c translation)
            if joint_X_c_np is not None:
                r_pos = np.array(joint_X_c_np[j][:3], dtype=np.float64)
            else:
                r_pos = np.zeros(3, dtype=np.float64)

            self._validate_ball_anchors(j, p_slot, p_id, c_slot, c_id, l_pos, r_pos)

            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            child_slots.append(c_slot)
            child_ids.append(c_id)
            l_positions.append(l_pos)
            r_positions.append(r_pos)
            strengths.append(100.0)
            joint_indices.append(j)

        jm = AffineBodySphericalJoint().create_geometry(
            np.array(l_positions, dtype=np.float64),
            np.array(r_positions, dtype=np.float64),
            parent_slots,
            np.array(parent_ids, dtype=np.int32),
            child_slots,
            np.array(child_ids, dtype=np.int32),
            np.array(strengths, dtype=np.float64),
        )

        jobj: Object = self._scene.objects().create("joints_ball")
        jslot: SimplicialComplexSlot = jobj.geometries().create(jm)[0]
        for j in joint_indices:
            self._mapping.joint_geo_slots[j] = jslot
            self._mapping.joint_mesh[j] = jm

    @staticmethod
    def _validate_revolute_anchors(
        joint_idx: int,
        p_slot: SimplicialComplexSlot,
        p_id: int,
        c_slot: SimplicialComplexSlot,
        c_id: int,
        lp0: np.ndarray,
        lp1: np.ndarray,
        rp0: np.ndarray,
        rp1: np.ndarray,
        atol: float = 1e-4,
    ) -> None:
        """Validate revolute joint: anchors and axis endpoints must coincide.

        Args:
            joint_idx: Newton joint index (for error messages).
            p_slot: Parent geometry slot.
            p_id: Parent instance index.
            c_slot: Child geometry slot.
            c_id: Child instance index.
            lp0: Parent-local anchor position (pos0).
            lp1: Parent-local axis endpoint (pos1).
            rp0: Child-local anchor position (pos0).
            rp1: Child-local axis endpoint (pos1).
            atol: Absolute tolerance for the comparison.

        Raises:
            RuntimeError: If the world-space positions do not match.
        """
        p_tf: np.ndarray = _view_attr(p_slot.geometry().transforms())[p_id]
        c_tf: np.ndarray = _view_attr(c_slot.geometry().transforms())[c_id]

        def to_world(tf: np.ndarray, p: np.ndarray) -> np.ndarray:
            return (tf @ np.append(p, 1.0))[:3]

        l_world_0: np.ndarray = to_world(p_tf, lp0)
        r_world_0: np.ndarray = to_world(c_tf, rp0)
        if not np.allclose(l_world_0, r_world_0, atol=atol):
            raise RuntimeError(
                f"Revolute joint {joint_idx}: parent/child anchor "
                f"mismatch in world space.\n"
                f"p_tf={p_tf}\n, c_tf={c_tf}\n, lp0={lp0}, rp0={rp0},\n "
                f"l_world={l_world_0}, r_world={r_world_0}, "
                f"diff={l_world_0 - r_world_0}"
            )

        l_world_1: np.ndarray = to_world(p_tf, lp1)
        r_world_1: np.ndarray = to_world(c_tf, rp1)
        if not np.allclose(l_world_1, r_world_1, atol=atol):
            raise RuntimeError(
                f"Revolute joint {joint_idx}: parent/child axis "
                f"endpoint mismatch in world space.\n"
                f"p_tf={p_tf}\n, c_tf={c_tf}\n, lp1={lp1}, rp1={rp1},\n "
                f"l_world={l_world_1}, r_world={r_world_1}, "
                f"diff={l_world_1 - r_world_1:6}"
            )

    @staticmethod
    def _validate_prismatic_anchors(
        joint_idx: int,
        p_slot: SimplicialComplexSlot,
        p_id: int,
        c_slot: SimplicialComplexSlot,
        c_id: int,
        lp0: np.ndarray,
        lp1: np.ndarray,
        rp0: np.ndarray,
        rp1: np.ndarray,
        atol: float = 1e-4,
    ) -> None:
        """Validate prismatic joint: axes must be parallel and anchors collinear.

        Unlike revolute joints, prismatic anchors need not coincide — they
        only need to lie on the same sliding axis.

        Args:
            joint_idx: Newton joint index (for error messages).
            p_slot: Parent geometry slot.
            p_id: Parent instance index.
            c_slot: Child geometry slot.
            c_id: Child instance index.
            lp0: Parent-local anchor position (pos0).
            lp1: Parent-local axis endpoint (pos1).
            rp0: Child-local anchor position (pos0).
            rp1: Child-local axis endpoint (pos1).
            atol: Absolute tolerance for the comparison.

        Raises:
            RuntimeError: If axes are not parallel or anchors not collinear.
        """
        p_tf: np.ndarray = _view_attr(p_slot.geometry().transforms())[p_id]
        c_tf: np.ndarray = _view_attr(c_slot.geometry().transforms())[c_id]

        def to_world(tf: np.ndarray, p: np.ndarray) -> np.ndarray:
            return (tf @ np.append(p, 1.0))[:3]

        l_world_0: np.ndarray = to_world(p_tf, lp0)
        l_world_1: np.ndarray = to_world(p_tf, lp1)
        r_world_0: np.ndarray = to_world(c_tf, rp0)
        r_world_1: np.ndarray = to_world(c_tf, rp1)

        # Axes must be parallel: cross product ≈ 0
        l_axis_u, _ = normalize_with_norm(wp.vec3d(*(l_world_1 - l_world_0)))
        r_axis_u, _ = normalize_with_norm(wp.vec3d(*(r_world_1 - r_world_0)))
        l_axis = np.asarray(l_axis_u, dtype=np.float64)
        r_axis = np.asarray(r_axis_u, dtype=np.float64)
        cross = np.cross(l_axis, r_axis)
        if not np.allclose(cross, 0.0, atol=atol):
            raise RuntimeError(
                f"Prismatic joint {joint_idx}: parent/child axes not parallel. "
                f"l_axis={l_axis}, r_axis={r_axis}, "
                f"cross={cross}"
            )

        # Anchors must be collinear along the axis: perpendicular offset ≈ 0
        offset = r_world_0 - l_world_0
        perp = offset - np.dot(offset, l_axis) * l_axis
        if not np.allclose(perp, 0.0, atol=atol):
            raise RuntimeError(
                f"Prismatic joint {joint_idx}: parent/child anchors not collinear. "
                f"l_world={l_world_0}, r_world={r_world_0}, "
                f"perp_offset={perp}, dist={np.linalg.norm(perp):.6f}"
            )

    @staticmethod
    def _validate_ball_anchors(
        joint_idx: int,
        p_slot: SimplicialComplexSlot,
        p_id: int,
        c_slot: SimplicialComplexSlot,
        c_id: int,
        l_pos: np.ndarray,
        r_pos: np.ndarray,
        atol: float = 1e-4,
    ) -> None:
        """Validate ball joint: anchor points must coincide in world space.

        Args:
            joint_idx: Newton joint index (for error messages).
            p_slot: Parent geometry slot.
            p_id: Parent instance index.
            c_slot: Child geometry slot.
            c_id: Child instance index.
            l_pos: Parent-local anchor position.
            r_pos: Child-local anchor position.
            atol: Absolute tolerance for the comparison.

        Raises:
            RuntimeError: If the world-space positions do not match.
        """
        p_tf: np.ndarray = _view_attr(p_slot.geometry().transforms())[p_id]
        c_tf: np.ndarray = _view_attr(c_slot.geometry().transforms())[c_id]

        def to_world(tf: np.ndarray, p: np.ndarray) -> np.ndarray:
            return (tf @ np.append(p, 1.0))[:3]

        l_world: np.ndarray = to_world(p_tf, l_pos)
        r_world: np.ndarray = to_world(c_tf, r_pos)
        if not np.allclose(l_world, r_world, atol=atol):
            raise RuntimeError(
                f"Ball joint {joint_idx}: parent/child anchor "
                f"mismatch in world space.\n"
                f"p_tf={p_tf}\n, c_tf={c_tf}\n, "
                f"l_pos={l_pos}, r_pos={r_pos},\n"
                f"l_world={l_world}, r_world={r_world}, "
                f"diff={np.linalg.norm(l_world - r_world):.6f}"
            )

    @staticmethod
    def _extract_limits(
        j: int,
        joint_qd_start: wp.array,
        joint_limit_lower: wp.array | None,
        joint_limit_upper: wp.array | None,
    ) -> tuple[float | None, float | None]:
        """Extract joint limits from model arrays.

        Args:
            j: Newton joint index.
            joint_qd_start: Joint DOF start indices (limits are per-DOF).
            joint_limit_lower: Lower limit array, shape ``[joint_dof_count]``, or ``None``.
            joint_limit_upper: Upper limit array, shape ``[joint_dof_count]``, or ``None``.

        Returns:
            ``(lower, upper)`` floats, either or both may be ``None``
            if no limit is defined.
        """
        qd_start = int(joint_qd_start.numpy()[j])
        lower = float(joint_limit_lower.numpy()[qd_start]) if joint_limit_lower is not None else None
        upper = float(joint_limit_upper.numpy()[qd_start]) if joint_limit_upper is not None else None
        return lower, upper

    @staticmethod
    def _extract_limit_strength(
        j: int,
        joint_qd_start: wp.array,
        joint_limit_ke: wp.array | None,
    ) -> float:
        """Extract UIPC joint-limit strength from Newton's per-DOF limit stiffness."""
        if joint_limit_ke is None:
            return 100.0
        qd_start = int(joint_qd_start.numpy()[j])
        return float(joint_limit_ke.numpy()[qd_start])

    @staticmethod
    def _extract_target_strength(
        j: int,
        joint_qd_start: wp.array,
        joint_target_ke: wp.array | None,
    ) -> float:
        """Extract UIPC joint drive strength from Newton's per-DOF target stiffness."""
        if joint_target_ke is None:
            return 100.0
        qd_start = int(joint_qd_start.numpy()[j])
        val = float(joint_target_ke.numpy()[qd_start])
        return val if val else 100.0

    # ------------------------------------------------------------------
    # Per-step interface (called by SolverUIPC.step)
    # ------------------------------------------------------------------

    def cache_joint_control(self, control: Control) -> None:
        """Cache Newton control values for all articulations.

        Extracts target positions, velocities, and forces from the Newton
        :class:`Control` object and distributes them to each
        :class:`Articulation`.

        Args:
            control: The Newton control input for this step.
        """
        model = self._model
        if model.joint_count == 0 or not self.articulations:
            return
        if model.joint_type is None or model.joint_q_start is None or model.joint_qd_start is None:
            return

        if model.joint_target_mode is None:
            return

        # Model + control arrays stay on the solver device — the
        # cache_control kernel runs on ``self._device``.
        joint_type = model.joint_type.to(self._device)
        joint_target_mode = model.joint_target_mode.to(self._device)
        target_pos = control.joint_target_pos.to(self._device) if control.joint_target_pos is not None else None
        target_vel = control.joint_target_vel.to(self._device) if control.joint_target_vel is not None else None
        joint_f = control.joint_f.to(self._device) if control.joint_f is not None else None

        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.cache_control(
                    joint_type,
                    joint_target_mode,
                    target_pos,
                    target_vel,
                    joint_f,
                )

        # Each Articulation.cache_control wp.copy's the device-side
        # output buffers into the CPU arrays consumed by the UIPC
        # animation callbacks. Those copies are async on the device
        # stream, so synchronise once before world.advance() runs the
        # animator on the host side.
        wp.synchronize_device(self._device)

    def read_joint_state_pre_advance(self) -> None:
        """Snapshot pre-advance edge attributes on each articulation.

        Call **once per step, before** ``world.advance()`` so each
        :class:`Articulation` records the start-of-step ``angle`` /
        ``distance`` for finite-difference velocity recovery in
        :meth:`read_joint_state_post_retrieve`.
        """
        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.read_pre_advance()

    def read_joint_state_post_retrieve(self) -> None:
        """Re-read UIPC edge attributes after ``world.retrieve()``.

        Pairs with :meth:`read_joint_state_pre_advance`: each articulation
        finite-differences the pre-advance and post-retrieve angle /
        distance to update ``joint_position`` and ``joint_velocity``,
        which the subsequent :meth:`write_joint_readback` consumes.
        """
        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.read_post_retrieve()

    def write_joint_readback(self, state_out: State) -> None:
        """Write cached joint readback values to Newton state arrays.

        Args:
            state_out: The output state to write joint positions and
                velocities into.
        """
        model = self._model
        if model.joint_count == 0 or not self.articulations:
            return
        if model.joint_q_start is None or model.joint_qd_start is None:
            return

        if state_out.joint_q is None:
            return

        # The scatter kernel writes directly into the solver-device
        # joint_q / joint_qd buffers — no CPU round trip required.
        joint_q = state_out.joint_q.to(self._device)
        joint_qd = state_out.joint_qd.to(self._device) if state_out.joint_qd is not None else None

        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.write_readback(joint_q, joint_qd)

        # If .to() returned a fresh allocation (i.e. the original lived
        # on a different device) propagate the result back.
        if joint_q is not state_out.joint_q:
            wp.copy(state_out.joint_q, joint_q)
        if joint_qd is not None and state_out.joint_qd is not None and joint_qd is not state_out.joint_qd:
            wp.copy(state_out.joint_qd, joint_qd)

    def increment_step(self) -> None:
        """Increment the step counter on all articulations."""
        for art in self.articulations.values():
            art.increment_step()
