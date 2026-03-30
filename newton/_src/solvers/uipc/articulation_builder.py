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
from uipc import view
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
from uipc.geometry import linemesh, pointcloud
from uipc.geometry import trimesh as uipc_trimesh
from uipc.unit import MPa

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
        contact_elem: Any | None = None,
        kappa: float = 100 * MPa,
    ) -> None:
        self._model = model
        self._scene = scene
        self._mapping = mapping
        self._dt = dt
        self._abd = AffineBodyConstitution()
        self._contact_elem = contact_elem
        self._kappa = kappa

        # Per-articulation runtime objects (populated by build_joints)
        self.articulations: dict[int, Articulation] = {}

        # Cache of anchor body geo slots for world-anchored joints
        self._anchor_slots: dict[str, Any] = {}

        # Transient subscene element set per build_joints call
        self._subscene_elem: Any | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_joints(
        self,
        body_transforms: np.ndarray | None,
        joint_range: tuple[int, int] | None = None,
        subscene_elem: Any | None = None,
    ) -> None:
        """Convert Newton joints to UIPC joint constitutions.

        Creates one :class:`Articulation` per Newton articulation, builds
        the UIPC geometry for each joint, and registers Animator callbacks.

        Args:
            body_transforms: Body world-frame transforms from the rigid-body
                builder, shape ``(body_count, 4, 4)``.
            joint_range: ``(start, end)`` slice of joints to process, or
                ``None`` for all joints.
            subscene_elem: UIPC subscene element for anchor bodies, or ``None``.
        """
        # Store subscene for use by _get_or_create_anchor
        self._subscene_elem = subscene_elem

        model = self._model
        if model.joint_count == 0:
            return

        # Validate required model arrays
        required = (
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_axis,
            model.joint_q_start,
            model.joint_qd_start,
        )
        if any(a is None for a in required):
            return

        # Determine joint iteration range
        jstart, jend = joint_range if joint_range else (0, model.joint_count)

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
            self.articulations[a] = Articulation(name=label, dt=self._dt)

        # Process each joint in range
        state = model.state()
        joint_X_p_np = model.joint_X_p.numpy()
        for j in range(jstart, jend):
            joint_type = JointType(model.joint_type.numpy()[j])
            parent_body = int(model.joint_parent.numpy()[j])
            child_body = int(model.joint_child.numpy()[j])

            if child_body not in self._mapping.body_geo_slots:
                continue

            child_slot = self._mapping.body_geo_slots[child_body]
            parent_slot = self._mapping.body_geo_slots.get(parent_body)

            # Joint world-frame transform
            jp = joint_X_p_np[j]
            jp_mat = newton_transform_to_mat4(wp.transform(jp[:3], jp[3:]))
            if parent_body >= 0 and body_transforms is not None:
                joint_world_mat = body_transforms[parent_body] @ jp_mat
            else:
                joint_world_mat = jp_mat
            pivot = joint_world_mat[:3, 3].copy()

            # Resolve owning articulation
            art_idx = int(joint_articulation[j])
            if art_idx not in self.articulations:
                self.articulations[art_idx] = Articulation(
                    name=f"articulation_{art_idx}",
                    dt=self._dt,
                )
            art = self.articulations[art_idx]

            # Dispatch by joint type
            if joint_type == JointType.REVOLUTE:
                self._build_revolute_joint(
                    art,
                    j,
                    pivot,
                    joint_world_mat,
                    parent_slot,
                    child_slot,
                    model.joint_axis,
                    model.joint_qd_start,
                    model.joint_q_start,
                    state.joint_q,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                )

            elif joint_type == JointType.PRISMATIC:
                self._build_prismatic_joint(
                    art,
                    j,
                    pivot,
                    joint_world_mat,
                    parent_body,
                    parent_slot,
                    child_body,
                    child_slot,
                    model.joint_axis,
                    model.joint_qd_start,
                    model.joint_q_start,
                    state.joint_q,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    body_transforms,
                )

            elif joint_type == JointType.FIXED:
                self._build_fixed_joint(
                    j,
                    pivot,
                    parent_body,
                    parent_slot,
                    child_body,
                    child_slot,
                    body_transforms,
                )

            elif joint_type == JointType.FREE:
                stc = SoftTransformConstraint()
                stc.apply_to(
                    child_slot.geometry(),
                    np.array([1000.0, 1000.0]),
                )

            elif joint_type == JointType.BALL:
                joint_X_c = model.joint_X_c.numpy() if model.joint_X_c is not None else None
                child_xform_tf = joint_X_c[j] if joint_X_c is not None else None
                self._build_ball_joint(
                    j,
                    pivot,
                    parent_body,
                    parent_slot,
                    child_body,
                    child_slot,
                    child_xform_tf,
                )

            elif joint_type in (JointType.DISTANCE, JointType.D6):
                warnings.warn(
                    f"Joint {j}: JointType {joint_type.name} is not yet supported by SolverUIPC, skipping",
                    stacklevel=2,
                )

        # Finalise all articulations that have active joints
        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.setup_state()

    # ------------------------------------------------------------------
    # Joint building helpers
    # ------------------------------------------------------------------

    def _get_or_create_anchor(
        self,
        name: str,
        position: np.ndarray,
    ) -> Any:
        """Get or create a fixed anchor body at *position*.

        Used when a revolute or prismatic joint is attached to the world
        (parent == -1).  Creates a tiny fixed AffineBody tetrahedron at
        the given position and returns its geometry slot.

        Args:
            name: Unique name for the anchor object.
            position: World-frame 3-D position of the anchor [m].

        Returns:
            The UIPC geometry slot for the anchor body.
        """
        if name in self._anchor_slots:
            return self._anchor_slots[name]

        # Create a minimal tetrahedron.  Offset it 0.5 m along the axis
        # that is furthest from other geometry so it doesn't collide with
        # the child link at the pivot.
        r = 0.005  # 5 mm half-extent
        verts = np.array(
            [
                [r, 0.0, 0.0],
                [-r, r, 0.0],
                [-r, -r, r],
                [-r, 0.0, -r],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
            dtype=np.int32,
        )
        sc = uipc_trimesh(verts, faces)

        # Offset the anchor 0.5 m along the local Y axis to avoid overlap
        # with the child link geometry sitting at the pivot.
        anchor_pos = position.copy()
        anchor_pos[1] += 0.5  # shift along Y
        mat4 = np.eye(4, dtype=np.float64)
        mat4[:3, 3] = anchor_pos
        view(sc.transforms())[:] = mat4

        # Create a dedicated contact element with all pairs disabled so the
        # anchor never participates in collision detection.
        if not hasattr(self, "_anchor_contact_elem"):
            tabular = self._scene.contact_tabular()
            self._anchor_contact_elem = tabular.create("_anchor")
            # Disable self-contact
            tabular.insert(self._anchor_contact_elem, self._anchor_contact_elem, 0.0, 0.0, False)
            # Disable contact with the default element
            if self._contact_elem is not None:
                tabular.insert(self._anchor_contact_elem, self._contact_elem, 0.0, 0.0, False)
        self._anchor_contact_elem.apply_to(sc)
        if self._subscene_elem is not None:
            self._subscene_elem.apply_to(sc)
        self._abd.apply_to(sc=sc, kappa=self._kappa, mass_density=1000.0)

        # Mark as fixed so it doesn't move
        view(sc.instances().find(uipc_builtin.is_fixed))[:] = 1  # type: ignore  # pyright: ignore[reportArgumentType]

        obj = self._scene.objects().create(name)
        geo_slot, _ = obj.geometries().create(sc)
        self._anchor_slots[name] = geo_slot
        return geo_slot

    @staticmethod
    def _rotate_around_pivot(
        tf: np.ndarray,
        pivot: np.ndarray,
        axis: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Rotate a 4x4 transform around *pivot* by *angle* about *axis*.

        Uses Rodrigues' rotation formula.  The rotation is applied in
        world space: translate to the pivot, rotate, translate back.

        Args:
            tf: 4x4 homogeneous transform (modified copy returned).
            pivot: World-space pivot point, shape (3,).
            axis: Unit rotation axis in world frame, shape (3,).
            angle: Rotation angle [rad].

        Returns:
            Rotated 4x4 transform.
        """
        c, s = np.cos(angle), np.sin(angle)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        R = np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)

        R4 = np.eye(4, dtype=np.float64)
        R4[:3, :3] = R

        T_pos = np.eye(4, dtype=np.float64)
        T_pos[:3, 3] = pivot
        T_neg = np.eye(4, dtype=np.float64)
        T_neg[:3, 3] = -pivot

        return T_pos @ R4 @ T_neg @ tf

    def _build_revolute_joint(
        self,
        art: Articulation,
        j: int,
        pivot: np.ndarray,
        joint_world_mat: np.ndarray,
        parent_slot: Any,
        child_slot: Any,
        joint_axis: wp.array,
        joint_qd_start: wp.array,
        joint_q_start: wp.array,
        joint_q: wp.array | None,
        joint_limit_lower: wp.array | None,
        joint_limit_upper: wp.array | None,
    ) -> None:
        """Create a UIPC revolute joint and register it with *art*."""
        if parent_slot is None:
            parent_slot = self._get_or_create_anchor(
                f"anchor_joint_{j}",
                pivot,
            )

        qd_start = int(joint_qd_start.numpy()[j])
        axis_world = joint_world_mat[:3, :3] @ joint_axis.numpy()[qd_start]
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
        q_start = int(joint_q_start.numpy()[j])

        vs = np.array([pivot, pivot + axis_world], dtype=np.float64)
        jm = linemesh(vs, np.array([[0, 1]], dtype=np.int32))

        AffineBodyRevoluteJoint().apply_to(
            jm,
            [parent_slot],
            [0],
            [child_slot],
            [0],
            [1000.0],
        )
        AffineBodyDrivingRevoluteJoint().apply_to(jm, [1000.0])
        AffineBodyRevoluteJointExternalForce().apply_to(jm, [0.0])

        # Apply joint limits via UIPC constitution
        lower, upper = self._extract_limits(
            j,
            joint_qd_start,
            joint_limit_lower,
            joint_limit_upper,
        )
        if lower is not None and upper is not None:
            AffineBodyRevoluteJointLimit().apply_to(jm, lower, upper)

        jobj = self._scene.objects().create(f"joint_{j}_revolute")
        jslot, _ = jobj.geometries().create(jm)

        # Register with the Articulation runtime
        init_angle = float(joint_q.numpy()[q_start]) if joint_q is not None else 0.0
        art.register_joint(j, q_start, qd_start, init_angle)
        art.joint_geo_slots[j] = jslot
        art.joint_mesh[j] = jm

        self._mapping.joint_geo_slots[j] = jslot
        self._mapping.joint_mesh[j] = jm

        self._scene.animator().insert(
            jobj,
            lambda info, a=art, ni=j: a.revolute_joint_anim(info, ni),
        )

    def _build_prismatic_joint(
        self,
        art: Articulation,
        j: int,
        pivot: np.ndarray,
        joint_world_mat: np.ndarray,
        parent_body: int,
        parent_slot: Any,
        child_body: int,
        child_slot: Any,
        joint_axis: wp.array,
        joint_qd_start: wp.array,
        joint_q_start: wp.array,
        joint_q: wp.array | None,
        joint_limit_lower: wp.array | None,
        joint_limit_upper: wp.array | None,
        body_transforms: np.ndarray | None,
    ) -> None:
        """Create a UIPC prismatic joint and register it with *art*."""
        if parent_slot is None:
            parent_slot = self._get_or_create_anchor(
                f"anchor_joint_{j}",
                pivot,
            )

        qd_start = int(joint_qd_start.numpy()[j])
        axis_world = joint_world_mat[:3, :3] @ joint_axis.numpy()[qd_start]
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)

        q_start = int(joint_q_start.numpy()[j])

        vs = np.array([pivot, pivot + axis_world], dtype=np.float64)
        jm = linemesh(vs, np.array([[0, 1]], dtype=np.int32))

        AffineBodyPrismaticJoint().apply_to(
            jm,
            [parent_slot],
            [0],
            [child_slot],
            [0],
            [100.0],
        )
        AffineBodyDrivingPrismaticJoint().apply_to(jm, [100.0])
        AffineBodyPrismaticJointExternalForce().apply_to(jm, [0.0])

        # Apply joint limits via UIPC constitution
        lower, upper = self._extract_limits(
            j,
            joint_qd_start,
            joint_limit_lower,
            joint_limit_upper,
        )
        if lower is not None and upper is not None:
            AffineBodyPrismaticJointLimit().apply_to(jm, lower, upper)

        # Compute init_distance from parent/child body world positions
        # projected onto the joint axis direction.
        if body_transforms is not None and parent_body >= 0:
            parent_pos = body_transforms[parent_body][:3, 3]
            child_pos = body_transforms[child_body][:3, 3]
            init_dist = float(np.dot(child_pos - parent_pos, axis_world))
        elif body_transforms is not None:
            # Parent is world (fixed anchor at pivot)
            child_pos = body_transforms[child_body][:3, 3]
            init_dist = float(np.dot(child_pos - pivot, axis_world))
        else:
            init_dist = 0.0
        view(jm.edges().find("init_distance"))[0] = init_dist  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]

        jobj = self._scene.objects().create(f"joint_{j}_prismatic")
        jslot, _ = jobj.geometries().create(jm)
        art.register_joint(j, q_start, qd_start, init_dist)
        art.joint_geo_slots[j] = jslot
        art.joint_mesh[j] = jm

        self._mapping.joint_geo_slots[j] = jslot
        self._mapping.joint_mesh[j] = jm

        self._scene.animator().insert(
            jobj,
            lambda info, a=art, ni=j: a.prismatic_joint_anim(info, ni),
        )

    def _build_fixed_joint(
        self,
        j: int,
        pivot: np.ndarray,
        parent_body: int,
        parent_slot: Any,
        child_body: int,
        child_slot: Any,
        body_transforms: np.ndarray | None,
    ) -> None:
        """Create a UIPC fixed joint, or mark child as fixed if no parent."""
        if parent_slot is None:
            view(child_slot.geometry().instances().find(uipc_builtin.is_fixed))[:] = 1
            return

        jm = pointcloud(np.array([], dtype=np.float64).reshape(0, 3))
        AffineBodyFixedJoint().apply_to(
            jm,
            [child_slot],
            [0],
            [parent_slot],
            [0],
            [100.0],
        )

        jobj = self._scene.objects().create(f"joint_{j}_fixed")
        jslot, _ = jobj.geometries().create(jm)
        self._mapping.joint_geo_slots[j] = jslot
        self._mapping.joint_mesh[j] = jm

    def _build_ball_joint(
        self,
        j: int,
        pivot: np.ndarray,
        parent_body: int,
        parent_slot: Any,
        child_body: int,
        child_slot: Any,
        child_xform: np.ndarray | None,
    ) -> None:
        """Create a UIPC spherical (ball) joint.

        The spherical joint constrains the anchor point to coincide on
        both bodies while allowing free relative rotation.  No driving
        or animation is needed — similar to a fixed joint but with
        rotational freedom.

        Args:
            child_xform: The child-frame joint transform
                (``wp.transform``: first 3 elements are translation).
                When provided, its translation is used as the anchor
                position in the child body's local frame.
        """
        if parent_slot is None:
            parent_slot = self._get_or_create_anchor(
                f"anchor_joint_{j}",
                pivot,
            )

        # Use child_xform translation as the anchor in child local frame
        if child_xform is not None:
            r_local = np.array(child_xform[:3], dtype=np.float64)
        else:
            r_local = np.zeros(3, dtype=np.float64)

        jm = pointcloud(np.array([], dtype=np.float64).reshape(0, 3))
        AffineBodySphericalJoint().apply_to(
            jm,
            [parent_slot],
            [child_slot],
            np.array([r_local], dtype=np.float64),
            100.0,
        )

        jobj = self._scene.objects().create(f"joint_{j}_ball")
        jslot, _ = jobj.geometries().create(jm)
        self._mapping.joint_geo_slots[j] = jslot
        self._mapping.joint_mesh[j] = jm

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

        # Ensure model arrays are on CPU for the kernel
        joint_type_cpu = model.joint_type.to("cpu")
        joint_target_mode_cpu = model.joint_target_mode.to("cpu")

        # Control arrays (per-step, may be None)
        target_pos_cpu = control.joint_target_pos.to("cpu") if control.joint_target_pos is not None else None
        target_vel_cpu = control.joint_target_vel.to("cpu") if control.joint_target_vel is not None else None
        joint_f_cpu = control.joint_f.to("cpu") if control.joint_f is not None else None

        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.cache_control(
                    joint_type_cpu,
                    joint_target_mode_cpu,
                    target_pos_cpu,
                    target_vel_cpu,
                    joint_f_cpu,
                )

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

        # CPU staging arrays for the kernel to scatter into
        joint_q_cpu = state_out.joint_q.to("cpu")
        joint_qd_cpu = state_out.joint_qd.to("cpu") if state_out.joint_qd is not None else None

        for art in self.articulations.values():
            if art.num_active_joints > 0:
                art.write_readback(joint_q_cpu, joint_qd_cpu)

        # Copy back to device
        wp.copy(state_out.joint_q, joint_q_cpu)
        if joint_qd_cpu is not None and state_out.joint_qd is not None:
            wp.copy(state_out.joint_qd, joint_qd_cpu)

    def increment_step(self) -> None:
        """Increment the step counter on all articulations."""
        for art in self.articulations.values():
            art.increment_step()
