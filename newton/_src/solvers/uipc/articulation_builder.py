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
from uipc.geometry import trimesh as uipc_trimesh
from uipc.unit import MPa

from ...sim import Control, JointType, Model, State
from .articulation import Articulation
from .converter import UIpcMappingInfo, newton_transform_to_mat4


def _mat4_to_transform(mat: np.ndarray) -> wp.transform:  # pyright: ignore[reportArgumentType]
    """Convert a 4x4 homogeneous matrix to a ``wp.transform``.

    Extracts position from the last column and quaternion from the
    rotation sub-matrix using Shepperd's method.

    Args:
        mat: 4x4 homogeneous transformation matrix (float64).

    Returns:
        A ``wp.transform(pos, quat)`` value.
    """
    pos = mat[:3, 3]
    R = mat[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm > 0.0:
        inv_norm = 1.0 / norm
        x, y, z, w = x * inv_norm, y * inv_norm, z * inv_norm, w * inv_norm

    return wp.transform(
        wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
        wp.quat(float(x), float(y), float(z), float(w)),
    )


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
        self._abd = AffineBodyConstitution()
        self._kappa = kappa

        # Per-articulation runtime objects (populated by build_joints)
        self.articulations: dict[int, Articulation] = {}

        # Per-body world-frame 4x4 transforms (populated by build_joints via FK)
        self._body_transforms: np.ndarray | None = None

        # Cache of anchor body geo slots for world-anchored joints
        self._anchor_slots: dict[str, Any] = {}

        # Transient subscene element set per build_joints call
        self._subscene_elem: Any | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_joint_transform(
        joint_type: JointType,
        joint_q_np: np.ndarray,
        joint_axis_np: np.ndarray,
        q_start: int,
        qd_start: int,
    ) -> wp.transform:  # pyright: ignore[reportArgumentType]
        """Compute the local joint transform from joint coordinates.

        Mirrors the per-joint FK logic in
        :func:`~newton._src.sim.articulation.eval_single_articulation_fk`.

        Args:
            joint_type: The type of joint.
            joint_q_np: Joint position coordinates (numpy).
            joint_axis_np: Joint axes (numpy, indexed by qd_start).
            q_start: Start index in ``joint_q`` for this joint.
            qd_start: Start index in ``joint_qd`` for this joint.

        Returns:
            A ``wp.transform`` representing the local joint displacement.
        """
        if joint_type == JointType.REVOLUTE:
            axis = joint_axis_np[qd_start]
            angle = float(joint_q_np[q_start])
            q = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
            return wp.transform(wp.vec3(0.0, 0.0, 0.0), q)

        elif joint_type == JointType.PRISMATIC:
            axis = joint_axis_np[qd_start].astype(np.float64)
            dist = float(joint_q_np[q_start])
            return wp.transform(wp.vec3(*(axis * dist)), wp.quat_identity())

        elif joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
            pos = wp.vec3(
                float(joint_q_np[q_start + 0]),
                float(joint_q_np[q_start + 1]),
                float(joint_q_np[q_start + 2]),
            )
            rot = wp.quat(
                float(joint_q_np[q_start + 3]),
                float(joint_q_np[q_start + 4]),
                float(joint_q_np[q_start + 5]),
                float(joint_q_np[q_start + 6]),
            )
            return wp.transform(pos, rot)

        elif joint_type == JointType.BALL:
            rot = wp.quat(
                float(joint_q_np[q_start + 0]),
                float(joint_q_np[q_start + 1]),
                float(joint_q_np[q_start + 2]),
                float(joint_q_np[q_start + 3]),
            )
            return wp.transform(wp.vec3(0.0, 0.0, 0.0), rot)

        # FIXED and unsupported types: identity
        return wp.transform_identity()

    def _apply_fk_for_joint(
        self,
        j: int,
        joint_type: JointType,
        parent_body: int,
        child_body: int,
        joint_X_p_np: np.ndarray,
        joint_X_c_np: np.ndarray | None,
        joint_q_np: np.ndarray | None,
        joint_axis_np: np.ndarray | None,
        q_start: int,
        qd_start: int,
    ) -> None:
        """Compute child body world transform via FK and store it.

        Uses the parent body transform (already in ``self._body_transforms``),
        the joint frame transforms (``joint_X_p``, ``joint_X_c``), and the
        joint coordinate (``joint_q``) to compute the child body's world-frame
        4x4 matrix.  The result is stored in ``self._body_transforms``.

        Args:
            j: Newton joint index.
            joint_type: Type of the joint.
            parent_body: Parent body index (``-1`` for world).
            child_body: Child body index.
            joint_X_p_np: Parent-frame joint transforms, shape ``(J, 7)``.
            joint_X_c_np: Child-frame joint transforms, shape ``(J, 7)``, or ``None``.
            joint_q_np: Joint position coordinates (numpy), or ``None``.
            joint_axis_np: Joint axes (numpy), or ``None``.
            q_start: Start index in ``joint_q`` for this joint.
            qd_start: Start index in ``joint_qd`` for this joint.
        """
        # Parent anchor in world: X_wpj = X_wp * X_pj
        jp = joint_X_p_np[j]
        X_pj = wp.transform(jp[:3], jp[3:])
        if parent_body >= 0 and self._body_transforms is not None:
            X_wp = self._body_transforms[parent_body]
            X_pj_mat = newton_transform_to_mat4(X_pj)
            X_wpj_mat = X_wp @ X_pj_mat
        else:
            X_wpj_mat = newton_transform_to_mat4(X_pj)

        # Local joint displacement from joint_q
        if joint_q_np is not None and joint_axis_np is not None:
            X_j = self._compute_joint_transform(
                joint_type,
                joint_q_np,
                joint_axis_np,
                q_start,
                qd_start,
            )
        else:
            X_j = wp.transform_identity()
        X_j_mat = newton_transform_to_mat4(X_j)

        # Child anchor in world: X_wcj = X_wpj * X_j
        X_wcj_mat = X_wpj_mat @ X_j_mat

        # Child body in world: X_wc = X_wcj * inv(X_cj)
        if joint_X_c_np is not None:
            jc = joint_X_c_np[j]
            X_cj = wp.transform(jc[:3], jc[3:])
            X_cj_inv_mat = newton_transform_to_mat4(wp.transform_inverse(X_cj))
            X_wc_mat = X_wcj_mat @ X_cj_inv_mat
        else:
            X_wc_mat = X_wcj_mat

        # Store
        assert self._body_transforms is not None
        self._body_transforms[child_body] = X_wc_mat

    def compute_fk(
        self,
        joint_range: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        """Compute body world transforms from joint coordinates (FK).

        Iterates joints in topological order and computes each child body's
        world-frame 4x4 transform from ``model.joint_q``.  Results are
        stored in ``self._body_transforms`` and synced back to
        ``model.body_q``.

        Must be called **before** :meth:`RigidBodyBuilder.build_affine_bodies`
        so that the pre-computed transforms can be used when creating
        UIPC geometries (transforms must be set before geometry creation).

        Args:
            joint_range: ``(start, end)`` slice of joints to process, or
                ``None`` for all joints.

        Returns:
            Body transforms array of shape ``(body_count, 4, 4)`` (float64),
            or ``None`` if no joints exist.
        """
        model = self._model
        if model.joint_count == 0 or model.body_count == 0:
            return None

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
            return None

        # Lazily allocate (shared across multi-world calls).
        # Seed from model.body_q so that bodies without joints (e.g.
        # kinematic bodies) already have a valid world-frame transform.
        if self._body_transforms is None:
            self._body_transforms = np.zeros(
                (model.body_count, 4, 4),
                dtype=np.float64,
            )
            if model.body_q is not None:
                body_q_np = model.body_q.numpy()
                for b in range(model.body_count):
                    tf = wp.transform(body_q_np[b, :3], body_q_np[b, 3:])
                    self._body_transforms[b] = newton_transform_to_mat4(tf)

        jstart, jend = joint_range if joint_range else (0, model.joint_count)

        # Pre-fetch numpy arrays
        joint_X_p_np = model.joint_X_p.numpy()
        joint_X_c_np = model.joint_X_c.numpy() if model.joint_X_c is not None else None
        joint_q_np = model.joint_q.numpy() if model.joint_q is not None else None
        joint_axis_np = model.joint_axis.numpy() if model.joint_axis is not None else None
        joint_type_np = model.joint_type.numpy()
        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()
        joint_q_start_np = model.joint_q_start.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()

        # Staging array for syncing FK results back to model.body_q
        body_q_staging = model.body_q.numpy().copy() if model.body_q is not None else None

        for j in range(jstart, jend):
            joint_type = JointType(joint_type_np[j])
            parent_body = int(joint_parent_np[j])
            child_body = int(joint_child_np[j])

            q_start = int(joint_q_start_np[j])
            qd_start = int(joint_qd_start_np[j])
            self._apply_fk_for_joint(
                j,
                joint_type,
                parent_body,
                child_body,
                joint_X_p_np,
                joint_X_c_np,
                joint_q_np,
                joint_axis_np,
                q_start,
                qd_start,
            )

            # Stage body_q sync
            if body_q_staging is not None:
                tf = _mat4_to_transform(self._body_transforms[child_body])
                body_q_staging[child_body, :3] = [tf.p[0], tf.p[1], tf.p[2]]
                body_q_staging[child_body, 3:] = [tf.q[0], tf.q[1], tf.q[2], tf.q[3]]

        # Flush to model.body_q
        if body_q_staging is not None and model.body_q is not None:
            wp.copy(
                model.body_q,
                wp.from_numpy(body_q_staging, dtype=model.body_q.dtype, device="cpu"),
            )

        return self._body_transforms

    def build_joints(
        self,
        contact_elem: Any,
        joint_range: tuple[int, int],
        subscene_elem: Any | None = None,
    ) -> None:
        """Convert Newton joints to UIPC joint constitutions.

        Creates one :class:`Articulation` per Newton articulation, builds
        the UIPC geometry for each joint, and registers Animator callbacks.

        Must be called **after** :meth:`compute_fk` so that
        ``self._body_transforms`` is populated.

        Args:
            contact_elem: Contact element for robot link geometries.
            joint_range: ``(start, end)`` slice of joints to process, or
                ``None`` for all joints.
            subscene_elem: UIPC subscene element for anchor bodies, or ``None``.
        """
        # Store for use by _get_or_create_anchor
        self._contact_elem = contact_elem
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
            self.articulations[a] = Articulation(name=label, dt=self._dt)

        # Pre-fetch numpy arrays
        state = model.state()
        joint_X_p_np = model.joint_X_p.numpy()
        joint_type_np = model.joint_type.numpy()
        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()

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

            if child_body not in self._mapping.body_geo_slots:
                continue

            child_slot = self._mapping.body_geo_slots[child_body]
            child_instance_id = self._mapping.body_instance_ids.get(child_body, 0)
            parent_slot = self._mapping.body_geo_slots.get(parent_body)
            parent_instance_id = self._mapping.body_instance_ids.get(parent_body, 0)

            # Joint world-frame transform (from the FK-computed body transforms)
            jp = joint_X_p_np[j]
            jp_mat = newton_transform_to_mat4(wp.transform(jp[:3], jp[3:]))
            if parent_body >= 0 and self._body_transforms is not None:
                joint_world_mat = self._body_transforms[parent_body] @ jp_mat
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

            jdata = {
                "j": j,
                "art": art,
                "pivot": pivot,
                "joint_world_mat": joint_world_mat,
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
                state,
                model,
            )
        if prismatic_joints:
            self._build_prismatic_joints_batch(
                prismatic_joints,
                state,
                model,
            )
        if fixed_joints:
            self._build_fixed_joints_batch(fixed_joints)
        for jdata in free_joints:
            stc = SoftTransformConstraint()
            stc.apply_to(jdata["child_slot"].geometry())
        if ball_joints:
            self._build_ball_joints_batch(ball_joints, model)

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

    def _build_revolute_joints_batch(
        self,
        joints: list[dict],
        state: Any,
        model: Any,
    ) -> None:
        """Create all revolute joints in a single batched linemesh."""
        all_verts: list[np.ndarray] = []
        parent_slots: list[Any] = []
        parent_ids: list[int] = []
        child_slots: list[Any] = []
        child_ids: list[int] = []
        strengths: list[float] = []
        drive_strengths: list[float] = []
        ext_forces: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        limit_strengths: list[float] = []
        has_any_limit = False

        joint_axis_np = model.joint_axis.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()
        joint_q_start_np = model.joint_q_start.numpy()
        joint_q_np = state.joint_q.numpy() if state.joint_q is not None else None

        # Dispatch list for animator callback: (art, newton_joint_idx, edge_idx)
        anim_dispatch: list[tuple[Articulation, int, int]] = []
        for edge_idx, jdata in enumerate(joints):
            j = jdata["j"]
            art = jdata["art"]
            pivot = jdata["pivot"]
            joint_world_mat = jdata["joint_world_mat"]
            p_slot = jdata["parent_slot"]
            p_id = jdata["parent_instance_id"]
            c_slot = jdata["child_slot"]
            c_id = jdata["child_instance_id"]

            if p_slot is None:
                p_slot = self._get_or_create_anchor(f"anchor_joint_{j}", pivot)
                p_id = 0

            qd_start = int(joint_qd_start_np[j])
            axis_world = joint_world_mat[:3, :3] @ joint_axis_np[qd_start]
            axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
            q_start = int(joint_q_start_np[j])

            all_verts.append(pivot)
            all_verts.append(pivot + axis_world)

            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            child_slots.append(c_slot)
            child_ids.append(c_id)
            strengths.append(1000.0)
            drive_strengths.append(1000.0)
            ext_forces.append(0.0)

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
                limit_strengths.append(1.0)
                has_any_limit = True
            else:
                lowers.append(-1e18)
                uppers.append(1e18)
                limit_strengths.append(1.0)

            # Register joint with its articulation
            init_angle = float(joint_q_np[q_start]) if joint_q_np is not None else 0.0
            art.register_joint(j, q_start, qd_start, init_angle)
            anim_dispatch.append((art, j, edge_idx))

        # Build batched linemesh via create_geometry
        pos0s = np.array(all_verts[0::2], dtype=np.float64)
        pos1s = np.array(all_verts[1::2], dtype=np.float64)
        jm = AffineBodyRevoluteJoint().create_geometry(
            pos0s,
            pos1s,
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
        AffineBodyRevoluteJointExternalForce().apply_to(
            jm,
            np.array(ext_forces, dtype=np.float64),
        )
        if has_any_limit:
            AffineBodyRevoluteJointLimit().apply_to(
                jm,
                np.array(lowers, dtype=np.float64),
                np.array(uppers, dtype=np.float64),
                np.array(limit_strengths, dtype=np.float64),
            )

        jobj = self._scene.objects().create("joints_revolute")
        jslot, _ = jobj.geometries().create(jm)

        # Record mappings for each joint
        for art, j, edge_idx in anim_dispatch:
            art.joint_geo_slots[j] = jslot
            art.joint_mesh[j] = jm
            self._mapping.joint_geo_slots[j] = jslot
            self._mapping.joint_mesh[j] = jm

        # Single animator callback dispatching to all revolute joints
        dispatch_copy = list(anim_dispatch)

        def _revolute_batch_anim(info: Any, dispatch: list = dispatch_copy) -> None:
            try:
                geo = info.geo_slots()[0].geometry()
            except (TypeError, IndexError):
                return
            for art, newton_j, edge_idx in dispatch:
                art.revolute_joint_anim(geo, newton_j, edge_idx)

        self._scene.animator().insert(jobj, _revolute_batch_anim)

    def _build_prismatic_joints_batch(
        self,
        joints: list[dict],
        state: Any,
        model: Any,
    ) -> None:
        """Create all prismatic joints in a single batched linemesh."""
        body_transforms = self._body_transforms

        all_verts: list[np.ndarray] = []
        parent_slots: list[Any] = []
        parent_ids: list[int] = []
        child_slots: list[Any] = []
        child_ids: list[int] = []
        strengths: list[float] = []
        drive_strengths: list[float] = []
        ext_forces: list[float] = []
        init_distances: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        limit_strengths: list[float] = []
        has_any_limit = False

        joint_axis_np = model.joint_axis.numpy()
        joint_qd_start_np = model.joint_qd_start.numpy()
        joint_q_start_np = model.joint_q_start.numpy()

        anim_dispatch: list[tuple[Articulation, int, int]] = []
        for edge_idx, jdata in enumerate(joints):
            j = jdata["j"]
            art = jdata["art"]
            pivot = jdata["pivot"]
            joint_world_mat = jdata["joint_world_mat"]
            parent_body = jdata["parent_body"]
            child_body = jdata["child_body"]
            p_slot = jdata["parent_slot"]
            p_id = jdata["parent_instance_id"]
            c_slot = jdata["child_slot"]
            c_id = jdata["child_instance_id"]

            if p_slot is None:
                p_slot = self._get_or_create_anchor(f"anchor_joint_{j}", pivot)
                p_id = 0

            qd_start = int(joint_qd_start_np[j])
            axis_world = joint_world_mat[:3, :3] @ joint_axis_np[qd_start]
            axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
            q_start = int(joint_q_start_np[j])

            all_verts.append(pivot)
            all_verts.append(pivot + axis_world)

            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            child_slots.append(c_slot)
            child_ids.append(c_id)
            strengths.append(100.0)
            drive_strengths.append(100.0)
            ext_forces.append(0.0)

            # Compute init_distance
            if body_transforms is not None and parent_body >= 0:
                parent_pos = body_transforms[parent_body][:3, 3]
                child_pos = body_transforms[child_body][:3, 3]
                init_dist = float(np.dot(child_pos - parent_pos, axis_world))
            elif body_transforms is not None:
                child_pos = body_transforms[child_body][:3, 3]
                init_dist = float(np.dot(child_pos - pivot, axis_world))
            else:
                init_dist = 0.0
            init_distances.append(init_dist)

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
                limit_strengths.append(1.0)
                has_any_limit = True
            else:
                lowers.append(-1e18)
                uppers.append(1e18)
                limit_strengths.append(1.0)

            art.register_joint(j, q_start, qd_start, init_dist)
            anim_dispatch.append((art, j, edge_idx))

        # Build batched linemesh via create_geometry
        pos0s = np.array(all_verts[0::2], dtype=np.float64)
        pos1s = np.array(all_verts[1::2], dtype=np.float64)
        jm = AffineBodyPrismaticJoint().create_geometry(
            pos0s,
            pos1s,
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
        AffineBodyPrismaticJointExternalForce().apply_to(
            jm,
            np.array(ext_forces, dtype=np.float64),
        )
        if has_any_limit:
            AffineBodyPrismaticJointLimit().apply_to(
                jm,
                np.array(lowers, dtype=np.float64),
                np.array(uppers, dtype=np.float64),
                np.array(limit_strengths, dtype=np.float64),
            )

        # Set per-edge init_distance
        dist_view = view(jm.edges().find("init_distance"))
        for i, d in enumerate(init_distances):
            dist_view[i] = d  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]

        jobj = self._scene.objects().create("joints_prismatic")
        jslot, _ = jobj.geometries().create(jm)

        for art, j, edge_idx in anim_dispatch:
            art.joint_geo_slots[j] = jslot
            art.joint_mesh[j] = jm
            self._mapping.joint_geo_slots[j] = jslot
            self._mapping.joint_mesh[j] = jm

        dispatch_copy = list(anim_dispatch)

        def _prismatic_batch_anim(info: Any, dispatch: list = dispatch_copy) -> None:
            try:
                geo = info.geo_slots()[0].geometry()
            except (TypeError, IndexError):
                return
            for art, newton_j, edge_idx in dispatch:
                art.prismatic_joint_anim(geo, newton_j, edge_idx)

        self._scene.animator().insert(jobj, _prismatic_batch_anim)

    def _build_fixed_joints_batch(
        self,
        joints: list[dict],
    ) -> None:
        """Create all fixed joints in a single batched pointcloud."""
        # Separate world-anchored (no parent) from inter-body fixed joints
        child_slots: list[Any] = []
        child_ids: list[int] = []
        parent_slots: list[Any] = []
        parent_ids: list[int] = []
        strengths: list[float] = []
        joint_indices: list[int] = []

        for jdata in joints:
            j = jdata["j"]
            p_slot = jdata["parent_slot"]
            p_id = jdata["parent_instance_id"]
            c_slot = jdata["child_slot"]
            c_id = jdata["child_instance_id"]

            if p_slot is None:
                # No parent → mark child instance as fixed directly
                view(c_slot.geometry().instances().find(uipc_builtin.is_fixed))[c_id] = 1
                continue

            child_slots.append(c_slot)
            child_ids.append(c_id)
            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            strengths.append(100.0)
            joint_indices.append(j)

        if not child_slots:
            return

        jm = AffineBodyFixedJoint().create_geometry(
            child_slots,
            np.array(child_ids, dtype=np.int32),
            parent_slots,
            np.array(parent_ids, dtype=np.int32),
            np.array(strengths, dtype=np.float64),
        )

        jobj = self._scene.objects().create("joints_fixed")
        jslot, _ = jobj.geometries().create(jm)
        for j in joint_indices:
            self._mapping.joint_geo_slots[j] = jslot
            self._mapping.joint_mesh[j] = jm

    def _build_ball_joints_batch(
        self,
        joints: list[dict],
        model: Any,
    ) -> None:
        """Create all spherical (ball) joints in a single batched pointcloud."""
        parent_slots: list[Any] = []
        parent_ids: list[int] = []
        child_slots: list[Any] = []
        child_ids: list[int] = []
        l_positions: list[np.ndarray] = []
        r_positions: list[np.ndarray] = []
        strengths: list[float] = []
        joint_indices: list[int] = []

        joint_X_p_np = model.joint_X_p.numpy()
        joint_X_c_np = model.joint_X_c.numpy() if model.joint_X_c is not None else None

        for jdata in joints:
            j = jdata["j"]
            p_slot = jdata["parent_slot"]
            p_id = jdata["parent_instance_id"]
            pivot = jdata["pivot"]
            c_slot = jdata["child_slot"]
            c_id = jdata["child_instance_id"]

            if p_slot is None:
                p_slot = self._get_or_create_anchor(f"anchor_joint_{j}", pivot)
                p_id = 0

            # Parent-side local anchor (joint_X_p translation)
            l_pos = np.array(joint_X_p_np[j][:3], dtype=np.float64)

            # Child-side local anchor (joint_X_c translation)
            if joint_X_c_np is not None:
                r_pos = np.array(joint_X_c_np[j][:3], dtype=np.float64)
            else:
                r_pos = np.zeros(3, dtype=np.float64)

            parent_slots.append(p_slot)
            parent_ids.append(p_id)
            child_slots.append(c_slot)
            child_ids.append(c_id)
            l_positions.append(l_pos)
            r_positions.append(r_pos)
            strengths.append(100.0)
            joint_indices.append(j)

        if not child_slots:
            return

        jm = AffineBodySphericalJoint().create_geometry(
            np.array(l_positions, dtype=np.float64),
            np.array(r_positions, dtype=np.float64),
            parent_slots,
            np.array(parent_ids, dtype=np.int32),
            child_slots,
            np.array(child_ids, dtype=np.int32),
            np.array(strengths, dtype=np.float64),
        )

        jobj = self._scene.objects().create("joints_ball")
        jslot, _ = jobj.geometries().create(jm)
        for j in joint_indices:
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
