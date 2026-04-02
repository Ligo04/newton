# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body (AffineBody) builder for the UIPC solver backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import uipc.builtin as uipc_builtin
import warp as wp
from uipc import view
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import halfplane, label_surface
from uipc.geometry import trimesh as uipc_trimesh

from ...geometry import GeoType
from ...sim import BodyFlags, Model
from .converter import (
    UIpcMappingInfo,
    build_body_mesh,
    newton_transform_to_mat4,
)


@dataclass
class _BodyInfo:
    """Per-body data collected before instanced geometry creation."""

    body_idx: int
    shape_key: tuple[Any, ...]
    transform: np.ndarray
    mass_density: float
    contact_elem: Any
    is_kinematic: bool


def _compute_shape_key(model: Model, body_idx: int) -> tuple[Any, ...] | None:
    """Compute a lightweight key that identifies a body's canonical shape.

    Two bodies with the same shape_key produce identical meshes from
    :func:`build_body_mesh`. The key is built from per-shape
    ``(geo_type, scale, shape_transform)`` tuples (and ``id(shape_source)``
    for mesh/convex-mesh types), avoiding the cost of full mesh generation.

    Returns:
        A hashable tuple, or ``None`` if the body has no shapes.
    """
    if (
        model.shape_body is None
        or model.shape_type is None
        or model.shape_transform is None
        or model.shape_scale is None
    ):
        return None

    shape_body_np = model.shape_body.numpy()
    shape_type_np = model.shape_type.numpy()
    shape_transform_np = model.shape_transform.numpy()
    shape_scale_np = model.shape_scale.numpy()

    parts: list[tuple[Any, ...]] = []
    for s in range(model.shape_count):
        if shape_body_np[s] != body_idx:
            continue
        geo_type = int(shape_type_np[s])
        if geo_type == int(GeoType.PLANE):
            continue
        scale = tuple(float(x) for x in shape_scale_np[s])
        tf = shape_transform_np[s].tobytes()
        if geo_type in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
            src_id = id(model.shape_source[s])
            parts.append((geo_type, scale, tf, src_id))
        else:
            parts.append((geo_type, scale, tf))

    return tuple(parts) if parts else None


class RigidBodyBuilder:
    """Build UIPC AffineBody geometries from Newton rigid bodies.

    Converts Newton :class:`~newton.Model` rigid bodies (links with shapes)
    into UIPC ``AffineBodyConstitution`` geometries. Also handles ground plane
    creation and body-shape index mapping.

    A single instance can be reused across multiple Newton worlds by calling
    the build methods with different ``body_range`` / ``subscene_elem`` arguments.
    """

    def __init__(
        self,
        model: Model,
        scene: Any,
        mapping: UIpcMappingInfo,
        kappa: float,
        default_mass_density: float,
    ):
        self._model = model
        self._scene = scene
        self._mapping = mapping
        self._kappa = kappa
        self._default_mass_density = default_mass_density

        # Cache host-side numpy views (computed lazily, shared across methods)
        self._shape_body_np: np.ndarray | None = None
        self._shape_type_np: np.ndarray | None = None
        self._shape_transform_np: np.ndarray | None = None

    def _ensure_shape_cache(self) -> bool:
        """Populate cached numpy views of shape arrays. Returns False if unavailable."""
        if self._shape_body_np is not None:
            return True
        model = self._model
        if (
            model.shape_count == 0
            or model.shape_body is None
            or model.shape_type is None
            or model.shape_transform is None
        ):
            return False
        self._shape_body_np = model.shape_body.numpy()
        self._shape_type_np = model.shape_type.numpy()
        self._shape_transform_np = model.shape_transform.numpy()
        return True

    def build_ground_planes(self, contact_elem: Any) -> None:
        """Create UIPC halfplanes for Newton ground plane shapes (body == -1).

        Args:
            contact_elem: Contact element to apply to ground geometries.
        """
        model = self._model
        if not self._ensure_shape_cache():
            return

        assert self._shape_body_np is not None
        assert self._shape_type_np is not None
        assert self._shape_transform_np is not None

        for s in range(model.shape_count):
            if self._shape_body_np[s] == -1 and GeoType(self._shape_type_np[s]) == GeoType.PLANE:
                tf_np = self._shape_transform_np[s]
                mat4 = newton_transform_to_mat4(wp.transform(tf_np[:3], tf_np[3:]))
                normal = mat4[:3, 2].copy()
                center = tf_np[:3].astype(np.float64)

                g = halfplane(center, normal)
                contact_elem.apply_to(g)
                ground_obj = self._scene.objects().create(f"ground_plane_{s}")
                ground_obj.geometries().create(g)

    def build_body_shape_mapping(
        self,
        body_range: tuple[int, int] | None = None,
    ) -> None:
        """Populate ``mapping.body_shapes``: body_idx -> list of shape indices.

        Args:
            body_range: ``(start, end)`` slice of bodies to process, or
                ``None`` for all bodies.
        """
        model = self._model
        if not self._ensure_shape_cache():
            return

        assert self._shape_body_np is not None
        bstart, bend = body_range if body_range else (0, model.body_count)
        for s in range(model.shape_count):
            b = self._shape_body_np[s]
            if bstart <= b < bend:
                self._mapping.body_shapes[b].append(s)

    @staticmethod
    def _mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
        """Compute the signed volume of a closed triangle mesh.

        Uses the divergence theorem: V = sum_i (v0 · (v1 \\times v2)) / 6.

        Args:
            verts: Vertex positions, shape ``(N, 3)``.
            faces: Triangle indices, shape ``(M, 3)``.

        Returns:
            Absolute volume of the mesh.
        """
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        return float(abs(np.sum(v0 * np.cross(v1, v2)) / 6.0))

    def _resolve_contact_elem(
        self,
        b: int,
        env_elem: Any,
        robo_elem: Any,
        actor_elem: Any,
        articulation_bodies: set[int],
        free_joint_bodies: set[int],
        body_element_overrides: dict[int, Any] | None,
    ) -> Any:
        """Return the resolved contact element for body *b*."""
        if body_element_overrides is not None and b in body_element_overrides:
            return body_element_overrides[b]
        if b in articulation_bodies:
            return robo_elem
        if b in free_joint_bodies:
            return actor_elem
        return env_elem

    def _compute_mass_density(
        self,
        b: int,
        verts: np.ndarray,
        faces: np.ndarray,
        body_mass_np: np.ndarray | None,
    ) -> float:
        """Return the effective mass density for body *b*."""
        density = self._default_mass_density
        if body_mass_np is not None:
            vol = self._mesh_volume(verts, faces)
            if vol > 1e-12:
                density = float(body_mass_np[b]) / vol
        return density

    def build_affine_bodies(
        self,
        env_elem: Any,
        robo_elem: Any,
        actor_elem: Any,
        articulation_bodies: set[int],
        free_joint_bodies: set[int],
        body_range: tuple[int, int],
        subscene_elem: Any,
        body_transforms: np.ndarray | None = None,
        body_element_overrides: dict[int, Any] | None = None,
        no_instance_bodies: set[int] | None = None,
    ) -> None:
        """Convert Newton rigid bodies to UIPC AffineBody geometries.

        Bodies with identical canonical meshes are grouped into a single UIPC
        geometry with multiple instances (``sc.instances().resize(N)``).  Bodies
        listed in *no_instance_bodies* are always placed in their own geometry.

        Per-instance attributes (transform, ``is_fixed``, ``mass_density``) are
        set individually for each instance within a shared geometry.

        Contact element assignment priority:
        1. Per-body overrides in ``body_element_overrides``.
        2. ``robo_elem`` for bodies in ``articulation_bodies`` (non-free joints).
        3. ``actor_elem`` for bodies in ``free_joint_bodies``.
        4. ``env_elem`` for all other bodies (non-articulated / kinematic).

        Args:
            env_elem: Contact element for non-articulated (environment) bodies.
            robo_elem: Contact element for articulated (robot) bodies.
            actor_elem: Contact element for free-joint bodies.
            articulation_bodies: Set of body indices that belong to non-free
                joint articulations.
            free_joint_bodies: Set of body indices attached via free joints.
            body_range: ``(start, end)`` slice of bodies to process.
            subscene_elem: UIPC subscene element to apply to geometries, or
                ``None`` to skip.
            body_transforms: Pre-computed body world-frame transforms from
                :meth:`ArticulationBuilder.compute_fk`, shape
                ``(body_count, 4, 4)``.  If ``None``, identity transforms
                are used.
            body_element_overrides: Mapping from body index to a custom contact
                element.  Overrides the default assignment for the specified
                bodies.
            no_instance_bodies: Set of body indices that must NOT be grouped
                into instanced geometries (e.g. children of ball joints).
        """
        model = self._model
        if model.body_count == 0:
            return
        if model.body_flags is None:
            return

        body_flags_np = model.body_flags.numpy()
        body_mass_np = model.body_mass.numpy() if model.body_mass is not None else None
        no_inst = no_instance_bodies or set()

        # --- Phase A: Collect per-body data ---------------------------------
        body_infos: list[_BodyInfo] = []
        for b in range(body_range[0], body_range[1]):
            sk = _compute_shape_key(model, b)
            if sk is None:
                continue

            tf = body_transforms[b] if body_transforms is not None else np.eye(4, dtype=np.float64)
            elem = self._resolve_contact_elem(
                b,
                env_elem,
                robo_elem,
                actor_elem,
                articulation_bodies,
                free_joint_bodies,
                body_element_overrides,
            )
            is_kin = (body_flags_np[b] & int(BodyFlags.KINEMATIC)) != 0
            body_infos.append(_BodyInfo(b, sk, tf, 0.0, elem, is_kin))

        # --- Phase B: Group by (shape_key, contact element) -----------------
        from collections import OrderedDict  # noqa: PLC0415

        groups: OrderedDict[tuple[Any, ...], list[_BodyInfo]] = OrderedDict()
        for info in body_infos:
            if info.body_idx in no_inst:
                # Force unique group for excluded bodies
                key = (info.shape_key, id(info.contact_elem), info.body_idx)
            else:
                key = (info.shape_key, id(info.contact_elem))
            groups.setdefault(key, []).append(info)

        # --- Phase C: Create instanced geometries ---------------------------
        for group_bodies in groups.values():
            n = len(group_bodies)
            ref = group_bodies[0]

            # Build mesh once per group using the representative body
            mesh_data = build_body_mesh(model, ref.body_idx)
            if mesh_data is None:
                continue
            verts, faces = mesh_data

            # Compute per-body mass density (needs mesh volume)
            mesh_vol = self._mesh_volume(verts, faces)
            for info in group_bodies:
                if body_mass_np is not None and mesh_vol > 1e-12:
                    info.mass_density = float(body_mass_np[info.body_idx]) / mesh_vol
                else:
                    info.mass_density = self._default_mass_density

            sc = uipc_trimesh(verts, faces)
            if n > 1:
                sc.instances().resize(n)

            # Per-instance transforms
            transforms_view = view(sc.transforms())
            for i, info in enumerate(group_bodies):
                transforms_view[i] = info.transform

            # Contact element (shared for all instances in this group)
            ref.contact_elem.apply_to(sc)
            if subscene_elem is not None:
                subscene_elem.apply_to(sc)

            # Constitution with first body's mass density as default
            AffineBodyConstitution().apply_to(
                sc=sc,
                kappa=self._kappa,
                mass_density=ref.mass_density,
            )

            # Override per-instance mass density where different from reference
            density_view = view(sc.meta().find(uipc_builtin.mass_density))  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]
            density_view[:] = ref.mass_density

            label_surface(sc)

            # Per-instance kinematic flag
            is_fixed_view = view(sc.instances().find(uipc_builtin.is_fixed))  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]
            for i, info in enumerate(group_bodies):
                if info.is_kinematic:
                    is_fixed_view[i] = 1  # pyright: ignore[reportArgumentType]

            # Create UIPC object and geometry slot
            body_labels = "_".join(str(info.body_idx) for info in group_bodies)
            obj = self._scene.objects().create(f"body_{body_labels}")
            geo_slot, _ = obj.geometries().create(sc)

            # Record mapping for each body in the group
            for i, info in enumerate(group_bodies):
                self._mapping.body_geo_slots[info.body_idx] = geo_slot
                self._mapping.body_instance_ids[info.body_idx] = i
                self._mapping.body_objects[info.body_idx] = obj
