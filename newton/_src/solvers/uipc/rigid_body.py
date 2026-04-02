# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body (AffineBody) builder for the UIPC solver backend."""

from __future__ import annotations

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
    ) -> None:
        """Convert Newton rigid bodies to UIPC AffineBody geometries.

        For each body, the effective mass density is computed from the model's
        ``body_mass`` and the mesh volume. If the mass or volume is unavailable,
        ``default_mass_density`` is used as a fallback.

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
            body_range: ``(start, end)`` slice of bodies to process, or
                ``None`` for all bodies.
            subscene_elem: UIPC subscene element to apply to geometries, or
                ``None`` to skip.
            body_transforms: Pre-computed body world-frame transforms from
                :meth:`ArticulationBuilder.compute_fk`, shape
                ``(body_count, 4, 4)``.  If ``None``, identity transforms
                are used.
            body_element_overrides: Mapping from body index to a custom contact
                element.  Overrides the default assignment for the specified
                bodies.
        """
        model = self._model
        if model.body_count == 0:
            return

        if model.body_flags is None:
            return

        body_flags_np = model.body_flags.numpy()
        body_mass_np = model.body_mass.numpy() if model.body_mass is not None else None

        for b in range(body_range[0], body_range[1]):
            mesh_data = build_body_mesh(model, b)
            if mesh_data is None:
                continue

            verts, faces = mesh_data
            sc = uipc_trimesh(verts, faces)

            if body_transforms is not None:
                view(sc.transforms())[:] = body_transforms[b]
            else:
                view(sc.transforms())[:] = np.eye(4, dtype=np.float64)

            # Compute per-body mass density from model mass and mesh volume
            mass_density = self._default_mass_density
            if body_mass_np is not None:
                vol = self._mesh_volume(verts, faces)
                if vol > 1e-12:
                    mass_density = float(body_mass_np[b]) / vol

            # Per-body override > articulation check > free joint > default env
            if body_element_overrides is not None and b in body_element_overrides:
                elem = body_element_overrides[b]
            elif b in articulation_bodies:
                elem = robo_elem
            elif b in free_joint_bodies:
                elem = actor_elem
            else:
                elem = env_elem
            elem.apply_to(sc)
            if subscene_elem is not None:
                subscene_elem.apply_to(sc)
            AffineBodyConstitution().apply_to(sc=sc, kappa=self._kappa, mass_density=mass_density)
            label_surface(sc)

            is_kinematic = (body_flags_np[b] & int(BodyFlags.KINEMATIC)) != 0
            if is_kinematic:
                view(sc.instances().find(uipc_builtin.is_fixed))[:] = 1  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]

            obj = self._scene.objects().create(f"body_{b}")
            geo_slot, _ = obj.geometries().create(sc)
            self._mapping.body_geo_slots[b] = geo_slot
            self._mapping.body_objects[b] = obj
