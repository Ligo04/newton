# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body (AffineBody) builder for the UIPC solver backend."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import uipc.builtin as uipc_builtin
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
    """

    def __init__(
        self,
        model: Model,
        scene: Any,
        contact_elem: Any,
        abd: AffineBodyConstitution,
        mapping: UIpcMappingInfo,
        kappa: float,
        default_mass_density: float,
    ):
        self._model = model
        self._scene = scene
        self._contact_elem = contact_elem
        self._abd = abd
        self._mapping = mapping
        self._kappa = kappa
        self._default_mass_density = default_mass_density

    def build_ground_planes(self) -> np.ndarray | None:
        """Create UIPC halfplanes for Newton ground plane shapes (body == -1).

        Returns:
            Pre-allocated body transform array of shape ``(body_count, 4, 4)``,
            or ``None`` if there are no bodies.
        """
        model = self._model
        if (
            model.shape_count == 0
            or model.shape_body is None
            or model.shape_type is None
            or model.shape_transform is None
        ):
            return None

        shape_body_np = model.shape_body.numpy()
        shape_type_np = model.shape_type.numpy()
        shape_transform_np = model.shape_transform.numpy()

        for s in range(model.shape_count):
            if shape_body_np[s] == -1 and GeoType(shape_type_np[s]) == GeoType.PLANE:
                tf = shape_transform_np[s]
                mat4 = newton_transform_to_mat4(tf[:3], tf[3:])
                normal = mat4[:3, 2].copy()
                center = tf[:3].copy()

                g = halfplane(center.astype(np.float64), normal.astype(np.float64))
                print(center, normal)
                ground_obj = self._scene.objects().create(f"ground_plane_{s}")
                ground_obj.geometries().create(g)

        if model.body_count > 0:
            return np.zeros((model.body_count, 4, 4), dtype=np.float64)
        return None

    def build_body_shape_mapping(self) -> None:
        """Populate ``mapping.body_shapes``: body_idx -> list of shape indices."""
        model = self._model
        if model.shape_count == 0 or model.shape_body is None:
            return

        shape_body_np = model.shape_body.numpy()
        for s in range(model.shape_count):
            b = shape_body_np[s]
            if b >= 0:
                self._mapping.body_shapes[b].append(s)

    def build_affine_bodies(self, body_transforms: np.ndarray | None) -> None:
        """Convert Newton rigid bodies to UIPC AffineBody geometries.

        Args:
            body_transforms: Pre-allocated array from :meth:`build_ground_planes`,
                populated with each body's world-frame 4x4 transform.
        """
        model = self._model
        if model.body_count == 0 or body_transforms is None:
            return

        import newton

        state = model.state()
        # Evaluate FK so articulated bodies have correct world-frame poses
        if model.joint_count > 0 and model.joint_q is not None:
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        if state.body_q is None or model.body_flags is None:
            return

        body_q_np = state.body_q.numpy()
        body_flags_np = model.body_flags.numpy()

        for b in range(model.body_count):
            mesh_data = build_body_mesh(model, b)
            if mesh_data is None:
                warnings.warn(f"Body {b}: no mesh shapes found, skipping", stacklevel=2)
                continue

            verts, faces = mesh_data
            sc = uipc_trimesh(verts, faces)

            tf = body_q_np[b]
            body_mat4 = newton_transform_to_mat4(tf[:3], tf[3:])
            body_transforms[b] = body_mat4
            view(sc.transforms())[:] = body_mat4

            self._contact_elem.apply_to(sc)
            self._abd.apply_to(sc=sc, kappa=self._kappa, mass_density=self._default_mass_density)
            label_surface(sc)

            is_kinematic = (body_flags_np[b] & int(BodyFlags.KINEMATIC)) != 0
            if is_kinematic:
                view(sc.instances().find(uipc_builtin.is_fixed))[:] = 1  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]

            obj = self._scene.objects().create(f"body_{b}")
            geo_slot, _ = obj.geometries().create(sc)
            self._mapping.body_geo_slots[b] = geo_slot
            self._mapping.body_objects[b] = obj
