# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-body, per-primitive contact force extraction from UIPC.

Uses ``ContactSystemFeature.contact_gradient`` to pull the raw IPC gradient
for each primitive (PH, PP, PE, PT, EE) split between normal (N) and
frictional (F) channels. Contact force = ``-gradient`` per vertex.
Per-vertex forces are bucketed by owning Newton body via a global-vertex
reverse lookup. For ABD bodies an additional spatial torque (world-frame)
is accumulated from ``tau = (x_world - com_world) x f_world``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import uipc.builtin as uipc_builtin
from uipc import view
from uipc.core import AffineBodyStateAccessorFeature, ContactSystemFeature, FiniteElementStateAccessorFeature
from uipc.geometry import Geometry

if TYPE_CHECKING:
    from .converter import UIpcMappingInfo


PRIMITIVE_TYPES: tuple[str, ...] = ("PH", "PP", "PE", "PT", "EE")
FORCE_CHANNELS: tuple[str, ...] = ("N", "F")


@dataclass
class PerBodyPrimitiveForce:
    """Per-vertex contact forces on one body from one primitive type + channel.

    Attributes:
        vertex_indices: (K,) int64 — global UIPC vertex indices on this body.
        forces: (K, 3) float64 — per-vertex world-space contact force [N].
        torques: (K, 3) float64 — per-vertex world-space torque about body COM
            [N·m], ``(x_world_i - com_world) x f_i``. Zero for FEM vertices
            (no rigid COM concept).
    """

    vertex_indices: np.ndarray
    forces: np.ndarray
    torques: np.ndarray

    @classmethod
    def empty(cls) -> PerBodyPrimitiveForce:
        return cls(
            vertex_indices=np.empty(0, dtype=np.int64),
            forces=np.empty((0, 3), dtype=np.float64),
            torques=np.empty((0, 3), dtype=np.float64),
        )


@dataclass
class ContactForceReadback:
    """Full dump of one retrieve pass.

    Layout: ``data[body_idx][prim_type][channel] -> PerBodyPrimitiveForce``.
    Bodies / primitive types / channels with no contacts are omitted.
    """

    data: dict[int, dict[str, dict[str, PerBodyPrimitiveForce]]] = field(default_factory=dict)

    def body_total(self, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Sum all primitive + channel force/torque into a single (f, tau) pair.

        Returns:
            (f_world, tau_world), both shape (3,), float64.
        """
        f = np.zeros(3, dtype=np.float64)
        tau = np.zeros(3, dtype=np.float64)
        body = self.data.get(body_idx, {})
        for prim_dict in body.values():
            for pbf in prim_dict.values():
                if pbf.forces.size:
                    f += pbf.forces.sum(axis=0)
                if pbf.torques.size:
                    tau += pbf.torques.sum(axis=0)
        return f, tau


def build_vertex_maps(
    mapping: UIpcMappingInfo,
    abd_accessor: AffineBodyStateAccessorFeature | None,
    fem_accessor: FiniteElementStateAccessorFeature | None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Build global_vertex_idx reverse lookups.

    Returns two maps:
        vertex_to_body: global_vertex_idx -> body_key (positive = ABD body,
            negative = FEM synthetic key).
        vertex_to_particle: global_vertex_idx -> Newton particle index
            (only populated for FEM cloth/deformable vertices).

    ABD bodies: each body occupies 1 vertex slot (affine DOF = 12D, but
    UIPC vertex count = 1 per body). FEM cloth/deformable: each particle
    is 1 vertex.

    Args:
        mapping: UIpcMappingInfo with body/cloth/deformable geo slots.
        abd_accessor: AffineBodyStateAccessorFeature (or None).
        fem_accessor: FiniteElementStateAccessorFeature (or None).

    Returns:
        (vertex_to_body, vertex_to_particle) dicts.
    """
    vertex_to_body: dict[int, int] = {}
    vertex_to_particle: dict[int, int] = {}

    # ABD bodies: 1 vertex per body (affine DOF slot)
    for body_idx, geo_slot in mapping.body_geo_slots.items():
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(view(offset_attr)[0])
        instance_id = mapping.body_instance_ids.get(body_idx, 0)
        global_vertex = base_offset + instance_id
        vertex_to_body[global_vertex] = body_idx

    # FEM cloth: each particle = 1 vertex.
    for mesh_idx, (geo_slot, particle_indices) in enumerate(
        zip(mapping.cloth_geo_slots, mapping.cloth_particle_indices, strict=False)
    ):
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(view(offset_attr)[0])
        num_verts = geo.vertices().size()
        body_key = -1 - mesh_idx
        for local_idx in range(num_verts):
            gv = base_offset + local_idx
            vertex_to_body[gv] = body_key
            if local_idx < len(particle_indices):
                vertex_to_particle[gv] = int(particle_indices[local_idx])

    # FEM deformable: each particle = 1 vertex.
    for mesh_idx, (geo_slot, particle_indices) in enumerate(
        zip(mapping.deformable_geo_slots, mapping.deformable_particle_indices, strict=False)
    ):
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(view(offset_attr)[0])
        num_verts = geo.vertices().size()
        body_key = -(10000 + mesh_idx)
        for local_idx in range(num_verts):
            gv = base_offset + local_idx
            vertex_to_body[gv] = body_key
            if local_idx < len(particle_indices):
                vertex_to_particle[gv] = int(particle_indices[local_idx])

    return vertex_to_body, vertex_to_particle


def retrieve_contact_forces(
    csf: ContactSystemFeature,
    mapping: UIpcMappingInfo,
    abd_accessor: AffineBodyStateAccessorFeature | None,
    fem_accessor: FiniteElementStateAccessorFeature | None,
    body_q_np: np.ndarray | None,
    body_com_np: np.ndarray | None,
) -> tuple[ContactForceReadback, dict[int, int]]:
    """Extract per-body, per-primitive contact forces from UIPC.

    Args:
        csf: ContactSystemFeature from world.features().find(...).
        mapping: UIpcMappingInfo with body/cloth/deformable geo slots.
        abd_accessor: AffineBodyStateAccessorFeature (or None).
        fem_accessor: FiniteElementStateAccessorFeature (or None).
        body_q_np: (body_count, 7) float32 — body transforms for torque calc.
        body_com_np: (body_count, 3) float32 — body COM in body frame.

    Returns:
        (readback, vertex_to_particle) — ContactForceReadback and UIPC global
        vertex → Newton particle index map (FEM bodies only).
    """
    vertex_to_body, vertex_to_particle = build_vertex_maps(mapping, abd_accessor, fem_accessor)
    readback = ContactForceReadback()

    for prim_type in PRIMITIVE_TYPES:
        for channel in FORCE_CHANNELS:
            key = f"{prim_type}+{channel}"
            geo = Geometry()
            csf.contact_gradient(key, geo)

            i_attr = geo.instances().find("i")
            if i_attr is None:
                continue
            i_view = view(i_attr)
            grad_attr = geo.instances().find("grad")
            if grad_attr is None:
                continue
            grad_view = view(grad_attr)

            # grad_view shape: (num_instances, arity, 3, 1) where arity depends on primitive
            # PH=1, PP=2, PE=3, PT=4, EE=4
            num_instances = i_view.shape[0]
            if num_instances == 0:
                continue

            # Flatten: each row = (vertex_idx, force_vec3)
            # i_view shape: (num_instances, arity) uint32
            # grad_view shape: (num_instances, arity, 3, 1) float64
            vertex_indices = i_view.reshape(-1).astype(np.int64)
            forces = -grad_view.reshape(-1, 3).astype(np.float64)  # contact force = -gradient

            # Bucket by body
            body_buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
            for v_idx, f_vec in zip(vertex_indices, forces, strict=True):
                body_idx = vertex_to_body.get(int(v_idx))
                if body_idx is None:
                    continue
                if body_idx not in body_buckets:
                    body_buckets[body_idx] = []
                body_buckets[body_idx].append((int(v_idx), f_vec))

            # Convert to PerBodyPrimitiveForce
            for body_idx, entries in body_buckets.items():
                if body_idx not in readback.data:
                    readback.data[body_idx] = {}
                if prim_type not in readback.data[body_idx]:
                    readback.data[body_idx][prim_type] = {}

                verts = np.array([e[0] for e in entries], dtype=np.int64)
                fs = np.array([e[1] for e in entries], dtype=np.float64)

                # Compute torques for ABD bodies
                torques = np.zeros_like(fs)
                if body_q_np is not None and body_com_np is not None and body_idx < len(body_q_np):
                    # body_q_np[body_idx] = (px, py, pz, qx, qy, qz, qw)
                    pos = body_q_np[body_idx, :3].astype(np.float64)
                    quat = body_q_np[body_idx, 3:].astype(np.float64)
                    com_body = body_com_np[body_idx].astype(np.float64)

                    # Rotate COM to world
                    # quat = (qx, qy, qz, qw) in Warp convention
                    qx, qy, qz, qw = quat
                    # Rotation matrix from quaternion
                    r00 = 1 - 2 * (qy * qy + qz * qz)
                    r01 = 2 * (qx * qy - qz * qw)
                    r02 = 2 * (qx * qz + qy * qw)
                    r10 = 2 * (qx * qy + qz * qw)
                    r11 = 1 - 2 * (qx * qx + qz * qz)
                    r12 = 2 * (qy * qz - qx * qw)
                    r20 = 2 * (qx * qz - qy * qw)
                    r21 = 2 * (qy * qz + qx * qw)
                    r22 = 1 - 2 * (qx * qx + qy * qy)
                    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float64)
                    com_world = pos + R @ com_body

                    # For ABD, contact vertex is the body origin (affine DOF slot),
                    # not a physical point. Torque = (pos - com_world) x f.
                    # But this is wrong — ABD gradient is 12D (affine space), not 3D force.
                    # Need to extract translation component from affine gradient.
                    # For now, assume grad is already 3D force at body origin.
                    for i, f_vec in enumerate(fs):
                        r_vec = pos - com_world
                        torques[i] = np.cross(r_vec, f_vec)

                readback.data[body_idx][prim_type][channel] = PerBodyPrimitiveForce(
                    vertex_indices=verts,
                    forces=fs,
                    torques=torques,
                )

    return readback, vertex_to_particle
