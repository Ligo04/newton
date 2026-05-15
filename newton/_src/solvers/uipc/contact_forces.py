# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-body contact force extraction from UIPC.

Two paths:
  - GPU (hot path): ``prepare_contact_gpu_data`` → warp kernels scatter forces
    directly into ``state.body_f`` / ``state.particle_f`` / ``Contacts``.
  - CPU (diagnostic): ``retrieve_contact_forces`` → ``ContactForceReadback``
    for ``get_contact_forces()`` readback API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import uipc.builtin as uipc_builtin
import warp as wp
from uipc.core import ContactSystemFeature
from uipc.geometry import Geometry

from newton._src.solvers.uipc.utils import _view_attr

from .converter import UIpcMappingInfo

PRIMITIVE_TYPES: tuple[str, ...] = ("PH", "PP", "PE", "PT", "EE")
FORCE_CHANNELS: tuple[str, ...] = ("N", "F")

_PRIM_SPLIT: dict[str, int] = {"PH": 1, "PP": 1, "PE": 1, "PT": 1, "EE": 2}

_VERTEX_UNMAPPED = -1


# ---------------------------------------------------------------------------
# Shared gradient reading
# ---------------------------------------------------------------------------


def _read_gradient(csf: ContactSystemFeature, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read one primitive+channel gradient from UIPC.

    Returns (i_view, grad_flat) or None if empty.
    i_view: (num_instances, arity) int array of vertex indices.
    grad_flat: (num_instances, arity, 3) float array of gradients.
    """
    geo = Geometry()
    csf.contact_gradient(key, geo)

    i_attr = geo.instances().find("i")
    grad_attr = geo.instances().find("grad")
    if i_attr is None or grad_attr is None:
        return None

    i_view = _view_attr(i_attr)
    grad_view = _view_attr(grad_attr)

    num_instances = i_view.shape[0]
    if num_instances == 0:
        return None

    if i_view.ndim == 1:
        i_view = i_view.reshape(-1, 1)
    arity = i_view.shape[1]
    grad_flat = grad_view.reshape(num_instances, arity, 3)
    return i_view, grad_flat


# ---------------------------------------------------------------------------
# CPU diagnostic path (ContactForceReadback)
# ---------------------------------------------------------------------------


@dataclass
class PerBodyPrimitiveForce:
    """Per-vertex contact forces on one body from one primitive type + channel."""

    vertex_indices: np.ndarray  # (K,) int64
    forces: np.ndarray  # (K, 3) float64
    torques: np.ndarray  # (K, 3) float64

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
    """

    data: dict[int, dict[str, dict[str, PerBodyPrimitiveForce]]] = field(default_factory=dict)

    def body_total(self, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Sum all force/torque into a single (f, tau) pair, both shape (3,)."""
        f = np.zeros(3, dtype=np.float64)
        tau = np.zeros(3, dtype=np.float64)
        for prim_dict in self.data.get(body_idx, {}).values():
            for pbf in prim_dict.values():
                if pbf.forces.size:
                    f += pbf.forces.sum(axis=0)
                if pbf.torques.size:
                    tau += pbf.torques.sum(axis=0)
        return f, tau


def _quat_rotate(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion (x, y, z, w)."""
    q_xyz = quat[:3]
    q_w = quat[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def retrieve_contact_forces(
    csf: ContactSystemFeature,
    mapping: UIpcMappingInfo,
    body_q_np: np.ndarray | None,
    body_com_np: np.ndarray | None,
    dt: float = 1.0,
) -> ContactForceReadback:
    """CPU diagnostic: read all contact gradients and bucket by body.

    Args:
        csf: UIPC contact system feature.
        mapping: Newton↔UIPC mapping info (needs vertex_to_body CPU data).
        body_q_np: (body_count, 7) body transforms [px,py,pz, qx,qy,qz,qw].
        body_com_np: (body_count, 3) body center-of-mass offsets.
        dt: Simulation time step [s] for gradient→force conversion.

    Returns:
        ContactForceReadback with per-body per-primitive forces and torques.
    """
    inv_dt2 = 1.0 / (dt * dt) if dt != 0.0 else 1.0
    result = ContactForceReadback()

    vtb_np = _build_vertex_to_body_np(mapping)
    if vtb_np is None:
        return result

    for prim in PRIMITIVE_TYPES:
        split_a = _PRIM_SPLIT[prim]
        for chan in FORCE_CHANNELS:
            key = f"{prim}+{chan}"
            data = _read_gradient(csf, key)
            if data is None:
                continue
            i_view, grad_flat = data
            n_inst = i_view.shape[0]
            arity = i_view.shape[1]

            for inst_idx in range(n_inst):
                verts = i_view[inst_idx]
                grads = grad_flat[inst_idx]

                for v_local in range(arity):
                    gv = int(verts[v_local])
                    if gv < 0 or gv >= len(vtb_np):
                        continue
                    body_idx = int(vtb_np[gv])
                    if body_idx == _VERTEX_UNMAPPED:
                        continue

                    force = -grads[v_local].astype(np.float64) * inv_dt2

                    torque = np.zeros(3, dtype=np.float64)
                    if body_q_np is not None and body_com_np is not None and 0 <= body_idx < len(body_q_np):
                        q = body_q_np[body_idx]
                        com_local = body_com_np[body_idx]
                        pos = q[:3]
                        quat = q[3:]
                        com_world = pos + _quat_rotate(quat, com_local)
                        # Approximate: use body COM for torque arm
                        # (vertex world position not available here without mesh readback)
                        torque = np.cross(-com_world + pos, force)

                    side = "A" if v_local < split_a else "B"
                    prim_key = f"{prim}.{side}"

                    if body_idx not in result.data:
                        result.data[body_idx] = {}
                    if prim_key not in result.data[body_idx]:
                        result.data[body_idx][prim_key] = {}
                    if chan not in result.data[body_idx][prim_key]:
                        result.data[body_idx][prim_key][chan] = PerBodyPrimitiveForce(
                            vertex_indices=np.array([gv], dtype=np.int64),
                            forces=force.reshape(1, 3),
                            torques=torque.reshape(1, 3),
                        )
                    else:
                        pbf = result.data[body_idx][prim_key][chan]
                        pbf.vertex_indices = np.append(pbf.vertex_indices, gv)
                        pbf.forces = np.vstack([pbf.forces, force])
                        pbf.torques = np.vstack([pbf.torques, torque])

    return result


def _build_vertex_to_body_np(mapping: UIpcMappingInfo) -> np.ndarray | None:
    """Build CPU vertex→body lookup from mapping (for diagnostic path)."""
    if not mapping.body_geo_slots:
        return None

    max_gv = 0
    body_vertex_ranges: list[tuple[int, int, int]] = []

    for body_idx, geo_slot in mapping.body_geo_slots.items():
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(_view_attr(offset_attr)[0])
        n_verts = geo.vertices().size()
        instance_id = mapping.body_instance_ids.get(body_idx, 0)
        start = base_offset + instance_id * n_verts
        end = start + n_verts
        body_vertex_ranges.append((body_idx, start, end))
        max_gv = max(max_gv, end)

    if max_gv == 0:
        return None

    vtb = np.full(max_gv, _VERTEX_UNMAPPED, dtype=np.int32)
    for body_idx, start, end in body_vertex_ranges:
        vtb[start:end] = body_idx

    return vtb


# ---------------------------------------------------------------------------
# GPU path: vertex map building
# ---------------------------------------------------------------------------


def build_gpu_vertex_maps(mapping: UIpcMappingInfo, body_count: int, device: wp.Device) -> None:
    """Build GPU lookup tables for contact force scatter.

    Populates ``mapping.vertex_to_body_wp``, ``mapping.vertex_to_particle_wp``,
    and ``mapping.body_to_first_shape_wp``.

    Args:
        mapping: The mapping info to populate (modified in-place).
        body_count: Number of rigid bodies in the model.
        device: Warp device for output arrays.
    """
    max_gv = 0
    body_vertex_ranges: list[tuple[int, int, int]] = []

    for body_idx, geo_slot in mapping.body_geo_slots.items():
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(_view_attr(offset_attr)[0])
        n_verts = geo.vertices().size()
        instance_id = mapping.body_instance_ids.get(body_idx, 0)
        start = base_offset + instance_id * n_verts
        end = start + n_verts
        body_vertex_ranges.append((body_idx, start, end))
        max_gv = max(max_gv, end)

    particle_vertex_ranges: list[tuple[np.ndarray, int, int]] = []
    for geo_slot, particle_indices in [
        *zip(mapping.cloth_geo_slots, mapping.cloth_particle_indices, strict=False),
        *zip(mapping.deformable_geo_slots, mapping.deformable_particle_indices, strict=False),
    ]:
        geo = geo_slot.geometry()
        offset_attr = geo.meta().find("backend_fem_vertex_offset") or geo.meta().find(uipc_builtin.global_vertex_offset)
        if offset_attr is None:
            continue
        base_offset = int(_view_attr(offset_attr)[0])
        pi = np.asarray(particle_indices, dtype=np.int32)
        end = base_offset + len(pi)
        particle_vertex_ranges.append((pi, base_offset, end))
        max_gv = max(max_gv, end)

    if max_gv == 0:
        mapping.max_global_vertex = 0
        return

    vtb_np = np.full(max_gv, _VERTEX_UNMAPPED, dtype=np.int32)
    vtp_np = np.full(max_gv, _VERTEX_UNMAPPED, dtype=np.int32)

    for body_idx, start, end in body_vertex_ranges:
        vtb_np[start:end] = body_idx

    for pi, start, end in particle_vertex_ranges:
        vtp_np[start:end] = pi

    mapping.vertex_to_body_wp = wp.from_numpy(vtb_np, dtype=wp.int32, device=device)
    mapping.vertex_to_particle_wp = wp.from_numpy(vtp_np, dtype=wp.int32, device=device)
    mapping.max_global_vertex = max_gv

    b2s_np = np.zeros(body_count, dtype=np.int32)
    for body_idx, shapes in mapping.body_shapes.items():
        if 0 <= body_idx < body_count and shapes:
            b2s_np[body_idx] = shapes[0]
    mapping.body_to_first_shape_wp = wp.from_numpy(b2s_np, dtype=wp.int32, device=device)


# ---------------------------------------------------------------------------
# GPU path: warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _scatter_contact_forces_kernel(
    vertex_indices: wp.array[wp.int32],
    forces: wp.array[wp.vec3],
    vertex_to_body: wp.array[wp.int32],
    vertex_to_particle: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_count: int,
    max_global_vertex: int,
    body_f: wp.array[wp.spatial_vector],
    particle_f: wp.array[wp.vec3],
):
    tid = wp.tid()
    gv = vertex_indices[tid]
    if gv < 0 or gv >= max_global_vertex:
        return

    f = forces[tid]

    body_idx = vertex_to_body[gv]
    if body_idx >= 0 and body_idx < body_count:
        q = body_q[body_idx]
        pos = wp.transform_get_translation(q)
        rot = wp.transform_get_rotation(q)
        com_local = body_com[body_idx]
        com_world = pos + wp.quat_rotate(rot, com_local)
        r = pos - com_world
        tau = wp.cross(r, f)
        wp.atomic_add(body_f, body_idx, wp.spatial_vector(f[0], f[1], f[2], tau[0], tau[1], tau[2]))
        return

    particle_idx = vertex_to_particle[gv]
    if particle_idx >= 0:
        wp.atomic_add(particle_f, particle_idx, f)


@wp.kernel
def _populate_contact_pairs_kernel(
    vert_a: wp.array[wp.int32],
    vert_b: wp.array[wp.int32],
    forces_n: wp.array[wp.vec3],
    forces_f: wp.array[wp.vec3],
    vertex_to_body: wp.array[wp.int32],
    body_to_first_shape: wp.array[wp.int32],
    body_count: int,
    max_global_vertex: int,
    ground_shape: int,
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_force: wp.array[wp.spatial_vector],  # ty:ignore[invalid-type-form]
    counter: wp.array[wp.int32],
    max_contacts: int,
):
    tid = wp.tid()
    gv_a = vert_a[tid]
    gv_b = vert_b[tid]

    if gv_a < 0 or gv_a >= max_global_vertex:
        return

    body_a = vertex_to_body[gv_a]
    if body_a < 0 or body_a >= body_count:
        return

    shape_a = body_to_first_shape[body_a]

    shape_b = ground_shape
    if gv_b >= 0 and gv_b < max_global_vertex:
        body_b = vertex_to_body[gv_b]
        if body_b >= 0 and body_b < body_count:
            shape_b = body_to_first_shape[body_b]

    slot = wp.atomic_add(counter, 0, 1)
    if slot >= max_contacts:
        return

    fn = forces_n[tid]
    ff = forces_f[tid]
    total_f = fn + ff
    n_len = wp.length(fn)
    normal = wp.vec3(0.0, 1.0, 0.0)
    if n_len > 1.0e-12:
        normal = fn / n_len

    contact_shape0[slot] = shape_a
    contact_shape1[slot] = shape_b
    contact_normal[slot] = normal
    contact_force[slot] = wp.spatial_vector(total_f[0], total_f[1], total_f[2], 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# GPU path: data preparation (vectorized numpy, no per-element Python loop)
# ---------------------------------------------------------------------------


@dataclass
class GpuContactData:
    """Packed contact data ready for GPU transfer."""

    vertex_indices: np.ndarray  # (N,) int32 — all vertices with forces
    forces: np.ndarray  # (N, 3) float32 — force per vertex

    vert_a: np.ndarray  # (M,) int32 — per-instance body-A representative vertex
    vert_b: np.ndarray  # (M,) int32 — per-instance body-B representative vertex
    forces_n: np.ndarray  # (M, 3) float32 — normal channel force sum per instance
    forces_f: np.ndarray  # (M, 3) float32 — friction channel force sum per instance


def prepare_contact_gpu_data(csf: ContactSystemFeature, dt: float = 1.0) -> GpuContactData | None:
    """Read all UIPC contact gradients and pack into flat arrays.

    UIPC gradients are derivatives of the incremental potential which
    includes a ``kappa · dt²`` scaling.  Dividing by ``dt²`` converts
    them to physical forces [N].

    No per-element Python loops — uses vectorized numpy operations.
    Returns None if no contact data is available.
    """
    inv_dt2 = 1.0 / (dt * dt) if dt != 0.0 else 1.0
    all_verts: list[np.ndarray] = []
    all_forces: list[np.ndarray] = []

    # Per-instance data for contact pairs (keyed by primitive type)
    inst_vert_a: list[np.ndarray] = []
    inst_vert_b: list[np.ndarray] = []
    inst_force_n: list[np.ndarray] = []
    inst_force_f: list[np.ndarray] = []

    for prim in PRIMITIVE_TYPES:
        split_a = _PRIM_SPLIT[prim]

        n_data = _read_gradient(csf, f"{prim}+N")
        f_data = _read_gradient(csf, f"{prim}+F")

        for chan_label, data in [("N", n_data), ("F", f_data)]:
            if data is None:
                continue
            i_view, grad_flat = data
            n_inst = i_view.shape[0]
            arity = i_view.shape[1]

            verts_flat = i_view.reshape(-1).astype(np.int32)
            forces_flat = (-grad_flat.reshape(-1, 3) * inv_dt2).astype(np.float32)
            all_verts.append(verts_flat)
            all_forces.append(forces_flat)

            if chan_label == "N":
                va = i_view[:, 0].astype(np.int32)
                inst_vert_a.append(va)
                if arity > split_a:
                    vb = i_view[:, split_a].astype(np.int32)
                else:
                    vb = np.full(n_inst, -1, dtype=np.int32)
                inst_vert_b.append(vb)
                force_sum = (-grad_flat.sum(axis=1) * inv_dt2).astype(np.float32)
                inst_force_n.append(force_sum)

            if chan_label == "F":
                if not inst_force_f or len(inst_force_f[-1]) != n_inst:
                    force_sum = (-grad_flat.sum(axis=1) * inv_dt2).astype(np.float32)
                    inst_force_f.append(force_sum)

    if not all_verts:
        return None

    vertex_indices = np.concatenate(all_verts)
    forces = np.concatenate(all_forces)

    n_pairs = sum(len(a) for a in inst_vert_a)
    if n_pairs == 0:
        return GpuContactData(
            vertex_indices=vertex_indices,
            forces=forces,
            vert_a=np.empty(0, dtype=np.int32),
            vert_b=np.empty(0, dtype=np.int32),
            forces_n=np.empty((0, 3), dtype=np.float32),
            forces_f=np.empty((0, 3), dtype=np.float32),
        )

    vert_a_cat = np.concatenate(inst_vert_a)
    vert_b_cat = np.concatenate(inst_vert_b)
    forces_n_cat = np.concatenate(inst_force_n)

    if inst_force_f:
        forces_f_cat = np.concatenate(inst_force_f)
        if len(forces_f_cat) < len(vert_a_cat):
            pad = np.zeros((len(vert_a_cat) - len(forces_f_cat), 3), dtype=np.float32)
            forces_f_cat = np.concatenate([forces_f_cat, pad])
    else:
        forces_f_cat = np.zeros_like(forces_n_cat)

    return GpuContactData(
        vertex_indices=vertex_indices,
        forces=forces,
        vert_a=vert_a_cat,
        vert_b=vert_b_cat,
        forces_n=forces_n_cat,
        forces_f=forces_f_cat,
    )
