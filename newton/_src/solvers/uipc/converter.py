# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conversion utilities between Newton Model and UIPC Scene objects."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import uipc.builtin as uipc_builtin
import warp as wp
from uipc import Quaternion, Transform

from ...geometry import GeoType, Mesh
from ...sim import Model

# ---------------------------------------------------------------------------
# Warp kernels for batch transform conversion
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _transform_to_mat44_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_indices: wp.array(dtype=wp.int32),
    out_transforms: wp.array(dtype=wp.mat44d, ndim=1),
):
    """Convert Newton wp.transform (pos + quat) to 4x4 double matrices in batch."""
    tid = wp.tid()
    body_idx = body_indices[tid]
    q = body_q[body_idx]
    p = wp.transform_get_translation(q)
    r = wp.transform_get_rotation(q)

    # Quaternion to rotation matrix (Warp quat: x, y, z, w)
    x = wp.float64(r[0])
    y = wp.float64(r[1])
    z = wp.float64(r[2])
    w = wp.float64(r[3])

    one = wp.float64(1.0)
    two = wp.float64(2.0)
    zero = wp.float64(0.0)
    px = wp.float64(p[0])
    py = wp.float64(p[1])
    pz = wp.float64(p[2])

    m = wp.mat44d(
        one - two * (y * y + z * z),
        two * (x * y - z * w),
        two * (x * z + y * w),
        px,
        two * (x * y + z * w),
        one - two * (x * x + z * z),
        two * (y * z - x * w),
        py,
        two * (x * z - y * w),
        two * (y * z + x * w),
        one - two * (x * x + y * y),
        pz,
        zero,
        zero,
        zero,
        one,
    )
    out_transforms[tid] = m


@wp.kernel(enable_backward=False)
def _spatial_to_vel_mat44_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_indices: wp.array(dtype=wp.int32),
    out_velocities: wp.array(dtype=wp.mat44d, ndim=1),
):
    """Convert Newton wp.spatial_vector (linear, angular) to 4x4 velocity matrices in batch.

    Newton spatial_vector layout: (vx, vy, vz, wx, wy, wz).
    UIPC velocity matrix: upper-left 3x3 = skew(angular), last column = linear.
    """
    tid = wp.tid()
    body_idx = body_indices[tid]
    qd = body_qd[body_idx]

    vx = wp.float64(qd[0])
    vy = wp.float64(qd[1])
    vz = wp.float64(qd[2])
    wx = wp.float64(qd[3])
    wy = wp.float64(qd[4])
    wz = wp.float64(qd[5])
    zero = wp.float64(0.0)

    m = wp.mat44d(
        zero,
        -wz,
        wy,
        vx,
        wz,
        zero,
        -wx,
        vy,
        -wy,
        wx,
        zero,
        vz,
        zero,
        zero,
        zero,
        zero,
    )
    out_velocities[tid] = m


@wp.kernel(enable_backward=False)
def _write_to_backend_kernel(
    backend_offsets: wp.array(dtype=wp.uint32),
    src_transforms: wp.array(dtype=wp.mat44d, ndim=1),
    src_velocities: wp.array(dtype=wp.mat44d, ndim=1),
    dst_transforms: wp.array(dtype=wp.mat44d, ndim=1),
    dst_velocities: wp.array(dtype=wp.mat44d, ndim=1),
):
    """Scatter body transforms/velocities into UIPC backend arrays by offset."""
    tid = wp.tid()
    dst_idx = backend_offsets[tid]
    dst_transforms[dst_idx] = src_transforms[tid]
    dst_velocities[dst_idx] = src_velocities[tid]


@wp.kernel(enable_backward=False)
def _read_from_backend_kernel(
    backend_offsets: wp.array(dtype=wp.uint32),
    src_transforms: wp.array(dtype=wp.mat44d, ndim=1),
    src_velocities: wp.array(dtype=wp.mat44d, ndim=1),
    body_indices: wp.array(dtype=wp.int32),
    out_body_q: wp.array(dtype=wp.transform),
    out_body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Gather UIPC backend transforms/velocities back into Newton state arrays."""
    tid = wp.tid()
    backend_idx = backend_offsets[tid]
    body_idx = body_indices[tid]

    m = src_transforms[backend_idx]

    # Extract position
    px = wp.float32(m[0, 3])
    py = wp.float32(m[1, 3])
    pz = wp.float32(m[2, 3])

    # Extract quaternion from rotation matrix (Shepperd's method)
    r00 = m[0, 0]
    r11 = m[1, 1]
    r22 = m[2, 2]
    trace = r00 + r11 + r22

    zero_d = wp.float64(0.0)
    one_d = wp.float64(1.0)
    two_d = wp.float64(2.0)
    quarter_d = wp.float64(0.25)

    qx = zero_d
    qy = zero_d
    qz = zero_d
    qw = one_d

    if trace > zero_d:
        s = wp.sqrt(trace + one_d) * two_d
        qw = quarter_d * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif r00 > r11 and r00 > r22:
        s = wp.sqrt(one_d + r00 - r11 - r22) * two_d
        qw = (m[2, 1] - m[1, 2]) / s
        qx = quarter_d * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif r11 > r22:
        s = wp.sqrt(one_d + r11 - r00 - r22) * two_d
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = quarter_d * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = wp.sqrt(one_d + r22 - r00 - r11) * two_d
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = quarter_d * s

    # Normalize quaternion
    norm = wp.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > zero_d:
        inv_norm = one_d / norm
        qx = qx * inv_norm
        qy = qy * inv_norm
        qz = qz * inv_norm
        qw = qw * inv_norm

    out_body_q[body_idx] = wp.transform(
        wp.vec3(px, py, pz),
        wp.quat(wp.float32(qx), wp.float32(qy), wp.float32(qz), wp.float32(qw)),
    )

    # Extract velocity: linear from last column, angular from skew-symmetric
    v = src_velocities[backend_idx]
    vx = wp.float32(v[0, 3])
    vy = wp.float32(v[1, 3])
    vz = wp.float32(v[2, 3])
    wx = wp.float32(v[2, 1])
    wy = wp.float32(v[0, 2])
    wz = wp.float32(v[1, 0])

    out_body_qd[body_idx] = wp.spatial_vector(vx, vy, vz, wx, wy, wz)


# ---------------------------------------------------------------------------
# Numpy-level helpers (used only during scene construction, not per-step)
# ---------------------------------------------------------------------------


def newton_transform_to_mat4(tf: wp.transform) -> np.ndarray:  # pyright: ignore[reportArgumentType]
    """Convert a Newton transform to a 4x4 homogeneous matrix.

    Args:
        tf: A ``wp.transform`` value (``p``: vec3, ``q``: quat).

    Returns:
        4x4 homogeneous transformation matrix (float64).
    """
    # Warp stores quaternions as (x, y, z, w) but UIPC Quaternion expects (w, x, y, z)
    q = tf.q
    tran = Transform.Identity()
    tran.rotate(Quaternion(wp.quat(q[3], q[0], q[1], q[2])))
    tran.pretranslate(tf.p)
    return tran.matrix()


# ---------------------------------------------------------------------------
# Mapping data structure
# ---------------------------------------------------------------------------


@dataclass
class UIpcMappingInfo:
    """Stores the mapping between Newton model indices and UIPC objects."""

    # body_idx -> UIPC geometry slot
    body_geo_slots: dict[int, Any] = field(default_factory=dict)
    # joint_idx -> UIPC joint geometry slot
    joint_geo_slots: dict[int, Any] = field(default_factory=dict)
    # joint_idx -> UIPC joint linemesh (for reading angle/position)
    joint_mesh: dict[int, Any] = field(default_factory=dict)
    # body_idx -> UIPC object
    body_objects: dict[int, Any] = field(default_factory=dict)
    # body_idx -> list of shape indices
    body_shapes: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))

    # Pre-computed warp arrays for batch GPU sync (populated after world.init)
    body_indices_wp: wp.array | None = None  # sorted body indices, int32
    backend_offsets_wp: wp.array | None = None  # UIPC backend offsets, uint32
    num_mapped_bodies: int = 0

    # Cloth geometry mappings: list of (particle_indices, geo_slot) per cloth mesh
    cloth_geo_slots: list[Any] = field(default_factory=list)
    cloth_particle_indices: list[Any] = field(default_factory=list)  # np.ndarray per mesh

    # Deformable geometry mappings: list of (particle_indices, geo_slot) per deformable mesh
    deformable_geo_slots: list[Any] = field(default_factory=list)
    deformable_particle_indices: list[Any] = field(default_factory=list)  # np.ndarray per mesh


# ---------------------------------------------------------------------------
# Build mesh for a Newton body
# ---------------------------------------------------------------------------


def _transform_points(points: np.ndarray, tf: wp.transform, scale: np.ndarray) -> np.ndarray:  # pyright: ignore[reportArgumentType]
    """Apply Newton shape transform to mesh points.

    Args:
        points: Vertex positions, shape ``(N, 3)``.
        tf: A ``wp.transform`` value.
        scale: Per-axis scale factors, shape ``(3,)``.
    """
    scaled = points * scale
    mat = newton_transform_to_mat4(tf)
    rot = mat[:3, :3]
    pos = mat[:3, 3]
    return (rot @ scaled.T).T + pos


def _mesh_to_vf(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Extract (vertices, faces) from a Newton Mesh as float64/int32."""
    return mesh._vertices.copy().astype(np.float64), mesh._indices.reshape(-1, 3).copy().astype(np.int32)


def _weld_vertices(
    verts: np.ndarray,
    faces: np.ndarray,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices so the mesh becomes a closed manifold.

    UV-sphere and similar generators duplicate vertices along seams for
    texture coordinates.  UIPC requires closed trimeshes; welding removes
    the seam duplicates.

    Args:
        verts: Vertex positions, shape ``(V, 3)``, float64.
        faces: Triangle indices, shape ``(F, 3)``, int32.
        tol: Euclidean distance below which two vertices are merged.

    Returns:
        ``(welded_verts, remapped_faces)`` with unique vertices only.
    """
    # Round to tolerance grid so coincident vertices hash identically
    quantized = np.ascontiguousarray(np.round(verts / tol).astype(np.int64))
    # Use structured view for unique-row detection
    view = quantized.view(np.dtype([("", np.int64)] * 3))
    _, unique_idx, inverse = np.unique(view, return_index=True, return_inverse=True)
    inverse = inverse.ravel()  # flatten structured-array extra dim
    if len(unique_idx) == len(verts):
        return verts, faces  # already manifold
    new_faces = inverse[faces].astype(np.int32)
    # Remove degenerate triangles (two or more identical vertices after merge)
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    return verts[unique_idx], new_faces[valid]


def build_body_mesh(model: Model, body_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a merged triangle mesh for a Newton body from its shapes.

    Uses Newton's :class:`~newton.Mesh` ``create_*`` factory methods for primitive
    shapes (BOX, SPHERE, CAPSULE, CYLINDER, CONE) to guarantee correct winding
    and closed manifold meshes.  Coincident vertices are welded so the result
    is always a closed manifold suitable for UIPC.

    Returns:
        (vertices, faces) arrays for a closed trimesh, or None if body has no mesh shapes.
    """
    if (
        model.shape_body is None
        or model.shape_type is None
        or model.shape_transform is None
        or model.shape_scale is None
    ):
        return None

    shape_body = model.shape_body.numpy()
    shape_type_np = model.shape_type.numpy()
    shape_transform_np = model.shape_transform.numpy()
    shape_scale_np = model.shape_scale.numpy()

    shape_indices = [s for s in range(model.shape_count) if shape_body[s] == body_idx]
    if not shape_indices:
        return None

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for s in shape_indices:
        geo_type = GeoType(shape_type_np[s])
        tf_np = shape_transform_np[s]
        scale = shape_scale_np[s]

        verts = None
        faces = None
        # Whether scale is already baked into the generated mesh
        scale_baked = False
        if geo_type in (GeoType.MESH, GeoType.CONVEX_MESH):
            geo_src = model.shape_source[s]
            if isinstance(geo_src, Mesh):
                verts, faces = _mesh_to_vf(geo_src)
            else:
                warnings.warn(f"Shape {s} (body {body_idx}): unsupported geo_src type {type(geo_src)}", stacklevel=2)
                continue
        elif geo_type == GeoType.BOX:
            m = Mesh.create_box(
                float(scale[0]), float(scale[1]), float(scale[2]), duplicate_vertices=False, compute_inertia=False
            )
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.SPHERE:
            m = Mesh.create_sphere(float(scale[0]), compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.CAPSULE:
            m = Mesh.create_capsule(float(scale[0]), float(scale[1]), compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.CYLINDER:
            m = Mesh.create_cylinder(float(scale[0]), float(scale[1]), compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.CONE:
            m = Mesh.create_cone(float(scale[0]), float(scale[1]), compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.PLANE:
            continue
        else:
            warnings.warn(f"Shape {s} (body {body_idx}): unsupported GeoType {geo_type}", stacklevel=2)
            continue

        if verts is not None and faces is not None:
            effective_scale = np.ones(3) if scale_baked else scale.astype(np.float64)
            is_identity = (
                np.allclose(tf_np[:3], 0.0, atol=1e-7)
                and np.allclose(tf_np[3:], [0, 0, 0, 1], atol=1e-7)
                and np.allclose(effective_scale, 1.0, atol=1e-7)
            )
            if not is_identity:
                tf_wp = wp.transform(tf_np[:3], tf_np[3:])
                verts = _transform_points(verts, tf_wp, effective_scale)
            all_faces.append(faces + vert_offset)
            all_verts.append(verts)
            vert_offset += len(verts)

    if not all_verts:
        return None

    merged_verts = np.vstack(all_verts).astype(np.float64)
    merged_faces = np.vstack(all_faces).astype(np.int32)
    return _weld_vertices(merged_verts, merged_faces)


def populate_backend_offsets(mapping: UIpcMappingInfo, device: wp.Device) -> None:
    """Pre-compute backend offset arrays after world.init() for GPU batch sync.

    Must be called after ``world.init()`` so that ``backend_abd_body_offset``
    attributes are available on the UIPC geometry objects.

    Args:
        mapping: The mapping info to populate.
        device: Warp device for the output arrays.
    """

    if not mapping.body_geo_slots:
        return

    body_indices = sorted(mapping.body_geo_slots.keys())
    n = len(body_indices)

    offsets_np = np.empty(n, dtype=np.uint32)
    indices_np = np.array(body_indices, dtype=np.int32)

    for i, body_idx in enumerate(body_indices):
        geo = mapping.body_geo_slots[body_idx].geometry()
        offset_attr = geo.meta().find(uipc_builtin.backend_abd_body_offset)
        if offset_attr is None:
            warnings.warn(
                f"Body {body_idx}: backend_abd_body_offset not found after world.init(), skipping backend mapping",
                stacklevel=2,
            )
            offsets_np[i] = 0
            continue
        offsets_np[i] = offset_attr.view()[0]

    mapping.body_indices_wp = wp.from_numpy(indices_np, dtype=wp.int32, device=device)
    mapping.backend_offsets_wp = wp.from_numpy(offsets_np, dtype=wp.uint32, device=device)
    mapping.num_mapped_bodies = n
