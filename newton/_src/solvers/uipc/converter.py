# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conversion utilities between Newton Model and UIPC Scene objects."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import uipc.builtin as uipc_builtin
import warp as wp
from uipc import Quaternion, Transform
from uipc.geometry import IAttributeSlot, SimplicialComplex, SimplicialComplexSlot, is_trimesh_closed
from uipc.geometry import trimesh as uipc_trimesh

from ...core.types import Axis
from ...geometry import GeoType, Mesh
from ...geometry.flags import ShapeFlags
from ...sim import Model

# ---------------------------------------------------------------------------
# Warp kernels for batch transform conversion
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _transform_to_mat44_kernel(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[wp.int32],
    out_transforms: wp.array[wp.mat44d],
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
    body_qd: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_indices: wp.array[wp.int32],
    out_velocities: wp.array[wp.mat44d],
):
    """Convert Newton ``(v_com_world, omega_world)`` body twists to UIPC's
    affine velocity matrix layout.

    Newton spatial_vector layout: ``(vx, vy, vz, wx, wy, wz)`` with the linear
    part referenced at the body COM. UIPC's affine velocity matrix encodes
    ``Ȧ`` (rate of the affine matrix) in the upper 3x3 and ``ṫ`` (rate of
    translation) in column 3, in Eigen column-major layout. Because ABD's
    mass-matrix already references velocity at the COM, ``ṫ = v_com``
    directly — no kinematic shift is needed for the linear part.

    The angular part requires ``Ȧ = [ωx] · R`` (rather than just ``[ωx]``)
    because UIPC stores the affine derivative, which equals ``[ωx]`` only
    when the body rotation is identity.
    """
    tid = wp.tid()
    body_idx = body_indices[tid]
    qd = body_qd[body_idx]
    r = wp.transform_get_rotation(body_q[body_idx])

    v_com = wp.vec3(qd[0], qd[1], qd[2])
    wx = qd[3]
    wy = qd[4]
    wz = qd[5]

    # Rotation matrix R (Warp row-major, R[i,j]) from quaternion (x, y, z, w).
    one = float(1.0)
    two = float(2.0)
    rx = r[0]
    ry = r[1]
    rz = r[2]
    rw = r[3]
    r00 = one - two * (ry * ry + rz * rz)
    r01 = two * (rx * ry - rz * rw)
    r02 = two * (rx * rz + ry * rw)
    r10 = two * (rx * ry + rz * rw)
    r11 = one - two * (rx * rx + rz * rz)
    r12 = two * (ry * rz - rx * rw)
    r20 = two * (rx * rz - ry * rw)
    r21 = two * (ry * rz + rx * rw)
    r22 = one - two * (rx * rx + ry * ry)

    # Ȧ = [ωx] · R in (i, j) row-major. Skew matrix:
    #   [ωx] = [[ 0, -wz,  wy],
    #           [ wz,   0, -wx],
    #           [-wy,  wx,   0]]
    a00 = -wz * r10 + wy * r20
    a01 = -wz * r11 + wy * r21
    a02 = -wz * r12 + wy * r22
    a10 = wz * r00 - wx * r20
    a11 = wz * r01 - wx * r21
    a12 = wz * r02 - wx * r22
    a20 = -wy * r00 + wx * r10
    a21 = -wy * r01 + wx * r11
    a22 = -wy * r02 + wx * r12

    # Pack into Eigen column-major mat44d. Warp ``mat44d(...)`` is row-major,
    # so each Warp row equals one Eigen column — ``warp[r,c] = eigen[c,r]``.
    zero = wp.float64(0.0)
    m = wp.mat44d(
        # Warp row 0 = Eigen column 0
        wp.float64(a00),
        wp.float64(a10),
        wp.float64(a20),
        zero,
        # Warp row 1 = Eigen column 1
        wp.float64(a01),
        wp.float64(a11),
        wp.float64(a21),
        zero,
        # Warp row 2 = Eigen column 2
        wp.float64(a02),
        wp.float64(a12),
        wp.float64(a22),
        zero,
        # Warp row 3 = Eigen column 3 (ṫ = v_com)
        wp.float64(v_com[0]),
        wp.float64(v_com[1]),
        wp.float64(v_com[2]),
        zero,
    )
    out_velocities[tid] = m


@wp.kernel(enable_backward=False)
def _write_to_backend_kernel(
    backend_offsets: wp.array[wp.uint32],
    src_transforms: wp.array[wp.mat44d],
    src_velocities: wp.array[wp.mat44d],
    dst_transforms: wp.array[wp.mat44d],
    dst_velocities: wp.array[wp.mat44d],
):
    """Scatter body transforms/velocities into UIPC backend arrays by offset."""
    tid = wp.tid()
    dst_idx = backend_offsets[tid]
    dst_transforms[dst_idx] = src_transforms[tid]
    dst_velocities[dst_idx] = src_velocities[tid]


@wp.kernel(enable_backward=False)
def _read_from_backend_kernel(
    backend_offsets: wp.array[wp.uint32],
    src_transforms: wp.array[wp.mat44d],
    src_velocities: wp.array[wp.mat44d],
    body_indices: wp.array[wp.int32],
    out_body_q: wp.array[wp.transform],
    out_body_qd: wp.array[wp.spatial_vector],
):
    """Gather UIPC backend transforms/velocities back into Newton state arrays.

    The source matrices are stored in **column-major** order (Eigen convention)
    but wrapped as Warp ``mat44d`` which uses row-major indexing.  Therefore
    every element access swaps row and column: ``m[i, j]`` in Warp reads the
    Eigen element at ``(j, i)``.

    UIPC's ABD parameterization references body velocity at the COM (the
    mass matrix internalizes the COM offset via the first moment), so the
    translation column already encodes ``v_com_world`` — no kinematic shift
    is required for the linear part. The angular part needs ``[ωx] = Ȧ · R^T``
    because UIPC stores ``Ȧ`` (not ``[ωx]``) in the upper 3x3.
    """
    tid = wp.tid()
    backend_idx = backend_offsets[tid]
    body_idx = body_indices[tid]

    m = src_transforms[backend_idx]

    # Extract position (Eigen column 3 → Warp row 3)
    px = wp.float32(m[3, 0])
    py = wp.float32(m[3, 1])
    pz = wp.float32(m[3, 2])

    # Extract quaternion from rotation matrix (Shepperd's method)
    # Eigen M[i,j] → Warp m[j,i]
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
        qx = (m[1, 2] - m[2, 1]) / s
        qy = (m[2, 0] - m[0, 2]) / s
        qz = (m[0, 1] - m[1, 0]) / s
    elif r00 > r11 and r00 > r22:
        s = wp.sqrt(one_d + r00 - r11 - r22) * two_d
        qw = (m[1, 2] - m[2, 1]) / s
        qx = quarter_d * s
        qy = (m[1, 0] + m[0, 1]) / s
        qz = (m[2, 0] + m[0, 2]) / s
    elif r11 > r22:
        s = wp.sqrt(one_d + r11 - r00 - r22) * two_d
        qw = (m[2, 0] - m[0, 2]) / s
        qx = (m[1, 0] + m[0, 1]) / s
        qy = quarter_d * s
        qz = (m[2, 1] + m[1, 2]) / s
    else:
        s = wp.sqrt(one_d + r22 - r00 - r11) * two_d
        qw = (m[0, 1] - m[1, 0]) / s
        qx = (m[2, 0] + m[0, 2]) / s
        qy = (m[2, 1] + m[1, 2]) / s
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

    # The affine velocity matrix stores ``Ȧ`` (rate of the affine matrix)
    # in the upper 3x3 and ``ṫ`` (rate of translation) in column 3, in
    # Eigen column-major layout — Warp ``v[i,j]`` reads Eigen ``v[j,i]``.
    #
    # UIPC's affine body parameterization already references velocity at
    # the COM (the mass matrix couples ṫ and Ȧ to COM via the first
    # moment ``m·x̄``), so ``ṫ`` is already ``v_com_world`` — no
    # ``ω x com`` shift is needed.
    #
    # The angular velocity follows from the rigid constraint ``Ȧ = [ωx] R``:
    #     [ωx] = Ȧ · R^T
    # Reading the skew components of ``Ȧ`` directly is only correct when
    # ``R = I``; we project through ``R^T`` to get true spatial ``ω``.
    v = src_velocities[backend_idx]

    # ṫ = v_com (Eigen column 3 → Warp row 3).
    v_com = wp.vec3(wp.float32(v[3, 0]), wp.float32(v[3, 1]), wp.float32(v[3, 2]))

    # [ωx] = Ȧ · A^T. With Eigen ``A[i,j]`` = Warp ``m[j,i]`` (rotation
    # only, so A = R), ``(Ȧ · A^T)_eigen[i,k]`` = Σ_j v[j,i] · m[j,k].
    # We only need three components: ω_x = [ωx][2,1], ω_y = [ωx][0,2],
    # ω_z = [ωx][1,0].
    m01 = wp.float32(m[0, 1])
    m02 = wp.float32(m[0, 2])
    m00 = wp.float32(m[0, 0])
    m11 = wp.float32(m[1, 1])
    m12 = wp.float32(m[1, 2])
    m10 = wp.float32(m[1, 0])
    m21 = wp.float32(m[2, 1])
    m22 = wp.float32(m[2, 2])
    m20 = wp.float32(m[2, 0])
    v02 = wp.float32(v[0, 2])
    v12 = wp.float32(v[1, 2])
    v22 = wp.float32(v[2, 2])
    v00 = wp.float32(v[0, 0])
    v10 = wp.float32(v[1, 0])
    v20 = wp.float32(v[2, 0])
    v01 = wp.float32(v[0, 1])
    v11 = wp.float32(v[1, 1])
    v21 = wp.float32(v[2, 1])
    omega_x = v02 * m01 + v12 * m11 + v22 * m21
    omega_y = v00 * m02 + v10 * m12 + v20 * m22
    omega_z = v01 * m00 + v11 * m10 + v21 * m20

    out_body_qd[body_idx] = wp.spatial_vector(v_com, wp.vec3(omega_x, omega_y, omega_z))


@wp.kernel(enable_backward=False)
def _write_fem_particles_to_backend_kernel(
    backend_offsets: wp.array[wp.uint32],
    particle_indices: wp.array[wp.int32],
    src_particle_q: wp.array[wp.vec3],
    src_particle_qd: wp.array[wp.vec3],
    dst_positions: wp.array[wp.vec3d],
    dst_velocities: wp.array[wp.vec3d],
):
    """Scatter Newton particle positions/velocities into UIPC FEM buffers."""
    tid = wp.tid()
    backend_idx = backend_offsets[tid]
    particle_idx = particle_indices[tid]

    q = src_particle_q[particle_idx]
    qd = src_particle_qd[particle_idx]
    dst_positions[backend_idx] = wp.vec3d(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]))
    dst_velocities[backend_idx] = wp.vec3d(wp.float64(qd[0]), wp.float64(qd[1]), wp.float64(qd[2]))


@wp.kernel(enable_backward=False)
def _write_fem_particle_positions_to_backend_kernel(
    backend_offsets: wp.array[wp.uint32],
    particle_indices: wp.array[wp.int32],
    src_particle_q: wp.array[wp.vec3],
    dst_positions: wp.array[wp.vec3d],
):
    """Scatter Newton particle positions into UIPC FEM buffers."""
    tid = wp.tid()
    backend_idx = backend_offsets[tid]
    particle_idx = particle_indices[tid]

    q = src_particle_q[particle_idx]
    dst_positions[backend_idx] = wp.vec3d(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]))


@wp.kernel(enable_backward=False)
def _read_fem_particles_from_backend_kernel(
    backend_offsets: wp.array[wp.uint32],
    src_positions: wp.array[wp.vec3d],
    src_velocities: wp.array[wp.vec3d],
    particle_indices: wp.array[wp.int32],
    out_particle_q: wp.array[wp.vec3],
    out_particle_qd: wp.array[wp.vec3],
):
    """Gather UIPC FEM vertex positions/velocities back into Newton particles."""
    tid = wp.tid()
    backend_idx = backend_offsets[tid]
    particle_idx = particle_indices[tid]

    q = src_positions[backend_idx]
    qd = src_velocities[backend_idx]
    out_particle_q[particle_idx] = wp.vec3(wp.float32(q[0]), wp.float32(q[1]), wp.float32(q[2]))
    out_particle_qd[particle_idx] = wp.vec3(wp.float32(qd[0]), wp.float32(qd[1]), wp.float32(qd[2]))


@wp.kernel(enable_backward=False)
def _read_fem_particle_positions_from_backend_kernel(
    backend_offsets: wp.array[wp.uint32],
    src_positions: wp.array[wp.vec3d],
    particle_indices: wp.array[wp.int32],
    out_particle_q: wp.array[wp.vec3],
):
    """Gather UIPC FEM vertex positions back into Newton particles."""
    tid = wp.tid()
    backend_idx = backend_offsets[tid]
    particle_idx = particle_indices[tid]

    q = src_positions[backend_idx]
    out_particle_q[particle_idx] = wp.vec3(wp.float32(q[0]), wp.float32(q[1]), wp.float32(q[2]))


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

    # body_idx -> UIPC geometry slot (shared when instanced)
    body_geo_slots: dict[int, SimplicialComplexSlot] = field(default_factory=dict)
    # body_idx -> instance index within its geometry (0 for non-instanced bodies)
    body_instance_ids: dict[int, int] = field(default_factory=dict)
    # joint_idx -> UIPC joint geometry slot
    joint_geo_slots: dict[int, SimplicialComplexSlot] = field(default_factory=dict)
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
    max_backend_count: int = 0  # max(backend_offsets) + 1, for buffer allocation

    # Cloth geometry mappings: list of (particle_indices, geo_slot) per cloth mesh
    cloth_geo_slots: list[Any] = field(default_factory=list)
    cloth_rest_geo_slots: list[Any] = field(default_factory=list)
    cloth_particle_indices: list[Any] = field(default_factory=list)  # np.ndarray per mesh

    # Deformable geometry mappings: list of (particle_indices, geo_slot) per deformable mesh
    deformable_geo_slots: list[Any] = field(default_factory=list)
    deformable_rest_geo_slots: list[Any] = field(default_factory=list)
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


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh via the divergence theorem.

    Positive when face normals point outward, negative when inward.  Used
    to detect and fix winding so UIPC ABD's volume sanity check passes.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def _orient_outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return ``faces`` flipped so the mesh has positive signed volume.

    Works for any closed manifold (convex or concave) — unlike a
    centroid-based per-face check which only handles star-shaped meshes.
    """
    if _signed_volume(verts, faces) < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
    return faces


def _aabb_box_mesh(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a closed axis-aligned bounding box trimesh around ``verts``.

    Last-resort substitute when both the merged mesh and its convex
    hull have non-positive volume (degenerate / coplanar geometry).
    """
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    box_verts = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )
    # Outward-facing CCW windings (right-hand rule).
    box_faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # -Z
            [4, 5, 6],
            [4, 6, 7],  # +Z
            [0, 1, 5],
            [0, 5, 4],  # -Y
            [2, 3, 7],
            [2, 7, 6],  # +Y
            [1, 2, 6],
            [1, 6, 5],  # +X
            [0, 4, 7],
            [0, 7, 3],  # -X
        ],
        dtype=np.int32,
    )
    return box_verts, box_faces


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


def build_body_mesh(model: Model, body_idx: int) -> tuple[SimplicialComplex, float] | None:
    """Build a merged triangle mesh for a Newton body from its shapes.

    Uses Newton's :class:`~newton.Mesh` ``create_*`` factory methods for primitive
    shapes (BOX, SPHERE, CAPSULE, CYLINDER, CONE) to guarantee correct winding
    and closed manifold meshes.  Coincident vertices are welded so the result
    is always a closed manifold suitable for UIPC.

    Returns:
        ``(sc, volume)`` — a :class:`~uipc.geometry.SimplicialComplex` and its
        absolute mesh volume, or ``None`` if body has no mesh shapes.
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
    shape_flags_np = model.shape_flags.numpy() if model.shape_flags is not None else None

    # Only use collision shapes (not visual-only shapes) to avoid duplicate
    # overlapping meshes that break the watertight topology after welding.
    shape_indices = [
        s
        for s in range(model.shape_count)
        if shape_body[s] == body_idx
        and (shape_flags_np is None or (int(shape_flags_np[s]) & int(ShapeFlags.COLLIDE_SHAPES)))
    ]
    if not shape_indices:
        return None

    # CAPSULE/CYLINDER/CONE primitives follow the model's up axis.
    primitive_up_axis = Axis.from_any(model.up_axis)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    # (shape_index, GeoType) for each shape that actually contributed a mesh;
    # surfaced in the non-watertight error message so users can pinpoint which
    # shapes to pass through ``ModelBuilder.approximate_meshes``.
    contributed_shapes: list[tuple[int, GeoType]] = []
    vert_offset = 0

    for s in shape_indices:
        geo_type = GeoType(shape_type_np[s])
        tf_np = shape_transform_np[s]
        scale = shape_scale_np[s]

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
                float(scale[0]),
                float(scale[1]),
                float(scale[2]),
                duplicate_vertices=False,
                compute_inertia=False,
            )
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.SPHERE:
            m = Mesh.create_sphere(float(scale[0]), compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.CAPSULE:
            # Capsule long axis follows ``model.up_axis``.
            m = Mesh.create_capsule(float(scale[0]), float(scale[1]), up_axis=primitive_up_axis, compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.CYLINDER:
            # Cylinder long axis follows ``model.up_axis``.
            m = Mesh.create_cylinder(float(scale[0]), float(scale[1]), up_axis=primitive_up_axis, compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.CONE:
            # Cone long axis (apex) follows ``model.up_axis``.
            m = Mesh.create_cone(float(scale[0]), float(scale[1]), up_axis=primitive_up_axis, compute_inertia=False)
            verts, faces = _mesh_to_vf(m)
            scale_baked = True
        elif geo_type == GeoType.PLANE:
            continue
        else:
            warnings.warn(f"Shape {s} (body {body_idx}): unsupported GeoType {geo_type}", stacklevel=2)
            continue

        # :meth:`newton.ModelBuilder.approximate_meshes`.
        if geo_type in (GeoType.BOX, GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE, GeoType.MESH):
            verts, faces = _weld_vertices(verts, faces)

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
        contributed_shapes.append((s, geo_type))
        vert_offset += len(verts)

    if not all_verts:
        return None

    verts = np.vstack(all_verts).astype(np.float64)
    faces = np.vstack(all_faces).astype(np.int32)

    # UIPC ABD's ``AffineBodyConstitution`` asserts ``volume > 0`` per body.
    # Two failure modes arise from real USD assets:
    #   (1) merged mesh is non-watertight (e.g. mixed mesh + primitive)
    #   (2) the closed mesh has near-zero signed volume (nearly coplanar
    #       vertices, or convex-hull fallback of a thin/flat link)
    # In both cases we need a positive-volume substitute or UIPC aborts.
    # Minimum absolute volume accepted before we fall back to an AABB box.
    _vol_eps = 1e-12
    body_name = model.body_label[body_idx] if body_idx < len(model.body_label) else "?"

    def _make_sc(v: np.ndarray, f: np.ndarray) -> tuple[SimplicialComplex, float]:
        """Create a :class:`SimplicialComplex` and compute its absolute volume.

        Falls back to an AABB box when the mesh has near-zero volume.

        IMPORTANT: ``uipc.geometry.trimesh`` expects C-contiguous (N, 3)
        arrays.  Pybind11 silently misreads strided / non-contiguous
        buffers as column-major, which transposes the mesh and produces
        a (usually negative-volume) garbage geometry.  Force contiguity.
        """
        f = _orient_outward(v, f)
        vol = abs(_signed_volume(v, f))
        if vol < _vol_eps:
            # Degenerate / coplanar — last-resort AABB box.
            warnings.warn(
                f"Body {body_idx} ({body_name}): collapsed to AABB box (near-zero volume mesh).",
                stacklevel=3,
            )
            bv, bf = _aabb_box_mesh(v)
            bf = _orient_outward(bv, bf)
            vol = abs(_signed_volume(bv, bf))
            v, f = bv, bf
        sc = uipc_trimesh(
            np.ascontiguousarray(v, dtype=np.float64),
            np.ascontiguousarray(f, dtype=np.int32),
        )
        return sc, vol

    # Build SC from the welded mesh and use UIPC's native watertight check.
    sc, vol = _make_sc(verts, faces)
    if is_trimesh_closed(sc):
        return sc, vol

    shapes_repr = ", ".join(f"shape[{s}]={gt.name}" for s, gt in contributed_shapes) or "<none>"
    raise RuntimeError(
        f"Body {body_idx} ({body_name}): merged collision mesh is not closed (non-watertight), "
        "which UIPC's affine-body constitution cannot accept. "
        f"Contributing shapes: {shapes_repr}. "
        "Convert the offending mesh shapes to a closed geometry at the builder level before "
        'passing the model to SolverUIPC — e.g. `builder.approximate_meshes(method="convex_hull")` '
        '(or `"coacd"` / `"bounding_box"`). See '
        ":meth:`newton.ModelBuilder.approximate_meshes` for the full list of methods."
    )


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
        # Stubs type ``find`` as non-optional; runtime may still omit the attribute.
        offset_attr = cast(
            IAttributeSlot | None,
            geo.meta().find(uipc_builtin.backend_abd_body_offset),
        )
        if offset_attr is None:
            warnings.warn(
                f"Body {body_idx}: backend_abd_body_offset not found after world.init(), skipping backend mapping",
                stacklevel=2,
            )
            offsets_np[i] = 0
            continue
        base_offset = offset_attr.view()[0]
        instance_id = mapping.body_instance_ids.get(body_idx, 0)
        offsets_np[i] = base_offset + instance_id

    mapping.body_indices_wp = wp.from_numpy(indices_np, dtype=wp.int32, device=device)
    mapping.backend_offsets_wp = wp.from_numpy(offsets_np, dtype=wp.uint32, device=device)
    mapping.num_mapped_bodies = n
    mapping.max_backend_count = int(offsets_np.max()) + 1 if n > 0 else 0
