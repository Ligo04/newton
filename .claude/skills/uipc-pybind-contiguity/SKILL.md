---
name: uipc-pybind-contiguity
description: Use when passing numpy arrays into any UIPC C++ binding (uipc.geometry.trimesh, simplicial complexes, position/topology buffers, AffineBody constitutions). Pybind11 silently misreads non-C-contiguous arrays as column-major, transposing meshes and producing negative-volume garbage that triggers `Assertion volume > 0 failed` in `affine_body_constitution.cpp`.
---

# UIPC pybind11 Contiguity Gotcha

## The Bug

`uipc.geometry.trimesh(Vs, Fs)` (and friends) takes numpy arrays via pybind11.
The binding **only checks dtype, not strides**. When the input is not
C-contiguous, pybind11 walks the underlying buffer **as if it were
column-major**, producing a transposed mesh — every vertex `(x_i, y_i, z_i)`
becomes `(x_0, x_1, x_2)`, etc.

The transposed mesh almost always has negative signed volume, which trips
UIPC's ABD sanity check at solver init:

```
[error] Assertion volume > 0 failed. Volume of the mesh is non-positive
(-1.36e-05), which is not allowed. src/constitution/affine_body_constitution.cpp(95)
```

This is **silent and data-dependent**: small bodies whose verts are still
contiguous (e.g. boxes from `Mesh.create_box`) work fine; bodies that went
through `_weld_vertices`, `np.vstack`, fancy indexing, or `_orient_outward`
end up with non-contiguous arrays and break.

## The Fix

Always coerce arrays before handing them to a UIPC C++ entrypoint:

```python
import numpy as np

Vs = np.ascontiguousarray(verts, dtype=np.float64)
Fs = np.ascontiguousarray(faces, dtype=np.int32)
sc = uipc_trimesh(Vs, Fs)
```

Even if `verts.dtype == float64`, you **must** call `ascontiguousarray` —
the dtype check passes but the stride check is missing on UIPC's side.

## How to Detect

Symptoms:
- `Assertion volume > 0 failed` in `affine_body_constitution.cpp`
- Newton-side `_signed_volume(verts, faces)` is positive but UIPC reports negative
- `verts.flags['C_CONTIGUOUS']` is `False` for the offending body
- `verts_in[0]` looks like `[x0, y0, z0]` but `sc.positions().view()[0]`
  comes back as `[x0, x1, x2]` — that's the smoking gun

Quick check inside a debug wrapper around `uipc.geometry.trimesh`:

```python
_orig = rb.uipc_trimesh
def _wrap(verts, faces):
    sc = _orig(verts, faces)
    pv = np.asarray(sc.positions().view()).reshape(-1, 3)
    if not np.allclose(pv, verts):
        print(f"MISMATCH! contig={verts.flags['C_CONTIGUOUS']} max_dv={np.abs(pv-verts).max()}")
    return sc
rb.uipc_trimesh = _wrap
```

## Where in Newton This Bites

- `newton/_src/solvers/uipc/converter.py::build_body_mesh` — return path
  must hand back C-contiguous arrays. Use a `_finalize` helper that all
  branches (welded, convex hull, AABB box) funnel through.
- Any code that builds verts via `np.vstack`, `verts[mask]`, `faces[:, [0, 2, 1]]`,
  `_weld_vertices`, or scipy `ConvexHull` outputs must coerce before UIPC.

## Related Gotcha: ConvexHull Winding

`scipy.spatial.ConvexHull.simplices` does **not** guarantee consistent
winding across faces — some come out CW, others CCW. A single global
`_orient_outward` flip cannot fix a mesh whose triangles disagree.

Reorient each face individually using `hull.equations[:, :3]` (the
outward plane normals scipy provides):

```python
normals = hull.equations[:, :3]
tri = hull_verts[faces]
face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
flip = np.einsum("ij,ij->i", face_normals, normals) < 0.0
faces[flip] = faces[flip][:, [0, 2, 1]]
```

## Defense in Depth

For robustness against degenerate / non-watertight USD assets, layer the
fallbacks in `build_body_mesh`:

1. Welded merged mesh (preferred)
2. Closed-manifold check via `_is_trimesh_closed` (every edge shared by exactly 2 faces)
3. Convex hull fallback (with per-face winding fix above)
4. AABB box (last resort for coplanar / degenerate inputs)
5. Final `_finalize` step: orient outward, check `|signed_volume| > 1e-12`,
   and **always** return `np.ascontiguousarray(...)`.

## Reference

- UIPC source: `src/constitution/affine_body_constitution.cpp:95`
- Newton fix: `newton/_src/solvers/uipc/converter.py::build_body_mesh::_finalize`
- Discovered while debugging G1 + `approximate_meshes("bounding_box")` on 2026-04-07
