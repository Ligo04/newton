# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deformable body (StableNeoHookean) builder for the UIPC solver backend."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

from ...sim import Model
from .converter import UIpcMappingInfo

from uipc import view
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.geometry import (
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
    tetmesh as uipc_tetmesh,
)


def _lame_to_youngs_poisson(k_mu: float, k_lambda: float) -> tuple[float, float]:
    """Convert Lame parameters to Young's modulus and Poisson's ratio.

    Args:
        k_mu: First Lame parameter (shear modulus) [Pa].
        k_lambda: Second Lame parameter [Pa].

    Returns:
        Tuple of (Young's modulus [Pa], Poisson's ratio).
    """
    if k_mu <= 0:
        return 1000.0, 0.45  # safe defaults
    denom = k_lambda + k_mu
    if abs(denom) < 1e-12:
        return 2.0 * k_mu * (1.0 + 0.0), 0.0  # lambda ~ -mu edge case
    youngs = k_mu * (3.0 * k_lambda + 2.0 * k_mu) / denom
    poisson = k_lambda / (2.0 * denom)
    # Clamp Poisson's ratio to valid range
    poisson = max(-1.0, min(poisson, 0.499))
    return youngs, poisson


class DeformableBodyBuilder:
    """Build UIPC deformable bodies (StableNeoHookean) from Newton particles and tetrahedra.

    Converts Newton :class:`~newton.Model` soft-body data (particles +
    tetrahedra) into UIPC ``StableNeoHookean`` constitutions.

    Newton stores deformable bodies as particles with tetrahedral connectivity
    and Lame material parameters (``k_mu``, ``k_lambda``, ``k_damp``). This
    builder converts those to UIPC's ``ElasticModuli`` (Young's modulus,
    Poisson's ratio).

    Material mapping:
        - ``k_mu``, ``k_lambda`` -> Young's modulus, Poisson's ratio via
          standard Lame-to-engineering conversion
        - ``particle_mass`` -> mass density [kg/m^3]
    """

    def __init__(
        self,
        model: Model,
        scene: Any,
        contact_elem: Any,
        mapping: UIpcMappingInfo,
        default_mass_density: float = 1000.0,
        particle_range: tuple[int, int] | None = None,
    ):
        self._model = model
        self._scene = scene
        self._contact_elem = contact_elem
        self._mapping = mapping
        self._default_mass_density = default_mass_density
        # When set, only particles in [start, end) are considered.
        self._particle_range = particle_range

    @property
    def has_deformable(self) -> bool:
        """Whether the Newton model contains deformable (tetrahedra) elements."""
        return self._model.tet_count > 0

    def build(self) -> None:
        """Convert Newton deformable particles and tetrahedra to UIPC deformable objects.

        Extracts tetrahedral connectivity and the referenced particles from the
        Newton model, remaps indices, creates a UIPC ``tetmesh``, and applies
        ``StableNeoHookean`` constitution with surface labeling and triangle
        orientation correction.
        """
        model = self._model
        if model.tet_count == 0 or model.tet_indices is None or model.particle_q is None:
            return

        # Get tetrahedra indices
        tet_indices_np = model.tet_indices.numpy()  # (tet_count, 4)
        tet_count = model.tet_count

        # Identify deformable particles: all particles referenced by tetrahedra
        tet_particle_set_raw = set(tet_indices_np.flatten())

        # Filter by particle range if specified
        if self._particle_range is not None:
            pstart, pend = self._particle_range
            tet_particle_set_raw = {p for p in tet_particle_set_raw if pstart <= p < pend}

        tet_particle_set = sorted(tet_particle_set_raw)
        if not tet_particle_set:
            return

        # Build particle index remapping: global -> local
        global_to_local = {g: l for l, g in enumerate(tet_particle_set)}
        deformable_particle_indices = np.array(tet_particle_set, dtype=np.int32)

        # Remap tetrahedra indices to local vertex indices
        local_tets = np.empty((tet_count, 4), dtype=np.int32)
        for t in range(tet_count):
            for v in range(4):
                local_tets[t, v] = global_to_local[tet_indices_np[t, v]]

        # Extract particle positions (already guarded by None check above)
        assert model.particle_q is not None
        particle_q_np = model.particle_q.numpy()  # (particle_count, 3)
        deformable_verts = particle_q_np[deformable_particle_indices].astype(np.float64)

        # Create UIPC tetmesh
        sc = uipc_tetmesh(deformable_verts, local_tets)

        # Apply contact and surface labels
        self._contact_elem.apply_to(sc)
        label_surface(sc)
        label_triangle_orient(sc)
        sc = flip_inward_triangles(sc)

        # Determine material parameters
        youngs_modulus, poisson_ratio = self._get_elastic_moduli(model)
        mass_density = self._estimate_mass_density(model, deformable_particle_indices, deformable_verts)

        # Apply StableNeoHookean constitution
        snk = StableNeoHookean()
        moduli = ElasticModuli.youngs_poisson(youngs_modulus, poisson_ratio)
        snk.apply_to(sc, moduli, mass_density=mass_density)

        # Create scene object
        obj = self._scene.objects().create("deformable_0")
        geo_slot, _ = obj.geometries().create(sc)

        # Store mapping for state sync
        self._mapping.deformable_geo_slots.append(geo_slot)
        self._mapping.deformable_particle_indices.append(deformable_particle_indices)

    def _get_elastic_moduli(self, model: Model) -> tuple[float, float]:
        """Extract elastic moduli from Newton tetrahedra material parameters.

        Converts Newton's Lame parameters (``k_mu``, ``k_lambda``) to Young's
        modulus and Poisson's ratio.
        """
        if model.tet_count > 0 and model.tet_materials is not None:
            tet_materials_np = model.tet_materials.numpy()
            # tet_materials layout: (tet_count, 3) -> [k_mu, k_lambda, k_damp]
            avg_k_mu = float(np.mean(tet_materials_np[:, 0]))
            avg_k_lambda = float(np.mean(tet_materials_np[:, 1]))
            if avg_k_mu > 0:
                return _lame_to_youngs_poisson(avg_k_mu, avg_k_lambda)
        return 1000.0, 0.45  # Default: 1 kPa, Poisson = 0.45

    def _estimate_mass_density(
        self,
        model: Model,
        particle_indices: np.ndarray,
        vertices: np.ndarray,
    ) -> float:
        """Estimate volumetric mass density [kg/m^3] from particle masses and tet volumes.

        Falls back to ``default_mass_density`` if estimation fails.
        """
        if model.particle_mass is not None and model.tet_count > 0 and model.tet_indices is not None:
            particle_mass_np = model.particle_mass.numpy()
            total_mass = float(np.sum(particle_mass_np[particle_indices]))

            if total_mass > 0:
                # Estimate total volume from tetrahedra
                tet_indices_np = model.tet_indices.numpy()
                global_to_local = {int(g): l for l, g in enumerate(particle_indices)}
                total_volume = 0.0
                for t in range(model.tet_count):
                    li = [global_to_local.get(int(tet_indices_np[t, v])) for v in range(4)]
                    if any(idx is None for idx in li):
                        continue
                    v0, v1, v2, v3 = [vertices[idx] for idx in li]
                    # Tetrahedron volume = |det([v1-v0, v2-v0, v3-v0])| / 6
                    mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
                    total_volume += abs(np.linalg.det(mat)) / 6.0

                if total_volume > 0:
                    return total_mass / total_volume

        return self._default_mass_density
