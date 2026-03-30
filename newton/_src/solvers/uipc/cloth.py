# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth (NeoHookeanShell) builder for the UIPC solver backend."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

from ...sim import Model
from .converter import UIpcMappingInfo

from uipc import view
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, NeoHookeanShell
from uipc.geometry import label_surface
from uipc.geometry import trimesh as uipc_trimesh


class ClothBuilder:
    """Build UIPC cloth (NeoHookeanShell) from Newton particles and triangles.

    Converts Newton :class:`~newton.Model` cloth data (particles + triangles +
    edges) into UIPC ``NeoHookeanShell`` and ``DiscreteShellBending``
    constitutions.

    Newton stores cloth as particles with triangle connectivity and material
    parameters (``tri_ke``, ``tri_ka``, ``tri_kd``, ``tri_drag``, ``tri_lift``).
    This builder maps those to UIPC's ``ElasticModuli2D`` (Young's modulus,
    Poisson's ratio) and shell bending stiffness.

    Material mapping:
        - ``tri_ke`` -> Young's modulus [Pa]
        - Poisson's ratio defaults to 0.3
        - Bending stiffness from ``edge_bending_properties`` or default
        - ``particle_mass`` -> mass density [kg/m^2]
    """

    def __init__(
        self,
        model: Model,
        scene: Any,
        contact_elem: Any,
        mapping: UIpcMappingInfo,
        default_thickness: float = 0.001,
        default_poisson_ratio: float = 0.3,
        default_bending_stiffness: float = 0.01,
        particle_range: tuple[int, int] | None = None,
    ):
        self._model = model
        self._scene = scene
        self._contact_elem = contact_elem
        self._mapping = mapping
        self._default_thickness = default_thickness
        self._default_poisson_ratio = default_poisson_ratio
        self._default_bending_stiffness = default_bending_stiffness
        # When set, only particles in [start, end) are considered.
        self._particle_range = particle_range

    @property
    def has_cloth(self) -> bool:
        """Whether the Newton model contains cloth (triangle) elements."""
        return self._model.tri_count > 0

    def build(self) -> None:
        """Convert Newton cloth particles and triangles to UIPC cloth objects.

        Groups all triangles that are NOT part of tetrahedra into cloth meshes.
        Extracts the referenced particles, remaps indices, creates a UIPC
        ``trimesh``, and applies ``NeoHookeanShell`` + ``DiscreteShellBending``.
        """
        model = self._model
        if model.tri_count == 0 or model.tri_indices is None or model.particle_q is None:
            return

        # Get triangle indices
        tri_indices_np = model.tri_indices.numpy()  # (tri_count, 3)
        tri_count = model.tri_count

        # Identify cloth particles: referenced by triangles but NOT by tetrahedra
        tri_particle_set = set(tri_indices_np.flatten())

        # Filter by particle range if specified
        if self._particle_range is not None:
            pstart, pend = self._particle_range
            tri_particle_set = {p for p in tri_particle_set if pstart <= p < pend}

        tet_particle_set: set[int] = set()
        if model.tet_count > 0 and model.tet_indices is not None:
            tet_indices_np = model.tet_indices.numpy()
            tet_particle_set = set(tet_indices_np.flatten())

        cloth_particles = sorted(tri_particle_set - tet_particle_set)
        if not cloth_particles:
            return

        # Build particle index remapping: global -> local
        global_to_local = {g: l for l, g in enumerate(cloth_particles)}
        cloth_particle_indices = np.array(cloth_particles, dtype=np.int32)

        # Filter triangles: only those with all vertices in cloth_particles
        cloth_tris = []
        for t in range(tri_count):
            i, j, k = tri_indices_np[t]
            if i in global_to_local and j in global_to_local and k in global_to_local:
                cloth_tris.append([global_to_local[i], global_to_local[j], global_to_local[k]])

        if not cloth_tris:
            return

        cloth_faces = np.array(cloth_tris, dtype=np.int32)

        # Extract particle positions (already guarded by None check above)
        assert model.particle_q is not None
        particle_q_np = model.particle_q.numpy()  # (particle_count, 3)
        cloth_verts = particle_q_np[cloth_particle_indices].astype(np.float64)

        # Create UIPC trimesh
        sc = uipc_trimesh(cloth_verts, cloth_faces)

        # Apply contact
        self._contact_elem.apply_to(sc)
        label_surface(sc)

        # Determine material parameters from Newton model
        youngs_modulus = self._get_youngs_modulus(model)
        poisson_ratio = self._default_poisson_ratio
        thickness = self._default_thickness
        bending_stiffness = self._get_bending_stiffness(model)

        # Compute mass density from particle masses and mesh area
        mass_density = self._estimate_mass_density(model, cloth_particle_indices, thickness)

        # Apply NeoHookeanShell constitution
        nhs = NeoHookeanShell()
        moduli = ElasticModuli2D.youngs_poisson(youngs_modulus, poisson_ratio)
        nhs.apply_to(sc, moduli, mass_density=mass_density, thickness=thickness)

        # Apply DiscreteShellBending constitution
        dsb = DiscreteShellBending()
        dsb.apply_to(sc, bending_stiffness)

        # Create scene object
        obj = self._scene.objects().create("cloth_0")
        geo_slot, _ = obj.geometries().create(sc)

        # Store mapping for state sync
        self._mapping.cloth_geo_slots.append(geo_slot)
        self._mapping.cloth_particle_indices.append(cloth_particle_indices)

    def _get_youngs_modulus(self, model: Model) -> float:
        """Extract Young's modulus from Newton triangle material parameters.

        Uses ``tri_ke`` (elastic stiffness) as the Young's modulus proxy.
        """
        if model.tri_count > 0 and model.tri_materials is not None:
            tri_materials_np = model.tri_materials.numpy()
            # tri_materials layout: (tri_count, 5) -> [ke, ka, kd, drag, lift]
            avg_ke = float(np.mean(tri_materials_np[:, 0]))
            if avg_ke > 0:
                return avg_ke
        return 1000.0  # Default: 1 kPa

    def _get_bending_stiffness(self, model: Model) -> float:
        """Extract bending stiffness from Newton edge parameters."""
        if model.edge_count > 0 and model.edge_bending_properties is not None:
            edge_props_np = model.edge_bending_properties.numpy()
            # edge_bending_properties layout: (edge_count, 2) -> [stiffness, damping]
            avg_ke = float(np.mean(edge_props_np[:, 0]))
            if avg_ke > 0:
                return avg_ke
        return self._default_bending_stiffness

    def _estimate_mass_density(
        self, model: Model, particle_indices: np.ndarray, thickness: float
    ) -> float:
        """Estimate surface mass density [kg/m^2] from particle masses and areas.

        Falls back to a default of 100 kg/m^3 volumetric density if estimation fails.
        """
        if model.particle_mass is not None:
            particle_mass_np = model.particle_mass.numpy()
            total_mass = float(np.sum(particle_mass_np[particle_indices]))
            if total_mass > 0 and model.tri_count > 0 and model.tri_areas is not None:
                total_area = float(np.sum(model.tri_areas.numpy()))
                if total_area > 0:
                    # Surface density = total_mass / total_area
                    # Volume density = surface_density / thickness
                    return total_mass / total_area / thickness
        return 100.0  # Default: 100 kg/m^3
