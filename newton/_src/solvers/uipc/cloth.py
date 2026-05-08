# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth builder for the UIPC solver backend."""

from __future__ import annotations

from typing import Any

import numpy as np
from uipc import view
from uipc.constitution import (
    DiscreteShellBending,
    ElasticModuli2D,
    NeoHookeanShell,
    SoftPositionConstraint,
    StrainLimitingBaraffWitkinShell,
)
from uipc.core import ContactElement, SubsceneElement
from uipc.geometry import is_trimesh_closed, label_surface, mesh_partition
from uipc.geometry import trimesh as uipc_trimesh

from ...sim import Model
from .converter import UIpcMappingInfo


class ClothBuilder:
    """Build UIPC cloth from Newton particles and triangles.

    Converts Newton :class:`~newton.Model` cloth data (particles + triangles +
    edges) into UIPC shell and bending constitutions.

    Newton stores cloth as particles with triangle connectivity and material
    parameters (``tri_ke``, ``tri_ka``, ``tri_kd``, ``tri_drag``, ``tri_lift``).
    This builder maps those to UIPC's ``ElasticModuli2D`` (Young's modulus,
    Poisson's ratio) and shell bending stiffness.

    Material mapping:
        - ``tri_ke`` -> Young's modulus [Pa]
        - Poisson's ratio defaults to 0.3
        - Bending stiffness from ``edge_bending_properties`` or default
        - ``particle_mass`` -> mass density [kg/m^2]

    By default the membrane model is UIPC's
    ``StrainLimitingBaraffWitkinShell``.  Pass ``"neo_hookean"`` through
    :class:`~newton.solvers.SolverUIPC` to use ``NeoHookeanShell`` instead.
    """

    CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN = "strain_limiting_baraff_witkin"
    CLOTH_MODEL_NEO_HOOKEAN = "neo_hookean"
    CLOTH_THICKNESS_ATTRIBUTE = "cloth_thick"

    def __init__(
        self,
        model: Model,
        scene: Any,
        mapping: UIpcMappingInfo,
        default_thickness: float = 0.001,
        default_poisson_ratio: float = 0.3,
        default_bending_stiffness: float = 0.01,
        cloth_model: str = CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
        enable_soft_position_constraint: bool = True,
        soft_position_strength_ratio: float = 100.0,
    ):
        self._model = model
        self._scene = scene
        self._mapping = mapping
        self._default_thickness = default_thickness
        self._default_poisson_ratio = default_poisson_ratio
        self._default_bending_stiffness = default_bending_stiffness
        self._cloth_model = self._normalize_cloth_model(cloth_model)
        self._enable_soft_position_constraint = enable_soft_position_constraint
        self._soft_position_strength_ratio = soft_position_strength_ratio

    @property
    def has_cloth(self) -> bool:
        """Whether the Newton model contains cloth (triangle) elements."""
        return self._model.tri_count > 0

    def build(
        self,
        contact_elem: ContactElement,
        particle_range: tuple[int, int] | None = None,
        subscene_elem: SubsceneElement | None = None,
    ) -> None:
        """Convert Newton cloth particles and triangles to UIPC cloth objects.

        Groups all triangles that are NOT part of tetrahedra into cloth meshes.
        Extracts the referenced particles, remaps indices, creates a UIPC
        ``trimesh``, and applies the configured membrane model plus
        ``DiscreteShellBending``.

        Args:
            contact_elem: Contact element to apply to cloth geometries.
            particle_range: ``(start, end)`` slice of particles, or ``None`` for all.
            subscene_elem: UIPC subscene element, or ``None``.
        """
        model = self._model
        if model.tri_count == 0 or model.tri_indices is None or model.particle_q is None:
            return

        if model.cloth_ranges:
            for cloth_range in model.cloth_ranges:
                if not self._range_in_particle_range(cloth_range.particle_range, particle_range):
                    continue
                selected_particles, cloth_faces, selected_tri_ids = self._select_range_cloth(model, cloth_range)
                self._build_geometry(
                    contact_elem,
                    selected_particles,
                    cloth_faces,
                    selected_tri_ids,
                    cloth_range.edge_range,
                    subscene_elem,
                )
            return

        selected_particles, cloth_faces, selected_tri_ids = self._select_legacy_cloth(model, particle_range)
        self._build_geometry(contact_elem, selected_particles, cloth_faces, selected_tri_ids, None, subscene_elem)

    def _select_range_cloth(self, model: Model, cloth_range: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return particles, local faces, and triangle IDs for one authored cloth range."""
        assert model.tri_indices is not None
        tri_indices_np = model.tri_indices.numpy().reshape(-1, 3)
        tstart, tend = cloth_range.tri_range
        selected_tri_ids = np.arange(tstart, tend, dtype=np.int32)
        selected_tris = tri_indices_np[tstart:tend]
        if selected_tris.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        selected_particles = np.unique(selected_tris.reshape(-1).astype(np.int32))
        global_to_local = {int(g): i for i, g in enumerate(selected_particles)}
        local_faces = np.empty_like(selected_tris, dtype=np.int32)
        for t, tri in enumerate(selected_tris):
            for v in range(3):
                local_faces[t, v] = global_to_local[int(tri[v])]
        return selected_particles, local_faces, selected_tri_ids

    def _select_legacy_cloth(
        self, model: Model, particle_range: tuple[int, int] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select cloth triangles using the historical particle-range heuristic."""
        # Get triangle indices
        assert model.tri_indices is not None
        tri_indices_np = model.tri_indices.numpy().reshape(-1, 3)
        tri_count = model.tri_count

        # Identify cloth particles: referenced by triangles but NOT by tetrahedra
        tri_particle_set = {int(i) for i in tri_indices_np.flatten()}

        # Filter by particle range if specified
        if particle_range is not None:
            pstart, pend = particle_range
            tri_particle_set = {p for p in tri_particle_set if pstart <= p < pend}

        tet_particle_set: set[int] = set()
        if model.tet_count > 0 and model.tet_indices is not None:
            tet_indices_np = model.tet_indices.numpy()
            tet_particle_set = {int(i) for i in tet_indices_np.flatten()}

        cloth_particles = sorted(tri_particle_set - tet_particle_set)
        if not cloth_particles:
            return (
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        # Build particle index remapping: global -> local
        global_to_local = {g: l for l, g in enumerate(cloth_particles)}
        cloth_particle_indices = np.array(cloth_particles, dtype=np.int32)

        # Filter triangles: only those with all vertices in cloth_particles
        cloth_tris = []
        selected_tri_ids = []
        for t in range(tri_count):
            i, j, k = tri_indices_np[t]
            if i in global_to_local and j in global_to_local and k in global_to_local:
                cloth_tris.append([global_to_local[i], global_to_local[j], global_to_local[k]])
                selected_tri_ids.append(t)

        if not cloth_tris:
            return (
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        return cloth_particle_indices, np.array(cloth_tris, dtype=np.int32), np.array(selected_tri_ids, dtype=np.int32)

    def _build_geometry(
        self,
        contact_elem: ContactElement,
        cloth_particle_indices: np.ndarray,
        cloth_faces: np.ndarray,
        selected_tri_ids: np.ndarray,
        edge_range: tuple[int, int] | None,
        subscene_elem: SubsceneElement | None,
    ) -> None:
        """Build one UIPC cloth geometry."""
        if cloth_particle_indices.size == 0 or cloth_faces.size == 0:
            return

        model = self._model
        # Extract particle positions (already guarded by None check above)
        assert model.particle_q is not None
        particle_q_np = model.particle_q.numpy()  # (particle_count, 3)
        cloth_verts = particle_q_np[cloth_particle_indices].astype(np.float64)

        # Create UIPC trimesh
        sc = uipc_trimesh(cloth_verts, cloth_faces)
        if is_trimesh_closed(sc):
            raise RuntimeError(
                "UIPC cloth expects an open triangle mesh, but the Newton cloth mesh is closed "
                "(watertight). Use a deformable/rigid representation for closed meshes, or open "
                "the surface before passing it to ModelBuilder.add_cloth_mesh()."
            )

        # Apply contact and subscene
        contact_elem.apply_to(sc)
        if subscene_elem is not None:
            subscene_elem.apply_to(sc)
        label_surface(sc)

        mesh_partition(sc, 16)

        thickness_values = self._get_thickness_values(model, cloth_particle_indices)
        density_thickness = (
            float(np.mean(thickness_values)) if thickness_values is not None else self._default_thickness
        )
        applied_thickness = self._default_thickness

        # Compute mass density from particle masses and mesh area
        mass_density = self._estimate_mass_density(model, cloth_particle_indices, selected_tri_ids, density_thickness)

        moduli = ElasticModuli2D.youngs_poisson(1000.0, self._default_poisson_ratio)
        if self._cloth_model == self.CLOTH_MODEL_NEO_HOOKEAN:
            membrane = NeoHookeanShell()
        else:
            membrane = StrainLimitingBaraffWitkinShell()
        membrane.apply_to(sc, moduli, mass_density=mass_density, thickness=applied_thickness)
        self._write_tri_elastic_moduli(sc, model, selected_tri_ids)
        if thickness_values is not None:
            self._write_vertex_thickness(sc, thickness_values, applied_thickness)

        # Apply DiscreteShellBending constitution
        dsb = DiscreteShellBending()
        dsb.apply_to(sc, self._default_bending_stiffness)
        self._write_edge_bending_stiffness(sc, model, cloth_particle_indices, edge_range)

        # Add dormant soft-position attributes.  Vertices remain unconstrained
        # until SolverUIPC.set_cloth_soft_position_constraints() toggles
        # ``is_constrained`` and writes ``aim_position``.
        if self._enable_soft_position_constraint:
            spc = SoftPositionConstraint()
            spc.apply_to(sc, self._soft_position_strength_ratio)

        # Create scene object
        obj = self._scene.objects().create(f"cloth_{len(self._mapping.cloth_geo_slots)}")
        geo_slot, rest_geo_slot = obj.geometries().create(sc)

        # Store mapping for state sync
        self._mapping.cloth_geo_slots.append(geo_slot)
        self._mapping.cloth_rest_geo_slots.append(rest_geo_slot)
        self._mapping.cloth_particle_indices.append(cloth_particle_indices)

    def _write_tri_elastic_moduli(self, sc: Any, model: Model, selected_tri_ids: np.ndarray) -> None:
        """Copy authored per-triangle membrane stiffness onto UIPC triangle attributes."""
        if model.tri_materials is None or selected_tri_ids.size == 0:
            return

        tri_materials_np = model.tri_materials.numpy()[selected_tri_ids]  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        mu_attr = sc.triangles().find("mu")
        lambda_attr = sc.triangles().find("lambda")
        if mu_attr is None or lambda_attr is None:
            raise RuntimeError("Cloth membrane apply_to() did not create triangle mu/lambda attributes.")

        mu_view = view(mu_attr)
        lambda_view = view(lambda_attr)
        mu_view[:] = np.asarray(tri_materials_np[:, 0], dtype=mu_view.dtype)
        lambda_view[:] = np.asarray(tri_materials_np[:, 1], dtype=lambda_view.dtype)

    def _write_edge_bending_stiffness(
        self,
        sc: Any,
        model: Model,
        particle_indices: np.ndarray,
        edge_range: tuple[int, int] | None,
    ) -> None:
        """Copy authored per-bending-edge stiffness onto UIPC edge attributes."""
        if model.edge_count == 0 or model.edge_indices is None or model.edge_bending_properties is None:
            return

        bending_attr = sc.edges().find("bending_stiffness")
        if bending_attr is None:
            raise RuntimeError("DiscreteShellBending.apply_to() did not create edge bending_stiffness attributes.")

        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(particle_indices)}
        uipc_edges = np.asarray(view(sc.edges().topo()), dtype=np.int32).reshape(-1, 2)
        local_edge_to_index = {tuple(sorted((int(edge[0]), int(edge[1])))): idx for idx, edge in enumerate(uipc_edges)}

        edge_indices_np = model.edge_indices.numpy().reshape(-1, 4)
        edge_props_np = model.edge_bending_properties.numpy()  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        if edge_range is None:
            edge_ids = range(edge_indices_np.shape[0])
        else:
            estart, eend = edge_range
            edge_ids = range(estart, eend)

        bending_view = view(bending_attr)
        for edge_id in edge_ids:
            _, _, global_k, global_l = edge_indices_np[edge_id]
            local_k = global_to_local.get(int(global_k))
            local_l = global_to_local.get(int(global_l))
            if local_k is None or local_l is None:
                continue
            local_edge_idx = local_edge_to_index.get(tuple(sorted((local_k, local_l))))
            if local_edge_idx is None:
                continue
            bending_view[local_edge_idx] = edge_props_np[edge_id, 0]

    def _estimate_mass_density(
        self, model: Model, particle_indices: np.ndarray, selected_tri_ids: np.ndarray, thickness: float
    ) -> float:
        """Estimate surface mass density [kg/m^2] from particle masses and areas.

        Falls back to a default of 100 kg/m^3 volumetric density if estimation fails.
        """
        if model.particle_mass is not None:
            particle_mass_np = model.particle_mass.numpy()
            total_mass = float(np.sum(particle_mass_np[particle_indices]))
            if total_mass > 0 and selected_tri_ids.size > 0 and model.tri_areas is not None:
                total_area = float(np.sum(model.tri_areas.numpy()[selected_tri_ids]))
                if total_area > 0:
                    # Surface density = total_mass / total_area
                    # Volume density = surface_density / thickness
                    return total_mass / total_area / thickness
        return 100.0  # Default: 100 kg/m^3

    @staticmethod
    def _range_in_particle_range(entity_range: tuple[int, int], particle_range: tuple[int, int] | None) -> bool:
        """Return whether an entity range belongs to the requested particle slice."""
        if particle_range is None:
            return True
        start, end = entity_range
        pstart, pend = particle_range
        return pstart <= start and end <= pend

    def _get_thickness_values(self, model: Model, particle_indices: np.ndarray) -> np.ndarray | None:
        """Return per-cloth-particle UIPC shell thickness values [m], if provided.

        Newton examples pass cloth thickness through the ``cloth_thick``
        particle custom attribute:

        .. code-block:: python

            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="cloth_thick",
                    dtype=wp.float32,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                )
            )
            builder.add_cloth_mesh(
                ...,
                custom_attributes_particles={"cloth_thick": [1.0e-4] * len(vertices)},
            )

        If the attribute is absent, the builder falls back to
        ``default_thickness``.
        """
        thickness_attr = getattr(model, self.CLOTH_THICKNESS_ATTRIBUTE, None)
        if thickness_attr is None:
            return None

        frequency = model.attribute_frequency.get(self.CLOTH_THICKNESS_ATTRIBUTE)
        if frequency != Model.AttributeFrequency.PARTICLE:
            raise ValueError(
                f"Custom attribute {self.CLOTH_THICKNESS_ATTRIBUTE!r} must have PARTICLE frequency, got {frequency}."
            )

        thickness_np = np.asarray(thickness_attr.numpy(), dtype=np.float64).reshape(-1)
        values = thickness_np[particle_indices]
        if np.any(values < 0.0):
            raise ValueError(
                f"Custom attribute {self.CLOTH_THICKNESS_ATTRIBUTE!r} must be non-negative for all UIPC cloth particles."
            )
        values = np.where(values > 0.0, values, self._default_thickness)
        return values

    @staticmethod
    def _write_vertex_thickness(sc: Any, thickness_values: np.ndarray, applied_thickness: float) -> None:
        """Write per-vertex thickness and keep shell volume consistent."""
        thickness_attr = sc.vertices().find("thickness")
        if thickness_attr is not None:
            view(thickness_attr)[:] = np.asarray(thickness_values, dtype=np.float64)

        volume_attr = sc.vertices().find("volume")
        if volume_attr is not None:
            volume = view(volume_attr)
            volume[:] = np.asarray(volume, dtype=np.float64) * (thickness_values / applied_thickness)

    @classmethod
    def _normalize_cloth_model(cls, cloth_model: str) -> str:
        """Return the canonical UIPC cloth membrane model key."""
        key = cloth_model.lower().replace("-", "_")
        aliases = {
            "strain_limiting": cls.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
            "strain_limiting_shell": cls.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
            "strain_limiting_baraff_witkin": cls.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
            "strain_limiting_baraff_witkin_shell": cls.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
            "baraff_witkin": cls.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
            "baraff_witkin_shell": cls.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
            "neo_hookean": cls.CLOTH_MODEL_NEO_HOOKEAN,
            "neo_hookean_shell": cls.CLOTH_MODEL_NEO_HOOKEAN,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            valid = ", ".join(sorted(set(aliases)))
            raise ValueError(f"Unknown UIPC cloth model {cloth_model!r}. Expected one of: {valid}") from exc
