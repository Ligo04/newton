# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deformable body builder for the UIPC solver backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import uipc.builtin as uipc_builtin
from uipc.constitution import ARAP, ElasticModuli, SoftPositionConstraint, StableNeoHookean
from uipc.core import ContactElement, SubsceneElement
from uipc.geometry import flip_inward_triangles, label_surface, label_triangle_orient
from uipc.geometry import tetmesh as uipc_tetmesh

from newton import Model

from .converter import UIpcMappingInfo
from .deformable_groups import deformable_body_groups_from_model
from .utils import _view_attr


class DeformableBodyBuilder:
    """Build UIPC deformable bodies from Newton particles and tetrahedra."""

    DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN = "stable_neo_hookean"
    DEFORMABLE_MODEL_ARAP = "arap"

    def __init__(
        self,
        model: Model,
        scene: Any,
        mapping: UIpcMappingInfo,
        default_mass_density: float = 1000.0,
        enable_soft_position_constraint: bool = True,
    ):
        self._model = model
        self._scene = scene
        self._mapping = mapping
        self._default_mass_density = default_mass_density
        self._deformable_model = self.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN
        self._enable_soft_position_constraint = enable_soft_position_constraint

    @property
    def has_deformable(self) -> bool:
        """Whether the Newton model contains deformable tetrahedral elements."""
        return self._model.tet_count > 0 and self._model.tet_indices is not None

    def build(
        self,
        contact_elem: ContactElement,
        particle_range: tuple[int, int] | None = None,
        subscene_elem: SubsceneElement | None = None,
    ) -> None:
        """Convert Newton deformable particles and tetrahedra to a UIPC object.

        Args:
            contact_elem: Contact element to apply to deformable geometries.
            particle_range: ``(start, end)`` particle slice, or ``None`` for all particles.
            subscene_elem: UIPC subscene element, or ``None``.
        """
        model = self._model
        particle_q = model.particle_q
        tet_indices = model.tet_indices
        if model.tet_count == 0 or particle_q is None or tet_indices is None:
            return

        deformable_groups = deformable_body_groups_from_model(model)
        if deformable_groups:
            for soft_index, soft_group in enumerate(deformable_groups):
                if not self._range_in_particle_range(soft_group.particle_range, particle_range):
                    continue
                selected_particle_indices, local_tets, selected_tet_ids = self._select_tetrahedra(
                    model, tet_range=soft_group.tetrahedron_range
                )
                self._build_tet_set(
                    contact_elem,
                    selected_particle_indices,
                    local_tets,
                    selected_tet_ids,
                    subscene_elem,
                    soft_group.density,
                    soft_index,
                )
            return

        selected_particle_indices, local_tets, selected_tet_ids = self._select_tetrahedra(
            model, particle_range=particle_range
        )
        self._build_tet_set(
            contact_elem, selected_particle_indices, local_tets, selected_tet_ids, subscene_elem, None, None
        )

    def _build_tet_set(
        self,
        contact_elem: ContactElement,
        selected_particle_indices: np.ndarray,
        local_tets: np.ndarray,
        selected_tet_ids: np.ndarray,
        subscene_elem: SubsceneElement | None,
        authored_density: float | None,
        soft_index: int | None,
    ) -> None:
        """Build one UIPC geometry for a selected tetrahedron set."""
        if selected_particle_indices.size == 0 or local_tets.size == 0:
            return

        self._build_geometry(
            contact_elem,
            selected_particle_indices,
            local_tets,
            selected_tet_ids,
            subscene_elem,
            authored_density,
            soft_index,
        )

    def _build_geometry(
        self,
        contact_elem: ContactElement,
        selected_particle_indices: np.ndarray,
        local_tets: np.ndarray,
        selected_tet_ids: np.ndarray,
        subscene_elem: SubsceneElement | None,
        authored_density: float | None,
        soft_index: int | None,
    ) -> None:
        """Build one UIPC deformable geometry from an authored tet group."""
        model = self._model
        particle_q = model.particle_q
        if particle_q is None:
            return

        particle_q_np = particle_q.numpy()
        deformable_verts = particle_q_np[selected_particle_indices].astype(np.float64, copy=False)

        sc = uipc_tetmesh(deformable_verts, local_tets)
        contact_elem.apply_to(sc)
        if subscene_elem is not None:
            subscene_elem.apply_to(sc)
        label_surface(sc)
        label_triangle_orient(sc)
        sc = flip_inward_triangles(sc)
        mass_density = self._estimate_mass_density(
            model, selected_particle_indices, local_tets, deformable_verts, authored_density
        )
        deformable_model = self._deformable_model_from_model(model, soft_index)
        self._apply_deformable_model(sc, deformable_model, mass_density)
        self._write_tet_elastic_moduli(sc, model, selected_tet_ids)

        fixed_local_indices = self._fixed_particle_indices(model, selected_particle_indices)
        if fixed_local_indices.size > 0:
            self._mark_fixed_vertices(sc, local_indices=fixed_local_indices)
        if self._enable_soft_position_constraint:
            spc = SoftPositionConstraint()
            spc.apply_to(sc)

        obj = self._scene.objects().create(f"deformable_{len(self._mapping.deformable_geo_slots)}")
        geo_slot, rest_geo_slot = obj.geometries().create(sc)

        self._mapping.deformable_geo_slots.append(geo_slot)
        self._mapping.deformable_rest_geo_slots.append(rest_geo_slot)
        self._mapping.deformable_particle_indices.append(selected_particle_indices)

    def _select_tetrahedra(
        self,
        model: Model,
        particle_range: tuple[int, int] | None = None,
        tet_range: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return selected particles, remapped local tetrahedra, and tet IDs."""
        assert model.tet_indices is not None
        tet_indices_np = model.tet_indices.numpy().reshape(-1, 4)
        tet_ids = np.arange(tet_indices_np.shape[0], dtype=np.int32)

        if tet_range is not None:
            tstart, tend = tet_range
            tet_indices_np = tet_indices_np[tstart:tend]
            tet_ids = tet_ids[tstart:tend]
        elif particle_range is not None:
            pstart, pend = particle_range
            in_range = (tet_indices_np >= pstart) & (tet_indices_np < pend)
            tet_mask = np.all(in_range, axis=1)
            tet_ids = tet_ids[tet_mask]
            tet_indices_np = tet_indices_np[tet_mask]

        if tet_indices_np.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty((0, 4), dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        selected_particles = np.unique(tet_indices_np.reshape(-1).astype(np.int32))
        if particle_range is not None:
            pstart, pend = particle_range
            selected_particles = selected_particles[(selected_particles >= pstart) & (selected_particles < pend)]
        if selected_particles.size == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty((0, 4), dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(selected_particles)}
        local_tets = np.empty_like(tet_indices_np, dtype=np.int32)
        for t in range(tet_indices_np.shape[0]):
            for v in range(4):
                local_tets[t, v] = global_to_local[int(tet_indices_np[t, v])]

        return selected_particles, local_tets, tet_ids

    @staticmethod
    def _range_in_particle_range(entity_range: tuple[int, int], particle_range: tuple[int, int] | None) -> bool:
        """Return whether an entity range belongs to the requested particle slice."""
        if particle_range is None:
            return True
        start, end = entity_range
        pstart, pend = particle_range
        return pstart <= start and end <= pend

    def _deformable_model_from_model(self, model: Model, soft_index: int | None) -> str:
        """Return the selected UIPC tet constitution for one deformable group."""
        uipc_attrs = getattr(model, "uipc", None)
        if uipc_attrs is None or not hasattr(uipc_attrs, "deformable_model"):
            return self._deformable_model

        deformable_model = uipc_attrs.deformable_model
        if isinstance(deformable_model, (list, tuple)):
            if len(deformable_model) == 0:
                return self._deformable_model
            if soft_index is None:
                deformable_model = deformable_model[0]
            elif soft_index >= len(deformable_model):
                return self._deformable_model
            else:
                deformable_model = deformable_model[soft_index]

        if not isinstance(deformable_model, str):
            raise TypeError("uipc:deformable_model must be a string custom attribute.")
        if not deformable_model:
            return self._deformable_model
        return self._normalize_deformable_model(deformable_model)

    @classmethod
    def _normalize_deformable_model(cls, deformable_model: str) -> str:
        """Return the canonical UIPC deformable constitution key."""
        key = deformable_model.lower().replace("-", "_")
        aliases = {
            "stable": cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
            "stable_neo": cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
            "stable_neo_hookean": cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
            "stable_neohookean": cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
            "stable_neohookean_fem": cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
            "stable_neo_hookean_fem": cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
            "stableneohookean": cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
            "arap": cls.DEFORMABLE_MODEL_ARAP,
        }
        try:
            return aliases[key]
        except KeyError as exc:
            supported = ", ".join(sorted({cls.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN, cls.DEFORMABLE_MODEL_ARAP}))
            raise ValueError(
                f"Unsupported UIPC deformable_model '{deformable_model}'. Supported values: {supported}."
            ) from exc

    def _apply_deformable_model(self, sc: Any, deformable_model: str, mass_density: float) -> None:
        """Apply the configured UIPC tetrahedral constitution."""
        if deformable_model == self.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN:
            model = StableNeoHookean()
            model.apply_to(sc, ElasticModuli.youngs_poisson(1000.0, 0.45), mass_density=mass_density)
        elif deformable_model == self.DEFORMABLE_MODEL_ARAP:
            model = ARAP()
            model.apply_to(sc, 1000.0, mass_density=mass_density)
        else:
            raise ValueError(f"Unsupported UIPC deformable_model '{deformable_model}'.")

    def _write_tet_elastic_moduli(self, sc: Any, model: Model, selected_tet_ids: np.ndarray) -> None:
        """Copy authored per-tet material parameters onto UIPC tet attributes."""
        if model.tet_materials is None or selected_tet_ids.size == 0:
            return

        tet_materials_np = model.tet_materials.numpy()[selected_tet_ids]  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        mu_attr = sc.tetrahedra().find("mu")
        lambda_attr = sc.tetrahedra().find("lambda")
        kappa_attr = sc.tetrahedra().find("kappa")
        if mu_attr is not None and lambda_attr is not None:
            snh_mu = (4.0 / 3.0) * tet_materials_np[:, 0]
            snh_lambda = tet_materials_np[:, 1] + (5.0 / 6.0) * tet_materials_np[:, 0]

            mu_view = _view_attr(mu_attr)
            lambda_view = _view_attr(lambda_attr)
            mu_view[:] = np.asarray(snh_mu, dtype=mu_view.dtype)
            lambda_view[:] = np.asarray(snh_lambda, dtype=lambda_view.dtype)
            return
        if kappa_attr is not None:
            kappa_view = _view_attr(kappa_attr)
            kappa_view[:] = np.asarray(tet_materials_np[:, 0], dtype=kappa_view.dtype)
            return

        raise RuntimeError("UIPC deformable constitution did not create supported tet material attributes.")

    def _fixed_particle_indices(self, model: Model, particle_indices: np.ndarray) -> np.ndarray:
        """Return local particle indices that should remain kinematic."""
        if model.particle_mass is None or particle_indices.size == 0:
            return np.empty(0, dtype=np.int32)

        particle_mass_np = model.particle_mass.numpy()[particle_indices]
        return np.flatnonzero(np.asarray(particle_mass_np <= 0.0, dtype=bool)).astype(np.int32)

    def _estimate_mass_density(
        self,
        model: Model,
        particle_indices: np.ndarray,
        local_tets: np.ndarray,
        vertices: np.ndarray,
        authored_density: float | None,
    ) -> float:
        """Estimate mass density [kg/m^3] from particle masses and tet volume."""
        if authored_density is not None:
            return float(authored_density)
        if model.particle_mass is None or particle_indices.size == 0 or local_tets.size == 0:
            return self._default_mass_density

        particle_mass_np = model.particle_mass.numpy()
        total_mass = float(np.sum(particle_mass_np[particle_indices]))
        if total_mass <= 0.0:
            return self._default_mass_density

        total_volume = 0.0
        for tet in local_tets:
            v0, v1, v2, v3 = vertices[tet]
            mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            total_volume += abs(np.linalg.det(mat)) / 6.0

        if total_volume <= 0.0:
            return self._default_mass_density

        return total_mass / total_volume

    @staticmethod
    def _mark_fixed_vertices(sc: Any, local_indices: np.ndarray) -> None:
        """Mirror fixed Newton particles into UIPC's ``is_fixed`` marker."""
        fixed_attr = sc.vertices().find(uipc_builtin.is_fixed)
        if fixed_attr is None:
            fixed_attr = sc.vertices().create(uipc_builtin.is_fixed, np.zeros(sc.vertices().size(), dtype=np.int32))
        _view_attr(fixed_attr)[local_indices] = 1
