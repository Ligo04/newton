# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""UIPC custom-frequency metadata for authored cloth and soft-body groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ...sim import Model, ModelBuilder

CLOTH_FREQUENCY = "uipc:cloth"
DEFORMABLE_BODY_FREQUENCY = "uipc:deformable_body"


def _cloth_group_count(builder: ModelBuilder) -> int:
    """Return the number of authored cloth groups."""
    return len(builder._cloth_label)


def _deformable_body_group_count(builder: ModelBuilder) -> int:
    """Return the number of authored soft-body groups."""
    return len(builder._soft_label)


_GROUP_ATTRIBUTE_SOURCES: dict[str, tuple[str, str | None, str]] = {
    "uipc:cloth_label": ("_cloth_label", None, "direct"),
    "uipc:cloth_world": ("_cloth_world", None, "direct"),
    "uipc:cloth_particle_first": ("_cloth_particle_start", "_cloth_particle_end", "first"),
    "uipc:cloth_particle_last": ("_cloth_particle_start", "_cloth_particle_end", "last"),
    "uipc:cloth_triangle_first": ("_cloth_tri_start", "_cloth_tri_end", "first"),
    "uipc:cloth_triangle_last": ("_cloth_tri_start", "_cloth_tri_end", "last"),
    "uipc:cloth_edge_first": ("_cloth_edge_start", "_cloth_edge_end", "first"),
    "uipc:cloth_edge_last": ("_cloth_edge_start", "_cloth_edge_end", "last"),
    "uipc:cloth_spring_first": ("_cloth_spring_start", "_cloth_spring_end", "first"),
    "uipc:cloth_spring_last": ("_cloth_spring_start", "_cloth_spring_end", "last"),
    "uipc:cloth_surface_density": ("_cloth_surface_density", None, "optional_float"),
    "uipc:deformable_body_label": ("_soft_label", None, "direct"),
    "uipc:deformable_body_world": ("_soft_world", None, "direct"),
    "uipc:deformable_body_particle_first": ("_soft_particle_start", "_soft_particle_end", "first"),
    "uipc:deformable_body_particle_last": ("_soft_particle_start", "_soft_particle_end", "last"),
    "uipc:deformable_body_tetrahedron_first": ("_soft_tet_start", "_soft_tet_end", "first"),
    "uipc:deformable_body_tetrahedron_last": ("_soft_tet_start", "_soft_tet_end", "last"),
    "uipc:deformable_body_triangle_first": ("_soft_tri_start", "_soft_tri_end", "first"),
    "uipc:deformable_body_triangle_last": ("_soft_tri_start", "_soft_tri_end", "last"),
    "uipc:deformable_body_edge_first": ("_soft_edge_start", "_soft_edge_end", "first"),
    "uipc:deformable_body_edge_last": ("_soft_edge_start", "_soft_edge_end", "last"),
    "uipc:deformable_body_density": ("_soft_density", None, "optional_float"),
}

_GROUP_ATTRIBUTE_SPECS = (
    ("cloth_label", CLOTH_FREQUENCY, str, "", None),
    ("cloth_world", CLOTH_FREQUENCY, wp.int32, -1, "world"),
    ("cloth_particle_first", CLOTH_FREQUENCY, wp.int32, -1, "particle"),
    ("cloth_particle_last", CLOTH_FREQUENCY, wp.int32, -1, "particle"),
    ("cloth_triangle_first", CLOTH_FREQUENCY, wp.int32, -1, "triangle"),
    ("cloth_triangle_last", CLOTH_FREQUENCY, wp.int32, -1, "triangle"),
    ("cloth_edge_first", CLOTH_FREQUENCY, wp.int32, -1, "edge"),
    ("cloth_edge_last", CLOTH_FREQUENCY, wp.int32, -1, "edge"),
    ("cloth_spring_first", CLOTH_FREQUENCY, wp.int32, -1, "spring"),
    ("cloth_spring_last", CLOTH_FREQUENCY, wp.int32, -1, "spring"),
    ("cloth_surface_density", CLOTH_FREQUENCY, wp.float32, -1.0, None),
    ("deformable_body_label", DEFORMABLE_BODY_FREQUENCY, str, "", None),
    ("deformable_body_world", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "world"),
    ("deformable_body_particle_first", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "particle"),
    ("deformable_body_particle_last", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "particle"),
    ("deformable_body_tetrahedron_first", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "tetrahedron"),
    ("deformable_body_tetrahedron_last", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "tetrahedron"),
    ("deformable_body_triangle_first", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "triangle"),
    ("deformable_body_triangle_last", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "triangle"),
    ("deformable_body_edge_first", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "edge"),
    ("deformable_body_edge_last", DEFORMABLE_BODY_FREQUENCY, wp.int32, -1, "edge"),
    ("deformable_body_density", DEFORMABLE_BODY_FREQUENCY, wp.float32, -1.0, None),
)


def _resolve_group_attribute_values(builder: ModelBuilder, attribute: ModelBuilder.CustomAttribute) -> list[Any]:
    """Resolve one UIPC metadata column from the builder group registry."""
    source_name, end_name, mode = _GROUP_ATTRIBUTE_SOURCES[attribute.key]
    source = getattr(builder, source_name)
    if mode == "direct":
        return list(source)
    if mode == "optional_float":
        return [-1.0 if value is None else float(value) for value in source]

    assert end_name is not None
    ends = getattr(builder, end_name)
    if len(source) != len(ends):
        raise ValueError(
            f"UIPC group metadata source length mismatch for '{attribute.key}': "
            f"{source_name} has {len(source)} rows but {end_name} has {len(ends)}"
        )
    if mode == "first":
        return [int(start) if end > start else -1 for start, end in zip(source, ends, strict=True)]
    return [int(end) - 1 if end > start else -1 for start, end in zip(source, ends, strict=True)]


def _finalize_group_attribute(
    builder: ModelBuilder,
    model: Model,
    attribute: ModelBuilder.CustomAttribute,
) -> None:
    """Materialize one resolver-owned UIPC group attribute on a model."""
    frequency = str(attribute.frequency)
    expected_count = model.custom_frequency_counts.get(frequency, 0)
    values = _resolve_group_attribute_values(builder, attribute)
    if len(values) != expected_count:
        raise ValueError(
            f"UIPC group metadata '{attribute.key}' has {len(values)} rows but "
            f"frequency '{frequency}' expects {expected_count}"
        )

    if attribute.dtype is str:
        result = [str(value) for value in values]
    else:
        result = wp.array(values, dtype=attribute.dtype, device=model.device)
    model.add_attribute(
        attribute.name,
        result,
        attribute.frequency,
        attribute.assignment,
        attribute.namespace,
        attribute.references,
    )


def register_deformable_group_attributes(builder: ModelBuilder) -> None:
    """Register UIPC cloth and soft-body group metadata on a builder."""
    builder.add_custom_frequency(
        ModelBuilder.CustomFrequency(
            name="cloth",
            namespace="uipc",
            label_attribute="uipc:cloth_label",
            count_resolver=_cloth_group_count,
        )
    )
    builder.add_custom_frequency(
        ModelBuilder.CustomFrequency(
            name="deformable_body",
            namespace="uipc",
            label_attribute="uipc:deformable_body_label",
            count_resolver=_deformable_body_group_count,
        )
    )
    for name, frequency, dtype, default, references in _GROUP_ATTRIBUTE_SPECS:
        attribute = ModelBuilder.CustomAttribute(
            name=name,
            frequency=frequency,
            assignment=Model.AttributeAssignment.MODEL,
            dtype=dtype,
            default=default,
            namespace="uipc",
            references=references,
        )
        builder.add_custom_attribute(attribute)
        builder._add_custom_attribute_model_finalizer(attribute.key, _finalize_group_attribute)


@dataclass(frozen=True)
class _ClothGroup:
    """One authored UIPC cloth group decoded from model metadata."""

    label: str
    """Group label."""
    world: int
    """Owning world index, or ``-1`` for a global group."""
    particle_range: tuple[int, int]
    """Half-open particle index range."""
    triangle_range: tuple[int, int]
    """Half-open triangle index range."""
    edge_range: tuple[int, int] | None
    """Optional half-open bending-edge index range."""
    spring_range: tuple[int, int] | None
    """Optional half-open spring index range."""
    surface_density: float | None
    """Authored surface density [kg/m^2]."""


@dataclass(frozen=True)
class _DeformableBodyGroup:
    """One authored UIPC soft-body group decoded from model metadata."""

    label: str
    """Group label."""
    world: int
    """Owning world index, or ``-1`` for a global group."""
    particle_range: tuple[int, int]
    """Half-open particle index range."""
    tetrahedron_range: tuple[int, int]
    """Half-open tetrahedron index range."""
    triangle_range: tuple[int, int] | None
    """Optional half-open surface-triangle index range."""
    edge_range: tuple[int, int] | None
    """Optional half-open surface-edge index range."""
    density: float | None
    """Authored volume density [kg/m^3]."""


def _read_column(model: Model, frequency: str, name: str) -> Any:
    """Return one validated UIPC metadata column."""
    count = model.custom_frequency_counts.get(frequency, 0)
    namespace = getattr(model, "uipc", None)
    if namespace is None or not hasattr(namespace, name):
        raise ValueError(f"Model frequency '{frequency}' is missing required UIPC metadata '{name}'")
    values = getattr(namespace, name)
    actual_count = len(values)
    if actual_count != count:
        raise ValueError(f"UIPC metadata '{name}' has {actual_count} rows but frequency '{frequency}' expects {count}")
    return values


def _read_numeric_column(model: Model, frequency: str, name: str) -> np.ndarray:
    """Return one numeric UIPC metadata column as a flat NumPy array."""
    values = _read_column(model, frequency, name)
    if not isinstance(values, wp.array):
        raise TypeError(f"UIPC metadata '{name}' must be a Warp array")
    return np.asarray(values.numpy()).reshape(-1)


def _decode_range(first: int, last: int, name: str, *, required: bool) -> tuple[int, int] | None:
    """Decode an inclusive first/last pair into a half-open range."""
    if first == -1 and last == -1:
        if required:
            raise ValueError(f"UIPC metadata '{name}' cannot be empty")
        return None
    if first < 0 or last < first:
        raise ValueError(f"UIPC metadata '{name}' has invalid inclusive range ({first}, {last})")
    return first, last + 1


def cloth_groups_from_model(model: Model) -> list[_ClothGroup]:
    """Decode authored UIPC cloth groups from model custom attributes."""
    count = model.custom_frequency_counts.get(CLOTH_FREQUENCY, 0)
    if count == 0:
        return []

    labels = _read_column(model, CLOTH_FREQUENCY, "cloth_label")
    worlds = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_world")
    particle_first = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_particle_first")
    particle_last = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_particle_last")
    triangle_first = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_triangle_first")
    triangle_last = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_triangle_last")
    edge_first = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_edge_first")
    edge_last = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_edge_last")
    spring_first = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_spring_first")
    spring_last = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_spring_last")
    surface_density = _read_numeric_column(model, CLOTH_FREQUENCY, "cloth_surface_density")

    groups = []
    for index in range(count):
        particle_range = _decode_range(
            int(particle_first[index]), int(particle_last[index]), "cloth_particle", required=True
        )
        triangle_range = _decode_range(
            int(triangle_first[index]), int(triangle_last[index]), "cloth_triangle", required=True
        )
        assert particle_range is not None and triangle_range is not None
        groups.append(
            _ClothGroup(
                label=str(labels[index]),
                world=int(worlds[index]),
                particle_range=particle_range,
                triangle_range=triangle_range,
                edge_range=_decode_range(int(edge_first[index]), int(edge_last[index]), "cloth_edge", required=False),
                spring_range=_decode_range(
                    int(spring_first[index]), int(spring_last[index]), "cloth_spring", required=False
                ),
                surface_density=None if surface_density[index] < 0.0 else float(surface_density[index]),
            )
        )
    return groups


def deformable_body_groups_from_model(model: Model) -> list[_DeformableBodyGroup]:
    """Decode authored UIPC soft-body groups from model custom attributes."""
    count = model.custom_frequency_counts.get(DEFORMABLE_BODY_FREQUENCY, 0)
    if count == 0:
        return []

    labels = _read_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_label")
    worlds = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_world")
    particle_first = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_particle_first")
    particle_last = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_particle_last")
    tet_first = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_tetrahedron_first")
    tet_last = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_tetrahedron_last")
    triangle_first = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_triangle_first")
    triangle_last = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_triangle_last")
    edge_first = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_edge_first")
    edge_last = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_edge_last")
    density = _read_numeric_column(model, DEFORMABLE_BODY_FREQUENCY, "deformable_body_density")

    groups = []
    for index in range(count):
        particle_range = _decode_range(
            int(particle_first[index]), int(particle_last[index]), "deformable_body_particle", required=True
        )
        tetrahedron_range = _decode_range(
            int(tet_first[index]), int(tet_last[index]), "deformable_body_tetrahedron", required=True
        )
        assert particle_range is not None and tetrahedron_range is not None
        groups.append(
            _DeformableBodyGroup(
                label=str(labels[index]),
                world=int(worlds[index]),
                particle_range=particle_range,
                tetrahedron_range=tetrahedron_range,
                triangle_range=_decode_range(
                    int(triangle_first[index]),
                    int(triangle_last[index]),
                    "deformable_body_triangle",
                    required=False,
                ),
                edge_range=_decode_range(
                    int(edge_first[index]), int(edge_last[index]), "deformable_body_edge", required=False
                ),
                density=None if density[index] < 0.0 else float(density[index]),
            )
        )
    return groups
