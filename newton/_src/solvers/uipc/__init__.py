# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING

from .solver_uipc import SolverUIPC

if TYPE_CHECKING:
    from .articulation import Articulation
    from .articulation_builder import ArticulationBuilder
    from .cloth import ClothBuilder
    from .deformable_body import DeformableBodyBuilder
    from .rigid_body import RigidBodyBuilder

__all__ = [
    "Articulation",
    "ArticulationBuilder",
    "ClothBuilder",
    "DeformableBodyBuilder",
    "RigidBodyBuilder",
    "SolverUIPC",
]

# These wrapper classes live in modules that hard-import ``uipc`` (libuipc) at
# load time. Expose them lazily so importing this package — the path by which
# ``import newton`` reaches the import-safe :class:`SolverUIPC` — does not pull
# in libuipc. :class:`SolverUIPC` itself is import-safe and imported eagerly.
_LAZY_MODULES = {
    "Articulation": ".articulation",
    "ArticulationBuilder": ".articulation_builder",
    "ClothBuilder": ".cloth",
    "DeformableBodyBuilder": ".deformable_body",
    "RigidBodyBuilder": ".rigid_body",
}


def __getattr__(name: str):
    module = _LAZY_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)
