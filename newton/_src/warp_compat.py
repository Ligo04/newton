# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fork-local Warp compatibility shim.

This fork pins ``warp-lang==1.14.0`` (see ``pyproject.toml``) for the UIPC
backend, while upstream solver code references the determinism API introduced
in Warp 1.15 (``wp.DeterministicMode``, ``wp.config.deterministic``). Those
references are evaluated both at import time (unstringized annotations in
solvers lacking ``from __future__ import annotations``) and at solver
construction. Back-fill inert stand-ins so the modules import and construct;
determinism is simply not enforced on 1.14.x. On Warp >= 1.15 this is a no-op.

Must run before any ``newton._src.solvers`` module is imported.
"""

import enum

import warp as wp


def is_shimmed() -> bool:
    """True when the determinism API is our back-filled stand-in (Warp < 1.15).

    The running Warp then has no real run-to-run determinism; tests asserting
    bit-identical rollouts skip on this.
    """
    return getattr(getattr(wp, "DeterministicMode", None), "__module__", None) == __name__


def apply() -> None:
    """Install Warp determinism stand-ins when the running Warp lacks them."""
    if hasattr(wp, "DeterministicMode"):
        return

    class DeterministicMode(enum.Enum):
        NOT_GUARANTEED = 0
        RUN_TO_RUN = 1

    wp.DeterministicMode = DeterministicMode
    # Fallback read in every solver __init__: `... else wp.config.deterministic`.
    if not hasattr(wp.config, "deterministic"):
        wp.config.deterministic = DeterministicMode.NOT_GUARANTEED
    if not hasattr(wp.config, "deterministic_max_records"):
        wp.config.deterministic_max_records = 0


# Apply on import so a single side-effecting import in newton/__init__ suffices.
apply()
