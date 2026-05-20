#!/usr/bin/env python3
"""Read-only MCP server for this repository's project memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = REPO_ROOT / ".omx" / "project-memory.json"

mcp = FastMCP("newton-project-memory")


def _load_memory() -> dict[str, Any]:
    with MEMORY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {MEMORY_PATH}")
    return data


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.resource("project-memory://newton/full")
def read_full_memory() -> str:
    """Return the full Newton project memory JSON."""
    return _dump(_load_memory())


@mcp.resource("project-memory://newton/observations")
def read_observations() -> str:
    """Return the structured observation list."""
    return _dump(_load_memory().get("observations", []))


@mcp.resource("project-memory://newton/migrations")
def read_migrations() -> str:
    """Return raw migrated memory blocks, including the original AGENTS.md block."""
    return _dump(_load_memory().get("migrations", []))


@mcp.tool()
def project_memory_summary() -> str:
    """Return a compact summary of the Newton project memory."""
    data = _load_memory()
    observations = data.get("observations", [])
    migrations = data.get("migrations", [])
    result = {
        "project": data.get("project"),
        "updated_at": data.get("updated_at"),
        "source": data.get("source"),
        "stats": data.get("summary", {}).get("stats", {}),
        "observation_count": len(observations) if isinstance(observations, list) else None,
        "migration_count": len(migrations) if isinstance(migrations, list) else None,
        "latest_observations": observations[-10:] if isinstance(observations, list) else [],
        "resources": [
            "project-memory://newton/full",
            "project-memory://newton/observations",
            "project-memory://newton/migrations",
        ],
    }
    return _dump(result)


@mcp.tool()
def project_memory_search(query: str, limit: int = 20) -> str:
    """Search Newton project memory observations and migrated raw blocks."""
    q = query.lower().strip()
    if not q:
        return _dump([])
    data = _load_memory()
    matches: list[dict[str, Any]] = []
    for obs in data.get("observations", []):
        haystack = _dump(obs).lower()
        if q in haystack:
            matches.append({"kind": "observation", **obs})
            if len(matches) >= limit:
                return _dump(matches)
    for index, migration in enumerate(data.get("migrations", [])):
        content = str(migration.get("content", ""))
        lower = content.lower()
        if q in lower:
            pos = lower.find(q)
            start = max(0, pos - 400)
            end = min(len(content), pos + 400)
            matches.append(
                {
                    "kind": "migration",
                    "index": index,
                    "source_path": migration.get("source_path"),
                    "excerpt": content[start:end],
                }
            )
            if len(matches) >= limit:
                return _dump(matches)
    return _dump(matches)


@mcp.tool()
def project_memory_get_observation(observation_id: str) -> str:
    """Return one structured observation by ID."""
    data = _load_memory()
    for obs in data.get("observations", []):
        if str(obs.get("id")) == observation_id:
            return _dump(obs)
    return _dump({"error": "not_found", "observation_id": observation_id})


if __name__ == "__main__":
    mcp.run()
