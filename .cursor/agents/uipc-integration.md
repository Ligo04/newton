---
name: uipc-integration
description: Project-local agent prompt for Newton SolverUIPC/UIPC integration work: backend behavior, docs, examples/tests, contact tabulars, AffineBody rigid bodies, cloth/deformable mapping, inertia sync, and pyuipc/libuipc caveats.
---

# UIPC Integration Agent

Use this agent when working on Newton's UIPC integration.

First load:
1. `.cursor/skills/uipc-integration/SKILL.md`
2. `.cursor/skills/uipc-integration/references/uipc-surface-map.md`
3. Current implementation files before making behavior claims.

Scope:
- `newton/_src/solvers/uipc/**`
- `newton/examples/uipc/**`
- `newton/tests/test_uipc*`
- `docs/integrations/uipc.md`
- `docs/guide/uipc_parameters.md`
- `pyproject.toml` / `CHANGELOG.md` only when UIPC behavior or dependency metadata changes.

Guardrails:
- Public docs/examples must not import from `newton._src`.
- Use unittest commands, not pytest.
- Do not add dependencies unless explicitly required.
- Preserve UIPC caveats: internal contact + `update_contacts`, fixed constructor `dt`, contact disabled by default, init-time baked geometry/actuators/inertials, active controls limited to revolute/prismatic.

Validation:
- Docs: `uv run --extra docs python -m sphinx -b dummy -q docs docs/_build/dummy` and `git diff --check`.
- UIPC tests: `uv run --extra dev -m newton.tests -k test_uipc` or narrower.
- Example discovery: `uv run --extra dev -m newton.tests -k test_example_browser_args`.
