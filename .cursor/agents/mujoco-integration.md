---
name: mujoco-integration
description: Project-local agent prompt for Newton SolverMuJoCo/MuJoCo integration work: backend behavior, docs, examples/tests, MJCF/USD import, custom attributes, contacts, tendons, actuators, sites, and multi-world behavior.
---

# MuJoCo Integration Agent

Use this agent when working on Newton's MuJoCo integration.

First load:
1. `.cursor/skills/mujoco-integration/SKILL.md`
2. `.cursor/skills/mujoco-integration/references/mujoco-surface-map.md`
3. Current implementation files before making behavior claims.

Scope:
- `newton/_src/solvers/mujoco/**`
- `newton/_src/utils/import_mjcf.py`
- `newton/_src/usd/schemas.py`
- `newton/examples/**` where `SolverMuJoCo` is used
- `newton/tests/test_mujoco*`, `test_import_mjcf.py`, site/tendon/actuator tests
- `docs/integrations/mujoco.md` and related concepts docs.

Guardrails:
- Public docs/examples must not import from `newton._src`.
- Use unittest commands, not pytest.
- Preserve MuJoCo caveats: default MuJoCo contacts, explicit `update_contacts`, construction-time option resolution, custom attribute registration, margin zeroing/multiccd constraints, and separate-world structural identity.

Validation:
- Docs: `uv run --extra docs python -m sphinx -b dummy -q docs docs/_build/dummy` and `git diff --check`.
- MuJoCo tests: `uv run --extra dev -m newton.tests -k test_mujoco` or narrower.
- Import/custom attributes: `uv run --extra dev -m newton.tests -k test_import_mjcf` plus relevant site/tendon/actuator tests.
