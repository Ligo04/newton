---
name: mujoco-integration
description: Work on Newton's MuJoCo integration, including SolverMuJoCo behavior, docs/integrations/mujoco.md, MuJoCo/mujoco-warp dependency pins, MJCF/USD import behavior, SchemaResolverMjc, MuJoCo custom attributes, contacts, multiccd/margin zeroing, tendons, actuators, sites, multi-world behavior, examples, and tests. Use when the user mentions MuJoCo, mujoco_warp, SolverMuJoCo, MJCF, mjc attributes, "mujoco integration/intergration", MuJoCo contacts, or asks another agent to inspect, document, fix, or extend the Newton MuJoCo backend.
---

# MuJoCo Integration

## Purpose

Use this skill to work on Newton's MuJoCo solver integration with repo-local evidence. It covers documentation, examples, tests, import/export behavior, and backend code for `SolverMuJoCo`.

## First load

1. Read `references/mujoco-surface-map.md` for the maintained MuJoCo file map and quick facts.
2. For docs tasks, read `docs/integrations/mujoco.md` before editing.
3. For code or behavior tasks, inspect the current implementation before trusting any summary.

## Core workflow

1. Classify the request:
   - **Docs:** update `docs/integrations/mujoco.md`, related concepts docs, or navigation.
   - **Backend:** edit `newton/_src/solvers/mujoco/*` or importer/schema code only when needed.
   - **Import/export:** inspect MJCF/USD import paths, `SchemaResolverMjc`, and custom attributes.
   - **Example:** edit examples that construct `SolverMuJoCo` or expose `--solver mujoco` / `--use-mujoco-contacts`.
   - **Tests:** add/update `unittest` tests under `newton/tests/`; do not use pytest.

2. Gather evidence:
   - Start from `SolverMuJoCo` and `newton/_src/solvers/mujoco/kernels.py`.
   - Check `docs/integrations/mujoco.md` for intended public behavior, then verify against code.
   - Check import tests (`test_import_mjcf.py`, site/tendon/general-actuator tests) before changing custom-attribute behavior.

3. Apply Newton repo rules:
   - Public docs/examples must not import from `newton._src`.
   - Use public imports such as `from newton.solvers import SolverMuJoCo`.
   - Preserve compatible-release MuJoCo pins unless release instructions say otherwise.
   - Prefer no new dependencies.
   - Update `CHANGELOG.md` only when user-facing behavior changes.

4. Validate with narrow checks:
   - Docs-only: `uv run --extra docs python -m sphinx -b dummy -q docs docs/_build/dummy` and `git diff --check`.
   - Solver behavior: `uv run --extra dev -m newton.tests -k test_mujoco` or a narrower target.
   - Import/custom attributes: `uv run --extra dev -m newton.tests -k test_import_mjcf`, plus site/tendon/general-actuator tests as relevant.
   - Example changes: run the relevant example test or `uv run --extra dev -m newton.tests -k test_examples` when practical.

## MuJoCo behavior guardrails

- `SolverMuJoCo` uses MuJoCo/mujoco_warp contacts by default; set `use_mujoco_contacts=False` to feed Newton contacts into the solver.
- Contacts are not pulled back automatically; call `update_contacts` when Newton `Contacts` readback is needed.
- `SolverMuJoCo.register_custom_attributes(builder)` must run before importing/building when MuJoCo-specific MJCF/USD attributes need to be preserved.
- MuJoCo solver options are resolved at construction; later edits to `model.mujoco.<option>` do not affect the compiled model.
- `enable_multiccd` and box/mesh constraints interact with MuJoCo/mujoco_warp margin limits; preserve margin-zeroing caveats.
- GPU multi-world `separate_worlds=True` requires structurally identical worlds.
- Kinematic roots and fixed roots have special MuJoCo handling; preserve mocap-body and armature caveats.
- MuJoCo-native unsupported concepts such as sensors, cameras/lights, keyframes, flex/composite, skins, user plugins, and arbitrary custom data should not be claimed as supported unless code/tests prove it.

## Final response

Report changed files, MuJoCo-specific assumptions/caveats, and validation evidence. If behavior changed, name the tests that would fail without the fix when practical.
