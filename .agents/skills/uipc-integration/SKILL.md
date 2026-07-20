---
name: uipc-integration
description: Work on Newton's UIPC integration, including SolverUIPC behavior, docs/integrations/uipc.md, docs/guide/uipc_parameters.md, UIPC examples/tests, pyuipc/libuipc setup, contact tabulars, AffineBody rigid bodies, cloth/deformable mappings, inertia sync, and UIPC-specific caveats. Use when the user mentions UIPC, SolverUIPC, pyuipc, libuipc, "uipc inte/integration", UIPC docs, UIPC examples, or asks another agent to inspect, document, fix, or extend the Newton UIPC backend.
---

# UIPC Integration

## Purpose

Use this skill to work on Newton's UIPC solver integration with repo-local evidence. It covers documentation, examples, tests, and backend code for `SolverUIPC`.

## First load

1. Read `references/uipc-surface-map.md` for the maintained UIPC file map and quick facts.
2. For docs tasks, read `docs/integrations/uipc.md` and the requested reference doc before editing.
3. For code or behavior tasks, inspect the current implementation before trusting any summary.

## Core workflow

1. Classify the request:
   - **Docs:** update `docs/integrations/uipc.md`, `docs/guide/uipc_parameters.md`, or navigation.
   - **Backend:** edit `newton/_src/solvers/uipc/*` and public exports only when needed.
   - **Example:** edit `newton/examples/uipc/**` and keep example-browser registration/tests in mind.
   - **Tests:** add/update `unittest` tests under `newton/tests/`; do not use pytest.

2. Gather evidence:
   - Start from `SolverUIPC` and the relevant builder (`rigid_body`, `articulation_builder`, `cloth`, `deformable_body`, `contact_forces`, `converter`).
   - Check examples for intended usage patterns before changing public guidance.
   - Check tests for established expectations and skip behavior when UIPC is unavailable.

3. Apply Newton repo rules:
   - Public docs/examples must not import from `newton._src`.
   - Expose user-facing symbols through public modules when adding APIs.
   - Prefer no new dependencies; use Warp, NumPy, stdlib, or existing UIPC dependency.
   - Use PEP 604 unions and bracket-style Warp array annotations in code.
   - Update `CHANGELOG.md` only when user-facing behavior changes.

4. Validate with the narrowest useful checks:
   - Docs-only: `uv run --extra docs python -m sphinx -b dummy -q docs docs/_build/dummy` and `git diff --check` on touched docs.
   - UIPC unit tests: `uv run --extra dev -m newton.tests -k test_uipc` or a narrower `-k` target.
   - Example-discovery changes: `uv run --extra dev -m newton.tests -k test_example_browser_args`.
   - If UIPC/GPU is unavailable, report the exact skipped/blocked validation and use static checks plus import/discovery tests.

## UIPC behavior guardrails

- Contact is computed inside UIPC; `step` does not consume Newton contacts. Use `update_contacts` for Newton-side force/contact buffers.
- Global UIPC contact is disabled by default; examples usually call `solver.set_contact(True, d_hat=...)`.
- UIPC uses a fixed constructor `dt`; do not imply per-step `dt` retunes UIPC.
- Active driven/read-back joints are revolute and prismatic. Ball joints constrain anchors but are not active UIPC articulation controls.
- Most shape, actuator, inertial, and joint-DOF changes are baked at initialization; recreate the solver unless `notify_model_changed` explicitly supports the flag.
- Rigid AffineBody meshes must be closed and positive-volume; use `ModelBuilder.approximate_meshes` guidance for non-watertight inputs.

## Final response

Report changed files, UIPC-specific assumptions/caveats, and validation evidence. If you changed behavior, name the tests that would fail without the fix when practical.
