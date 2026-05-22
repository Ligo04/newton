# UIPC integration surface map

## Primary docs

- `docs/integrations/uipc.md` — high-level UIPC integration page.
- `docs/guide/uipc_parameters.md` — parameter mapping table, currently Chinese.
- `docs/integrations/index.md` — integrations toctree.
- `docs/api/newton_solvers.rst` — generated API page listing `SolverUIPC`.
- `CHANGELOG.md` — recent UIPC feature/fix history.

## Public import surface

- `newton/solvers.py` exports `SolverUIPC` from `newton.solvers`.
- Public docs/examples should import `from newton.solvers import SolverUIPC` or use `newton.solvers.SolverUIPC`, not `newton._src`.

## Backend code anchors

- `newton/_src/solvers/uipc/solver_uipc.py` — `SolverUIPC`, scene config, contact setup, initialize/step, contact readback, inertia sync, notify flags, soft-position APIs.
- `newton/_src/solvers/uipc/rigid_body.py` — AffineBody construction, static colliders, ground halfplanes, contact-element assignment, instancing.
- `newton/_src/solvers/uipc/articulation_builder.py` — revolute/prismatic/fixed/free/ball joint conversion and UIPC animator registration.
- `newton/_src/solvers/uipc/articulation.py` — active joint control cache, animator callbacks, joint q/qd readback.
- `newton/_src/solvers/uipc/cloth.py` — cloth triangle mesh, membrane model, bending stiffness, fixed vertices, soft-position constraints.
- `newton/_src/solvers/uipc/deformable_body.py` — tet mesh, Stable Neo-Hookean materials, fixed vertices, soft-position constraints.
- `newton/_src/solvers/uipc/contact_forces.py` — UIPC contact-gradient readback and Warp scatter into Newton state/contacts.
- `newton/_src/solvers/uipc/converter.py` — shape-to-mesh conversion, backend offset maps, state conversion kernels.
- `newton/_src/solvers/uipc/utils.py` — small UIPC helper utilities.

## Examples

- `newton/examples/uipc/basic/` — hello world, joints, USD loading, conveyor.
- `newton/examples/uipc/robot/` — Cartpole, Panda, UR10, humanoids, Allegro hand.
- `newton/examples/uipc/contacts/` — brick stacking, nut/bolt, pyramid, two-brick stack.
- `newton/examples/uipc/cloth/` — cloth twist, cloth Franka, poker cards.
- `newton/examples/uipc/softbody/` — deformable body and softbody demos.
- `newton/examples/uipc/multiphysics/` — softbody-to-cloth coupling.
- `newton/examples/uipc/sensors/` — UIPC contact sensor readback.

## Tests and diagnostics

- `newton/tests/test_uipc_cloth.py` — cloth material and soft-position behavior.
- `newton/tests/test_uipc_ur10_force.py` — UIPC UR10 force/stable-PD parity helpers.
- `newton/tests/test_uipc_sanity_check_backend.py` — backend sanity-check behavior.
- `newton/tests/test_example_browser_args.py` — grouped UIPC example discovery.
- `newton/tests/test_examples.py` — example source expectations for UIPC contact/configuration.
- `scripts/compare_uipc_newton_inertia.py` — inertia bridge diagnostic.
- `scripts/compare_uipc_fd_vs_ik_joint_qd.py` — joint qd readback diagnostic.

## Quick facts to verify before documenting

- Dependency: `pyuipc>=0.0.25` in `pyproject.toml` for Python 3.10-3.13.
- Default backend: `backend="cuda"`; some tests use `backend="none"` for configuration checks.
- Scene defaults in `SolverUIPC.__init__`: contact disabled, `d_hat=0.001`, Newton velocity tolerance `0.001`, translation tolerance `0.01`, gravity copied from the model.
- Contact elements: shared ground plus per-world env/robot/actor; default pairs use friction `0.5` and stiffness `1 GPa` with selected pairs disabled.
- Active joint target modes: `POSITION`/`POSITION_VELOCITY` write position targets; `EFFORT` writes external force/torque; `VELOCITY` is passive.
- Runtime notify support: body properties, joint properties via FK/state push, and model gravity. Other notify flags warn and require solver reconstruction.
