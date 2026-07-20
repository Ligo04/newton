# MuJoCo integration surface map

## Primary docs

- `docs/integrations/mujoco.md` — high-level MuJoCo integration page.
- `docs/integrations/index.md` — integrations toctree.
- `docs/concepts/custom_attributes.rst` — MuJoCo custom-attribute namespace and custom frequencies.
- `docs/concepts/collisions.rst` — MuJoCo vs Newton contact pipeline behavior.
- `docs/concepts/sites.rst` — site export behavior for SolverMuJoCo.
- `docs/concepts/usd_parsing.rst` — USD schema resolver ordering, including `SchemaResolverMjc`.
- `docs/concepts/extended_attributes.rst` — SolverMuJoCo extended state attributes.
- `docs/api/newton_solvers.rst` — generated API page listing `SolverMuJoCo`.
- `pyproject.toml` — `mujoco` and `mujoco-warp` dependency pins under the `sim` extra.

## Public import surface

- `newton/solvers.py` exports `SolverMuJoCo` from `newton.solvers`.
- `newton/usd.py` exports `SchemaResolverMjc` from `newton.usd`.
- Public docs/examples should use public modules, not `newton._src` imports.

## Backend code anchors

- `newton/_src/solvers/mujoco/solver_mujoco.py` — `SolverMuJoCo`, custom attributes, Newton→MuJoCo conversion, contacts, tendons, actuators, runtime sync, notify behavior, separate-world validation.
- `newton/_src/solvers/mujoco/kernels.py` — Warp kernels for coordinate conversion, contact conversion, geom property updates, and state sync.
- `newton/_src/utils/import_mjcf.py` — MJCF import and MuJoCo attribute capture/mapping.
- `newton/_src/utils/import_usd.py` — USD import path for MuJoCo schema attributes.
- `newton/_src/usd/schemas.py` — `SchemaResolverMjc` mapping from USD `mjc:*` attributes to Newton built-ins/custom attributes.
- `newton/examples/__init__.py` — helper arguments such as `add_mujoco_contacts_arg`.

## Tests

- `newton/tests/test_mujoco_solver.py` — core SolverMuJoCo behavior.
- `newton/tests/test_mujoco_margin_zeroing.py` — margin zeroing and contact mode caveats.
- `newton/tests/test_mujoco_general_actuators.py` — MuJoCo general actuator custom attributes.
- `newton/tests/test_import_mjcf.py` — MJCF import, custom attributes, actuators, sites, options.
- `newton/tests/test_sites_mjcf_import.py` and `test_sites_mujoco_export.py` — site import/export behavior.
- `newton/tests/test_fixed_tendon.py` and `test_spatial_tendon.py` — tendon behavior.
- `newton/tests/test_menagerie_mujoco.py` and `test_menagerie_usd_mujoco.py` — menagerie import/integration checks.
- `newton/tests/test_mujoco_version_check.py` — dependency/version compatibility expectations.
- `newton/tests/test_solver_mujoco_planar_mesh.py` — planar mesh handling.
- `newton/tests/test_anymal_reset.py` — reset/notify behavior around MuJoCo solver usage.

## Examples

- Robot examples: `newton/examples/robot/*` frequently call `SolverMuJoCo.register_custom_attributes` before importing assets.
- Contact examples: `newton/examples/contacts/*` often compare MuJoCo with Newton contact pipelines via `use_mujoco_contacts=False`.
- Selection/sensor examples: `newton/examples/selection/*`, `newton/examples/sensors/*` exercise SolverMuJoCo state/contact behavior.
- UIPC comparison examples may expose `--solver mujoco` for solver parity checks.

## Quick facts to verify before documenting

- Dependencies: `mujoco-warp>=3.10.0.2,~=3.10.0` and `mujoco~=3.10.0` currently live in the `sim` extra.
- Default contact mode: `use_mujoco_contacts=True` uses MuJoCo/mujoco_warp collision; `False` expects Newton contacts passed into `step`.
- `update_contacts` explicitly pulls MuJoCo contacts into Newton `Contacts`; default stepping does not automatically populate a caller's contacts object.
- `enable_multiccd=True` allows up to four contact points per geom pair, but margin constraints can force single-contact behavior.
- Solver options resolve from constructor args, then `model.mujoco.<option>`, then defaults at solver construction.
- `separate_worlds=True` builds from the first world and replicates through mujoco_warp; all worlds must be structurally identical.
- Fixed roots may become MuJoCo mocap bodies; kinematic non-fixed roots use large internal armature.
- `SolverMuJoCo.register_custom_attributes` is the authoritative registry for MuJoCo-specific custom attrs/frequencies.
