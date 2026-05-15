# Newton Development Guidelines

- `newton/_src/` is internal. Examples and docs must not import from `newton._src`. Expose user-facing symbols via public modules (`newton/geometry.py`, `newton/solvers.py`, etc.).
- Breaking changes require a deprecation first. Do not remove or rename public API symbols without deprecating them in a prior release.
- Prefix-first naming for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`).
- Prefer nested classes for self-contained helper types/enums.
- PEP 604 unions (`x | None`, not `Optional[x]`).
- Annotate Warp arrays with bracket syntax (`wp.array[wp.vec3]`, `wp.array2d[float]`, `wp.array[Any]`), not the parenthesized form (`wp.array(dtype=...)`). Use `wp.array[X]` for 1-D arrays, not `wp.array1d[X]`.
- Follow Google-style docstrings. Types in annotations, not docstrings. `Args:` use `name: description`.
  - Sphinx cross-refs (`:class:`, `:meth:`) with shortest possible targets. Prefer public API paths; never use `newton._src`.
  - SI units for physical quantities in public API docstrings: `"""Particle positions [m], shape [particle_count, 3]."""`. Joint-dependent: `[m or rad]`. Spatial vectors: `[N, N·m]`. Compound arrays: per-component. Skip non-physical fields.
- Run `docs/generate_api.py` when adding public API symbols.
- Avoid new required dependencies. Strongly prefer not adding optional ones — use Warp, NumPy, or stdlib.
- Create a feature branch before committing — never commit directly to `main`. Use `<username>/feature-desc`.
- Imperative mood in commit messages ("Fix X", not "Fixed X"), ~50 char subject, body wraps at 72 chars explaining _what_ and _why_.
- Verify regression tests fail without the fix before committing.
- Pin GitHub Actions by SHA: `action@<sha>  # vX.Y.Z`. Check `.github/workflows/` for allowlisted hashes.
- In SPDX copyright lines, use the year the file was first created. Do not create date ranges or update the year when modifying a file.

Run `uvx pre-commit run -a` to lint/format before committing. Use `uv` for all commands; fall back to `venv`/`conda` if unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum
```

## Tests

Always use `unittest`, not pytest.

```bash
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch
```

### Testing guidelines

- Never call `wp.synchronize()` or `wp.synchronize_device()` right before `.numpy()` on a Warp array. This is redundant as `.numpy()` performs a synchronous device-to-host copy that completes all outstanding work.

```bash
# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- If a change modifies user-facing behavior, insert an entry at a random position within the correct category (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`) in `CHANGELOG.md`'s `[Unreleased]` section. Use imperative present tense ("Add X") and avoid internal implementation details.
- For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance: "Deprecate `Model.geo_meshes` in favor of `Model.shapes`".

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` — runs after the example completes to verify simulation state is valid.
  - Optionally implement `test_post_step()` — runs after every `step()` for per-step validation.
- Register in `README.md` with `python -m newton.examples <name>` command and a 320x320 jpg screenshot.


<claude-mem-context>
# Memory Context

# [newton] recent context, 2026-05-15 11:18am GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (15,851t read) | 624,492t work | 97% savings

### May 7, 2026
1992 12:27p 🟣 ClothRange dataclass added to Newton simulation model
1993 " 🟣 ClothRange exported through full newton package hierarchy
1995 12:29p 🟣 65/65 TestCloth tests pass including all three new ClothRange tests
1996 12:30p ✅ API docs regenerated; newton.rst now 38 symbols after ClothRange addition
S579 How does deformable_body.py distinguish soft bodies — full implementation of per-body UIPC geometry building using SoftBodyRange/ClothRange now applied to both UIPC solvers (May 7, 12:33 PM)
1998 12:33p 🟣 UIPC cloth solver refactored to use ClothRange for per-body geometry building
1999 " 🔴 UIPC cloth hardcoded "cloth_0" object name fixed for multi-cloth scenes
1997 " 🔵 label_prefix only applied when both prefix and range label are non-None
S582 Per-body UIPC solver tracking using SoftBodyRange/ClothRange: implemented, then simplified deformable_body.py by removing BFS tet grouping (May 7, 2:18 PM)
2000 2:19p 🔵 test_uipc_cloth 5/5 pass; test_uipc_deformable 0/3 fail with NameError in _build_tet_groups
2001 " 🔴 NameError in _build_tet_groups fixed by adding model = self._model
2002 " 🔴 cloth.py type safety fixes: set() → int comprehension, pyright ignores for dynamic attrs
2003 " 🔴 test_uipc_deformable 3/3 green after model scope fix
S583 Per-body UIPC solver tracking: SoftBodyRange/ClothRange implemented + BFS tet grouping removed from DeformableBodyBuilder after authored ranges made it redundant (May 7, 2:19 PM)
S580 Per-body index tracking for UIPC solvers: implement SoftBodyRange/ClothRange and refactor deformable_body.py + cloth.py to use exact authored ranges instead of particle heuristics (May 7, 2:19 PM)
S581 Per-body UIPC solver tracking: SoftBodyRange/ClothRange implementation + deformable_body.py refactor to use exact authored ranges; then simplified to remove BFS grouping (May 7, 2:19 PM)
2004 2:20p 🟣 deformable_body.py fully reworked: connected-component grouping, fixed-vertex marking, mesh_partition
2005 " 🟣 mesh_partition(sc, 16) added to cloth UIPC geometry building
2006 2:35p 🟣 New tests: single SoftBodyRange with disconnected tets + explicit fallback path
2008 2:36p 🔵 test_uipc_deformable reverted to 3 tests — new tests not persisted
S584 Per-body UIPC solver tracking: SoftBodyRange/ClothRange implementation + DeformableBodyBuilder simplified to one-range-one-geometry (no BFS grouping) (May 7, 2:37 PM)
2007 2:39p 🟣 test_uipc_deformable expanded to 5 tests, all passing
2009 2:41p 🔄 DeformableBodyBuilder: _build_tet_groups renamed to _build_tet_set
S586 Query: does example_cloth_franka.py have fixed vertices for cloth mesh? (May 7, 2:42 PM)
S587 Query: does example_uipc_cloth_franka.py have fixed cloth vertices? (May 7, 4:33 PM)
S585 Query: does example_cloth_franka.py have fixed vertices for cloth mesh? (May 7, 4:33 PM)
2010 4:38p 🟣 Test added: UIPC cloth-franka must use robot contact, not soft handle API
2012 " 🔄 example_uipc_cloth_franka.py refactored: soft-handle API removed, robot contact activated
2011 4:39p 🔵 example_uipc_cloth_franka.py currently uses soft-handle API — TDD red phase confirmed
2013 4:40p 🔴 TDD cycle complete: test now green, quat_to_vec4 helper removed
2014 " 🔵 UIPC init fails: Franka geometry intersects cloth and body_0 at t=0
2015 " 🔴 Reverted Franka base position and cloth spawn height to fix UIPC init intersection
2016 4:41p 🔵 UIPC workspace caches old sanity-check logs; example now runs with original Franka base
2017 4:42p ✅ Final confirmed git diff: example_uipc_cloth_franka.py and test_examples.py
2018 " 🔵 Only uv available; ruff/black/mypy/pylint not installed in this environment
2019 " 🔵 Keyframe sequence totals 70.5s = 4230 frames; num_frames=3850 ends simulation early
2020 " 🔵 Franka FR3 body structure: 9 links, fr3_link7 at index 6 in URDF-only builder
2022 4:44p 🟣 num_frames constant extracted; cloth particle tracking removed; positive assertions and runtime test added
2023 4:45p 🔴 Two test failures fixed: assertIn on missing string and --test-timeout unrecognized argument
2021 4:46p 🔵 Quaternion linear interpolation in _target_at_time produces non-unit quaternions at transitions
2024 4:48p 🔴 Both uipc_cloth_franka tests green: static assertion + runtime CUDA integration
2025 " 🔵 assertIn("configure_contact_tabular") fails — API not present in example and may not exist yet
2026 4:50p 🟣 configure_contact_tabular added to example_uipc_cloth_franka.py for selective robot contact
2027 " 🔴 Ruff reports SyntaxError at line 1:0 — example file starts with blank line, may be corrupted
2028 " ✅ Final complete git diff confirmed: configure_contact_tabular + all refactor changes in example
2029 4:51p 🔴 Ruff E999 syntax error was transient — py_compile confirms file is syntactically valid
2032 " 🔵 Black disagrees with ruff formatting on both files; Python version mismatch warning
2033 " 🔵 Project ruff vs latest uvx ruff disagree: 2 import-sort errors in committed files
S589 Refactor newton/examples/uipc/example_uipc_cloth_franka.py: replace soft-constraint cloth manipulation with physical robot contact; fix uvx ruff I001 import-sort violations (May 7, 4:51 PM)
### May 8, 2026
2072 10:46a 🔵 libuipc: Exploring clamp_close_activation_val and Contact Activation Code
2077 " 🔵 UIPC/libuipc lacks .omx/wiki — sphere-driving code search initiated
2074 " ⚖️ Robot Component Should Use free_joint for Movement
2073 10:47a 🔵 OMX Explore Shell Execution Blocked by bwrap Sandbox in Newton Repo
2078 10:50a 🔵 OMX Explore shell backend blocked by bwrap sandbox in Newton-Isaac/newton
2079 2:41p 🔵 Studied: Adjustable Constrained Soft-Tissue Dynamics (Wang, Zheng, Barbic 2020)
2080 " ✅ Paper archived to ~/claude-papers study library
2081 " 🔵 claude-papers library structure confirmed at ~/claude-papers/
2082 2:42p ✅ meta.json created for Wang-Zheng-Barbic 2020 in claude-papers library
2083 " ✅ README.md study guide written for Wang-Zheng-Barbic 2020
2084 2:43p ✅ summary.md created for Wang-Zheng-Barbic 2020 study materials

Access 624k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>