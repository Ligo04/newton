<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# UIPC Integration

`SolverUIPC` wraps `libuipc` (the `uipc` / `pyuipc` Python package) behind
Newton's standard solver interface. UIPC provides implicit IPC-style contact
resolution for rigid bodies, cloth, and deformable bodies. Newton converts a
`Model` into a UIPC `Scene` made of AffineBody rigid objects, shell cloth
objects, tetrahedral deformables, contact-tabular entries, and optional
subscenes for multi-world isolation.

Because UIPC has its own scene graph and constitutions, many Newton properties
are baked into UIPC objects during initialization. The sections below describe
what the solver supports, how state and controls are synchronized, where contact
configuration lives, and which runtime updates can be pushed into an already
initialized UIPC world. For the per-parameter mapping table, see the
[UIPC parameter guide](../guide/uipc_parameters.md).

> **Note**
> `SolverUIPC` uses a fixed time step. Construct the solver with the `dt` you
> intend to simulate, and pass the same value to `step(..., dt=dt)`.

## Basic workflow

`SolverUIPC` supports deferred initialization. This lets callers customize the
scene configuration, contact table, subscene table, and inertia bridge before
UIPC finalizes the world:

```python
import newton
from newton.solvers import SolverUIPC

builder = newton.ModelBuilder()
# ...build or import a model...
model = builder.finalize()

dt = 1.0 / 60.0
solver = SolverUIPC(model, dt=dt, workspace="/tmp/newton_uipc")

# UIPC contact is disabled by default in Newton's scene config.
solver.set_contact(True, d_hat=0.001)

# Optional: tune UIPC's JSON-like scene config before initialize().
solver.configure_scene(
    {
        "newton": {"velocity_tol": 1.0e-3},
        "line_search": {"max_iter": 8},
    }
)

# Optional: customize contact pairs before initialize().
def setup_contacts(tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem):
    gripper_elem = tabular.create(f"gripper_{world_index}")
    tabular.insert(gripper_elem, actor_elem, 0.8, 1.0e9, True)
    tabular.insert(gripper_elem, ground_elem, 0.8, 1.0e9, True)


solver.configure_contact_tabular(setup_contacts)
solver.initialize()

state_in = model.state()
state_out = model.state()
control = model.control()
contacts = model.contacts()

for _ in range(100):
    solver.step(state_in, state_out, control, dt=dt)
    solver.update_contacts(contacts, state_out)
    state_in, state_out = state_out, state_in
```

Calling `step` before `initialize` implicitly initializes the solver, but
explicit initialization is recommended whenever you use `configure_scene`,
`configure_contact_tabular`, `configure_subscene_tabular`, or
`sync_uipc_inertia_with_model`.

## Joint types

| Newton type | UIPC equivalent | Notes |
| --- | --- | --- |
| `REVOLUTE` | `AffineBodyRevoluteJoint` + driving / external-torque / limit constitutions | Active joint. `Control.joint_target` can drive `aim_angle`; UIPC's `angle` edge attribute is read back into Newton `joint_q`. |
| `PRISMATIC` | `AffineBodyPrismaticJoint` + driving / external-force / limit constitutions | Active joint. `Control.joint_target` can drive `aim_distance`; UIPC's `distance` edge attribute is read back into Newton `joint_q`. |
| `FIXED` | `AffineBodyFixedJoint`, or fixed child instance for world anchors | A fixed joint to world marks the child instance as fixed. Inter-body fixed joints are emitted as batched UIPC fixed-joint geometry. |
| `FREE` | AffineBody object with `SoftTransformConstraint` | The body is simulated as a free actor. Body pose / velocity are synchronized through Newton `body_q` / `body_qd`, not through an active generalized-coordinate joint. |
| `BALL` | `AffineBodySphericalJoint` | Constrains parent and child anchors, but is not an active driven/read-back joint in Newton's UIPC articulation state. |
| `DISTANCE`, `D6` | *unsupported* | Skipped with a warning during UIPC joint construction. |
| `CABLE` | *unsupported* | No UIPC conversion path. |

For active `REVOLUTE` and `PRISMATIC` joints, parent/child anchors are
validated in world space before UIPC geometry is created. Revolute anchors and
axis endpoints must coincide; prismatic axes must be parallel and their anchors
must be collinear. Invalid authored joint frames raise at initialization rather
than failing later inside UIPC.

### Mimic joint coupling

Mimic constraints (`Model.constraint_mimic_*`, imported from URDF `<mimic>`,
MJCF joint equalities, or USD `PhysxMimicJointAPI` / `NewtonMimicAPI`) enforce
`joint0 = coef0 + coef1 * joint1` (follower = offset + scale × leader). Each
step, before `world.advance()`, the follower's position target is set to
`coef0 + coef1 * q_leader` and the follower is forced into position-driving
mode, so the UIPC animator drives it toward the coupled target. The leader value
is its commanded target when the leader is itself position-driven (no lag),
otherwise its measured start-of-step position (one-step lag).

The coupling is **soft**: the follower tracks its target through the driving
joint's `drive_strength_ratio` and may lag under load, unlike a hard equality
constraint. Both follower and leader must be active
`REVOLUTE` / `PRISMATIC` joints; a constraint whose follower or leader is not an
active UIPC joint is skipped with a warning. Coefficients are baked at
initialization — editing them requires reconstructing the solver.

## Geometry types

Rigid bodies are represented as UIPC AffineBody geometries. Newton gathers each
body's colliding shapes, converts them into one closed triangle mesh, and
applies `AffineBodyConstitution`. Only shapes with the `COLLIDE_SHAPES` flag
contribute to UIPC collision geometry.

| Newton type | UIPC equivalent | Notes |
| --- | --- | --- |
| `BOX` | Closed triangle mesh | Generated from Newton primitive dimensions. |
| `SPHERE` | Closed triangle mesh | Generated from Newton primitive dimensions. |
| `CAPSULE` | Closed triangle mesh | Capsule axis follows `model.up_axis`. |
| `CYLINDER` | Closed triangle mesh | Cylinder axis follows `model.up_axis`. |
| `CONE` | Closed triangle mesh | Cone axis follows `model.up_axis`. |
| `MESH` / `CONVEX_MESH` | Triangle mesh from the shape source | The merged body mesh must be closed and have positive volume. Non-watertight meshes raise with a suggestion to call `ModelBuilder.approximate_meshes`. Near-zero-volume meshes fall back to an AABB box. |
| `PLANE` | `uipc.geometry.halfplane` for `shape_body == -1` | World-attached planes become infinite UIPC halfplanes. Body-attached planes do not contribute to AffineBody meshes. |
| `HFIELD`, `ELLIPSOID`, `GAUSSIAN`, `NONE` | *unsupported* | Ignored by the UIPC rigid-body mesh builder, with warnings for unsupported geometry where applicable. |

World-space non-plane shapes (`shape_body == -1`) are emitted as fixed static
colliders with UIPC's `Empty` constitution. Ground planes share UIPC's default
ground contact element; other static colliders use the per-world environment
contact element.

Newton's SDF, hydroelastic, and other Newton-native collision pipelines are not
fed into UIPC. `SolverUIPC` computes contact internally through UIPC's contact
system.

## Rigid bodies and inertia

Dynamic rigid bodies use UIPC's AffineBody formulation. Newton forwards:

- the initial `body_q` / `body_qd` transforms and velocities,
- kinematic flags as UIPC `is_fixed` instance attributes,
- mesh-derived or authored mass density,
- the solver-level AffineBody stiffness `kappa`, with optional per-body
  overrides through the `uipc:abd_kappa` custom attribute.

By default, `initialize` syncs UIPC's final AffineBody mass, center of mass, and
inertia back into the Newton `Model` via `sync_model_inertia_from_uipc`. This
keeps host-side Newton utilities, such as mass-matrix evaluation for Stable PD,
consistent with the body properties UIPC actually uses.

If you need UIPC to preserve Newton-authored inertial values instead, call
`sync_uipc_inertia_with_model(body_indices)` before `initialize`. To keep
Newton-authored values untouched after initialization, construct the solver with
`auto_sync_inertia=False`.

### Joint armature

AffineBody dynamics has no joint-space mass slot, so `Model.joint_armature` on
a revolute joint is folded into the child link's ABD inertia as
`armature * axis ⊗ axis` about the joint axis. This is exact for rotation
about the joint's own axis, but the extra inertia also resists other rotations
of that link (upstream joints, contact impulses) — unlike true joint-space
armature. Armature children are automatically built through the
Newton-authored mass-matrix path described above (their mass and COM then come
from the Newton model rather than `mass_density * mesh_volume`). Armature on
non-revolute joints has no ABD equivalent and is dropped with a warning.
`sync_model_inertia_from_uipc` subtracts the folded armature on the way back,
so `Model.body_inertia` always stays armature-free and
`eval_mass_matrix(include_armature=True)` does not double-count.

(uipc-custom-attributes)=
## UIPC-specific custom attributes

`SolverUIPC.register_custom_attributes` registers Newton's UIPC namespace on a
`ModelBuilder`. Call it before adding or importing bodies when you want to
author UIPC-specific values:

```python
import newton
from newton.solvers import SolverUIPC

builder = newton.ModelBuilder()
SolverUIPC.register_custom_attributes(builder)
# ...add bodies...
```

Currently registered:

| Attribute | Frequency | Default | Meaning |
| --- | --- | --- | --- |
| `model.uipc.abd_kappa` | Body | `-1.0` | Per-body AffineBody stiffness override [Pa]. Negative values inherit the solver-level `kappa`. |
| `model.uipc.cloth_model` | `uipc:cloth` | `"strain_limiting_baraff_witkin"` | Per-`cloth_ranges` membrane constitution. Supported values are `"strain_limiting_baraff_witkin"` and `"neo_hookean"`. |
| `model.uipc.deformable_model` | `uipc:deformable_body` | `"stable_neo_hookean"` | Per-`soft_body_ranges` tetrahedral deformable constitution. Supported values are `"stable_neo_hookean"` and `"arap"`. |

## Cloth

Newton cloth is built from particles, triangle connectivity, triangle material
properties, and bending-edge properties. UIPC cloth objects use an open
triangle mesh plus a membrane constitution and `DiscreteShellBending`.

| Newton source | UIPC target | Notes |
| --- | --- | --- |
| `particle_q` | Cloth mesh vertices | World-space particle positions [m]. |
| `tri_indices` / `cloth_ranges` | Cloth mesh faces and grouping | Authored `cloth_ranges` are preferred; otherwise the legacy selector uses triangle connectivity not claimed by tetrahedra. |
| `particle_mass` / `tri_areas` | Membrane mass density | Estimated as total mass divided by total area and shell thickness, unless an authored cloth surface density is present. |
| `particle_radius` | Vertex shell thickness | Cloth collision thickness follows Newton particle radius. |
| `tri_materials[:, 0]` / `tri_materials[:, 1]` | Triangle `mu` / `lambda` attributes | Written directly to UIPC membrane attributes after applying the chosen membrane model. |
| `edge_bending_properties[:, 0]` | Edge `bending_stiffness` | Damping (`edge_kd`) is not forwarded. |
| `particle_mass <= 0.0` | UIPC `is_fixed` vertex marker | Marks kinematic cloth vertices. |

The default membrane model is `StrainLimitingBaraffWitkinShell`. Call
`SolverUIPC.register_custom_attributes(builder)` before adding cloth, then set
`model.uipc.cloth_model[cloth_index]` before constructing `SolverUIPC` to choose
`NeoHookeanShell` for one authored `model.cloth_ranges` entry. Closed /
watertight triangle meshes are rejected as cloth; use a deformable or rigid
representation for closed volumes.

When `enable_soft_position_constraint=True` (the default), the builder adds
dormant UIPC `SoftPositionConstraint` attributes. Use
`set_cloth_soft_position_constraints` to enable selected particle handles and
`clear_cloth_soft_position_constraints` to disable them.

## Deformable bodies

Newton deformables are built from particles and tetrahedra. UIPC deformables
use `tetmesh` geometry, surface labels for contact, and
`StableNeoHookean` elasticity by default. Call
`SolverUIPC.register_custom_attributes(builder)` before adding soft bodies, then
set `model.uipc.deformable_model[soft_index]` before constructing `SolverUIPC`
to choose a different supported constitution for one authored
`model.soft_body_ranges` entry.

| Newton source | UIPC target | Notes |
| --- | --- | --- |
| `particle_q` | Tet mesh vertices | World-space particle positions [m]. |
| `tet_indices` / `soft_body_ranges` | Tet mesh topology and grouping | Authored `soft_body_ranges` keep multiple soft bodies separate. |
| `particle_mass` + tet volume | Mass density | Falls back to `default_mass_density` when the estimate is unavailable. |
| `tet_materials[:, 0]` / `tet_materials[:, 1]` | Deformable material attributes | Stable Neo-Hookean writes converted `mu` / `lambda`; ARAP writes `tet_materials[:, 0]` to UIPC `kappa`. |
| `particle_mass <= 0.0` | UIPC `is_fixed` vertex marker | Marks kinematic deformable vertices. |

When soft-position constraints are enabled, use
`set_deformable_soft_position_constraints` and
`clear_deformable_soft_position_constraints` to drive selected deformable
particles.

## Actuators and controls

UIPC joint control is applied through UIPC `Animator` callbacks registered for
active revolute and prismatic joints. Before each `world.advance()`, Newton
copies the current `Control` buffers into CPU arrays consumed by those
callbacks.

| Newton target mode | UIPC behavior |
| --- | --- |
| `POSITION` | Enables UIPC driving and writes `aim_angle` / `aim_distance` from `Control.joint_target`. |
| `POSITION_VELOCITY` | Same as `POSITION`; the position target is forwarded. |
| `EFFORT` | Enables UIPC external torque / force and writes from `Control.joint_f`. |
| `VELOCITY` | Passive; no UIPC velocity-only drive is written. |
| `NONE` | Passive. |

The aim-drive strength is a pure solver constraint-stiffness knob, deliberately
decoupled from `joint_target_ke` / `joint_target_kd` (which UIPC does not
consume — they remain portable metadata for other backends). Position-driven
joints get the solver-level `drive_strength_ratio` parameter: a global float
(default `100.0`, near-rigid tracking) or a per-joint mapping keyed by Newton
joint index, e.g. `SolverUIPC(model, drive_strength_ratio={3: 10.0})`; joints
missing from the mapping fall back to `100.0`. Non-position target modes get
no drive. The joint anchoring constraints themselves use the independent
solver-level `joint_strength_ratio` parameter (default `100.0`).

Alternatively, `SolverUIPC(implicit_pd=True)` opts position-driven joints into
implicit PD with physical gain semantics: `joint_target_ke` /
`joint_target_kd` [N·m/rad, N·m·s/rad] replace `drive_strength_ratio`. The PD
is expressed inside the incremental potential — the stiffness spring acts on
the new-state position error and the damping spring on the new-state velocity
error (an aim toward `q_prev + dt * dq_ref`, merged with the position spring
into the single drive channel) — so it is co-solved with contact,
unconditionally stable, and equivalent to `SolverKamino`'s implicit joint PD:
steady-state sag under load is `tau / ke`, and `kd` damps transients.
`POSITION_VELOCITY` mode feeds `joint_target_qd` as the damping reference;
plain `POSITION` damps toward rest. Gains are baked at initialization and
cannot be changed at runtime.
`joint_limit_lower` / `joint_limit_upper` create UIPC joint-limit
constitutions whose strength comes from the solver-level
`limit_strength_ratio` parameter (default `10.0`, same global-or-per-joint
form as `drive_strength_ratio`), decoupled from `joint_limit_ke`.

## Contact pipeline

`SolverUIPC` computes contact inside UIPC. Contacts passed to `step` are ignored;
call `update_contacts` after `step` when Newton-side force or contact buffers
are needed.

Global contact is disabled in the default Newton UIPC scene config:

```python
solver = SolverUIPC(model)
solver.set_contact(True, d_hat=0.001)
```

`d_hat` is UIPC's IPC barrier distance [m]. Smaller values allow tighter
contacts but may require smaller time steps or more Newton iterations for
stability.

### Contact elements and default pairs

During initialization, Newton creates one shared ground contact element and
three contact elements per Newton world:

- `ground_elem` — ground planes.
- `env_elem` — environment / non-articulated rigid bodies and kinematic bodies.
- `robo_elem` — articulated robot links.
- `actor_elem` — free-joint actors, cloth, and deformables.

Default contact pairs use friction `0.5` and stiffness `1 GPa`:

| Pair | Enabled by default |
| --- | --- |
| `env` ↔ `env` | No |
| `env` ↔ `robot` | Yes |
| `env` ↔ `actor` | Yes |
| `ground` ↔ `env` | No |
| `ground` ↔ `robot` | Yes |
| `ground` ↔ `actor` | Yes |
| `robot` ↔ `robot` | No |
| `robot` ↔ `actor` | Yes |
| `actor` ↔ `actor` | Yes |

Use `configure_contact_tabular` before initialization to insert additional
elements, override friction / stiffness, or change which pairs are enabled. The
callback may also return a `{body_index: contact_element}` mapping to override
individual body assignment.

### Contact readback

`update_contacts(contacts, state)` pulls UIPC contact gradients back into
Newton form:

- `state.body_f` receives per-rigid-body contact wrenches when allocated.
- `state.particle_f` receives per-particle cloth/deformable contact forces when
  allocated.
- `contacts.force` and rigid-contact pair arrays are populated when those
  buffers are present.

For CPU diagnostics, `_get_contact_forces` returns the lower-level per-body,
per-primitive UIPC contact-force buckets. It is intentionally an internal
debugging entry point rather than the normal public readback path.

## Multi-world support

For models produced by `ModelBuilder.replicate`, `SolverUIPC` builds all Newton
worlds into one UIPC scene. It creates one UIPC subscene per Newton world and
places each world's bodies, cloth, and deformables into that subscene.

By default, cross-subscene contact is disabled, while each world can still
contact the default subscene used by ground planes and global objects. Use
`configure_subscene_tabular` before initialization to opt into cross-world
contact or other custom subscene rules.

## Runtime state synchronization

Each call to `SolverUIPC.step` uses the same three-phase cycle:

1. **Push control / snapshot joints.** Active joint controls are cached for
   UIPC animator callbacks. Current revolute `angle` / prismatic `distance`
   attributes are snapshotted so post-step joint velocities can be finite
   differenced.
2. **Advance UIPC.** `world.advance()` runs UIPC's nonlinear solve, contact
   handling, and animator callbacks. `world.retrieve()` makes the integrated
   state available for readback.
3. **Pull UIPC → Newton.** AffineBody transforms / velocities and FEM particle
   positions / velocities are copied into `state_out`; active joint positions
   and velocities are written into `state_out.joint_q` / `joint_qd`; force
   buffers are cleared until `update_contacts` is called.

`dt` is fixed by the constructor. If the first `step` receives a different
`dt`, Newton warns; UIPC still uses the constructor-configured time step.

Per-world reset is supported at runtime via
:meth:`~newton.solvers.SolverUIPC.reset`, which re-pushes the selected worlds'
rigid-body (and FEM particle) state into the live IPC scene without rebuilding
the solver. After a body-state push, UIPC's internal revolute/prismatic
joint-angle tracker lags by one step; read ``Model.joint_q`` rather than
``State.joint_q`` for articulated bodies until the next ``step``.

## Runtime model updates

Most model properties are baked into UIPC geometry during initialization. After
the solver is initialized, `notify_model_changed` supports only:

| Flag | Runtime effect |
| --- | --- |
| `BODY_PROPERTIES` | Pushes updated `body_q` / `body_qd` and FEM particle state into UIPC. |
| `JOINT_PROPERTIES` | Runs Newton FK from `joint_q` / `joint_qd` / joint frames, then pushes the resulting body state into UIPC. |
| `MODEL_PROPERTIES` | Propagates `model.gravity` into the live UIPC `scene.config()`. |

Unsupported flags (`JOINT_DOF_PROPERTIES`, `BODY_INERTIAL_PROPERTIES`,
`SHAPE_PROPERTIES`, `CONSTRAINT_PROPERTIES`, `TENDON_PROPERTIES`, and
`ACTUATOR_PROPERTIES`) produce one aggregated warning. Recreate the solver when
those properties change.

`set_contact` is safe before or after initialization. `set_animator_substep`
can be called after initialization to change how many times UIPC animator
callbacks fire within one `world.advance()`.

## Solver options

Important constructor arguments:

| Argument | Default | Notes |
| --- | --- | --- |
| `backend` | `"cuda"` | Passed to `uipc.Engine(backend_name=...)`. |
| `workspace` | `"/tmp/newton_uipc"` | UIPC engine workspace, surface-dump directory, and default performance-report root. |
| `dt` | `1.0 / 60.0` | Fixed UIPC scene time step [s]. |
| `scene_config` | `uipc.Scene.default_config()` | Mutated with Newton defaults for `dt`, gravity, contact, and Newton tolerances. |
| `kappa` | `1 GPa` | Solver-level AffineBody stiffness [Pa]. |
| `default_mass_density` | `1000.0` | Fallback density [kg/m³] for rigid/deformable construction. |
| `dump_enable` | `False` | Dumps UIPC surface OBJ snapshots before each physics advance. |
| `require_profile` | `False` | Collects UIPC timer data; use `save_performance_report`. |
| `auto_sync_inertia` | `True` | Sync final UIPC AffineBody inertia back into Newton after initialization. |
| `enable_soft_position_constraint` | `True` | Adds dormant cloth/deformable soft-position attributes. |

Use `configure_scene` for UIPC scene configuration not exposed as constructor
arguments. It deep-merges nested dictionaries before initialization.

## Unsupported Newton features and caveats

Smaller limitations are documented inline above. The most common ones are:

- **Global contact is off by default.** Call `set_contact(True, d_hat=...)` for
  contact-driven examples.
- **Rigid AffineBody meshes must be closed.** Non-watertight rigid meshes raise
  during initialization. Use `ModelBuilder.approximate_meshes` to generate a
  closed convex hull, COACD approximation, or bounding box before constructing
  the solver.
- **Only revolute and prismatic joints are active driven/read-back joints.**
  Ball joints constrain anchors but do not participate in the active joint
  control/readback arrays. Distance, D6, and cable joints are unsupported.
- **Mimic constraints are enforced softly via position driving.** The follower
  tracks `coef0 + coef1 * q_leader` through its driving-joint stiffness and may
  lag under load; this is not a hard equality constraint. Follower and leader
  must both be active revolute/prismatic joints, and coefficients are baked at
  initialization.
- **Velocity-only joint drives are not forwarded.** `VELOCITY` target mode is
  passive in UIPC; `POSITION_VELOCITY` forwards only the position target.
- **Most edits require solver reconstruction.** Shape changes, actuator changes,
  inertial changes, constraints, tendons, and joint-DOF property changes are not
  pushed into live UIPC objects.
- **Newton-native contact features are separate.** SDF and hydroelastic contact
  pipelines are not fed into UIPC.

(uipc-code-pointers)=
## Code pointers

For readers navigating the source, the following symbols are the most useful
entry points. Symbols with a leading underscore are **internal entry points** —
stable enough to navigate to, but not part of the public API and subject to
change.

- `SolverUIPC.register_custom_attributes` — UIPC-specific custom attribute
  registration (`uipc:abd_kappa`).
- `SolverUIPC.__init__` — scene-config defaults and constructor options.
- `SolverUIPC.configure_scene`, `set_contact`, `configure_contact_tabular`, and
  `configure_subscene_tabular` — pre-initialization scene/contact setup.
- `SolverUIPC.initialize` — Newton `Model` → UIPC `Engine` / `World` / `Scene`
  construction and multi-world setup.
- `SolverUIPC.step` — per-step UIPC integration entry point.
- `SolverUIPC.update_contacts` — explicit pull of UIPC contact forces into
  Newton `State` and `Contacts` buffers.
- `SolverUIPC.sync_model_inertia_from_uipc` /
  `sync_uipc_inertia_with_model` — bidirectional AffineBody mass/inertia bridge.
- `SolverUIPC.notify_model_changed` — runtime updates supported by the UIPC
  backend.
- `SolverUIPC.set_cloth_soft_position_constraints` /
  `set_deformable_soft_position_constraints` — soft-position handles for FEM
  particles.
- `newton/_src/solvers/uipc/rigid_body.py` — AffineBody, ground-plane, and
  static-collider construction.
- `newton/_src/solvers/uipc/articulation_builder.py` and
  `articulation.py` — joint conversion, UIPC animator callbacks, and joint
  state readback.
- `newton/_src/solvers/uipc/cloth.py` — cloth shell and bending conversion.
- `newton/_src/solvers/uipc/deformable_body.py` — Stable Neo-Hookean tet
  conversion.
- `newton/_src/solvers/uipc/contact_forces.py` — UIPC contact-gradient readback
  and Warp scatter kernels.
- `newton/_src/solvers/uipc/converter.py` — shape-to-mesh conversion, state
  mapping, and UIPC backend-offset tables.
