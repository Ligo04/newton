---
name: uipc-contact-tabular
description: Use when configuring, tuning, or debugging contact behavior in SolverUIPC — friction/stiffness per pair, enabling/disabling contact between specific body categories, adding custom contact elements (e.g. a separate "gripper" element), or overriding the default element of a specific body. Covers the four built-in elements (ground/env/robo/actor), the default tabular inserted by the solver, the `configure_contact_tabular` callback signature, and per-body overrides via the return dict.
---

# UIPC Contact Tabular

SolverUIPC uses UIPC's `ContactTabular` to declare which pairs of bodies may
interact, and with what friction / stiffness. This skill captures everything
you need to configure it correctly without reading the solver source.

## The Four Built-in Elements

The solver creates **one** shared ground element and **three** per-world
elements (suffix `_{world_index}` only when `world_count > 1`):

| Element      | Applied to                                                              | Source (`rigid_body.py:194-211`) |
|--------------|-------------------------------------------------------------------------|----------------------------------|
| `ground_elem`| Ground planes. Shared across all worlds.                                | `rb.build_ground_planes`         |
| `env_elem`   | Non-articulated bodies (kinematic tables, static mesh colliders, cloth, deformables). **Default fallback.** | `_resolve_contact_elem` returns this when body is in neither articulation nor free-joint set |
| `robo_elem`  | Articulated robot links — any body reachable through a non-free joint.  | Set membership in `articulation_bodies` |
| `actor_elem` | Bodies attached via a **free joint** (dynamic free rigid bodies — cubes, pens, objects to be manipulated). | Set membership in `free_joint_bodies`   |

**Priority for per-body resolution** (`rigid_body.py:205-211`):
1. `body_element_overrides[b]` if provided (highest — see per-body override below)
2. `robo_elem` if `b in articulation_bodies`
3. `actor_elem` if `b in free_joint_bodies`
4. `env_elem` otherwise

## The Default Tabular

Inserted automatically by `SolverUIPC.initialize()` in
`newton/_src/solvers/uipc/solver_uipc.py:388-410`:

```python
# Friction μ = 0.5, stiffness k = 1 GPa for every pair.
# The third positional arg after the two elements is μ, then k, then
# `enable` (True = contact enabled, False = disabled).

contact_tabular.insert(ground_elem, ground_elem, 0.5, 1.0 * GPa, False)

# Per-world (repeated for every Newton world):
contact_tabular.insert(env_elem,    env_elem,    0.5, 1.0 * GPa, False)
contact_tabular.insert(env_elem,    robo_elem,   0.5, 1.0 * GPa, True )
contact_tabular.insert(env_elem,    actor_elem,  0.5, 1.0 * GPa, True )
contact_tabular.insert(ground_elem, env_elem,    0.5, 1.0 * GPa, False)  # ← note: False!
contact_tabular.insert(ground_elem, robo_elem,   0.5, 1.0 * GPa, True )
contact_tabular.insert(ground_elem, actor_elem,  0.5, 1.0 * GPa, True )
contact_tabular.insert(robo_elem,   robo_elem,   0.5, 1.0 * GPa, False)  # ← self-collision off
contact_tabular.insert(robo_elem,   actor_elem,  0.5, 1.0 * GPa, True )
contact_tabular.insert(actor_elem,  actor_elem,  0.5, 1.0 * GPa, True )
```

**Critical defaults to remember**
- **`robo_elem ↔ robo_elem` is OFF** → robot self-collision is disabled by
  default. Turn it on in your callback if you need it.
- **`env_elem ↔ env_elem` is OFF** → two kinematic tables don't push each
  other. (They are usually disjoint anyway.)
- **`ground_elem ↔ env_elem` is OFF** → a kinematic table resting on the
  ground does NOT generate contacts with it. The table is kinematic so it
  won't fall, but if you stack dynamic bodies on it and need ground as a
  backstop, the actor/ground pair (which IS on) will catch them.
- Every pair that involves at least one dynamic category (robo/actor) is
  **ON**.
- Unlisted pairs are treated by UIPC as "not declared" and will not
  generate contacts — always call `insert` for any new element pair.

## `insert` Signature

```python
contact_tabular.insert(
    element_a,   # ContactElement
    element_b,   # ContactElement (order does not matter, symmetric)
    mu,          # float — Coulomb friction coefficient
    kappa,       # float — barrier stiffness in Pa. `GPa = 1e9` is imported
                 #         from `newton._src.solvers.uipc.solver_uipc`.
    enable,      # bool — True to enable contact, False to skip this pair
)
```

Insert the **same** pair again to override the previous entry — later calls
win. This is exactly how the user callback patches defaults.

## The `configure_contact_tabular` Callback

Call **before** `initialize()`. Requires `auto_init=False` when constructing
the solver.

```python
def setup_contacts(tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem):
    # Create a new element for this world.
    gripper_elem = tabular.create(f"gripper_{world_index}")

    # Patch defaults and declare new pairs.
    tabular.insert(gripper_elem, env_elem,    0.8, 1e9, True)
    tabular.insert(gripper_elem, ground_elem, 0.8, 1e9, True)
    tabular.insert(gripper_elem, actor_elem,  0.8, 1e9, True)

    # (Optional) return a dict mapping body_index → element to override
    # the default element assignment of specific bodies.
    return {
        left_finger_body_idx:  gripper_elem,
        right_finger_body_idx: gripper_elem,
    }


solver = newton.solvers.SolverUIPC(model, auto_init=False)
solver.configure_contact_tabular(setup_contacts)
solver.initialize()
```

Key details
- **`world_index`** — the callback is called once per Newton world. For
  multi-world models, `create(f"gripper_{world_index}")` is mandatory:
  elements must be named uniquely across the whole tabular.
- **Return value** — `None` or a `dict[int, ContactElement]`. Returned bodies
  are looked up by their **global** (replicated) body index and consumed by
  `body_element_overrides` in `build_affine_bodies`. Non-returned bodies
  keep the default priority rules.
- **Body indices are global** — if you want to override a body that lives
  in the single-world builder before replication, remember to shift by
  `world_index * bodies_per_world` when computing indices.
- **Calling `configure_contact_tabular` after `initialize()` raises
  `RuntimeError`.**

## Reading the `enable` Flag — Design Rules

When you need to decide whether a pair should be `True` or `False`:

- **Turn OFF** pairs that are physically impossible or would cost cycles:
  - `ground_elem ↔ env_elem` when your environment is static and known to
    not touch the ground (matches solver default).
  - `robo_elem ↔ robo_elem` when self-collision is off (default — URDF
    meshes from visual colliders are often not airtight and would produce
    spurious self-contacts).
- **Turn ON** any pair where either element is a dynamic category (actor
  or robo) and the other element is something the dynamic body can
  actually touch.
- Pairs involving a new custom element default to "not declared" until
  you `insert` them — do not rely on the defaults carrying over.

## Common Patterns

### Higher-friction gripper pads
Assign gripper bodies to a dedicated element with higher μ so fingers grip
harder without affecting everything else.

```python
def setup(tabular, wi, ground, env, robo, actor):
    pad = tabular.create(f"pad_{wi}")
    tabular.insert(pad, actor,  0.9, 1e9, True)   # grip objects
    tabular.insert(pad, env,    0.9, 1e9, True)   # grip table edge
    tabular.insert(pad, ground, 0.9, 1e9, True)
    tabular.insert(pad, robo,   0.5, 1e9, False)  # no self-collision with arm
    return {left_finger_idx + wi * bodies_per_world: pad,
            right_finger_idx + wi * bodies_per_world: pad}
```

### Enabling robot self-collision on a subset of links
```python
def setup(tabular, wi, ground, env, robo, actor):
    # Turn on robo-robo at lower stiffness to avoid jitter.
    tabular.insert(robo, robo, 0.3, 1e8, True)
```

### Two separate "teams" of free bodies that do NOT collide
```python
def setup(tabular, wi, ground, env, robo, actor):
    team_a = tabular.create(f"team_a_{wi}")
    team_b = tabular.create(f"team_b_{wi}")
    for t in (team_a, team_b):
        tabular.insert(t, ground, 0.5, 1e9, True)
        tabular.insert(t, env,    0.5, 1e9, True)
        tabular.insert(t, robo,   0.5, 1e9, True)
        tabular.insert(t, t,      0.5, 1e9, True)
    # team_a ↔ team_b NOT inserted → no contact between teams.
    return { **{b: team_a for b in team_a_bodies},
             **{b: team_b for b in team_b_bodies} }
```

### Disabling a noisy default pair
```python
def setup(tabular, wi, ground, env, robo, actor):
    tabular.insert(env, actor, 0.5, 1e9, False)  # override default (was True)
```

## Debugging Checklist

- **Body not colliding with anything it should** — print its resolved
  element. The most common mistake is forgetting that a body connected by
  a fixed joint to a non-free root lands in `robo_elem`, not `env_elem`.
- **ABD init fails with "Assertion volume > 0"** — that's a mesh-orientation
  bug, not a tabular bug. See the `uipc-pybind-contiguity` skill.
- **IPC sanity check fails with "d_hat violation"** — bodies are initially
  too close. Fix the initial transforms (see the `uipc_gap` pattern in
  `newton/examples/uipc/example_uipc_panda_hydro.py`). Do NOT attempt to
  fix this by disabling the pair in the tabular — disabled pairs still
  trip the sanity check at build time because d_hat is a global scene
  parameter.
- **Silent no-op override** — returning a `body_element_overrides` dict
  whose keys use the *pre-replication* body index when `world_count > 1`.
  Always convert to global indices.
- **New element with no inserts** — `tabular.create("foo")` without any
  `insert` calls silently disables contact for every body you assign to
  `foo`. At least insert `foo ↔ ground` and `foo ↔ env` to get baseline
  behavior.

## File Map

- `newton/_src/solvers/uipc/solver_uipc.py:261-312` —
  `configure_contact_tabular` public API.
- `newton/_src/solvers/uipc/solver_uipc.py:385-421` — the default tabular
  is constructed here. Source of truth for which pairs are on/off.
- `newton/_src/solvers/uipc/rigid_body.py:194-211` —
  `_resolve_contact_elem`: the priority rules for per-body element
  assignment.
- `newton/_src/solvers/uipc/rigid_body.py:228-305` —
  `build_affine_bodies`: where `body_element_overrides` is consumed.
- `newton/examples/uipc/example_uipc_panda_hydro.py` — example where the
  default tabular is sufficient (panda arm is robo, table/cup are env,
  pen/cube is actor, ground is ground).
