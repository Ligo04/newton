---
name: uipc-subscene-tabular
description: Use when configuring multi-world contact isolation in SolverUIPC — subscene creation, cross-world contact enable/disable, subscene-contact element relationship, and the `configure_subscene_tabular` callback. Covers the default isolation behavior, the C++ semantics (cross-subscene disabled by default), and common patterns for enabling selective cross-world interaction.
---

# UIPC Subscene Tabular

SolverUIPC uses UIPC's `SubsceneTabular` to isolate contact between Newton
worlds in multi-world simulations. This skill covers the C++ semantics, the
Newton wrapper, and how subscenes relate to contact elements.

## Key Rules

- `(subscene, subscene)` self-interaction → **enabled by default**
- `(subscene_a, subscene_b)` cross-interaction → **disabled by default**
- `(default_element, subscene)` → **must be explicitly enabled** via `insert`
- `insert` with the same pair overwrites (upsert), same as `contact_tabular`

You do NOT need `insert(subscene_a, subscene_b, false)` to disable
cross-world contact — it's already off. Only call `insert` to **enable**
cross-subscene interaction.

## Newton's Default Setup

In `SolverUIPC.initialize()` (`solver_uipc.py`), for every model:

```python
tabular = self.scene.subscene_tabular()
default_subscene_elem = tabular.default_element()

for world_index in range(model.world_count):
    se = tabular.create(f"world_{world_index}")
    subscene_elements.append(se)

# Only enable each world ↔ default (ground).
# Cross-subscene is disabled by default in UIPC — no explicit False needed.
for i in range(model.world_count):
    tabular.insert(default_subscene_elem, subscene_elements[i], True)
```

**What goes into each subscene:**
- `default_element` — ground planes (ground belongs to the default subscene element)
- `world_{i}` — all bodies, joints, cloth, deformables belonging to world `i`

Assignment happens via `subscene_elem.apply_to(sc)` in each builder
(`rigid_body.py`, `articulation_builder.py`, `cloth.py`, `deformable_body.py`).

## Subscene ↔ Contact Element Relationship

These are **orthogonal but complementary** systems:

| System | Controls | Granularity |
|--------|----------|-------------|
| `contact_tabular` | Friction, stiffness, enable per element pair | Per body category (env/robo/actor/custom) |
| `subscene_tabular` | Whether two groups can interact at all | Per Newton world |

A contact pair must pass **both** gates:
1. The subscene tabular must allow interaction between the two worlds
2. The contact tabular must enable the specific element pair

**Example:** Two robots in world 0 and world 1 both use `robo_elem`. Even if
`robo ↔ robo` is enabled in the contact tabular, they won't interact unless
`subscene_tabular.insert(world_0, world_1, True)` is also called.

## `configure_subscene_tabular` Callback

Call **before** `initialize()`.

```python
def setup_subscenes(tabular, world_subscenes, default_elem):
    # Enable contact between world 0 and world 1
    tabular.insert(world_subscenes[0], world_subscenes[1], True)


solver = newton.solvers.SolverUIPC(model)
solver.configure_subscene_tabular(setup_subscenes)
solver.initialize()
```

**Callback signature:**
```python
fn(tabular: SubsceneTabular,
   world_subscenes: list[SubsceneElement],
   default_element: SubsceneElement) -> None
```

## Common Patterns

### Enable all worlds to interact (global arena)
```python
def setup(tabular, worlds, default_elem):
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            tabular.insert(worlds[i], worlds[j], True)
```

### Pairwise neighbor interaction only
```python
def setup(tabular, worlds, default_elem):
    for i in range(len(worlds) - 1):
        tabular.insert(worlds[i], worlds[i + 1], True)
```

### Single world (world_count == 1)
A `world_0` subscene is created and enabled against the default ground element.
The `configure_subscene_tabular` callback is invoked with a one-element
`world_subscenes` list when configured.

## Debugging Checklist

- **Bodies in different worlds not colliding** — check that
  `subscene_tabular.insert(world_i, world_j, True)` is called. Cross-subscene
  is disabled by default.
- **Bodies in the same world not colliding** — this is NOT a subscene issue.
  Check the `contact_tabular` (see `uipc-contact-tabular` skill).
- **Ground not interacting with a world** — verify
  `tabular.insert(default_subscene_elem, world_subscene, True)` is present.
  The solver does this automatically, but a user callback might override it.
- **`configure_subscene_tabular` callback not firing** — verify it was
  configured before `initialize()`.

## Execution Order in `initialize()`

```
1. subscene_tabular setup
   └─ create subscene elements
   └─ insert default ↔ world_i (True)
   └─ user callback (configure_subscene_tabular)
2. contact_tabular setup
   └─ create ground/env/robo/actor elements
   └─ insert default pairs
   └─ user callback (configure_contact_tabular)
3. Build geometries (assign subscene + contact elements via apply_to)
4. world.init(scene)
```

## File Map

- `solver_uipc.py` — subscene tabular creation and default setup
- `solver_uipc.py` — `configure_subscene_tabular` public API and initialization
- `rigid_body.py` — `subscene_elem.apply_to(sc)` for rigid bodies
- `articulation_builder.py` — subscene element for joint anchors
- `cloth.py` — `subscene_elem.apply_to(sc)` for cloth
- `deformable_body.py` — `subscene_elem.apply_to(sc)` for deformables
