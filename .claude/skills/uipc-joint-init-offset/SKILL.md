---
name: uipc-joint-init-offset
description: Use when writing, reading, or debugging UIPC revolute/prismatic joint edge attributes (`angle`, `aim_angle`, `distance`, `aim_distance`, limits) or when confused about why `init_angle` / `init_distance` on UIPC joint edges is stored with a negated sign in `articulation_builder.py`. Explains the coordinate-offset algebra that lets the Newton animator read/write Newton-absolute joint coordinates without any per-step offset and keeps `AffineBodyRevoluteJointLimit` / `AffineBodyPrismaticJointLimit` in Newton space.
---

# UIPC Joint Init Offset — Why the Negative Sign

SolverUIPC stores an `init_angle` (revolute) or `init_distance` (prismatic)
field on every UIPC joint **edge**. These are written with a **negated** copy
of Newton's initial joint coordinate:

```python
# articulation_builder.py:827-828 (revolute)
init_angle_view[:] = -np.array(init_angles, dtype=np.float64)

# articulation_builder.py:978-979 (prismatic)
init_distance_view[:] = -np.array(init_qs, dtype=np.float64)
```

This skill explains **why** the sign is negated so that future edits don't
"fix" it and silently break the animator.

## UIPC Kernel Convention

UIPC measures the joint coordinate as a signed delta from the
**construction-time rest pose** (zero at build). All UIPC kernels that touch
joint edges apply the stored `init_*` as an offset:

```
current_angle_edge = raw_angle  - init_angle
effective_target   = aim_angle  + init_angle
actual_limit       = limit      + init_angle
```

- `raw_angle` is the live geometric angle measured by UIPC (zero at build pose).
- `current_angle_edge` is the value the solver reports to the outside world.
- `aim_angle` / `limit` are the values **we** write into the edge attributes.

The same three formulas apply to `distance` / `aim_distance` for prismatic
joints.

## Newton's Convention

Newton stores joint state in **absolute** coordinates — `init_q_Newton` is
whatever Newton considers the starting joint position, and all downstream
code (animator, limits API, user-facing getters) speaks Newton-absolute.

We want the UIPC edge attributes (`angle`, `aim_angle`, lower/upper limits)
to **also** be readable/writable in Newton-absolute coordinates, so the
animator doesn't have to add/subtract an offset on every step and so the
values passed to `AffineBodyRevoluteJointLimit` / `AffineBodyPrismaticJointLimit`
stay in Newton space.

## Why the Negative Sign — the Algebra

Substitute `init_angle = -init_q_Newton` into the three UIPC kernel formulas:

| UIPC formula | With `init_angle = -init_q_Newton` | Result |
|---|---|---|
| `current_angle_edge = raw_angle - init_angle` | `raw_angle - (-init_q_Newton)` = `raw_angle + init_q_Newton` | == Newton_q ✓ |
| `effective_target = aim_angle + init_angle` | `aim_angle + (-init_q_Newton)` = `aim_angle - init_q_Newton` | when `aim_angle` is Newton-absolute, the raw target lands in raw space ✓ |
| `actual_limit = limit + init_angle` | `limit + (-init_q_Newton)` = `limit - init_q_Newton` | Newton-space limit → raw-space limit ✓ |

The minus sign is the **algebraic inverse of a coordinate-frame offset**,
not an arbitrary choice. UIPC's formulas add the stored offset; to cancel
Newton's absolute zero we must store the negative.

### Intuition in one line

> UIPC starts counting at zero; Newton starts counting at `init_q`. To glue
> the two frames together we store the **difference** — and because UIPC
> adds this difference while we need to subtract it, the stored value is
> `-init_q_Newton`.

## Consequences for Callers

- The animator (`revolute_joint_anim`, `prismatic_joint_anim`) can write
  `aim_angle` / `aim_distance` directly from `Model.joint_target`, with no
  per-step arithmetic.
- Limits passed into `AffineBodyRevoluteJointLimit` /
  `AffineBodyPrismaticJointLimit` must be in **Newton absolute space** —
  do not pre-subtract `init_q` at the call site.
- Reading `current_angle_edge` back from UIPC gives you a Newton-absolute
  value directly.

## Red Flags — Do Not "Fix"

If you see code that looks like:

```python
init_angle_view[:] = -np.array(init_angles, ...)   # looks weird, "is this a bug?"
```

it is **not** a bug. Flipping it to the positive sign will:

1. Double-offset the limits (the UIPC limit kernel will read
   `limit + init_q_Newton` instead of cancelling `init_q_Newton`).
2. Cause the animator to drive the joint to `aim + 2*init_q` in raw space.
3. Make `current_angle_edge` report `raw - init_q_Newton`, which is neither
   Newton space nor UIPC raw space.

Before touching either `init_angle_view[:] = ...` or `init_distance_view[:] = ...`,
re-derive the three kernel formulas above and verify the substitution
still collapses to Newton-absolute coordinates.

## Relevant Code

- `newton/_src/solvers/uipc/articulation_builder.py:776-828` — revolute
  joint builder, `init_angle` negation and commentary.
- `newton/_src/solvers/uipc/articulation_builder.py:964-979` — prismatic
  joint builder, identical derivation for `init_distance`.
- The animator methods `revolute_joint_anim` / `prismatic_joint_anim`
  rely on the negated convention to avoid per-step offset math.
