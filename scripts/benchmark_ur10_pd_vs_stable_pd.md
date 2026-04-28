# UR10 Stability Benchmark — `ControllerPD` vs `ControllerStablePD`

**Solver:** `newton.solvers.SolverMuJoCo`
**Robot:** UR10 (6 revolute DOFs), pedestaled with a fixed base
**Asset:** `universal_robots_ur10/usd/ur10_instanceable.usda`
**Script:** [`scripts/benchmark_ur10_pd_vs_stable_pd.py`](./benchmark_ur10_pd_vs_stable_pd.py)

## Setup

- **Initial pose:** `HOME_POSE = [0, -π/3, π/2, -π/6, π/2, 0]` (rad)
- **Step setpoint:** `HOME_POSE + [0.30, 0.20, -0.25, 0.40, -0.30, 0.50]` (rad), held constant
- **Joint mode:** every DOF in `JointTargetMode.EFFORT`, `joint_armature = 1e-2`
- **PD gains:** `Kp = [300, 300, 200, 100, 60, 30]`, `Kd = [40, 40, 30, 15, 10, 5]`
- **Effort clamp:** `[330, 330, 150, 54, 54, 54]` N·m (UR10 nominal limits)
- **Stable-PD bias inputs (each substep):** `M = newton.eval_mass_matrix`,
  `bias_forces = -tau_g + tau_c` with `tau_g` from Jacobian-transpose gravity
  and `tau_c` from finite-difference Christoffel Coriolis (same recipe as
  `examples/uipc/example_uipc_ur10_force.py`).
- **Run length per cell:** 3.0 s simulated
- **Steady-state window:** last 0.5 s (RMS over joints × steps)
- **Divergence criterion:** any non-finite value, or `|q| > 20 rad`, or
  `max|qd| > 1000 rad/s`
- **Hardware:** NVIDIA RTX 5090 (sm_120), Warp 1.13.0rc1, CUDA 12.9

## Results

| dt (ms) | rate (Hz) | PD diverge | PD rms_err (rad) | PD ss_err (rad) | PD max\|qd\| (rad/s) | SPD diverge | SPD rms_err (rad) | SPD ss_err (rad) | SPD max\|qd\| (rad/s) |
|--------:|----------:|:----------:|-----------------:|----------------:|---------------------:|:-----------:|------------------:|-----------------:|----------------------:|
|   2.083 |       480 | no         |           0.1205 |          0.1089 |                 3.50 | no          |            0.1251 |           0.1120 |                  2.51 |
|   4.167 |       240 | no         |           0.2001 |          0.2165 |                11.66 | no          |            0.1292 |           0.1152 |                  2.32 |
|   8.333 |       120 | no         |           0.8966 |          0.9858 |                26.70 | no          |            0.1373 |           0.1218 |                  2.03 |
|  16.667 |        60 | no         |           1.5453 |          1.5572 |                69.80 | no          |            0.1533 |           0.1360 |                  1.70 |
|  33.333 |        30 | **yes @ step 8** |          inf |             inf |               699.10 | no          |            0.1863 |           0.1717 |                  1.80 |
|  66.667 |        15 | **yes @ step 1** |          inf |             inf |               111.88 | no          |            0.2665 |           0.2663 |                  1.99 |

### Plain-language read

- `ControllerPD` becomes numerically unstable at `dt ≥ 33 ms` — at 30 Hz the
  arm blows up at step 8 (`max|qd| = 699 rad/s`); at 15 Hz it diverges on
  the first step.
- `ControllerStablePD` is bounded across the whole sweep: steady-state error
  drifts only mildly with `dt` (0.11 rad → 0.27 rad) and the peak joint
  velocity stays below 2.6 rad/s. There is no observable instability up to
  `dt = 66.7 ms`.
- At fine `dt` (480 Hz) the two controllers are within 0.005 rad of each
  other — Stable-PD's advantage shows up only once plain PD starts paying
  for the explicit error projection across an integration step.
- Plain-PD's `max|qd|` curve (`3.5 → 11.7 → 26.7 → 69.8 rad/s` over
  `2.1 → 16.7 ms`) is the classic high-gain implicit-integration ringing
  signature; Stable-PD's `max|qd|` actually *decreases* with `dt`
  (`2.51 → 1.70 rad/s`) because the implicit `(M + diag(Kd)·dt)·qddot`
  solve damps the projected error in proportion to `dt`.

## Reproducing

```bash
uv run python scripts/benchmark_ur10_pd_vs_stable_pd.py

# Custom sweep
uv run python scripts/benchmark_ur10_pd_vs_stable_pd.py \
    --duration 5.0 \
    --dts 0.002 0.005 0.01 0.02 0.04

# Single controller (e.g. only Stable-PD)
uv run python scripts/benchmark_ur10_pd_vs_stable_pd.py --controllers stable_pd
```

## Notes

- The Stable-PD path requires `M(q)` and `C(q,q̇)` populated into
  `act_state.controller_state` each substep. The script uses the same
  `newton.eval_mass_matrix` / `newton.eval_jacobian` + finite-difference
  Christoffel Coriolis pipeline as `example_uipc_ur10_force.py`, so the
  results are directly comparable to that example's `--stable-pd` mode.
- Plain-PD divergence at coarse `dt` is a property of the controller, not
  the solver. MuJoCo's own constraint solver is stable at these `dt`s
  (Stable-PD finishes cleanly under the same solver), so the explosion is
  the high-gain PD law overshooting between integration steps.
- Effort saturation (`max_torque` clamp) is enforced for both controllers
  via the same `ClampingMaxEffort` post-layer; it does not by itself
  prevent divergence under plain PD because the clamp acts on the
  *commanded* torque, not on the integrated joint state.
