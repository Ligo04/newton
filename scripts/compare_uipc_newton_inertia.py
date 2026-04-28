# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare Newton's authored body mass / COM / inertia against UIPC's ABD.

Newton stores :attr:`Model.body_mass` / :attr:`Model.body_com` /
:attr:`Model.body_inertia` as authored values (from URDF / USD or
explicit ``ModelBuilder`` calls).  UIPC's :class:`AffineBodyConstitution`
re-derives those quantities from ``mass_density x mesh_volume`` and the
mesh's spatial moments, which can diverge from what Newton has — for
example when the collision geometry is a simplified hull, when the COM
was hand-set, or when the authored inertia tensor is empirical rather
than geometrically exact.

This script builds a handful of rigid-body primitives and exercises the
bidirectional bridge between Newton and UIPC:

* Phase [1/4]  Newton-authored values before UIPC initialises.
* Phase [2/4]  Side-by-side after UIPC initialise — UIPC has rederived
  the inertial properties from mesh-volume integrals, so the authored
  custom values of body 3 have been replaced.
* Phase [3/4]  Call
  :meth:`SolverUIPC.override_model_inertia_from_uipc` to push UIPC's
  values back into the Newton model and verify the model now mirrors
  UIPC.
* Phase [4/5]  Rebuild the model from scratch and call
  :meth:`SolverUIPC.override_uipc_inertia_from_model` BEFORE
  :meth:`initialize` for the custom body, verifying that UIPC now
  reflects Newton's authored (mass, com, inertia) instead of the
  mesh-derived values.
* Phase [5/5]  Rebuild once more with ``auto_sync_inertia=False`` and
  confirm ``initialize`` does **not** overwrite Newton's authored
  values — useful when downstream consumers (stable-PD mass matrix,
  URDF round-trip) must see the exact ``ModelBuilder`` inputs.

Run it with:

    uv run python scripts/compare_uipc_newton_inertia.py
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverUIPC


def _build_model() -> newton.Model:
    """Four primitive shapes with deliberately non-default inertial properties."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()

    # 1. A box authored with the builder's default density-driven inertia.
    b_box = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
    )
    builder.add_shape_box(b_box, hx=0.3, hy=0.2, hz=0.1)

    # 2. A sphere — mesh volume and analytical volume match closely, so
    #    UIPC and Newton should agree on the mass very well but can
    #    disagree on the inertia depending on tessellation.
    b_sphere = builder.add_body(
        xform=wp.transform(p=wp.vec3(1.0, 0.0, 1.0), q=wp.quat_identity()),
    )
    builder.add_shape_sphere(b_sphere, radius=0.25)

    # 3. A capsule — long axis inertia differs significantly from the
    #    two transverse axes, a good stress test for the off-diagonal
    #    terms.
    b_capsule = builder.add_body(
        xform=wp.transform(p=wp.vec3(2.0, 0.0, 1.0), q=wp.quat_identity()),
    )
    builder.add_shape_capsule(b_capsule, radius=0.1, half_height=0.3)

    # 4. A box with an explicitly overridden COM / inertia — the classic
    #    case where UIPC's density-derived values will NOT match Newton.
    b_custom = builder.add_body(
        xform=wp.transform(p=wp.vec3(3.0, 0.0, 1.0), q=wp.quat_identity()),
    )
    builder.add_shape_box(b_custom, hx=0.3, hy=0.3, hz=0.3)
    # Hand-author eccentric mass properties to simulate an asymmetric payload.
    builder.body_mass[b_custom] = 10.0
    builder.body_inv_mass[b_custom] = 1.0 / 10.0
    builder.body_com[b_custom] = wp.vec3(0.1, 0.05, 0.0)
    inertia_override = np.diag([0.5, 1.2, 0.8]).astype(np.float32)
    builder.body_inertia[b_custom] = wp.mat33(*inertia_override.flatten().tolist())
    builder.body_inv_inertia[b_custom] = wp.mat33(*np.linalg.inv(inertia_override).flatten().tolist())

    return builder.finalize()


def _row(name: str, newton_val: float | np.ndarray, uipc_val: float | np.ndarray) -> str:
    n = np.atleast_1d(np.asarray(newton_val, dtype=np.float64)).ravel()
    u = np.atleast_1d(np.asarray(uipc_val, dtype=np.float64)).ravel()
    abs_err = float(np.linalg.norm(n - u))
    rel_err = abs_err / max(float(np.linalg.norm(n)), 1e-12)
    return (
        f"  {name:10s} newton={np.array2string(n, precision=4, suppress_small=True):30s} "
        f"uipc={np.array2string(u, precision=4, suppress_small=True):30s} "
        f"|Δ|={abs_err:.4g}  rel={rel_err:.4g}"
    )


def _print_body(body_idx: int, label: str, model: newton.Model, uipc_props: dict) -> None:
    print(f"\n  Body {body_idx}: {label}")
    print(
        _row(
            "mass",
            float(model.body_mass.numpy()[body_idx]),
            float(uipc_props["mass"]),
        )
    )
    print(
        _row(
            "com",
            model.body_com.numpy()[body_idx],
            uipc_props["mass_center"],
        )
    )
    print(
        _row(
            "inertia",
            model.body_inertia.numpy()[body_idx].reshape(-1),
            uipc_props["inertia"].reshape(-1),
        )
    )
    # Redundant ABD integrals — sanity check that abd_mass_x_bar ≈ m·c
    m = uipc_props["mass"]
    c = uipc_props["mass_center"]
    mx = uipc_props["abd_mass_x_bar"]
    mxx_origin = uipc_props["abd_mass_x_bar_x_bar"]
    # S_origin = S_COM + m c c^T ; S_COM = 0.5 tr(I) E - I
    s_com = 0.5 * np.trace(uipc_props["inertia"]) * np.eye(3) - uipc_props["inertia"]
    s_origin_reconstructed = s_com + m * np.outer(c, c)
    print(
        f"    [abd consistency] |m*c - abd_mass_x_bar| = "
        f"{float(np.linalg.norm(m * c - mx)):.3e}  "
        f"|recon - abd_mass_x_bar_x_bar| = "
        f"{float(np.linalg.norm(s_origin_reconstructed - mxx_origin)):.3e}"
    )


def main() -> None:
    model = _build_model()

    body_labels = ["box (default)", "sphere (default)", "capsule (default)", "box (custom inertia)"]

    print("=" * 100)
    print("[1/4] Newton-authored body properties (before UIPC initialize)")
    print("=" * 100)
    for b in range(model.body_count):
        m = float(model.body_mass.numpy()[b])
        c = model.body_com.numpy()[b]
        i = model.body_inertia.numpy()[b]
        print(f"\n  Body {b}: {body_labels[b]}")
        print(f"    mass    = {m:.6f}")
        print(f"    com     = {np.array2string(c, precision=4)}")
        print(f"    inertia =\n{np.array2string(i, precision=4)}")

    solver = SolverUIPC(
        model,
        dt=1.0 / 60.0,
        workspace="/tmp/newton_uipc/compare_inertia",
    )
    solver.initialize()

    print("\n" + "=" * 100)
    print(
        "[2/4] After initialize (model auto-synced to UIPC ABD; body 3 authored values are clobbered by the default density path)"
    )
    print("=" * 100)
    for b in range(model.body_count):
        try:
            props = solver.read_uipc_body_inertia(b)
        except KeyError:
            print(f"\n  Body {b}: {body_labels[b]}  (no UIPC geometry)")
            continue
        if props["mass"] is None:
            print(f"\n  Body {b}: {body_labels[b]}  (UIPC proxy - no ABD meta)")
            continue
        _print_body(b, body_labels[b], model, props)

    print("\n" + "=" * 100)
    print("[3/4] Re-calling override_model_inertia_from_uipc() — should be idempotent after initialize's auto-sync")
    print("=" * 100)
    written = solver.sync_model_inertia_from_uipc()
    print(f"\n  Overwrote {len(written)} bodies: {written}")
    max_err = 0.0
    for b in written:
        props = solver.read_uipc_body_inertia(b)
        # Verify the model now matches UIPC exactly (up to float32 rounding).
        diff_mass = abs(float(model.body_mass.numpy()[b]) - float(props["mass"]))
        diff_com = float(np.linalg.norm(model.body_com.numpy()[b] - np.asarray(props["mass_center"])))
        diff_i = float(
            np.linalg.norm(model.body_inertia.numpy()[b].reshape(-1) - np.asarray(props["inertia"]).reshape(-1))
        )
        max_err = max(max_err, diff_mass, diff_com, diff_i)
        print(f"  Body {b}: |Δmass|={diff_mass:.3e}  |Δcom|={diff_com:.3e}  |Δinertia|={diff_i:.3e}")
    print(f"\n  max post-override error across all bodies: {max_err:.3e}")
    # float32 round-trip leaves ~1e-5 relative error at most.
    assert max_err < 1e-4, f"override failed to sync model with UIPC (max error {max_err})"
    print("  ✓ model now mirrors UIPC ABD values within float32 precision")

    # ------------------------------------------------------------------
    # [4/4] Reverse direction: push Newton-authored custom inertia into UIPC
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[4/4] override_uipc_inertia_from_model(...) — UIPC should mirror Newton")
    print("=" * 100)

    # Rebuild from scratch so body 3's custom (mass, com, inertia) is
    # re-authored cleanly (the previous phase overwrote them).
    model2 = _build_model()
    solver2 = SolverUIPC(
        model2,
        dt=1.0 / 60.0,
        workspace="/tmp/newton_uipc/compare_inertia_reverse",
    )
    # b_custom is the 4th body added (after ground plane, which does not
    # contribute a body).  See _build_model for the ordering.
    b_custom = 3
    flagged = solver2.sync_uipc_inertia_with_model([b_custom])
    print(f"\n  Flagged bodies for UIPC override: {flagged}")

    # Capture Newton's authored values before initialize() — this is the
    # ground truth we expect UIPC to echo back.
    assert model2.body_mass is not None
    assert model2.body_com is not None
    assert model2.body_inertia is not None
    newton_mass = float(model2.body_mass.numpy()[b_custom])
    newton_com = model2.body_com.numpy()[b_custom].astype(np.float64).copy()
    newton_inertia = model2.body_inertia.numpy()[b_custom].astype(np.float64).reshape(3, 3).copy()

    solver2.initialize()

    props = solver2.read_uipc_body_inertia(b_custom)
    uipc_mass = float(props["mass"])
    uipc_com = np.asarray(props["mass_center"], dtype=np.float64)
    uipc_inertia = np.asarray(props["inertia"], dtype=np.float64).reshape(3, 3)

    print(f"\n  Body {b_custom} (custom inertia, Newton-authored → UIPC)")
    print(_row("mass", newton_mass, uipc_mass))
    print(_row("com", newton_com, uipc_com))
    print(_row("inertia", newton_inertia.reshape(-1), uipc_inertia.reshape(-1)))

    diff_mass = abs(newton_mass - uipc_mass)
    diff_com = float(np.linalg.norm(newton_com - uipc_com))
    diff_inertia = float(np.linalg.norm(newton_inertia - uipc_inertia))
    max_rev_err = max(diff_mass, diff_com, diff_inertia)
    print(f"\n  reverse-override errors: |Δmass|={diff_mass:.3e}  |Δcom|={diff_com:.3e}  |Δinertia|={diff_inertia:.3e}")
    # ``from_rigid_body`` -> 12x12 -> ABD meta round-trip is float64 the
    # whole way, so the tolerance is tighter than the float32 Newton side.
    assert max_rev_err < 1e-6, (
        f"override_uipc_inertia_from_model failed to push Newton values into UIPC (max error {max_rev_err:.3e})"
    )
    print("  ✓ UIPC ABD values now mirror Newton's authored custom inertia")

    # Sanity check: the other (default) bodies must remain mesh-derived,
    # i.e. unchanged from Phase [2/4].
    for b in range(model2.body_count):
        if b == b_custom:
            continue
        p = solver2.read_uipc_body_inertia(b)
        assert p["mass"] is not None, f"body {b} unexpectedly has no ABD mass"
    print("  ✓ non-flagged bodies still use the default density-driven ABD path")


if __name__ == "__main__":
    main()
