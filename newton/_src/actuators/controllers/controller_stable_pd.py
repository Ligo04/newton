# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .base import Controller

# ``...solvers.kamino._src.linalg.factorize`` is intentionally imported lazily
# inside methods (not at module import time). Top-level import here would
# trigger ``newton._src.solvers.__init__`` which eagerly loads Featherstone,
# which in turn imports ``newton._src.sim`` — and at the moment this file
# is first imported, ``sim/__init__.py`` is still mid-initialization, so the
# eager top-level variant hits a circular import.


def _next_block_multiple(n: int, block_size: int) -> int:
    """Round ``n`` up to the nearest multiple of ``block_size``."""
    return ((n + block_size - 1) // block_size) * block_size


def _alloc_identity_padded_2d(n: int, n_padded: int, device: wp.Device) -> wp.array:
    """Allocate an ``(n_padded, n_padded)`` float32 array with identity padding.

    The real block ``[:n, :n]`` is zero — callers are expected to overwrite it
    each step (e.g. via :func:`_stable_pd_build_A_kernel`). The phantom block
    ``[n:, n:]`` is initialised to the identity so the Cholesky factor of the
    composite matrix is ``diag(L_real, I)``, which keeps the forward/backward
    substitution on the padded rows equal to ``b[n:]`` — and since
    :func:`_stable_pd_build_rhs_kernel` only writes ``b[:n]`` with
    ``b[n:]`` kept at zero by construction, ``qddot[:n]`` is the exact
    solution of the real ``[:n, :n]`` system.
    """
    buf = np.zeros((n_padded, n_padded), dtype=np.float32)
    if n_padded > n:
        idx = np.arange(n, n_padded)
        buf[idx, idx] = 1.0
    return wp.array(buf, dtype=wp.float32, device=device)


@wp.kernel
def _stable_pd_build_rhs_kernel(
    current_pos: wp.array[float],
    current_vel: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    kd: wp.array[float],
    bias_forces: wp.array[float],
    dt: float,
    b: wp.array[float],
):
    """b[i] = kp*(target_pos - q - q̇·dt) + kd*(target_vel - q̇) - bias_forces[i]."""
    i = wp.tid()
    pos_idx = pos_indices[i]
    vel_idx = vel_indices[i]
    tgt_pos_idx = target_pos_indices[i]
    tgt_vel_idx = target_vel_indices[i]

    q_i = current_pos[pos_idx]
    qd_i = current_vel[vel_idx]
    qt_i = target_pos[tgt_pos_idx]
    qdt_i = target_vel[tgt_vel_idx]

    p_term = kp[i] * (qt_i - q_i - qd_i * dt)
    d_term = kd[i] * (qdt_i - qd_i)

    c_i = float(0.0)
    if bias_forces:
        c_i = bias_forces[i]

    b[i] = p_term + d_term - c_i


@wp.kernel
def _stable_pd_build_A_kernel(
    mass_matrix: wp.array2d[float],
    kd: wp.array[float],
    dt: float,
    A: wp.array2d[float],
):
    """A[i, j] = M[i, j] + (i == j) * kd[i] * dt over the real [:n, :n] block.

    The phantom region ``[n:, n:]`` of ``A`` is left untouched here — the
    kamino blocked Cholesky kernel applies identity padding to out-of-range
    tiles on the fly (see ``llt_blocked.py``), so callers only need to
    ensure the physical allocation covers ``(n_padded, n_padded)``.
    """
    i, j = wp.tid()
    a_ij = mass_matrix[i, j]
    if i == j:
        a_ij = a_ij + kd[i] * dt
    A[i, j] = a_ij


@wp.kernel
def _stable_pd_effort_kernel(
    current_pos: wp.array[float],
    current_vel: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    feedforward: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    kp: wp.array[float],
    kd: wp.array[float],
    const_effort: wp.array[float],
    qddot: wp.array[float],
    dt: float,
    efforts: wp.array[float],
):
    """effort = const_effort + feedforward + kp·err_pos + kd·err_vel - kd·qddot·dt."""
    i = wp.tid()
    pos_idx = pos_indices[i]
    vel_idx = vel_indices[i]
    tgt_pos_idx = target_pos_indices[i]
    tgt_vel_idx = target_vel_indices[i]

    q_i = current_pos[pos_idx]
    qd_i = current_vel[vel_idx]
    qt_i = target_pos[tgt_pos_idx]
    qdt_i = target_vel[tgt_vel_idx]

    p_term = kp[i] * (qt_i - q_i - qd_i * dt)
    d_term = kd[i] * (qdt_i - qd_i)

    const_e = float(0.0)
    if const_effort:
        const_e = const_effort[i]

    ff = float(0.0)
    if feedforward:
        ff = feedforward[tgt_vel_idx]

    efforts[i] = const_e + ff + p_term + d_term - kd[i] * qddot[i] * dt


class ControllerStablePD(Controller):
    """Stateful stable-PD controller (Tan 2011) with implicit mass-matrix solve.

    Implements the Tan, Liu & Turk 2011 "stable proportional-derivative"
    control law::

        p_term = kp · (target_pos - q - q̇·dt)
        d_term = kd · (target_vel - q̇)
        (M + diag(kd)·dt) · q̈ = p_term + d_term - C
        effort = const_effort + feedforward + p_term + d_term - kd·q̈·dt

    The ``q̈`` unknown is solved implicitly on the GPU via a blocked
    Cholesky factorization of ``A = M + diag(kd)·dt`` (reusing the kamino
    blocked LLT kernels). The entire step — rhs assembly, A-matrix
    assembly, factorize, solve, and per-actuator effort composition — is a
    sequence of ``wp.launch`` / ``wp.launch_tiled`` calls with no host↔device
    transfers, making the pipeline CUDA-graph capturable.

    The caller is responsible for populating ``State.mass_matrix`` and
    ``State.bias_forces`` before each ``compute()`` call from the physics
    engine (:mod:`newton.sim`, MuJoCo, PhysX, UIPC, ...). This class makes
    no assumption about the source of those quantities.

    Unlike the historical ``ActuatorStablePD`` in ``newton_actuators``,
    this class delegates effort clamping to the downstream
    :class:`~newton.actuators.Clamping` layer and does **not** accumulate
    a separate ``control_input`` channel inside the kernel, so the effort
    law is layer-consistent with :class:`ControllerPD` and
    :class:`ControllerPID`.

    Reference:
        Tan, J., Liu, K., & Turk, G. (2011). "Stable proportional-derivative
        controllers." IEEE Computer Graphics and Applications, 31(4):34-44.
        DOI: 10.1109/MCG.2011.30
    """

    DEFAULT_BLOCK_SIZE: int = 32
    """Tile edge length for the blocked Cholesky (compile-time constant baked into the kernels)."""

    DEFAULT_TILE_BLOCK_DIM: int = 128
    """Launch-time thread-block size for the tile kernels."""

    @dataclass
    class State(Controller.State):
        """Per-step inputs and scratch buffers for the Tan 2011 solve.

        User-populated each step (before :meth:`ControllerStablePD.compute`):
            mass_matrix: Effective mass/inertia ``M(q)`` for this actuator
                subsystem, shape ``(N, N)``, dtype ``float32``.
            bias_forces: Coriolis + gravity + external force compensation
                ``C(q, q̇)``, shape ``(N,)``, dtype ``float32``.

        Internal scratch (allocated by :meth:`ControllerStablePD.state`):
            A: Augmented mass matrix ``M + diag(kd)·dt``, shape
                ``(N_padded, N_padded)``, dtype ``float32``. Only the real
                ``[:N, :N]`` block is written each step; the phantom region
                is left as zeros and identity-padded on the fly inside the
                kamino LLT kernel.
            L: Cholesky factor of ``A`` in the same layout.
            b: Rhs ``p_term + d_term - C``, shape ``(N_padded,)``; only
                ``b[:N]`` is written.
            y: Forward-substitution intermediate ``L · y = b``, shape
                ``(N_padded,)``.
            qddot: Solution of ``Lᵀ · qddot = y`` (implicit acceleration),
                shape ``(N_padded,)``; only ``qddot[:N]`` is read.
        """

        mass_matrix: wp.array2d[float] | None = None
        bias_forces: wp.array[float] | None = None
        A: wp.array2d[float] | None = None
        L: wp.array2d[float] | None = None
        b: wp.array[float] | None = None
        y: wp.array[float] | None = None
        qddot: wp.array[float] | None = None

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Zero all buffers in-place. ``mask`` is ignored (scratch is per-actuator-batch)."""
            for arr in (self.mass_matrix, self.bias_forces, self.A, self.L, self.b, self.y, self.qddot):
                if arr is not None:
                    arr.zero_()

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        kp = args.get("kp", 0.0)
        if kp < 0:
            raise ValueError(f"kp must be non-negative, got {kp}")
        kd = args.get("kd", 0.0)
        if kd < 0:
            raise ValueError(f"kd must be non-negative, got {kd}")
        return {"kp": kp, "kd": kd, "const_effort": args.get("const_effort", 0.0)}

    def __init__(
        self,
        kp: wp.array[float],
        kd: wp.array[float],
        const_effort: wp.array[float] | None = None,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        tile_block_dim: int = DEFAULT_TILE_BLOCK_DIM,
    ):
        """Initialize stable-PD controller.

        Args:
            kp: Proportional gains [N/m or N·m/rad]. Shape ``(N,)``.
            kd: Derivative gains [N·s/m or N·m·s/rad]. Shape ``(N,)``.
            const_effort: Constant bias effort [N or N·m]. Shape ``(N,)``. ``None`` to skip.
            block_size: Tile edge length for the blocked Cholesky. Default 32.
            tile_block_dim: CUDA thread-block size for the tile kernels. Default 128.
        """
        if kp.shape != kd.shape:
            raise ValueError(f"kp shape {kp.shape} must match kd shape {kd.shape}")
        if const_effort is not None and const_effort.shape != kp.shape:
            raise ValueError(f"const_effort shape {const_effort.shape} must match kp shape {kp.shape}")
        self.kp = kp
        self.kd = kd
        self.const_effort = const_effort
        self._block_size = int(block_size)
        self._tile_block_dim = int(tile_block_dim)
        # Lazy import: top-level would trigger newton._src.solvers.__init__ →
        # Featherstone → sim circular dependency at module-load time. By the
        # time a ControllerStablePD is actually instantiated the newton package
        # is fully initialised, so the import is safe here.
        from ...solvers.kamino._src.linalg.factorize import (  # noqa: PLC0415
            llt_blocked_factorize,
            llt_blocked_solve,
            make_llt_blocked_factorize_kernel,
            make_llt_blocked_solve_kernel,
        )

        # @cache-memoized on block_size — repeat instances share compiled kernels.
        self._factorize_kernel = make_llt_blocked_factorize_kernel(self._block_size)
        self._solve_kernel = make_llt_blocked_solve_kernel(self._block_size)
        # Stash launchers as instance attributes so compute() avoids a second import.
        self._launch_factorize = llt_blocked_factorize
        self._launch_solve = llt_blocked_solve
        # Populated in finalize() once the device is known.
        self._n: int = 0
        self._n_padded: int = 0
        self._dim: wp.array[wp.int32] | None = None
        self._mio: wp.array[wp.int32] | None = None
        self._vio: wp.array[wp.int32] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._n = int(num_actuators)
        self._n_padded = _next_block_multiple(self._n, self._block_size)
        # Single-matrix batch layout for the kamino LLT kernels. We declare the
        # matrix as the full padded size so kamino interprets the flat view
        # with the correct row stride (``n_padded`` floats per row). The
        # phantom ``[n:, n:]`` region is identity-padded by :meth:`state`, so
        # the composite system still resolves to the real ``[:n, :n]`` solve.
        self._dim = wp.array([self._n_padded], dtype=wp.int32, device=device)
        self._mio = wp.array([0], dtype=wp.int32, device=device)
        self._vio = wp.array([0], dtype=wp.int32, device=device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> ControllerStablePD.State:
        n = int(num_actuators)
        n_padded = _next_block_multiple(n, self._block_size)
        return ControllerStablePD.State(
            mass_matrix=wp.zeros((n, n), dtype=wp.float32, device=device),
            bias_forces=wp.zeros(n, dtype=wp.float32, device=device),
            A=_alloc_identity_padded_2d(n, n_padded, device=device),
            L=_alloc_identity_padded_2d(n, n_padded, device=device),
            b=wp.zeros(n_padded, dtype=wp.float32, device=device),
            y=wp.zeros(n_padded, dtype=wp.float32, device=device),
            qddot=wp.zeros(n_padded, dtype=wp.float32, device=device),
        )

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerStablePD.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        if state is None:
            raise ValueError("ControllerStablePD requires a State; got None.")
        if state.mass_matrix is None:
            raise ValueError("ControllerStablePD.State.mass_matrix must be populated each step.")
        if state.bias_forces is None:
            raise ValueError("ControllerStablePD.State.bias_forces must be populated each step.")
        if state.A is None or state.L is None or state.b is None or state.y is None or state.qddot is None:
            raise ValueError(
                "ControllerStablePD.State is missing scratch buffers (A, L, b, y, qddot). "
                "Allocate via ControllerStablePD.state() rather than constructing State() directly."
            )

        n = self._n

        # Step 1: b[:n] = p_term + d_term - C (per-actuator, parallel).
        wp.launch(
            kernel=_stable_pd_build_rhs_kernel,
            dim=n,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self.kp,
                self.kd,
                state.bias_forces,
                dt,
            ],
            outputs=[state.b],
            device=device,
        )

        # Step 2: A[:n, :n] = M + diag(kd)·dt (per-entry, parallel).
        wp.launch(
            kernel=_stable_pd_build_A_kernel,
            dim=(n, n),
            inputs=[state.mass_matrix, self.kd, dt],
            outputs=[state.A],
            device=device,
        )

        # Step 3: blocked Cholesky factorize A → L (tile-parallel on GPU).
        # kamino kernels expect flat 1D views of the (n_padded, n_padded) buffers;
        # reshape is a zero-copy alias so this does not break graph capture.
        self._launch_factorize(
            kernel=self._factorize_kernel,
            dim=self._dim,
            mio=self._mio,
            A=state.A.reshape(-1),
            L=state.L.reshape(-1),
            num_blocks=1,
            block_dim=self._tile_block_dim,
            device=device,
        )

        # Step 4: forward + backward substitution → qddot (tile-parallel on GPU).
        self._launch_solve(
            kernel=self._solve_kernel,
            dim=self._dim,
            mio=self._mio,
            vio=self._vio,
            L=state.L.reshape(-1),
            b=state.b,
            y=state.y,
            x=state.qddot,
            num_blocks=1,
            block_dim=self._tile_block_dim,
            device=device,
        )

        # Step 5: effort[i] = const_e + ff + kp·err_pos + kd·err_vel - kd·qddot·dt.
        wp.launch(
            kernel=_stable_pd_effort_kernel,
            dim=n,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                feedforward,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self.kp,
                self.kd,
                self.const_effort,
                state.qddot,
                dt,
            ],
            outputs=[forces],
            device=device,
        )
