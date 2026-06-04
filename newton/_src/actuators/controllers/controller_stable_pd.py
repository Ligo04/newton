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


def _alloc_identity_padded_blocks(num_worlds: int, n: int, n_padded: int, device: wp.Device) -> wp.array:
    """Allocate ``W`` contiguous ``(n_padded, n_padded)`` blocks, each identity-padded.

    Layout is ``(W, n_padded, n_padded)`` row-major; the kamino blocked LLT
    kernels read it as a flat 1D float buffer with ``mio[w] = w·n_padded²``.

    Inside each per-world block, the real ``[:n, :n]`` sub-block is zero —
    callers are expected to overwrite it each step (e.g. via
    :func:`_stable_pd_build_A_kernel`). The phantom region ``[n:, n:]`` is
    initialised to the identity, so the Cholesky factor of the composite
    block is ``diag(L_real, I)``. Combined with
    :func:`_stable_pd_build_rhs_kernel` only writing ``b[w, :n]`` (the
    rest stays zero by construction), the padded substitution on
    ``b[w, n:]`` returns zero and ``qddot[w, :n]`` is the exact solution
    of the real ``[:n, :n]`` system.
    """
    buf = np.zeros((num_worlds, n_padded, n_padded), dtype=np.float32)
    if n_padded > n:
        idx = np.arange(n, n_padded)
        buf[:, idx, idx] = 1.0
    return wp.array(buf, dtype=wp.float32, device=device)


@wp.kernel
def _stable_pd_reset_identity_padded_kernel(n_per_world: int, blocks: wp.array3d[float]):
    """Reset each ``(n_padded, n_padded)`` block to its identity-padded initial state.

    Restores the invariant established by :func:`_alloc_identity_padded_blocks`:
    the real ``[:n, :n]`` sub-block is zeroed and the phantom diagonal ``[n:, n:]``
    is set to the identity. A blanket ``zero_()`` would wipe the phantom identity
    and leave the per-world composite ``diag(A_real, 0)`` singular, so the blocked
    Cholesky takes ``sqrt`` of a zero pivot and the whole solve returns NaN.
    Launched on a ``(W, n_padded, n_padded)`` grid (the trailing dims are the
    padded block size, not ``n_per_world``).
    """
    w, i, j = wp.tid()
    if i >= n_per_world and i == j:
        blocks[w, i, j] = 1.0
    else:
        blocks[w, i, j] = 0.0


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
    bias_forces: wp.array2d[float],
    n_per_world: int,
    dt: float,
    b: wp.array2d[float],
):
    """b[w, j] = kp·(target_pos - q - q̇·dt) + kd·(target_vel - q̇) - bias_forces[w, j].

    Launched on a 2D ``(W, n_per_world)`` grid. ``b`` is shape
    ``(W, n_padded)``; only the real ``[w, :n_per_world]`` slice is
    written. The phantom ``[w, n_per_world:]`` columns stay zero from
    allocation (and the Cholesky composite resolves them trivially).
    """
    w, j = wp.tid()
    flat_i = w * n_per_world + j
    pos_idx = pos_indices[flat_i]
    vel_idx = vel_indices[flat_i]
    tgt_pos_idx = target_pos_indices[flat_i]
    tgt_vel_idx = target_vel_indices[flat_i]

    q_i = current_pos[pos_idx]
    qd_i = current_vel[vel_idx]
    qt_i = target_pos[tgt_pos_idx]
    qdt_i = target_vel[tgt_vel_idx]

    p_term = kp[flat_i] * (qt_i - q_i - qd_i * dt)
    d_term = kd[flat_i] * (qdt_i - qd_i)

    c_i = float(0.0)
    if bias_forces:
        c_i = bias_forces[w, j]

    b[w, j] = p_term + d_term - c_i


@wp.kernel
def _stable_pd_build_A_kernel(
    mass_matrix: wp.array3d[float],
    kd: wp.array[float],
    n_per_world: int,
    dt: float,
    A: wp.array3d[float],
):
    """A[w, i, j] = M[w, i, j] + (i == j) · kd[w·n + i] · dt over the real block.

    Launched on a 3D ``(W, n_per_world, n_per_world)`` grid. ``A`` is
    shape ``(W, n_padded, n_padded)``; only the real ``[w, :n, :n]``
    sub-block is written. The phantom ``[w, n:, n:]`` region keeps the
    identity-pad set at allocation (see
    :func:`_alloc_identity_padded_blocks`), so the per-world composite
    Cholesky reduces to the real ``[:n, :n]`` solve.
    """
    w, i, j = wp.tid()
    a_ij = mass_matrix[w, i, j]
    if i == j:
        a_ij = a_ij + kd[w * n_per_world + i] * dt
    A[w, i, j] = a_ij


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
    qddot: wp.array2d[float],
    n_per_world: int,
    dt: float,
    efforts: wp.array[float],
):
    """effort[w·n + j] = const_effort + feedforward + kp·err_pos + kd·err_vel - kd·qddot[w, j]·dt."""
    w, j = wp.tid()
    flat_i = w * n_per_world + j
    pos_idx = pos_indices[flat_i]
    vel_idx = vel_indices[flat_i]
    tgt_pos_idx = target_pos_indices[flat_i]
    tgt_vel_idx = target_vel_indices[flat_i]

    q_i = current_pos[pos_idx]
    qd_i = current_vel[vel_idx]
    qt_i = target_pos[tgt_pos_idx]
    qdt_i = target_vel[tgt_vel_idx]

    p_term = kp[flat_i] * (qt_i - q_i - qd_i * dt)
    d_term = kd[flat_i] * (qdt_i - qd_i)

    const_e = float(0.0)
    if const_effort:
        const_e = const_effort[flat_i]

    ff = float(0.0)
    if feedforward:
        ff = feedforward[tgt_vel_idx]

    efforts[flat_i] = const_e + ff + p_term + d_term - kd[flat_i] * qddot[w, j] * dt


class ControllerStablePD(Controller):
    """Stateful stable-PD controller (Tan 2011) with implicit mass-matrix solve.

    Implements the Tan, Liu & Turk 2011 "stable proportional-derivative"
    control law per world::

        p_term = kp · (target_pos - q - q̇·dt)
        d_term = kd · (target_vel - q̇)
        (M_w + diag(kd_w)·dt) · q̈_w = p_term_w + d_term_w - C_w
        effort = const_effort + feedforward + p_term + d_term - kd·q̈·dt

    The ``q̈`` unknowns are solved implicitly on the GPU via a blocked
    Cholesky factorization of ``A_w = M_w + diag(kd_w)·dt`` for each
    world ``w``, dispatched as a single batched kamino LLT launch with
    ``num_blocks = num_worlds``. Every world's Cholesky operates on its
    own ``n_per_world * n_per_world`` block independently — the mass
    coupling between worlds is zero by construction (each world is a
    disjoint articulation), so there is no need to factorize one big
    ``(W·n) * (W·n)`` dense matrix. The entire step — rhs assembly,
    A-matrix assembly, factorize, solve, and per-actuator effort
    composition — is a sequence of ``wp.launch`` / ``wp.launch_tiled``
    calls with no host↔device transfers, making the pipeline CUDA-graph
    capturable.

    The caller is responsible for populating ``State.mass_matrix`` and
    ``State.bias_forces`` before each ``compute()`` call from the physics
    engine (:mod:`newton.sim`, MuJoCo, PhysX, UIPC, ...). This class makes
    no assumption about the source of those quantities. The leading
    batch dimension on both quantities matches the
    ``(articulation_count, max_dofs, max_dofs)`` output of
    :func:`newton.eval_mass_matrix` directly, so multi-world callers can
    typically slice and assign without any reshape.

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
            mass_matrix: Per-world effective mass/inertia ``M_w(q)`` for
                this actuator subsystem, shape
                ``(num_worlds, n_per_world, n_per_world)``, dtype
                ``float32``. The leading batch dimension matches
                :func:`newton.eval_mass_matrix` output (``(articulation_count,
                max_dofs, max_dofs)``), so common usage is a direct slice
                + assign.
            bias_forces: Per-world Coriolis + gravity + external force
                compensation ``C_w(q, q̇)``, shape
                ``(num_worlds, n_per_world)``, dtype ``float32``.

        Internal scratch (allocated by :meth:`ControllerStablePD.state`):
            A: Augmented per-world mass matrices ``M_w + diag(kd_w)·dt``,
                shape ``(num_worlds, n_padded, n_padded)``, dtype
                ``float32``. Only the real ``[:, :n, :n]`` sub-blocks
                are written each step; the phantom regions are
                identity-padded at allocation and never touched.
            L: Per-world Cholesky factors of ``A``, same layout as ``A``.
            b: Rhs ``p_term + d_term - C`` per world, shape
                ``(num_worlds, n_padded)``; only ``b[:, :n]`` is written.
            y: Forward-substitution intermediate ``L · y = b``, shape
                ``(num_worlds, n_padded)``.
            qddot: Solution of ``Lᵀ · qddot = y`` (implicit acceleration),
                shape ``(num_worlds, n_padded)``; only ``qddot[:, :n]``
                is read.
        """

        mass_matrix: wp.array3d[float] | None = None
        bias_forces: wp.array2d[float] | None = None
        A: wp.array3d[float] | None = None
        L: wp.array3d[float] | None = None
        b: wp.array2d[float] | None = None
        y: wp.array2d[float] | None = None
        qddot: wp.array2d[float] | None = None

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset all buffers to their initial state. ``mask`` is ignored (scratch is per-world).

            ``mass_matrix``, ``bias_forces``, ``b``, ``y`` and ``qddot`` are
            zeroed (they are user-populated or fully recomputed each step). ``A``
            and ``L`` are restored to their identity-padded initial state rather
            than zeroed: a blanket ``zero_()`` wipes the phantom ``[n:, n:]``
            identity that :func:`_alloc_identity_padded_blocks` installs and that
            :func:`_stable_pd_build_A_kernel` never rewrites, leaving the per-world
            composite ``diag(A_real, 0)`` singular so the blocked Cholesky returns
            NaN on the next step (only when ``n_padded > n_per_world``).
            """
            for arr in (self.mass_matrix, self.bias_forces, self.b, self.y, self.qddot):
                if arr is not None:
                    arr.zero_()
            # mass_matrix is (W, n_per_world, n_per_world); A/L are (W, n_padded, n_padded).
            n_per_world = self.mass_matrix.shape[1] if self.mass_matrix is not None else 0
            for blocks in (self.A, self.L):
                if blocks is not None:
                    wp.launch(
                        _stable_pd_reset_identity_padded_kernel,
                        dim=blocks.shape,
                        inputs=[n_per_world],
                        outputs=[blocks],
                        device=blocks.device,
                    )

    SHARED_PARAMS = frozenset({"num_worlds", "block_size", "tile_block_dim"})  # pyright: ignore[reportAssignmentType]
    """Controller-construction kwargs shared across every actuator in the entry.

    The :class:`~newton.ModelBuilder` per-DOF / shared split inspects this set
    to decide which kwargs flow through ``controller_shared_kwargs`` instead
    of being stacked into per-actuator arrays.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        kp = args.get("kp", 0.0)
        if kp < 0:
            raise ValueError(f"kp must be non-negative, got {kp}")
        kd = args.get("kd", 0.0)
        if kd < 0:
            raise ValueError(f"kd must be non-negative, got {kd}")
        num_worlds = int(args.get("num_worlds", 1))
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
        resolved: dict[str, Any] = {
            "kp": kp,
            "kd": kd,
            "const_effort": args.get("const_effort", 0.0),
            "num_worlds": num_worlds,
        }
        # block_size / tile_block_dim pass through opt-in: include only when the
        # user authored them so the entry-key shared-tuple stays minimal and
        # groups still merge across calls that didn't override the defaults.
        if "block_size" in args:
            resolved["block_size"] = int(args["block_size"])
        if "tile_block_dim" in args:
            resolved["tile_block_dim"] = int(args["tile_block_dim"])
        return resolved

    def __init__(
        self,
        kp: wp.array[float],
        kd: wp.array[float],
        const_effort: wp.array[float] | None = None,
        *,
        num_worlds: int = 1,
        block_size: int = DEFAULT_BLOCK_SIZE,
        tile_block_dim: int = DEFAULT_TILE_BLOCK_DIM,
    ):
        """Initialize stable-PD controller.

        Args:
            kp: Proportional gains [N/m or N·m/rad]. Shape ``(num_worlds·n_per_world,)``.
            kd: Derivative gains [N·s/m or N·m·s/rad]. Same shape as ``kp``.
            const_effort: Constant bias effort [N or N·m]. Same shape as ``kp``. ``None`` to skip.
            num_worlds: Number of independent worlds (block-diagonal blocks). Default 1.
                The total actuator count must be divisible by ``num_worlds``;
                each world contributes the same ``n_per_world`` DOFs in a
                contiguous chunk of the flat per-DOF arrays
                (``[w0_dof0, ..., w0_dofN-1, w1_dof0, ..., w1_dofN-1, ...]``,
                which is exactly the layout :meth:`newton.ModelBuilder.replicate`
                produces).
            block_size: Tile edge length for the blocked Cholesky. Default 32.
            tile_block_dim: CUDA thread-block size for the tile kernels. Default 128.
        """
        if kp.shape != kd.shape:
            raise ValueError(f"kp shape {kp.shape} must match kd shape {kd.shape}")
        if const_effort is not None and const_effort.shape != kp.shape:
            raise ValueError(f"const_effort shape {const_effort.shape} must match kp shape {kp.shape}")
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
        self.kp = kp
        self.kd = kd
        self.const_effort = const_effort
        self._num_worlds = int(num_worlds)
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
        self._n_per_world: int = 0
        self._n_padded: int = 0
        self._dim: wp.array[wp.int32] | None = None
        self._mio: wp.array[wp.int32] | None = None
        self._vio: wp.array[wp.int32] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        if num_actuators % self._num_worlds != 0:
            raise ValueError(
                f"num_actuators ({num_actuators}) must be divisible by num_worlds "
                f"({self._num_worlds}); each world must contribute the same DOF count"
            )
        self._n_per_world = num_actuators // self._num_worlds
        self._n_padded = _next_block_multiple(self._n_per_world, self._block_size)
        # Per-world batched layout for the kamino LLT kernels: each world's
        # matrix occupies ``n_padded × n_padded`` floats laid out contiguously  # noqa: RUF003
        # in a flat 1D buffer, and each world's vector occupies ``n_padded``
        # floats. ``dim[w] = n_padded`` (composite system, identity-padded
        # by :meth:`state`); ``mio[w] = w·n_padded²``; ``vio[w] = w·n_padded``.
        self._dim = wp.array([self._n_padded] * self._num_worlds, dtype=wp.int32, device=device)
        self._mio = wp.array(
            [w * self._n_padded * self._n_padded for w in range(self._num_worlds)],
            dtype=wp.int32,
            device=device,
        )
        self._vio = wp.array(
            [w * self._n_padded for w in range(self._num_worlds)],
            dtype=wp.int32,
            device=device,
        )

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return True

    def state(self, num_actuators: int, device: wp.Device) -> ControllerStablePD.State:
        if num_actuators % self._num_worlds != 0:
            raise ValueError(f"num_actuators ({num_actuators}) must be divisible by num_worlds ({self._num_worlds})")
        n = num_actuators // self._num_worlds
        n_padded = _next_block_multiple(n, self._block_size)
        W = self._num_worlds
        return ControllerStablePD.State(
            mass_matrix=wp.zeros((W, n, n), dtype=wp.float32, device=device),
            bias_forces=wp.zeros((W, n), dtype=wp.float32, device=device),
            A=_alloc_identity_padded_blocks(W, n, n_padded, device=device),
            L=_alloc_identity_padded_blocks(W, n, n_padded, device=device),
            b=wp.zeros((W, n_padded), dtype=wp.float32, device=device),
            y=wp.zeros((W, n_padded), dtype=wp.float32, device=device),
            qddot=wp.zeros((W, n_padded), dtype=wp.float32, device=device),
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

        n = self._n_per_world
        W = self._num_worlds

        # Step 1: b[w, :n] = p_term + d_term - C (per-world per-actuator, parallel).
        wp.launch(
            kernel=_stable_pd_build_rhs_kernel,
            dim=(W, n),
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
                n,
                dt,
            ],
            outputs=[state.b],
            device=device,
        )

        # Step 2: A[w, :n, :n] = M_w + diag(kd_w)·dt (per-world per-entry, parallel).
        wp.launch(
            kernel=_stable_pd_build_A_kernel,
            dim=(W, n, n),
            inputs=[state.mass_matrix, self.kd, n, dt],
            outputs=[state.A],
            device=device,
        )

        # Step 3: blocked Cholesky factorize A → L (tile-parallel on GPU, batched over worlds).
        # kamino kernels expect flat 1D views of the per-world (n_padded, n_padded) and
        # n_padded buffers; reshape is a zero-copy alias so this does not break graph capture.
        self._launch_factorize(
            kernel=self._factorize_kernel,
            dim=self._dim,
            mio=self._mio,
            A=state.A.reshape(-1),
            L=state.L.reshape(-1),
            num_blocks=W,
            block_dim=self._tile_block_dim,
        )

        # Step 4: forward + backward substitution → qddot (tile-parallel, batched over worlds).
        self._launch_solve(
            kernel=self._solve_kernel,
            dim=self._dim,
            mio=self._mio,
            vio=self._vio,
            L=state.L.reshape(-1),
            b=state.b.reshape(-1),
            y=state.y.reshape(-1),
            x=state.qddot.reshape(-1),
            num_blocks=W,
            block_dim=self._tile_block_dim,
        )

        # Step 5: effort[w·n + j] = const_e + ff + kp·err_pos + kd·err_vel - kd·qddot[w, j]·dt.
        wp.launch(
            kernel=_stable_pd_effort_kernel,
            dim=(W, n),
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
                n,
                dt,
            ],
            outputs=[forces],
            device=device,
        )
