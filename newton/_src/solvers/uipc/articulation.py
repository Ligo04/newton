# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation runtime state for the UIPC solver backend.

Each :class:`Articulation` manages the per-joint state, animation callbacks,
control caching, and state readback for one Newton articulation.  The build
logic that creates these objects lives in :mod:`.articulation_builder`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from uipc.core import Animation
from uipc.geometry import SimplicialComplex, SimplicialComplexSlot

from newton._src.solvers.uipc.utils import _view_attr

from ...sim import JointTargetMode, JointType

# -- Warp kernels (CPU) ---------------------------------------------------


@wp.kernel
def _cache_control_kernel(
    active_joints: wp.array[wp.int32],
    local_q_start: wp.array[wp.int32],
    local_qd_start: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_target_mode: wp.array[wp.int32],
    target_pos: wp.array[wp.float32],
    target_vel: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
    has_target_pos: int,
    has_target_vel: int,
    has_joint_f: int,
    # outputs (double precision for UIPC)
    out_target_pos: wp.array[wp.float64],
    out_target_vel: wp.array[wp.float64],
    out_target_force: wp.array[wp.float64],
    out_is_constrained: wp.array[wp.int32],
    out_is_force_constrained: wp.array[wp.int32],
):
    local = wp.tid()
    newton_idx = active_joints[local]
    jtype = joint_type[newton_idx]

    if jtype != JointType.REVOLUTE and jtype != JointType.PRISMATIC:
        return

    q_idx = local_q_start[local]
    qd_idx = local_qd_start[local]
    mode = joint_target_mode[qd_idx]

    # Position driving (POSITION or POSITION_VELOCITY)
    if mode == JointTargetMode.POSITION or mode == JointTargetMode.POSITION_VELOCITY:
        out_is_constrained[local] = 1
        if has_target_pos != 0:
            out_target_pos[local] = wp.float64(target_pos[q_idx])
    else:
        out_is_constrained[local] = 0

    # Velocity target
    if has_target_vel != 0:
        out_target_vel[local] = wp.float64(target_vel[qd_idx])

    # Force/torque control (EFFORT mode)
    if mode == JointTargetMode.EFFORT and has_joint_f != 0:
        out_target_force[local] = wp.float64(joint_f[qd_idx])
        out_is_force_constrained[local] = 1
    else:
        out_is_force_constrained[local] = 0


@wp.kernel
def _write_readback_kernel(
    local_q_start: wp.array[wp.int32],
    local_qd_start: wp.array[wp.int32],
    joint_position: wp.array[wp.float64],
    joint_velocity: wp.array[wp.float64],
    joint_q_out: wp.array[wp.float32],
    joint_qd_out: wp.array[wp.float32],
    has_qd: int,
):
    local = wp.tid()
    joint_q_out[local_q_start[local]] = wp.float32(joint_position[local])
    if has_qd != 0:
        joint_qd_out[local_qd_start[local]] = wp.float32(joint_velocity[local])


# -- Placeholder for empty warp arrays passed to kernels -------------------
# One placeholder per device, allocated lazily so kernel launches with optional
# inputs can match the launch device without re-allocating each step.
_EMPTY_F32_CACHE: dict[str, wp.array] = {}
_EMPTY_F64_CACHE: dict[str, wp.array] = {}


def _empty_f32(device: Any) -> wp.array:
    key = str(device)
    arr = _EMPTY_F32_CACHE.get(key)
    if arr is None:
        arr = wp.zeros(1, dtype=wp.float32, device=device)
        _EMPTY_F32_CACHE[key] = arr
    return arr


def _empty_f64(device: Any) -> wp.array:
    key = str(device)
    arr = _EMPTY_F64_CACHE.get(key)
    if arr is None:
        arr = wp.zeros(1, dtype=wp.float64, device=device)
        _EMPTY_F64_CACHE[key] = arr
    return arr


class Articulation:
    """Runtime state and animation callbacks for a single articulation.

    Manages UIPC joint geometry references, per-joint control caching,
    UIPC :class:`Animator` callbacks, and state readback for every joint
    that belongs to one Newton articulation.

    Animator-facing state and control cache arrays are stored as
    ``wp.array(device="cpu")``.  Numpy views (zero-copy on CPU) are
    maintained for fast element-wise access inside UIPC animation
    callbacks.  Mapping arrays and per-step kernel I/O buffers live on
    the solver device (``self._device``, typically CUDA); :meth:`cache_control`
    and :meth:`write_readback` ``wp.copy`` between the two as needed.

    State arrays have shape ``(J,)`` where *J* is the number of *active*
    (driven) joints — currently :attr:`~newton.JointType.REVOLUTE` and
    :attr:`~newton.JointType.PRISMATIC`.  Joint limits are enforced by
    UIPC constitutions (``AffineBodyRevoluteJointLimit`` /
    ``AffineBodyPrismaticJointLimit``) at the physics level.

    Lifecycle
    ---------
    1. ``ArticulationBuilder`` creates an ``Articulation`` and calls
       :meth:`register_joint` / :meth:`set_joint_limits` for every
       active joint discovered during the build.
    2. After all joints are registered, :meth:`setup_state` allocates
       the warp arrays and numpy views.
    3. Each simulation step:

       a. :meth:`cache_control` launches a warp kernel to copy Newton
          ``Control`` values into the internal cache arrays.
       b. UIPC ``world.advance()`` fires the registered animation
          callbacks (:meth:`revolute_joint_anim`, :meth:`prismatic_joint_anim`).
       c. :meth:`write_readback` launches a warp kernel to scatter the
          latest joint positions and velocities back into Newton arrays.
       d. :meth:`increment_step` bumps the internal frame counter.
    """

    def __init__(self, name: str, dt: float, device: Any) -> None:
        self.name = name
        self._dt = dt
        self._step_count = 0
        self._device = device
        """Solver device — kernel launches and mapping arrays live here."""

        # -- Joint metadata (populated by ArticulationBuilder) ----------
        self.active_joint_indices: list[int] = []
        """Newton joint indices for active (driven) joints."""

        self._joint_to_local: dict[int, int] = {}
        """Newton joint index → local (0-based) index."""

        # Newton model index mapping (populated by register_joint)
        self._joint_q_start: dict[int, int] = {}
        self._joint_qd_start: dict[int, int] = {}

        # -- UIPC geometry references (populated by ArticulationBuilder) --
        self.joint_geo_slots: dict[int, SimplicialComplexSlot] = {}
        self.joint_mesh: dict[int, Any] = {}
        # Per-joint edge index and type for post-retrieve readback.
        self._joint_edge_idx: dict[int, int] = {}
        self._joint_is_revolute: dict[int, bool] = {}

        # -- Animator-facing CPU arrays (allocated by setup_state) -----
        # These are read/written by UIPC animation callbacks via numpy
        # views, so they MUST stay on CPU.
        self.joint_position: wp.array | None = None  # (J,) float64
        self.joint_velocity: wp.array | None = None  # (J,) float64
        self.target_position: wp.array | None = None  # (J,) float64
        self.target_velocity: wp.array | None = None  # (J,) float64
        self.target_force: wp.array | None = None  # (J,) float64
        self.is_constrained: wp.array | None = None  # (J,) int32
        self.is_force_constrained: wp.array | None = None  # (J,) int32

        # -- Mapping arrays for kernel dispatch (allocated by setup_state)
        # These live on ``self._device`` (typically CUDA).
        self._active_joints_wp: wp.array | None = None  # (J,) int32
        self._local_q_start_wp: wp.array | None = None  # (J,) int32
        self._local_qd_start_wp: wp.array | None = None  # (J,) int32

        # -- Device-side mirrors (allocated by setup_state) ------------
        # ``cache_control`` writes the target/constraint arrays on the
        # solver device and then ``wp.copy`` them into the CPU arrays
        # above for the animator. ``write_readback`` does the reverse:
        # ``wp.copy`` the animator-updated CPU joint state into these
        # mirrors before launching the scatter kernel.
        self._joint_position_dev: wp.array | None = None
        self._joint_velocity_dev: wp.array | None = None
        self._target_position_dev: wp.array | None = None
        self._target_velocity_dev: wp.array | None = None
        self._target_force_dev: wp.array | None = None
        self._is_constrained_dev: wp.array | None = None
        self._is_force_constrained_dev: wp.array | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_active_joints(self) -> int:
        """Number of active (driven) joints in this articulation."""
        return len(self.active_joint_indices)

    # ------------------------------------------------------------------
    # Build-time registration
    # ------------------------------------------------------------------

    def register_joint(
        self,
        newton_idx: int,
        q_start: int,
        qd_start: int,
    ) -> int:
        """Register an active joint and return its local index.

        Args:
            newton_idx: Newton joint index.
            q_start: Start index in ``joint_q`` for this joint.
            qd_start: Start index in ``joint_qd`` for this joint.

        Returns:
            Local (0-based) index within this articulation.
        """
        local = len(self.active_joint_indices)
        self.active_joint_indices.append(newton_idx)
        self._joint_to_local[newton_idx] = local
        self._joint_q_start[newton_idx] = q_start
        self._joint_qd_start[newton_idx] = qd_start
        return local

    # ------------------------------------------------------------------
    # State allocation
    # ------------------------------------------------------------------

    def setup_state(self) -> None:
        """Allocate warp arrays and numpy views.

        Must be called after all joints are registered via
        :meth:`register_joint`.
        """
        num_activate_joints = self.num_active_joints
        device = self._device

        # -- Mapping arrays (on solver device) -----------------------------
        active_np = np.array(self.active_joint_indices, dtype=np.int32)
        q_starts = np.array(
            [self._joint_q_start[idx] for idx in self.active_joint_indices],
            dtype=np.int32,
        )
        qd_starts = np.array(
            [self._joint_qd_start[idx] for idx in self.active_joint_indices],
            dtype=np.int32,
        )
        self._active_joints_wp = wp.array(active_np, dtype=wp.int32, device=device)
        self._local_q_start_wp = wp.array(q_starts, dtype=wp.int32, device=device)
        self._local_qd_start_wp = wp.array(qd_starts, dtype=wp.int32, device=device)

        # -- Animator-facing CPU arrays ------------------------------------
        self.joint_position = wp.zeros(num_activate_joints, dtype=wp.float64, device="cpu")
        self.joint_velocity = wp.zeros(num_activate_joints, dtype=wp.float64, device="cpu")
        self.target_position = wp.zeros(num_activate_joints, dtype=wp.float64, device="cpu")
        self.target_velocity = wp.zeros(num_activate_joints, dtype=wp.float64, device="cpu")
        self.target_force = wp.zeros(num_activate_joints, dtype=wp.float64, device="cpu")
        self.is_constrained = wp.zeros(num_activate_joints, dtype=wp.int32, device="cpu")
        self.is_force_constrained = wp.zeros(num_activate_joints, dtype=wp.int32, device="cpu")

        # -- Device-side mirrors for kernel I/O ----------------------------
        self._joint_position_dev = wp.zeros(num_activate_joints, dtype=wp.float64, device=device)
        self._joint_velocity_dev = wp.zeros(num_activate_joints, dtype=wp.float64, device=device)
        self._target_position_dev = wp.zeros(num_activate_joints, dtype=wp.float64, device=device)
        self._target_velocity_dev = wp.zeros(num_activate_joints, dtype=wp.float64, device=device)
        self._target_force_dev = wp.zeros(num_activate_joints, dtype=wp.float64, device=device)
        self._is_constrained_dev = wp.zeros(num_activate_joints, dtype=wp.int32, device=device)
        self._is_force_constrained_dev = wp.zeros(num_activate_joints, dtype=wp.int32, device=device)

    def seed_initial_targets(self, joint_q_np: np.ndarray) -> None:
        """Seed ``target_position`` from ``model.joint_q`` at build time.

        UIPC's animator may fire once during ``world.init(scene)`` before
        :meth:`~SolverUIPC.step` has a chance to call ``cache_control``.
        Without seeding, ``target_position`` reads as zero on that first
        callback and UIPC drives the joint toward ``aim = 0`` (with the
        ``-init_q`` edge offset, that collapses prismatic gripper fingers
        to their fully-closed state before step 0).

        Args:
            joint_q_np: Host-side view of ``model.joint_q`` indexed by
                ``q_start`` that was recorded during ``register_joint``.
        """
        if self.target_position is None:
            return
        target_np = self.target_position.numpy()
        for newton_idx in self.active_joint_indices:
            local = self._joint_to_local[newton_idx]
            q_start = self._joint_q_start[newton_idx]
            target_np[local] = float(joint_q_np[q_start])

    def increment_step(self) -> None:
        """Increment internal step counter (call once per simulation step)."""
        self._step_count += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_state(self) -> bool:
        """Return ``True`` if state arrays have been allocated."""
        return self.joint_position is not None

    # ------------------------------------------------------------------
    # Animation callbacks (called by UIPC inside world.advance())
    # ------------------------------------------------------------------

    def revolute_joint_anim(
        self,
        info: Animation.UpdateInfo,
        geo: SimplicialComplex,
        newton_joint_idx: int,
        edge_idx: int = 0,
    ) -> None:
        """UIPC Animator callback for a revolute joint.

        Writes constraint flags and target angle into the UIPC geometry
        based on the cached control mode.  State readback is **not** done
        here — the animator may fire multiple times during one
        ``world.advance()`` (e.g. line search, Newton iterations), so the
        ``angle`` value seen here is not the canonical pre/post-step
        position.  See :meth:`read_pre_advance` and
        :meth:`read_post_retrieve`.

        Args:
            info: UIPC animation update info for this dispatch (frame
                number, substep, etc.). The per-callback ``geo`` is
                extracted once in the caller and passed in separately
                to avoid re-querying the geometry slot per joint.
            geo: UIPC geometry object (from ``info.geo_slots()[0].geometry()``).
            newton_joint_idx: Newton joint index.
            edge_idx: Edge index within the batched linemesh.
        """
        if not self._ensure_state():
            return
        assert self.is_constrained is not None
        assert self.is_force_constrained is not None
        assert self.target_force is not None
        assert self.target_position is not None

        local = self._joint_to_local[newton_joint_idx]

        # Constraint and force flags
        driving = bool(self.is_constrained.numpy()[local])
        is_force_constrained = bool(self.is_force_constrained.numpy()[local])
        force_only = is_force_constrained and not driving
        external_torque = self.target_force.numpy()[local]
        aim_angle = self.target_position.numpy()[local]

        _view_attr(geo.edges().find("driving/is_constrained"))[edge_idx] = int(driving)
        _view_attr(geo.edges().find("external_torque/is_constrained"))[edge_idx] = int(force_only)

        # Force/torque control
        if force_only:
            _view_attr(geo.edges().find("external_torque"))[edge_idx] = external_torque

        # Position/velocity driving — ``aim_angle`` is in Newton absolute
        # space thanks to the ``init_angle`` edge offset, so write the
        # Newton target directly.
        if driving:
            _view_attr(geo.edges().find("aim_angle"))[edge_idx] = aim_angle

    def prismatic_joint_anim(
        self,
        info: Animation.UpdateInfo,
        geo: SimplicialComplex,
        newton_joint_idx: int,
        edge_idx: int = 0,
    ) -> None:
        """UIPC Animator callback for a prismatic joint.

        Same structure as :meth:`revolute_joint_anim` but operates on
        distance / aim_distance attributes.

        Args:
            info: UIPC animation update info for this dispatch (frame
                number, substep, etc.).
            geo: UIPC geometry object (from ``info.geo_slots()[0].geometry()``).
            newton_joint_idx: Newton joint index.
            edge_idx: Edge index within the batched linemesh.
        """
        if not self._ensure_state():
            return
        assert self.is_constrained is not None
        assert self.is_force_constrained is not None
        assert self.target_force is not None
        assert self.target_position is not None

        local = self._joint_to_local[newton_joint_idx]

        # Constraint and force flags
        driving = bool(self.is_constrained.numpy()[local])
        is_force_constrained = bool(self.is_force_constrained.numpy()[local])
        force_only = is_force_constrained and not driving
        external_force = self.target_force.numpy()[local]
        aim_distance = self.target_position.numpy()[local]

        _view_attr(geo.edges().find("driving/is_constrained"))[edge_idx] = int(driving)
        _view_attr(geo.edges().find("external_force/is_constrained"))[edge_idx] = int(force_only)

        if force_only:
            _view_attr(geo.edges().find("external_force"))[edge_idx] = external_force

        if driving:
            _view_attr(geo.edges().find("aim_distance"))[edge_idx] = aim_distance

    # ------------------------------------------------------------------
    # Per-step control caching & state readback
    # ------------------------------------------------------------------

    def cache_control(
        self,
        joint_type: wp.array,
        joint_target_mode: wp.array,
        target_pos: wp.array | None,
        target_vel: wp.array | None,
        joint_f: wp.array | None,
    ) -> None:
        """Cache Newton control values via a warp kernel.

        Called once per step **before** ``world.advance()``.

        The :class:`~newton.JointTargetMode` per DOF determines which
        control path is active:

        - ``POSITION`` / ``POSITION_VELOCITY``: position driving
          (``is_constrained``).
        - ``EFFORT``: force/torque control (``is_force_constrained``).
        - ``NONE`` / ``VELOCITY``: passive, no constraint written.

        Args:
            joint_type: Joint types array from the model (CPU).
            joint_target_mode: Per-DOF target mode from the model (CPU).
            target_pos: Target positions from :class:`Control` (CPU),
                or ``None``.
            target_vel: Target velocities from :class:`Control` (CPU),
                or ``None``.
            joint_f: Joint forces from :class:`Control` (CPU),
                or ``None``.
        """
        if not self._ensure_state():
            return

        J = self.num_active_joints
        device = self._device
        empty_f32 = _empty_f32(device)
        wp.launch(
            _cache_control_kernel,
            dim=J,
            inputs=[
                self._active_joints_wp,
                self._local_q_start_wp,
                self._local_qd_start_wp,
                joint_type,
                joint_target_mode,
                target_pos if target_pos is not None else empty_f32,
                target_vel if target_vel is not None else empty_f32,
                joint_f if joint_f is not None else empty_f32,
                int(target_pos is not None),
                int(target_vel is not None),
                int(joint_f is not None),
            ],
            outputs=[
                self._target_position_dev,
                self._target_velocity_dev,
                self._target_force_dev,
                self._is_constrained_dev,
                self._is_force_constrained_dev,
            ],
            device=device,
        )
        # Mirror device-side results into the CPU arrays consumed by the
        # UIPC animation callbacks. wp.copy is enqueued on the device
        # stream; the caller is responsible for synchronising before the
        # animator runs (see ArticulationBuilder.cache_joint_control).
        wp.copy(self.target_position, self._target_position_dev)
        wp.copy(self.target_velocity, self._target_velocity_dev)
        wp.copy(self.target_force, self._target_force_dev)
        wp.copy(self.is_constrained, self._is_constrained_dev)
        wp.copy(self.is_force_constrained, self._is_force_constrained_dev)

    def write_readback(
        self,
        joint_q_out: wp.array,
        joint_qd_out: wp.array | None,
    ) -> None:
        """Scatter cached positions/velocities into Newton arrays via kernel.

        Called once per step **after** ``world.advance()``.

        Args:
            joint_q_out: Mutable joint-position array on the solver device.
            joint_qd_out: Mutable joint-velocity array on the solver
                device, or ``None``.
        """
        if not self._ensure_state():
            return

        J = self.num_active_joints
        device = self._device

        # Mirror animator-updated CPU joint state up to the solver device.
        # The kernel launch below is enqueued on the same device stream so
        # it observes these copies without an explicit sync.
        wp.copy(self._joint_position_dev, self.joint_position)
        if joint_qd_out is not None:
            wp.copy(self._joint_velocity_dev, self.joint_velocity)

        wp.launch(
            _write_readback_kernel,
            dim=J,
            inputs=[
                self._local_q_start_wp,
                self._local_qd_start_wp,
                self._joint_position_dev,
                self._joint_velocity_dev,
                joint_q_out,
                joint_qd_out if joint_qd_out is not None else _empty_f32(device),
                int(joint_qd_out is not None),
            ],
            device=device,
        )

    def read_pre_advance(self) -> None:
        """Snapshot ``angle`` / ``distance`` into ``joint_position``.

        Call **once per step, before** :meth:`world.advance`.  This captures
        the start-of-step joint position ``q_t`` so that
        :meth:`read_post_retrieve` can compute
        ``qd = (q_{t+dt} - q_t) / dt`` from two well-defined edge-attribute
        snapshots — instead of from values seen inside the animator
        callback, which UIPC may invoke multiple times per step (line
        search, Newton iterations).
        """
        if not self._ensure_state():
            return
        assert self.joint_position is not None

        pos_np = self.joint_position.numpy()
        for newton_j in self.active_joint_indices:
            if newton_j not in self._joint_edge_idx:
                continue
            local = self._joint_to_local[newton_j]
            edge_idx = self._joint_edge_idx[newton_j]
            geo: SimplicialComplex = self.joint_geo_slots[newton_j].geometry()
            if self._joint_is_revolute[newton_j]:
                pos_np[local] = float(_view_attr(geo.edges().find("angle"))[edge_idx])
            else:
                pos_np[local] = float(_view_attr(geo.edges().find("distance"))[edge_idx])

    def read_post_retrieve(self) -> None:
        """Re-read edge attributes after ``world.retrieve()`` and finite-difference.

        Pairs with :meth:`read_pre_advance`: with ``pos_np`` holding the
        pre-advance angle/distance, this reads the post-retrieve value and
        sets ``vel = (post - pre) / dt``, then updates ``pos_np`` to the
        post-retrieve value.
        """
        if not self._ensure_state():
            return
        assert self.joint_position is not None
        assert self.joint_velocity is not None

        pos_np = self.joint_position.numpy()
        vel_np = self.joint_velocity.numpy()

        for newton_j in self.active_joint_indices:
            local = self._joint_to_local[newton_j]
            if newton_j not in self._joint_edge_idx:
                continue
            edge_idx = self._joint_edge_idx[newton_j]
            geo: SimplicialComplex = self.joint_geo_slots[newton_j].geometry()

            if self._joint_is_revolute[newton_j]:
                curr_val: float = float(_view_attr(geo.edges().find("angle"))[edge_idx])
            else:
                curr_val = float(_view_attr(geo.edges().find("distance"))[edge_idx])

            pre_val = pos_np[local]
            vel_np[local] = (curr_val - pre_val) / self._dt
            pos_np[local] = curr_val
