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
from uipc import view
from uipc.core import Animation as UIPCAnimation

from ...sim import JointTargetMode, JointType

# -- Warp kernels (CPU) ---------------------------------------------------


@wp.kernel
def _cache_control_kernel(
    active_joints: wp.array(dtype=wp.int32),
    local_q_start: wp.array(dtype=wp.int32),
    local_qd_start: wp.array(dtype=wp.int32),
    joint_type: wp.array(dtype=wp.int32),
    joint_target_mode: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.float32),
    target_vel: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
    has_target_pos: int,
    has_target_vel: int,
    has_joint_f: int,
    # outputs (double precision for UIPC)
    out_target_pos: wp.array(dtype=wp.float64),
    out_target_vel: wp.array(dtype=wp.float64),
    out_target_force: wp.array(dtype=wp.float64),
    out_is_constrained: wp.array(dtype=wp.int32),
    out_is_force_constrained: wp.array(dtype=wp.int32),
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
    local_q_start: wp.array(dtype=wp.int32),
    local_qd_start: wp.array(dtype=wp.int32),
    joint_position: wp.array(dtype=wp.float64),
    joint_velocity: wp.array(dtype=wp.float64),
    joint_q_out: wp.array(dtype=wp.float32),
    joint_qd_out: wp.array(dtype=wp.float32),
    has_qd: int,
):
    local = wp.tid()
    joint_q_out[local_q_start[local]] = wp.float32(joint_position[local])
    if has_qd != 0:
        joint_qd_out[local_qd_start[local]] = wp.float32(joint_velocity[local])


# -- Placeholder for empty warp arrays passed to kernels -------------------
_EMPTY_F32 = wp.zeros(1, dtype=wp.float32, device="cpu")
_EMPTY_F64 = wp.zeros(1, dtype=wp.float64, device="cpu")


class Articulation:
    """Runtime state and animation callbacks for a single articulation.

    Manages UIPC joint geometry references, per-joint control caching,
    UIPC :class:`Animator` callbacks, and state readback for every joint
    that belongs to one Newton articulation.

    All internal state and control cache arrays are stored as
    ``wp.array(device="cpu")``.  Numpy views (zero-copy on CPU) are
    maintained for fast element-wise access inside UIPC animation
    callbacks.

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

    def __init__(self, name: str, dt: float) -> None:
        self.name = name
        self._dt = dt
        self._step_count = 0

        # -- Joint metadata (populated by ArticulationBuilder) ----------
        self.active_joint_indices: list[int] = []
        """Newton joint indices for active (driven) joints."""

        self._joint_to_local: dict[int, int] = {}
        """Newton joint index → local (0-based) index."""

        # Newton model index mapping (populated by register_joint)
        self._joint_q_start: dict[int, int] = {}
        self._joint_qd_start: dict[int, int] = {}

        # -- UIPC geometry references (populated by ArticulationBuilder) --
        self.joint_geo_slots: dict[int, Any] = {}
        self.joint_mesh: dict[int, Any] = {}

        # -- Warp arrays on CPU (allocated by setup_state) --------------
        self.joint_position: wp.array | None = None  # (J,) float64
        self.joint_velocity: wp.array | None = None  # (J,) float64
        self.target_position: wp.array | None = None  # (J,) float64
        self.target_velocity: wp.array | None = None  # (J,) float64
        self.target_force: wp.array | None = None  # (J,) float64
        self.is_constrained: wp.array | None = None  # (J,) int32
        self.is_force_constrained: wp.array | None = None  # (J,) int32

        # -- Mapping arrays for kernel dispatch (allocated by setup_state)
        self._active_joints_wp: wp.array | None = None  # (J,) int32
        self._local_q_start_wp: wp.array | None = None  # (J,) int32
        self._local_qd_start_wp: wp.array | None = None  # (J,) int32

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
        init_value: float = 0.0,
    ) -> int:
        """Register an active joint and return its local index.

        Args:
            newton_idx: Newton joint index.
            q_start: Start index in ``joint_q`` for this joint.
            qd_start: Start index in ``joint_qd`` for this joint.
            init_value: Initial angle [rad] or distance [m].

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
        J = self.num_active_joints

        # -- Mapping arrays for kernel dispatch ----------------------------
        active_np = np.array(self.active_joint_indices, dtype=np.int32)
        q_starts = np.array(
            [self._joint_q_start[idx] for idx in self.active_joint_indices],
            dtype=np.int32,
        )
        qd_starts = np.array(
            [self._joint_qd_start[idx] for idx in self.active_joint_indices],
            dtype=np.int32,
        )
        self._active_joints_wp = wp.array(active_np, dtype=wp.int32, device="cpu")
        self._local_q_start_wp = wp.array(q_starts, dtype=wp.int32, device="cpu")
        self._local_qd_start_wp = wp.array(qd_starts, dtype=wp.int32, device="cpu")

        # -- State arrays (wp.array on CPU) --------------------------------
        self.joint_position = wp.zeros(J, dtype=wp.float64, device="cpu")
        self.joint_velocity = wp.zeros(J, dtype=wp.float64, device="cpu")

        # Initialise positions from build-time values
        # pos_np = self.joint_position.numpy()
        # for newton_idx in self.active_joint_indices:
        #     local = self._joint_to_local[newton_idx]
        #     pos_np[local] = self._init_values.get(newton_idx, 0.0)

        # -- Control cache arrays (wp.array on CPU) ------------------------
        self.target_position = wp.zeros(J, dtype=wp.float64, device="cpu")
        self.target_velocity = wp.zeros(J, dtype=wp.float64, device="cpu")
        self.target_force = wp.zeros(J, dtype=wp.float64, device="cpu")
        self.is_constrained = wp.zeros(J, dtype=wp.int32, device="cpu")
        self.is_force_constrained = wp.zeros(J, dtype=wp.int32, device="cpu")

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
        info: UIPCAnimation.UpdateInfo,
        newton_joint_idx: int,
    ) -> None:
        """UIPC Animator callback for a revolute joint.

        Reads the current angle from the UIPC geometry, updates local
        readback state, then writes constraint flags and target angle
        back into the geometry based on the cached control mode.
        """
        if not self._ensure_state():
            return
        assert self.joint_position is not None
        assert self.joint_velocity is not None
        assert self.is_constrained is not None
        assert self.is_force_constrained is not None
        assert self.target_force is not None
        assert self.target_position is not None

        try:
            geo = info.geo_slots()[0].geometry()
        except (TypeError, IndexError):
            return

        local = self._joint_to_local[newton_joint_idx]
        pos_np = self.joint_position.numpy()
        vel_np = self.joint_velocity.numpy()
        curr_angle = view(geo.edges().find("angle"))[0]

        # Update readback (numpy view writes through to wp.array on CPU)
        if self._step_count > 0:
            vel_np[local] = (curr_angle - pos_np[local]) / self._dt
        pos_np[local] = curr_angle

        # Constraint and force flags
        driving = bool(self.is_constrained.numpy()[local])
        is_force_constrained = bool(self.is_force_constrained.numpy()[local])
        force_only = is_force_constrained and not driving

        view(geo.edges().find("driving/is_constrained"))[:] = int(driving)
        view(geo.edges().find("external_torque/is_constrained"))[:] = int(force_only)

        # Force/torque control
        if force_only:
            view(geo.edges().find("external_force"))[:] = self.target_force.numpy()[local]

        # Position/velocity driving
        if driving:
            view(geo.edges().find("aim_angle"))[:] = self.target_position.numpy()[local]

    def prismatic_joint_anim(
        self,
        info: UIPCAnimation.UpdateInfo,
        newton_joint_idx: int,
    ) -> None:
        """UIPC Animator callback for a prismatic joint.

        Same structure as :meth:`revolute_joint_anim` but operates on
        distance / aim_distance attributes.
        """
        if not self._ensure_state():
            return
        assert self.joint_position is not None
        assert self.joint_velocity is not None
        assert self.is_constrained is not None
        assert self.is_force_constrained is not None
        assert self.target_force is not None
        assert self.target_position is not None

        try:
            geo = info.geo_slots()[0].geometry()
        except (TypeError, IndexError):
            return

        local = self._joint_to_local[newton_joint_idx]
        pos_np = self.joint_position.numpy()
        vel_np = self.joint_velocity.numpy()
        curr_dist = view(geo.edges().find("distance"))[0]
        # print("curr_dist", curr_dist)
        # Update readback (numpy view writes through to wp.array on CPU)
        if self._step_count > 0:
            vel_np[local] = (curr_dist - pos_np[local]) / self._dt
        pos_np[local] = curr_dist

        # Constraint and force flags
        driving = bool(self.is_constrained.numpy()[local])
        is_force_constrained = bool(self.is_force_constrained.numpy()[local])
        force_only = is_force_constrained and not driving

        view(geo.edges().find("driving/is_constrained"))[:] = int(driving)
        view(geo.edges().find("external_force/is_constrained"))[:] = int(force_only)

        if force_only:
            view(geo.edges().find("external_force"))[:] = self.target_force.numpy()[local]

        if driving:
            view(geo.edges().find("aim_distance"))[:] = self.target_position.numpy()[local]

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
        wp.launch(
            _cache_control_kernel,
            dim=J,
            inputs=[
                self._active_joints_wp,
                self._local_q_start_wp,
                self._local_qd_start_wp,
                joint_type,
                joint_target_mode,
                target_pos if target_pos is not None else _EMPTY_F32,
                target_vel if target_vel is not None else _EMPTY_F32,
                joint_f if joint_f is not None else _EMPTY_F32,
                int(target_pos is not None),
                int(target_vel is not None),
                int(joint_f is not None),
            ],
            outputs=[
                self.target_position,
                self.target_velocity,
                self.target_force,
                self.is_constrained,
                self.is_force_constrained,
            ],
            device="cpu",
        )

    def write_readback(
        self,
        joint_q_out: wp.array,
        joint_qd_out: wp.array | None,
    ) -> None:
        """Scatter cached positions/velocities into Newton arrays via kernel.

        Called once per step **after** ``world.advance()``.

        Args:
            joint_q_out: Mutable joint-position array on CPU.
            joint_qd_out: Mutable joint-velocity array on CPU, or ``None``.
        """
        if not self._ensure_state():
            return

        J = self.num_active_joints
        wp.launch(
            _write_readback_kernel,
            dim=J,
            inputs=[
                self._local_q_start_wp,
                self._local_qd_start_wp,
                self.joint_position,
                self.joint_velocity,
                joint_q_out,
                joint_qd_out if joint_qd_out is not None else _EMPTY_F32,
                int(joint_qd_out is not None),
            ],
            device="cpu",
        )
