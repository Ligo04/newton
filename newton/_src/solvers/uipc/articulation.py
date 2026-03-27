# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulation runtime state for the UIPC solver backend.

Each :class:`Articulation` manages the per-joint state, animation callbacks,
control caching, and state readback for one Newton articulation.  The build
logic that creates these objects lives in :mod:`.articulation_builder`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from uipc import view
from uipc.core import Animation as UIPCAnimation

from ...sim import JointType


class JointControlMode(IntEnum):
    """Control mode for UIPC-backed joints.

    - ``NONE``:     No position/velocity driving.  Use effort for force/torque.
    - ``POSITION``: Position-target driving.
    - ``VELOCITY``: Velocity-target driving.
    """

    NONE = 0
    POSITION = 1
    VELOCITY = 2


class Articulation:
    """Runtime state and animation callbacks for a single articulation.

    Manages UIPC joint geometry references, per-joint control caching,
    UIPC :class:`Animator` callbacks, and state readback for every joint
    that belongs to one Newton articulation.

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
       the numpy arrays.
    3. Each simulation step:

       a. :meth:`cache_control` copies Newton ``Control`` values into
          the internal cache arrays.
       b. UIPC ``world.advance()`` fires the registered animation
          callbacks (:meth:`revolute_joint_anim`, :meth:`prismatic_joint_anim`).
       c. :meth:`write_readback` copies the latest joint positions
          and velocities back into Newton ``State`` arrays.
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

        # -- Build-time temporaries (consumed by setup_state) -----------
        self._init_values: dict[int, float] = {}

        # -- State arrays (allocated by setup_state) --------------------
        self.joint_position: np.ndarray | None = None  # (J,)
        self.joint_velocity: np.ndarray | None = None  # (J,)

        # -- Control cache arrays (allocated by setup_state) ------------
        self.target_position: np.ndarray | None = None  # (J,)
        self.target_velocity: np.ndarray | None = None  # (J,)
        self.target_force: np.ndarray | None = None  # (J,)
        self.is_constrained: np.ndarray | None = None  # (J,) bool
        self.has_force: np.ndarray | None = None  # (J,) bool

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
        self._init_values[newton_idx] = init_value
        return local

    # ------------------------------------------------------------------
    # State allocation
    # ------------------------------------------------------------------

    def setup_state(self) -> None:
        """Allocate state and control arrays.

        Must be called after all joints are registered via
        :meth:`register_joint`.
        """
        J = self.num_active_joints

        # Readback state
        self.joint_position = np.zeros(J, dtype=np.float32)
        self.joint_velocity = np.zeros(J, dtype=np.float32)

        # Initialise positions from build-time values
        for newton_idx in self.active_joint_indices:
            local = self._joint_to_local[newton_idx]
            self.joint_position[local] = self._init_values.get(newton_idx, 0.0)

        # Control cache
        self.target_position = np.zeros(J, dtype=np.float32)
        self.target_velocity = np.zeros(J, dtype=np.float32)
        self.target_force = np.zeros(J, dtype=np.float32)
        self.is_constrained = np.zeros(J, dtype=bool)
        self.has_force = np.zeros(J, dtype=bool)

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
        assert self.has_force is not None
        assert self.target_force is not None
        assert self.target_velocity is not None
        assert self.target_position is not None

        try:
            geo = info.geo_slots()[0].geometry()
        except (TypeError, IndexError):
            return

        local = self._joint_to_local[newton_joint_idx]
        curr_angle = float(view(geo.edges().find("angle"))[0])

        # Update readback
        if self._step_count > 0:
            self.joint_velocity[local] = (curr_angle - self.joint_position[local]) / self._dt
        self.joint_position[local] = curr_angle

        # Constraint and force flags
        driving = self.is_constrained[local]
        has_force = self.has_force[local]
        force_only = has_force and not driving

        view(geo.edges().find("driving/is_constrained"))[:] = driving.astype(int)
        view(geo.edges().find("external_torque/is_constrained"))[:] = int(force_only)

        # Force/torque control
        if force_only:
            view(geo.edges().find("external_force"))[:] = float(self.target_force[local])

        # Position/velocity driving
        if driving:
            target = float(self.target_position[local])
            view(geo.edges().find("aim_angle"))[:] = target

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
        assert self.has_force is not None
        assert self.target_force is not None
        assert self.target_velocity is not None
        assert self.target_position is not None

        try:
            geo = info.geo_slots()[0].geometry()
        except (TypeError, IndexError):
            return

        local = self._joint_to_local[newton_joint_idx]
        curr_dist = float(view(geo.edges().find("distance"))[0])

        # Update readback
        if self._step_count > 0:
            self.joint_velocity[local] = (curr_dist - self.joint_position[local]) / self._dt
        self.joint_position[local] = curr_dist

        # Constraint and force flags
        driving = bool(self.is_constrained[local])
        has_force = bool(self.has_force[local])
        force_only = has_force and not driving

        view(geo.edges().find("driving/is_constrained"))[:] = int(driving)
        view(geo.edges().find("external_force/is_constrained"))[:] = int(force_only)

        if force_only:
            view(geo.edges().find("external_force"))[:] = float(self.target_force[local])

        if driving:
            target = float(self.target_position[local])
            view(geo.edges().find("aim_distance"))[:] = target

    # ------------------------------------------------------------------
    # Per-step control caching & state readback
    # ------------------------------------------------------------------

    def cache_control(
        self,
        joint_type_np: np.ndarray,
        joint_q_start_np: np.ndarray,
        joint_qd_start_np: np.ndarray,
        target_pos_np: np.ndarray | None,
        target_vel_np: np.ndarray | None,
        joint_f_np: np.ndarray | None,
    ) -> None:
        """Cache Newton control values for the animation callbacks.

        Called once per step **before** ``world.advance()``.

        Args:
            joint_type_np: Joint types array from the model.
            joint_q_start_np: Joint position-coordinate start indices.
            joint_qd_start_np: Joint velocity-DOF start indices.
            target_pos_np: Target positions from :class:`Control`, or ``None``.
            target_vel_np: Target velocities from :class:`Control`, or ``None``.
            joint_f_np: Joint forces from :class:`Control`, or ``None``.
        """
        if not self._ensure_state():
            return
        assert self.target_position is not None
        assert self.target_velocity is not None
        assert self.target_force is not None
        assert self.is_constrained is not None
        assert self.has_force is not None

        for newton_idx in self.active_joint_indices:
            local = self._joint_to_local[newton_idx]
            jtype = JointType(joint_type_np[newton_idx])

            if jtype not in (JointType.REVOLUTE, JointType.PRISMATIC):
                continue

            q_idx = joint_q_start_np[newton_idx]
            qd_idx = joint_qd_start_np[newton_idx]

            # Position target
            if target_pos_np is not None:
                self.target_position[local] = target_pos_np[q_idx]
                self.is_constrained[local] = True
            else:
                self.is_constrained[local] = False

            # Velocity target (used as fallback when no position target)
            if target_vel_np is not None:
                self.target_velocity[local] = float(target_vel_np[qd_idx])

            # Force/torque
            if joint_f_np is not None:
                f = float(joint_f_np[qd_idx])
                self.target_force[local] = f
                self.has_force[local] = abs(f) > 1e-12
            else:
                self.has_force[local] = False

    def write_readback(
        self,
        joint_q_out: np.ndarray,
        joint_qd_out: np.ndarray | None,
    ) -> None:
        """Write cached joint positions and velocities to Newton arrays.

        Called once per step **after** ``world.advance()``.

        Args:
            joint_q_out: Mutable joint-position array (modified in-place).
            joint_qd_out: Mutable joint-velocity array, or ``None``.
        """
        if not self._ensure_state():
            return
        assert self.joint_position is not None
        assert self.joint_velocity is not None

        for newton_idx in self.active_joint_indices:
            local = self._joint_to_local[newton_idx]
            q_start = self._joint_q_start[newton_idx]
            joint_q_out[q_start] = self.joint_position[local]

            if joint_qd_out is not None:
                qd_start = self._joint_qd_start[newton_idx]
                joint_qd_out[qd_start] = self.joint_velocity[local]
