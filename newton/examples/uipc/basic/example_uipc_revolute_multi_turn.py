# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Revolute Multi-Turn
#
# Probes the aim-driven hinge energy E = 1/2 * kappa * (theta_a - aim)^2 of
# a single revolute joint when the target angle leaves (-pi, pi].  The
# measured angle theta_a inside UIPC comes from atan2 and is inherently
# wrapped to (-pi, pi], while ``control.joint_target_q`` is an unbounded
# absolute angle coming from the upper layer (Newton wrapper / IsaacLab).
# Without unwrapping, the energy jumps by O(2*pi) at the wrap point, line
# search never accepts the crossing, and the joint pins just before 180
# degrees.  The ``angle`` readback shares the same wrap.
#
# The rotor spins about the world Z axis so gravity exerts no torque about
# the joint and tracking error isolates the drive itself.  The commanded
# target ramps at constant speed through several full turns; the example
# unwraps the (possibly wrapped) readback by nearest continuation purely
# for measurement and asserts the joint keeps tracking past +/-pi.
#
# Command: python -m newton.examples uipc_revolute_multi_turn
#
###########################################################################

import math

import uipc
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.viewer = viewer

        # Constant-speed ramp: half a turn per second, so the target exits
        # (-pi, pi] after two seconds and completes several turns per run.
        self.target_speed = math.pi  # [rad/s]

        # joint_target_ke/kd are cross-solver metadata only: UIPC's aim
        # drive strength comes from the solver's drive_strength_ratio
        # (default 100) and has no damping channel, independent of kp/kd.
        self.kp = 1.0e6
        self.kd = 200.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        q_id = wp.quat_identity(dtype=wp.float64)  # pyright: ignore[reportArgumentType]
        hub_z = 1.2
        hub_hz = 0.1
        arm_hx = 0.4

        hub = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, hub_z), q=q_id),
            label="hub",
        )
        arm = builder.add_link(
            xform=wp.transform(p=wp.vec3(arm_hx, 0.0, hub_z - hub_hz * 2.0), q=q_id),
            label="arm",
        )
        builder.add_shape_box(hub, hx=0.1, hy=0.1, hz=hub_hz)
        builder.add_shape_box(arm, hx=arm_hx, hy=0.05, hz=0.05)

        j_fixed = builder.add_joint_fixed(
            parent=-1,
            child=hub,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, hub_z), q=q_id),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=q_id),
            label="fixed_hub_anchor",
        )
        j_revolute = builder.add_joint_revolute(
            parent=hub,
            child=arm,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -hub_hz * 2.0), q=q_id),
            child_xform=wp.transform(p=wp.vec3(-arm_hx, 0.0, 0.0), q=q_id),
            label="revolute_hub_arm",
        )
        builder.add_articulation([j_fixed, j_revolute], label="rotor")

        # Single DOF in the model: the revolute axis (the fixed joint has none).
        self.rev_dof = len(builder.joint_target_mode) - 1
        builder.joint_target_mode[self.rev_dof] = int(JointTargetMode.POSITION)
        builder.joint_target_ke[self.rev_dof] = self.kp
        builder.joint_target_kd[self.rev_dof] = self.kd
        builder.joint_target_q[self.rev_dof] = 0.0

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            workspace="/tmp/newton_uipc/revolute_multi_turn",
            dt=self.sim_dt,
            logger_level=uipc.Logger.Error,
        )
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.solver.initialize(self.state_0)

        # Nearest-continuation unwrap state for the (possibly wrapped) readback.
        self.raw_angle = 0.0
        self.unwrapped_angle = 0.0
        self.target_angle = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -3.0, 3.0), pitch=-35.0, yaw=-90.0)

    def _read_angle(self) -> None:
        """Read the joint angle back and unwrap it by nearest continuation.

        The per-frame true motion (target_speed / fps) is far below pi, so
        adding the 2*pi multiple that keeps the reading closest to the
        previous unwrapped value reconstructs the absolute angle whether or
        not the readback wraps.
        """
        q_start = int(self.model.joint_q_start.numpy()[1])  # revolute joint is joint index 1
        raw = float(self.state_0.joint_q.numpy()[q_start])
        delta = math.remainder(raw - self.raw_angle, math.tau)
        self.unwrapped_angle += delta
        self.raw_angle = raw

    def simulate(self):
        self.target_angle = self.target_speed * (self.sim_time + self.frame_dt)
        target_np = self.control.joint_target_q.numpy()
        target_np[self.rev_dof] = self.target_angle
        self.control.joint_target_q.assign(target_np)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._read_angle()
        if round(self.sim_time * self.fps) % self.fps == 0:
            print(
                f"t={self.sim_time:5.2f}s  target={math.degrees(self.target_angle):8.2f}°  "
                f"raw={math.degrees(self.raw_angle):8.2f}°  "
                f"unwrapped={math.degrees(self.unwrapped_angle):8.2f}°  "
                f"error={math.degrees(self.unwrapped_angle - self.target_angle):+.2f}°"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        # Allow a spin-up transient, then require continuous tracking. A
        # wrapped hinge energy pins the joint just before pi, so the error
        # grows without bound and trips this immediately after t ~ 2 s.
        if self.sim_time > 1.0:
            error = abs(self.unwrapped_angle - self.target_angle)
            assert error < 0.5, (
                f"joint stopped tracking at t={self.sim_time:.2f}s: "
                f"target={self.target_angle:.3f} unwrapped={self.unwrapped_angle:.3f}"
            )

    def test_final(self):
        # The joint must have genuinely crossed the atan2 wrap point --
        # a run too short to leave (-pi, pi] would not probe anything.
        assert self.target_angle > math.pi + 1.0, (
            f"run too short to cross the wrap point: final target={self.target_angle:.3f}"
        )
        assert self.unwrapped_angle > math.pi + 0.5, (
            f"joint never crossed pi: unwrapped={self.unwrapped_angle:.3f} (pinned at the wrap point?)"
        )
        error = abs(self.unwrapped_angle - self.target_angle)
        assert error < 0.5, (
            f"final tracking error too large: target={self.target_angle:.3f} unwrapped={self.unwrapped_angle:.3f}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=360)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
