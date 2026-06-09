# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cartpole PD — Force (Effort) Control via ControllerPD
#
# Drives the cart (prismatic DOF) with a PD controller implemented as a
# Newton ``ControllerPD`` composed under ``newton.actuators.Actuator`` with
# a ``ClampingMaxEffort`` layer for the force limit. The actuator reads
# ``state.joint_q`` / ``state.joint_qd`` and ``control.joint_target_q`` /
# ``joint_target_qd``, computes the PD torque, clamps it, and writes it
# into ``control.joint_f``. The cart DOF is configured as
# ``JointTargetMode.EFFORT`` so the UIPC solver forwards that force as
# generalized effort on the joint.
#
# Compared to ``example_uipc_cartpole_pd_position`` which uses a solver-
# native position drive, this variant shows the canonical "external
# actuator" pipeline:
#   1. ``builder.add_actuator(ControllerPD, ...)`` registers a PD actuator
#   2. every step: write the position target into ``control.joint_target_q``
#   3. every step: ``actuator.step(state, control, ...)`` computes joint_f
#   4. every step: ``solver.step(...)`` consumes joint_f via EFFORT mode
#
# Command: python -m newton.examples uipc_cartpole_pd_force --world-count 1
#
###########################################################################

import math

import numpy as np
import uipc
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.actuators import ClampingMaxEffort, ControllerPD
from newton.selection import ArticulationView


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.world_count = args.world_count
        self.viewer = viewer

        # Target trajectory for the cart — identical to the position-control
        # variant so the two examples can be visually compared.
        self.cart_amplitude = 0.8  # [m]
        self.cart_frequency = 0.5  # [Hz]

        # PD gains for the cart actuator.
        self.kp = 2000.0
        self.kd = 200.0
        self.max_force = 1.0e4

        cartpole = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1
        cartpole.default_body_armature = 0.1

        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # DOF layout after collapse_fixed_joints:
        #   d=0 : prismatic cart slider
        #   d=1 : revolute pole1
        #   d=2 : revolute pole2
        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        # Cart DOF is EFFORT-mode (fed by the external PD actuator); the
        # poles stay passive.
        cart_dof = len(cartpole.joint_target_mode) - 3
        cartpole.joint_target_mode[cart_dof] = int(JointTargetMode.EFFORT)
        cartpole.joint_target_mode[cart_dof + 1] = int(JointTargetMode.NONE)
        cartpole.joint_target_mode[cart_dof + 2] = int(JointTargetMode.NONE)

        # Register the PD actuator on the cart DOF. The actuator reads
        # control.joint_target_q / joint_target_qd, the ControllerPD
        # kernel computes the PD torque, ClampingMaxEffort clamps it to
        # ±max_force, and the Actuator scatter-adds the result into
        # control.joint_f.
        cartpole.add_actuator(
            ControllerPD,
            index=cart_dof,
            kp=self.kp,
            kd=self.kd,
            clamping=[(ClampingMaxEffort, {"max_effort": self.max_force})],
        )

        if self.world_count > 1:
            builder = newton.ModelBuilder(newton.Axis.Z)
            builder.replicate(cartpole, self.world_count, spacing=(1.0, 2.0, 0.0))
        else:
            builder = cartpole

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            workspace="/tmp/newton_uipc/cartpole_pd_force",
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
        )

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Selection view over every replicated cartpole — lets us read/write
        # per-DOF control/state tensors as (world_count, 1, dof_per_arti)
        # without hand-rolled strided indexing.
        self.cartpoles = ArticulationView(self.model, "/cartPole")
        assert self.cartpoles.count == self.world_count, (
            f"expected one /cartPole per world, got {self.cartpoles.count} for {self.world_count} worlds"
        )
        self.dofs_per_world = self.cartpoles.joint_dof_count
        # DOF layout after collapse_fixed_joints: [cart_slider, pole1, pole2].
        self.cart_dof = 0

        # Evaluate forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(9.5, 5, 3.5),
            pitch=-10.0,
            yaw=-160.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

    def _update_cart_target(self):
        """Write the sinusoidal cart position/velocity target into control.

        Uses :class:`ArticulationView` so every replica's cart DOF is
        addressed through a single strided tensor instead of hand-rolled
        per-world offsets.
        """
        t = self.sim_time
        w = 2.0 * math.pi * self.cart_frequency
        target_pos = self.cart_amplitude * math.sin(w * t)
        target_vel = self.cart_amplitude * w * math.cos(w * t)

        # Shape: (world_count, 1, dofs_per_arti)
        pos = self.cartpoles.get_attribute("joint_target_q", self.control).numpy()
        vel = self.cartpoles.get_attribute("joint_target_qd", self.control).numpy()
        pos[:, 0, self.cart_dof] = target_pos
        vel[:, 0, self.cart_dof] = target_vel
        self.cartpoles.set_attribute("joint_target_q", self.control, pos)
        self.cartpoles.set_attribute("joint_target_qd", self.control, vel)

    def _apply_actuators(self):
        """Run every registered actuator so joint_f gets filled before solver.step.

        ``Actuator.step`` scatter-adds into ``control.joint_f`` (``out += force``),
        so we must clear the buffer each frame or the PD force will grow unboundedly
        across frames and blow past the clamping ceiling.
        """
        self.control.joint_f.zero_()
        for actuator in self.model.actuators:
            actuator.step(
                sim_state=self.state_0,
                sim_control=self.control,
                current_act_state=None,
                next_act_state=None,
                dt=self.sim_dt,
            )

    def simulate(self):
        self._update_cart_target()
        self._apply_actuators()
        self._log_applied_force()
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _log_applied_force(self):
        """Print the PD torque ControllerPD produced for the cart this frame."""
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0).numpy()
        joint_f = self.cartpoles.get_attribute("joint_f", self.control).numpy()
        target = self.cart_amplitude * math.sin(2.0 * math.pi * self.cart_frequency * self.sim_time)
        # Report world-0 cart slider only; replicas follow the same trajectory.
        cart_q = float(joint_q[0, 0, self.cart_dof])
        cart_f = float(joint_f[0, 0, self.cart_dof])
        print(f"[t={self.sim_time:6.3f}s] cart_q={cart_q:+.4f} m  target={target:+.4f} m  joint_f={cart_f:+9.2f} N")

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # After simulation the cart should be tracking the commanded sine
        # (looser tolerance than the position-driven version since this PD
        # loop runs at the outer frame rate), and both poles should still be
        # above the ground.
        num_bodies_per_world = self.model.body_count // self.world_count

        # Shape: (world_count, 1, dofs_per_arti) → cart slider at dof `cart_dof`.
        joint_q = self.cartpoles.get_attribute("joint_q", self.state_0).numpy()
        expected = self.cart_amplitude * math.sin(2.0 * math.pi * self.cart_frequency * self.sim_time)
        cart_qs = joint_q[:, 0, self.cart_dof]
        tracking_errs = np.abs(cart_qs - expected)
        worst = int(np.argmax(tracking_errs))
        assert tracking_errs[worst] < 0.5, (
            f"world {worst} cart tracking error too large: cart_q={float(cart_qs[worst]):.3f} expected={expected:.3f}"
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 above ground",
            lambda q, qd: float(q[2]) > -0.5,
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 above ground",
            lambda q, qd: float(q[2]) > -0.5,
            indices=[i * num_bodies_per_world + 2 for i in range(self.world_count)],
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
