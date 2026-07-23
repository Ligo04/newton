# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cartpole PD — Position Control
#
# Drives the cart (prismatic DOF) with UIPC's built-in position-control PD:
# ``JointTargetMode.POSITION`` sets up an aim constraint whose strength comes
# from the solver's ``drive_strength_ratio`` (default 100), not from
# ``joint_target_ke`` / ``joint_target_kd`` on the builder.
# The user just writes a scalar target into ``control.joint_target_q``
# every step and UIPC takes care of the PD law internally. The poles stay
# passive so they swing as the cart accelerates.
#
# Compare with ``example_uipc_cartpole_pd_force`` which keeps the same task
# but implements the PD law in user space via ``newton.actuators.ControllerPD``
# and feeds the computed torque into ``JointTargetMode.EFFORT``.
#
# Command: python -m newton.examples uipc_cartpole_pd_position --world-count 1
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

        self.world_count = args.world_count
        self.viewer = viewer

        # Trajectory for the cart — sine wave on the prismatic DOF.
        self.cart_amplitude = 0.8  # [m]
        self.cart_frequency = 0.5  # [Hz]

        # joint_target_ke/kd are cross-solver metadata only: UIPC's aim
        # drive strength comes from the solver's drive_strength_ratio
        # (default 100) and has no damping channel, independent of kp/kd.
        self.kp = 2000.0
        self.kd = 200.0

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
        # Initial state — cart at origin, pole1 tilted, pole2 upright.
        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        # Configure the cart DOF as a position-driven actuator. Poles stay
        # passive (JointTargetMode.NONE — no constraint written).
        cart_dof = len(cartpole.joint_target_mode) - 3
        cartpole.joint_target_mode[cart_dof] = int(JointTargetMode.POSITION)
        cartpole.joint_target_mode[cart_dof + 1] = int(JointTargetMode.NONE)
        cartpole.joint_target_mode[cart_dof + 2] = int(JointTargetMode.NONE)

        # PD gains on the cart drive. UIPC tracks the target via its own aim
        # constraint; the same gains are read by EFFORT-consuming solvers
        # (MuJoCo, etc.) so the model stays portable across backends.
        cartpole.joint_target_ke[cart_dof] = self.kp
        cartpole.joint_target_kd[cart_dof] = self.kd

        # Initial target matches the rest pose so the first step is consistent.
        cartpole.joint_target_q[cart_dof] = 0.0

        if self.world_count > 1:
            builder = newton.ModelBuilder(newton.Axis.Z)
            builder.replicate(cartpole, self.world_count, spacing=(1.0, 2.0, 0.0))
        else:
            builder = cartpole

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            workspace="/tmp/newton_uipc/cartpole_pd_position",
            dt=self.sim_dt,
            logger_level=uipc.Logger.Info,
            dump_enable=True,
        )

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        # DOFs per world, used to index the cart slider in every replica.
        self.dofs_per_world = self.model.joint_dof_count // self.world_count
        self.cart_dof_indices = [w * self.dofs_per_world for w in range(self.world_count)]

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
        """Write a sinusoidal cart position target into the control buffer."""
        target = self.cart_amplitude * math.sin(2.0 * math.pi * self.cart_frequency * self.sim_time)
        # Copy into host-side numpy first so we hit one H→D transfer per step
        # instead of one per world.
        target_np = self.control.joint_target_q.numpy()
        for idx in self.cart_dof_indices:
            target_np[idx] = target
        self.control.joint_target_q.assign(target_np)

    def simulate(self):
        self._update_cart_target()
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

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # After simulation the cart should be tracking the commanded sine
        # within a tolerance determined by the drive stiffness, and both
        # poles should still be above the ground.
        num_bodies_per_world = self.model.body_count // self.world_count

        joint_q = self.state_0.joint_q.numpy()
        expected = self.cart_amplitude * math.sin(2.0 * math.pi * self.cart_frequency * self.sim_time)
        for w in range(self.world_count):
            cart_q = float(joint_q[w * self.dofs_per_world])
            assert abs(cart_q - expected) < 0.2, (
                f"world {w} cart tracking error too large: cart_q={cart_q:.3f} expected={expected:.3f}"
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
