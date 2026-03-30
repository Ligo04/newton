# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a USD stage using the SolverUIPC backend.
#
# Command: python -m newton.examples uipc_cartpole --world-count 4
#
###########################################################################

import uipc
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.world_count = args.world_count
        self.viewer = viewer

        cartpole = newton.ModelBuilder(up_axis=newton.Axis.Y)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1
        cartpole.default_body_armature = 0.1

        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        # Set initial joint positions
        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        if self.world_count > 1:
            builder = newton.ModelBuilder(newton.Axis.Y)
            builder.replicate(cartpole, self.world_count, spacing=(0.1, 0.0, 0.2))
        else:
            builder = cartpole

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(self.model, dt=self.sim_dt, logger_level=uipc.Logger.Critical)

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(9.5, 3.5, 5),
            pitch=-10.0,
            yaw=-160.0,
        )
        self.viewer._paused = True

    def simulate(self):
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
        # After simulation the cart should have moved and poles should be swinging.
        # Bodies: 0=cart, 1=pole1, 2=pole2
        # (rail is collapsed into the fixed joint since collapse_fixed_joints=True)
        num_bodies_per_world = self.model.body_count // self.world_count

        # Cart should remain near ground level with correct orientation
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart is near ground level",
            lambda q, qd: abs(float(q[1])) < 1.0,
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )

        # Poles should have non-zero angular velocity from swinging
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 is swinging",
            lambda q, qd: abs(qd[3]) > 0.01 or abs(qd[4]) > 0.01 or abs(qd[5]) > 0.01,
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
        )

        # Poles should still be above ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 above ground",
            lambda q, qd: float(q[1]) > -0.5,
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 above ground",
            lambda q, qd: float(q[1]) > -0.5,
            indices=[i * num_bodies_per_world + 2 for i in range(self.world_count)],
        )

        # Verify cross-world consistency when using multiple worlds
        if self.world_count > 1:
            qd = self.state_0.body_qd.numpy()
            world0_cart_vel = wp.spatial_vector(*qd[0])
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "cart velocities match across worlds",
                lambda q, qd: newton.math.vec_allclose(qd, world0_cart_vel),
                indices=[i * num_bodies_per_world for i in range(self.world_count)],
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
