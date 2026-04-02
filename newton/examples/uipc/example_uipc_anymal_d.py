# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Anymal D
#
# Shows how to simulate Anymal D with multiple worlds using SolverUIPC.
#
# Command: python -m newton.examples uipc_anymal_d --world-count 4
#
###########################################################################

import uipc
import warp as wp

import newton
import newton.examples
import newton.utils
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

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        articulation_builder.default_shape_cfg.ke = 2.0e3
        articulation_builder.default_shape_cfg.kd = 1.0e2
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        articulation_builder.add_usd(
            asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.62]
        if len(articulation_builder.joint_q) > 6:
            articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

        for i in range(articulation_builder.joint_dof_count):
            articulation_builder.joint_target_ke[i] = 150
            articulation_builder.joint_target_kd[i] = 5
            articulation_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if articulation_builder.joint_type[i] == newton.JointType.REVOLUTE:
                articulation_builder.joint_armature[i] = 1e-2

        if self.world_count > 1:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            builder.replicate(articulation_builder, self.world_count, spacing=(2.0, 2.0, 0.0))
        else:
            builder = articulation_builder

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Info,
        )

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(5.0, 5.0, 3.0),
            pitch=-20.0,
            yaw=-135.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
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
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
