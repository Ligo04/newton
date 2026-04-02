# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC H1
#
# Shows how to set up a simulation of a H1 articulation
# from a USD file using the SolverUIPC backend.
#
# Command: python -m newton.examples uipc_h1 --world-count 4
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

        h1 = newton.ModelBuilder()
        h1.default_shape_cfg.ke = 2.0e3
        h1.default_shape_cfg.kd = 1.0e2
        h1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_h1")
        asset_file = str(asset_path / "usd_structured" / "h1.usda")
        h1.add_usd(
            asset_file,
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
        )
        # Approximate meshes for faster collision detection
        h1.approximate_meshes("bounding_box")

        for i in range(len(h1.joint_target_ke)):
            h1.joint_target_ke[i] = 150
            h1.joint_target_kd[i] = 5
            h1.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if h1.joint_type[i] == newton.JointType.REVOLUTE:
                h1.joint_armature[i] = 1e-2

        if self.world_count > 1:
            builder = newton.ModelBuilder()
            builder.replicate(h1, self.world_count, spacing=(2.0, 2.0, 0.0))
        else:
            builder = h1

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
            lambda q, qd: q[2] > 0.0,
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
