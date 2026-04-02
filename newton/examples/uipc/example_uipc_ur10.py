# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC UR10
#
# Shows how to set up a simulation of a UR10 robot arm
# from a USD file using the SolverUIPC backend, and applies a sinusoidal
# trajectory to the joint targets.
#
# Command: python -m newton.examples uipc_ur10 --world-count 4
#
###########################################################################

import numpy as np
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

        ur10 = newton.ModelBuilder()

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        # Create a pedestal
        ur10.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0, 0, height / 2)), half_height=height / 2, radius=0.08)

        for i in range(len(ur10.joint_target_ke)):
            ur10.joint_target_ke[i] = 500
            ur10.joint_target_kd[i] = 50
            ur10.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if ur10.joint_type[i] == newton.JointType.REVOLUTE:
                ur10.joint_armature[i] = 1e-2

        if self.world_count > 1:
            builder = newton.ModelBuilder()
            builder.replicate(ur10, self.world_count, spacing=(2.0, 2.0, 0.0))
        else:
            builder = ur10

        # Set random joint configurations
        rng = np.random.default_rng(42)
        joint_q = rng.uniform(-wp.pi, wp.pi, builder.joint_dof_count)
        builder.joint_q = joint_q.tolist()

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

        # Cache joint limit arrays on CPU for target clamping
        self.joint_limit_lower = self.model.joint_limit_lower.numpy()
        self.joint_limit_upper = self.model.joint_limit_upper.numpy()
        self.joint_qd_start = self.model.joint_qd_start.numpy()

        # Prepare sinusoidal trajectory parameters
        self.dof_per_world = self.model.joint_dof_count // self.world_count if self.world_count > 0 else 0

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(5.0, 5.0, 3.0),
            pitch=-20.0,
            yaw=-135.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

    def _update_targets(self):
        """Apply sinusoidal trajectory to joint targets."""
        target_pos = self.control.joint_target_pos.numpy()
        t = self.sim_time

        for w in range(self.world_count):
            dof_start = w * self.dof_per_world
            for i in range(self.dof_per_world):
                di = dof_start + i
                lower = self.joint_limit_lower[di]
                upper = self.joint_limit_upper[di]
                if not np.isfinite(lower) or abs(lower) > 6.0:
                    lower = -wp.pi
                    upper = wp.pi
                mid = 0.5 * (upper + lower)
                amp = 0.4 * (upper - lower)
                # Offset phase per DOF and per world for visual variety
                val = mid + amp * np.sin(t * 1.5 + i * 0.8 + w * 0.3)
                target_pos[di] = np.clip(val, lower, upper)

        self.control.joint_target_pos.assign(target_pos)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._update_targets()
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
        pass

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
