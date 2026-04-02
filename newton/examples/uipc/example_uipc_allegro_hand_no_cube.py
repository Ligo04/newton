# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Allegro Hand (No Cube)
#
# Shows how to set up a simulation of an Allegro hand articulation
# from a USD file using the SolverUIPC backend, without loading the
# cube object.  A sinusoidal trajectory is applied to the finger
# joint targets each step.
#
# Command: python -m newton.examples uipc_allegro_hand_no_cube --world-count 1
#
###########################################################################

import numpy as np
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

        allegro_hand = newton.ModelBuilder(up_axis=newton.Axis.Z)
        allegro_hand.default_shape_cfg.ke = 1.0e3
        allegro_hand.default_shape_cfg.kd = 1.0e2
        allegro_hand.default_shape_cfg.margin = 0.005
        allegro_hand.default_shape_cfg.gap = 0.015

        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        allegro_hand.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0, 0, 0.5)),
            enable_self_collisions=False,
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*object", ".*goal"],
            hide_collision_shapes=True,
        )

        self.finger_dof_count = allegro_hand.joint_dof_count

        # Set joint targets and drive gains for the hand joints
        for i in range(self.finger_dof_count):
            allegro_hand.joint_target_ke[i] = 150
            allegro_hand.joint_target_kd[i] = 5
            allegro_hand.joint_q[i] = 0.3
            allegro_hand.joint_target_pos[i] = 0.3
            if allegro_hand.joint_label[i][-2:] == "_0":
                allegro_hand.joint_q[i] = 0.6
                allegro_hand.joint_target_pos[i] = 0.6
            allegro_hand.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if allegro_hand.joint_type[i] == newton.JointType.REVOLUTE:
                allegro_hand.joint_armature[i] = 1e-2

        if self.world_count > 1:
            builder = newton.ModelBuilder(newton.Axis.Z)
            builder.replicate(allegro_hand, self.world_count, spacing=(1.0, 2.0, 0.0))
        else:
            builder = allegro_hand

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

        # Cache joint limit arrays on CPU for target clamping
        self.joint_limit_lower = self.model.joint_limit_lower.numpy()
        self.joint_limit_upper = self.model.joint_limit_upper.numpy()
        self.joint_qd_start = self.model.joint_qd_start.numpy()

        # Compute joints per world for the hand-only articulation
        self.joints_per_world = self.model.joint_count // self.world_count
        self.bodies_per_world = self.model.body_count // self.world_count

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(-0.5, 1.0, 0.6),
            pitch=-15.0,
            yaw=-160.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        self.viewer._paused = True

    def _update_targets(self):
        """Apply sinusoidal trajectory to finger joint targets."""
        target_pos = self.control.joint_target_pos.numpy()
        t = self.sim_time

        dof_per_world = self.finger_dof_count

        for w in range(self.world_count):
            root_joint_id = w * self.joints_per_world
            root_dof_start = self.joint_qd_start[root_joint_id]

            for i in range(dof_per_world):
                di = root_dof_start + i
                val = np.sin(t + i * 0.6) * 0.1 + 0.3
                target_pos[di] = np.clip(
                    val,
                    self.joint_limit_lower[di],
                    self.joint_limit_upper[di],
                )

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
        for i in range(self.world_count):
            world_offset = i * self.bodies_per_world

            # Hand bodies should remain in a reasonable volume
            hand_body_indices = np.arange(self.bodies_per_world, dtype=np.int32) + world_offset
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                f"hand bodies from world {i} are within bounds",
                lambda q, qd: float(q[2]) > -0.5 and float(q[2]) < 2.0,
                indices=hand_body_indices,
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
