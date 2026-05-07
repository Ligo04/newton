# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Softbody Franka
#
# Demonstrates a Franka Panda robot grasping a deformable rubber duck
# on a table. The robot is driven with Newton's GPU IK solver while
# SolverUIPC integrates the articulated robot and the Stable Neo-Hookean
# deformable body in the same scene.
#
# Command: python -m newton.examples uipc_softbody_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import uipc
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode, ModelBuilder, eval_fk
from newton.solvers import SolverUIPC


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


class Example:
    def __init__(self, viewer, args=None):
        self.sim_substeps = 1
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.particle_radius = 0.005
        self.soft_body_contact_margin = 0.001
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.005

        self.viewer = viewer
        self.scene = ModelBuilder(up_axis=newton.Axis.Z)

        self.create_articulation(self.scene)
        self.scene.add_ground_plane()

        table_body = self.scene.add_link(
            xform=wp.transform(p=wp.vec3(0.0, -0.5, 0.1), q=wp.quat_identity()),
            is_kinematic=True,
            label="uipc_softbody_franka_table",
        )
        table_cfg = ModelBuilder.ShapeConfig(mu=0.8, ke=1.0e4, kd=1.0e1)
        self.scene.add_shape_box(table_body, hx=0.4, hy=0.4, hz=0.1, cfg=table_cfg, label="table")

        duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
        usd_stage = Usd.Stage.Open(str(duck_path / "model.usda"))
        prim = usd_stage.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh = newton.TetMesh.create_from_usd(prim)

        self.scene.add_soft_mesh(
            pos=wp.vec3(0.0, -0.5, 0.23),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            density=100.0,
            k_mu=1.0e6,
            k_lambda=1.0e6,
            k_damp=1.0e-6,
            particle_radius=self.particle_radius,
        )

        self.scene.color()
        self.model = self.scene.finalize(requires_grad=False)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.set_up_ik()

        self.solver = SolverUIPC(
            workspace="/tmp/newton_uipc/uipc_softbody_franka",
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            auto_sync_inertia=False,
        )
        self.solver.set_contact(enable=True, d_hat=self.soft_body_contact_margin)
        self.solver.initialize(self.state_0)

        self.joint_targets_flat = wp.zeros_like(self.control.joint_target_pos)
        wp.copy(self.joint_targets_flat, self.model.joint_q)
        wp.copy(self.control.joint_target_pos, self.joint_targets_flat)

        self.viewer._paused = True

    def set_up_ik(self):
        """Set up GPU IK solver for end-effector pose tracking."""
        state = self.model.state()
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)

        self.n_coords = self.model.joint_coord_count
        self.n_dofs = self.model.joint_dof_count
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.n_coords))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.zeros(1, dtype=float)
        self.target_joint_q = wp.zeros(self.n_coords, dtype=float)

        target_pos = wp.vec3(*self.targets[0][:3].tolist())
        target_rot = wp.vec4(*self.targets[0][3:7].tolist())
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=wp.vec3(0.0, 0.0, 0.22),
            target_positions=wp.array([target_pos], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([target_rot], dtype=wp.vec4),
        )
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
        for d in range(min(9, len(builder.joint_q))):
            builder.joint_target_pos[d] = builder.joint_q[d]
            builder.joint_target_mode[d] = int(JointTargetMode.POSITION)

        gripper_open = 1.0
        gripper_close = 0.5
        self.robot_key_poses = np.array(
            [
                # approach: move above the duck
                [2.5, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_open],
                # descend: lower to duck body
                [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_open],
                # pinch: close gripper on duck
                [2.5, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_close],
                # lift: raise duck off table
                [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_close],
                # hold: pause in air
                [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_close],
                # place: lower back to table
                [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_close],
                # release: open gripper
                [1.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_open],
                # retract: move away
                [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_open],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3

    def update_ik_targets(self):
        """Interpolate keyframes and update IK targets."""
        if self.sim_time >= self.robot_key_poses_time[-1]:
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        t_start = self.robot_key_poses_time[current_interval - 1] if current_interval > 0 else 0.0
        t_end = self.robot_key_poses_time[current_interval]
        alpha = float(np.clip((self.sim_time - t_start) / (t_end - t_start), 0.0, 1.0))

        target_cur = self.targets[current_interval]
        target_prev = self.targets[current_interval - 1] if current_interval > 0 else target_cur
        target_interp = (1.0 - alpha) * target_prev + alpha * target_cur

        self.pos_obj.set_target_position(0, wp.vec3(*target_interp[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target_interp[3:7].tolist()))
        self.finger_pos_buf.fill_(float(target_interp[-1]) * 0.04)

    def simulate(self):
        self.update_ik_targets()
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
        )
        wp.copy(self.joint_targets_flat, self.ik_joint_q, dest_offset=0, src_offset=0, count=self.n_coords)
        wp.copy(self.control.joint_target_pos, self.joint_targets_flat)

        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.5, -1.0, -0.05)
        p_upper = wp.vec3(0.5, 0.0, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
