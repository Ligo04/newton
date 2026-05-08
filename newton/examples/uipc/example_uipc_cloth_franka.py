# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cloth Franka
#
# Franka-guided cloth manipulation with SolverUIPC.  Cloth defaults to
# StrainLimitingBaraffWitkinShell + DiscreteShellBending; pass
# ``--cloth-model neo_hookean`` to use NeoHookeanShell.  The Franka follows
# the same end-effector keyframe sequence as ``cloth_franka`` and manipulates
# the cloth through UIPC contact.
#
# Command: python -m newton.examples uipc_cloth_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import uipc
import warp as wp
from pxr import Usd
from warp._src.types import vec3f

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import JointTargetMode


@wp.kernel
def write_joint_targets_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array[wp.float32],
    gripper_activation: float,
    gripper_activation_scale: float,
):
    tid = wp.tid()
    if tid == 0:
        for j in range(7):
            joint_targets[j] = ik_solution[0, j]
        finger_target = gripper_activation * gripper_activation_scale
        joint_targets[7] = finger_target
        joint_targets[8] = finger_target


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.viewer = viewer
        self.robot_contact_mu = 1.5

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self._build_franka(builder)
        self._build_table(builder)
        self._build_cloth(builder)
        builder.add_ground_plane()
        # builder.gravity = 0.0
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._setup_ik()

        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/cloth_franka",
            dump_enable=True,
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            cloth_model=args.cloth_model,
            enable_soft_position_constraint=False,
            auto_sync_inertia=False,
        )
        self.solver.set_contact(enable=True, d_hat=0.0005)
        self.solver.configure_scene(
            {
                "linear_system": {"precond": {"mas": {"contact_aware": True}}},
            }
        )
        self.solver.configure_contact_tabular(self._configure_contact_tabular)
        self.solver.initialize(self.state_0)

        self.joint_targets_flat = wp.zeros_like(self.control.joint_target_pos)
        wp.copy(self.joint_targets_flat, self.model.joint_q)
        wp.copy(self.control.joint_target_pos, self.joint_targets_flat)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.0, 0.4, 0.9), pitch=-30.0, yaw=-145.0)
        self.viewer._paused = True  # start paused to allow shader cache to populate before starting the sim

    def _build_franka(self, builder: newton.ModelBuilder) -> None:
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((0.0, -1.20, 0.02), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

        init_q = [
            0.0,
            0.0,
            0.0,
            -1.59695,
            0.0,
            2.5307,
            0.0,
        ]
        self.gripper_activation_scale = 0.04
        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8
        builder.joint_q[:9] = [
            *init_q,
            clamp_open_activation_val * self.gripper_activation_scale,
            clamp_open_activation_val * self.gripper_activation_scale,
        ]

        for d in range(9):
            builder.joint_target_pos[d] = builder.joint_q[d]
            builder.joint_target_mode[d] = int(JointTargetMode.POSITION)

        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (position [m], quaternion), gripper activation
                # descend to working height before approaching the cloth
                [4, 0.31, -0.60, 0.40, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top left
                [2, 0.31, -0.60, 0.20, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [2, 0.31, -0.60, 0.20, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 0.26, -0.60, 0.26, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 0.12, -0.60, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -0.06, -0.60, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -0.06, -0.60, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # bottom left
                [2, 0.15, -0.33, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 0.15, -0.33, 0.21, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 0.15, -0.33, 0.21, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 0.15, -0.33, 0.28, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -0.02, -0.33, 0.28, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -0.02, -0.33, 0.28, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top right
                [2, -0.28, -0.60, 0.28, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.28, -0.60, 0.20, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.28, -0.60, 0.20, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.18, -0.60, 0.31, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, 0.05, -0.60, 0.31, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.05, -0.60, 0.31, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [3, -0.18, -0.30, 0.205, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [3, -0.18, -0.30, 0.205, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.03, -0.30, 0.31, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, -0.03, -0.30, 0.31, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -0.03, -0.30, 0.31, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom
                [2, 0.0, -0.20, 0.30, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -0.20, 0.195, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -0.20, 0.195, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.0, -0.20, 0.35, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.0, -0.30, 0.35, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -0.30, 0.35, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -0.40, 0.35, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -0.40, 0.35, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -0.28, -0.60, 0.28, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

    def _configure_contact_tabular(
        self,
        contact_tabular,
        _world_index,
        ground_elem,
        env_elem,
        robo_elem,
        actor_elem,
    ) -> None:
        contact_tabular.insert(env_elem, robo_elem, self.robot_contact_mu, 1.0e9, True)
        contact_tabular.insert(ground_elem, robo_elem, self.robot_contact_mu, 1.0e9, True)
        contact_tabular.insert(robo_elem, actor_elem, self.robot_contact_mu, 1.0e9, True)

    def _build_table(self, builder: newton.ModelBuilder) -> None:
        table_body = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, -0.5, 0.1), q=wp.quat_identity()),
            is_kinematic=True,
            label="uipc_cloth_franka_table",
        )
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=1.0e4, kd=1.0e1)
        builder.add_shape_box(table_body, hx=0.45, hy=0.45, hz=0.1, cfg=cfg, label="table")

    def _build_cloth(self, builder: newton.ModelBuilder) -> None:
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        vertices: list[vec3f] = [wp.vec3(v) for v in shirt_mesh.vertices]
        if not builder.has_custom_attribute("cloth_thick"):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="cloth_thick",
                    dtype=wp.float32,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                    default=0.005,
                )
            )
        builder.add_cloth_mesh(
            vertices=vertices,
            indices=shirt_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 0.7, 0.45),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            scale=0.01,
            tri_ke=2.0e4,
            tri_ka=2.0e4,
            tri_kd=1.0e-4,
            edge_ke=1.0e-2,
            edge_kd=1.0e-4,
            particle_radius=0.004,
            custom_attributes_particles={"cloth_thick": [0.0005] * len(vertices)},
        )
        builder.color()

    def _setup_ik(self) -> None:
        self.ee_index = next(i for i, lbl in enumerate(self.model.body_label) if lbl.endswith("/fr3_link7"))

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.22),
            target_positions=wp.array([wp.vec3(*self.targets[0][:3].tolist())], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(dtype=wp.float32),
            target_rotations=wp.array([wp.vec4(*self.targets[0][3:7].tolist())], dtype=wp.vec4),
        )
        self.joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        joint_q_np = self.model.joint_q.numpy().astype(np.float32).reshape(1, self.model.joint_coord_count)
        self.joint_q_ik = wp.array(joint_q_np, dtype=wp.float32)
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def _target_at_time(self, time: float) -> np.ndarray:
        if time >= self.robot_key_poses_time[-1]:
            return self.targets[-1]

        current_interval = int(np.searchsorted(self.robot_key_poses_time, time))
        t_start = self.robot_key_poses_time[current_interval - 1] if current_interval > 0 else 0.0
        t_end = self.robot_key_poses_time[current_interval]
        target_prev = self.targets[current_interval - 1] if current_interval > 0 else self.targets[current_interval]
        target_cur = self.targets[current_interval]
        alpha = float(np.clip((time - t_start) / (t_end - t_start), 0.0, 1.0))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return (1.0 - alpha) * target_prev + alpha * target_cur

    def _solve_ik_and_push_control(self, time: float) -> None:
        target = self._target_at_time(time)
        self.pos_obj.set_target_position(0, wp.vec3(*target[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target[3:7].tolist()))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        wp.launch(
            write_joint_targets_kernel,
            dim=1,
            inputs=[
                self.joint_q_ik,
                self.joint_targets_flat,
                float(target[-1]),
                self.gripper_activation_scale,
            ],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_flat)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.8, -1.2, -0.05)
        p_upper = wp.vec3(0.8, 0.2, 1.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 10.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--cloth-model",
        default="strain_limiting_baraff_witkin",
        choices=("strain_limiting_baraff_witkin", "strain_limiting", "neo_hookean"),
        help="UIPC cloth membrane model.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
