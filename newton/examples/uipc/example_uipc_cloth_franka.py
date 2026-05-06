# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cloth Franka
#
# Franka-guided cloth manipulation with SolverUIPC.  Cloth defaults to
# StrainLimitingBaraffWitkinShell + DiscreteShellBending; pass
# ``--cloth-model neo_hookean`` to use NeoHookeanShell.  The grasped cloth
# patch is controlled through UIPC SoftPositionConstraint targets while the
# Franka hand follows the same handle with IK-driven position actuators.
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


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def write_joint_targets_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array[wp.float32],
    gripper_value: float,
):
    tid = wp.tid()
    if tid == 0:
        for j in range(7):
            joint_targets[j] = ik_solution[0, j]
        joint_targets[7] = gripper_value
        joint_targets[8] = gripper_value


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.soft_strength = float(args.soft_strength)
        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self._build_franka(builder)
        self._build_table(builder)
        self._build_cloth(builder)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._setup_cloth_handle()
        self._setup_ik()

        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/cloth_franka",
            # dump_enable=True,
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            cloth_model=args.cloth_model,
            cloth_soft_position_strength_ratio=self.soft_strength,
            auto_sync_inertia=False,
        )
        self.solver.set_contact(enable=True, d_hat=0.005)
        self.solver.initialize(self.state_0)
        self._push_cloth_handle(0.0)

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
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        builder.joint_q[:9] = [*init_q, 0.04, 0.04]

        for d in range(9):
            builder.joint_target_pos[d] = builder.joint_q[d]
            builder.joint_target_ke[d] = 650.0
            builder.joint_target_kd[d] = 100.0
            builder.joint_target_mode[d] = int(JointTargetMode.POSITION)
            builder.joint_armature[d] = 1e-2 if d < 7 else 5e-2

    def _build_table(self, builder: newton.ModelBuilder) -> None:
        table_body = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, -0.5, 0.1), q=wp.quat_identity()),
            is_kinematic=True,
            label="uipc_cloth_franka_table",
        )
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=1.0e4, kd=1.0e1)
        builder.add_shape_box(table_body, hx=0.45, hy=0.45, hz=0.1, cfg=cfg, label="table")

    def _build_cloth(self, builder: newton.ModelBuilder) -> None:
        self.cloth_particle_start = builder.particle_count
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        vertices: list[vec3f] = [wp.vec3(v) for v in shirt_mesh.vertices]
        if not builder.has_custom_attribute("cloth_thick"):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="cloth_thick",
                    dtype=wp.float32,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                    default=0.001,
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
            custom_attributes_particles={"cloth_thick": [1.0e-4] * len(vertices)},
        )
        self.cloth_particle_end = builder.particle_count
        builder.color()

    def _setup_cloth_handle(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        cloth_indices = np.arange(self.cloth_particle_start, self.cloth_particle_end, dtype=np.int32)
        cloth_q = particle_q[cloth_indices]

        desired = np.array([0.24, -0.60, 0.63], dtype=np.float64)
        nearest = np.argsort(np.linalg.norm(cloth_q - desired, axis=1))[:32]
        self.grasp_particle_indices = cloth_indices[nearest]
        grasp_positions = particle_q[self.grasp_particle_indices].astype(np.float64)
        self.handle_start = grasp_positions.mean(axis=0)
        self.handle_offsets = grasp_positions - self.handle_start

    def _setup_ik(self) -> None:
        self.ee_index = next(i for i, lbl in enumerate(self.model.body_label) if lbl.endswith("/fr3_link7"))
        body_q_np = self.state_0.body_q.numpy()
        ee_tf = wp.transform(*body_q_np[self.ee_index])
        self.ee_rotation = wp.transform_get_rotation(ee_tf)
        self.ee_lift = np.array([0.0, 0.0, 0.14], dtype=np.float64)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*(self.handle_start + self.ee_lift))], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(dtype=wp.float32),
            target_rotations=wp.array([quat_to_vec4(self.ee_rotation)], dtype=wp.vec4),
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
        self.gripper_value = 0.01

    def _handle_center(self, time: float) -> np.ndarray:
        key_times = np.array([0.0, 1.0, 2.5, 4.0, 5.5], dtype=np.float64)
        key_positions = np.vstack(
            (
                self.handle_start,
                self.handle_start + np.array([0.0, 0.0, 0.08]),
                self.handle_start + np.array([-0.12, 0.04, 0.16]),
                self.handle_start + np.array([-0.22, 0.15, 0.14]),
                self.handle_start + np.array([-0.22, 0.15, 0.14]),
            )
        )
        if time <= key_times[0]:
            return key_positions[0]
        if time >= key_times[-1]:
            return key_positions[-1]
        segment = int(np.searchsorted(key_times, time) - 1)
        alpha = (time - key_times[segment]) / (key_times[segment + 1] - key_times[segment])
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return (1.0 - alpha) * key_positions[segment] + alpha * key_positions[segment + 1]

    def _push_cloth_handle(self, time: float) -> None:
        center = self._handle_center(time)
        self.solver.set_cloth_soft_position_constraints(
            self.grasp_particle_indices,
            center + self.handle_offsets,
            strength_ratio=self.soft_strength,
        )

    def _solve_ik_and_push_control(self, time: float) -> None:
        target_pos = self._handle_center(time) + self.ee_lift
        self.pos_obj.set_target_position(0, wp.vec3(*target_pos))
        self.rot_obj.set_target_rotation(0, quat_to_vec4(self.ee_rotation))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        wp.launch(
            write_joint_targets_kernel,
            dim=1,
            inputs=[self.joint_q_ik, self.joint_targets_flat, self.gripper_value],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_flat)

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self.state_0.clear_forces()
            self._push_cloth_handle(t)
            # self._solve_ik_and_push_control(t)
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
    parser.add_argument("--soft-strength", type=float, default=200.0, help="SoftPositionConstraint strength ratio.")
    parser.set_defaults(num_frames=360)

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
