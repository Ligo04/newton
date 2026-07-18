# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cloth Franka Stable-PD Force
#
# Stable-PD force-control version of the UIPC cloth Franka example.  Cloth shell thickness
# is taken directly from ``particle_radius`` so UIPC thickness and Newton
# particle contact radius stay aligned.  Cloth defaults to
# StrainLimitingBaraffWitkinShell + DiscreteShellBending; pass
# ``--cloth-model neo_hookean`` to set every ``model.uipc.cloth_model`` entry to
# NeoHookeanShell.  The Franka follows
# the same end-effector keyframe sequence as ``cloth_franka`` and manipulates
# the cloth through UIPC contact.
#
# Command: python -m newton.examples uipc_cloth_franka_stable_pd_force
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
from newton.actuators import Actuator as _NewtonActuator
from newton.actuators import ClampingMaxEffort, ControllerStablePD
from newton.selection import ArticulationView


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def _sum_bias_forces_to_worlds(
    gravity_force: wp.array[float],
    coriolis_force: wp.array[float],
    dofs_per_world: int,
    # outputs
    bias_forces: wp.array2d[float],
):
    """Fold the flat inverse-dynamics bias ``g(q) + C(q,q̇)q̇`` into the controller's
    per-world ``(world, dof)`` layout. Valid when every world owns a contiguous,
    equal-width DOF block (one articulation per world, no padding)."""
    w, j = wp.tid()
    idx = w * dofs_per_world + j
    bias_forces[w, j] = gravity_force[idx] + coriolis_force[idx]


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.action_delay_frames = 25
        self.action_delay = self.action_delay_frames * self.frame_dt
        self.sim_time = 0.0
        self.frame = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.robot_contact_mu = 1.5
        self.gripper_activation_scale = 0.04

        # Stable-PD force-control gains for the 7 arm joints plus two
        # Franka finger sliders.  UIPC consumes these through
        # ``control.joint_f`` because every robot DOF is set to EFFORT mode.
        # Gains tuned below UIPC-ABD stability limit (see UR10 force example
        # notes on implicit-solve chatter at high kp). Proximal joints carry
        # more inertia so keep higher kp; distal wrist DOFs are heavily
        # reduced to stop the implicit-solve oscillation at ±effort limits.
        self.stable_pd_kp = np.array([350.0, 350.0, 300.0, 250.0, 150.0, 120.0, 70.0, 700.0, 700.0], dtype=np.float32)
        self.stable_pd_kd = np.array([42.0, 42.0, 36.0, 30.0, 18.0, 14.0, 8.0, 84.0, 84.0], dtype=np.float32)
        self.stable_pd_max_effort = np.array(
            [300.0, 300.0, 300.0, 300.0, 150.0, 150.0, 150.0, 70.0, 70.0], dtype=np.float32
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        self._build_franka(builder)
        self._build_table(builder)
        self._build_cloth(builder)
        builder.add_ground_plane()
        # builder.gravity = 0.0
        self.model = builder.finalize()
        if hasattr(self.model.uipc, "cloth_model"):
            self.model.uipc.cloth_model[:] = [args.cloth_model] * len(self.model.cloth_ranges)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._setup_ik()

        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/cloth_franka_stable_pd_force",
            dump_enable=True,
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            enable_soft_position_constraint=False,
            auto_sync_inertia=False,
        )
        self.solver.set_contact(enable=True, d_hat=0.001)
        self.solver.configure_scene(
            {
                "linear_system": {"precond": {"mas": {"contact_aware": True}}},
            }
        )
        self.solver.configure_contact_tabular(self._configure_contact_tabular)
        self.solver.initialize(self.state_0)

        self.joint_targets_flat = wp.zeros_like(self.control.joint_target_q)
        wp.copy(self.joint_targets_flat, self.model.joint_q)
        wp.copy(self.control.joint_target_q, self.joint_targets_flat)
        self.control.joint_target_qd.zero_()

        self.robot = ArticulationView(self.model, "fr3")
        self.dofs_per_world = self.robot.joint_dof_count

        self._gravity_np = self.model.gravity.numpy()
        self._body_mass_np = self.model.body_mass.numpy()
        self._J_buf: wp.array | None = None

        self.pd_actuator = next(
            actuator
            for actuator in self.model.actuators
            if isinstance(actuator, _NewtonActuator) and isinstance(actuator.controller, ControllerStablePD)
        )
        self._act_state = self.pd_actuator.state()
        if self._act_state is None or self._act_state.controller_state is None:
            raise ValueError("ControllerStablePD actuator state was not initialized")
        self._H_buf: wp.array | None = None
        # Bias-force output buffers (gravity + Coriolis), sized for the
        # model and reused every control step.
        self._id_gravity_force = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.model.device)
        self._id_coriolis_force = wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.model.device)
        self.home_q = self.robot.get_attribute("joint_q", self.state_0).numpy()[0, 0].copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.0, 0.4, 0.9), pitch=-30.0, yaw=-145.0)
        self.viewer._paused = True  # start paused to allow shader cache to populate before starting the sim

    def _build_franka(self, builder: newton.ModelBuilder) -> None:
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-0.5, -0.5, 0.0),
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,  # URDF is in meters, scale to cm
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
        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8
        builder.joint_q[:9] = [
            *init_q,
            clamp_open_activation_val * self.gripper_activation_scale,
            clamp_open_activation_val * self.gripper_activation_scale,
        ]
        for d in range(9):
            builder.joint_target_q[d] = builder.joint_q[d]
            builder.joint_target_qd[d] = 0.0
            builder.joint_target_ke[d] = 100.0 if d < 7 else self.stable_pd_kp[d]
            builder.joint_target_kd[d] = 10.0 if d < 7 else self.stable_pd_kd[d]
            builder.joint_target_mode[d] = int(JointTargetMode.EFFORT)
            builder.add_actuator(
                ControllerStablePD,
                index=d,
                kp=float(self.stable_pd_kp[d]),
                kd=float(self.stable_pd_kd[d]),
                num_worlds=1,
                clamping=[(ClampingMaxEffort, {"max_effort": float(self.stable_pd_max_effort[d])})],
            )

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
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith("/fr3_link7"))
        self.finger_idx0 = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith("/fr3_finger_joint1"))
        self.finger_idx1 = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith("/fr3_finger_joint2"))
        self.endeffector_offset = wp.transform(
            [
                0.0,
                0.0,
                22.0,
            ],
            wp.quat_identity(),
        )

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
        builder.add_shape_box(table_body, hx=0.40, hy=0.40, hz=0.1, cfg=cfg, label="table")

    def _build_cloth(self, builder: newton.ModelBuilder) -> None:
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        vertices: list[vec3f] = [wp.vec3(v) for v in shirt_mesh.vertices]
        builder.add_cloth_mesh(
            vertices=vertices,
            indices=shirt_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 0.70, 0.45),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            scale=0.01,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.5e-6,
            edge_ke=5,
            edge_kd=1e-2,
            particle_radius=0.0005,
        )
        builder.color()

    def _setup_ik(self) -> None:
        self.ee_index = self.endeffector_id
        ee_tf = self.state_0.body_q.numpy()[self.ee_index]
        self.n_coords = self.model.joint_coord_count
        self.initial_target = np.array(
            [
                float(ee_tf[0]),
                float(ee_tf[1]),
                float(ee_tf[2]),
                float(ee_tf[3]),
                float(ee_tf[4]),
                float(ee_tf[5]),
                float(ee_tf[6]),
                0.8,
            ],
            dtype=np.float32,
        )

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
        self.finger_pos_buf = wp.zeros(1, dtype=float)
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def _target_at_time(self, time: float, is_delayed: bool) -> np.ndarray:
        if is_delayed:
            return self.initial_target

        time -= self.action_delay
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

    def _solve_ik_and_push_control(self, time: float, is_delayed: bool) -> None:
        if is_delayed:
            self.robot.set_attribute("joint_target_q", self.control, self.home_q.reshape(1, 1, self.dofs_per_world))
            self.control.joint_target_qd.zero_()
            return

        target = self._target_at_time(time, is_delayed)
        self.pos_obj.set_target_position(0, wp.vec3(*target[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target[3:7].tolist()))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)
        self.finger_pos_buf.fill_(float(target[-1]) * self.gripper_activation_scale)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.joint_q_ik, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
        )
        wp.copy(self.joint_targets_flat, self.joint_q_ik, dest_offset=0, src_offset=0, count=self.n_coords)
        wp.copy(self.control.joint_target_q, self.joint_targets_flat)
        self.control.joint_target_qd.zero_()

    def _compute_gravity_feedforward(self) -> np.ndarray:
        """Compute Jacobian-transpose gravity compensation for the Franka DOFs."""
        self._J_buf = newton.eval_jacobian(self.model, self.state_0, self._J_buf)
        if self._J_buf is None:
            raise ValueError("eval_jacobian unexpectedly returned None for the Franka articulation")

        J_np = self._J_buf.numpy()[0]
        gravity = self._gravity_np[0]
        body_mass = self._body_mass_np
        joint_child = self.model.joint_child.numpy()
        articulation_start = self.model.articulation_start.numpy()

        joint_start = int(articulation_start[0])
        joint_end = int(articulation_start[1])
        dof_count = self.model.joint_dof_count
        tau_g = np.zeros(dof_count, dtype=np.float32)
        for local_link_idx, joint_idx in enumerate(range(joint_start, joint_end)):
            body_idx = int(joint_child[joint_idx])
            mass = float(body_mass[body_idx])
            if mass <= 0.0:
                continue
            J_lin = J_np[6 * local_link_idx : 6 * local_link_idx + 3, :dof_count]
            tau_g -= J_lin.T @ (mass * gravity)
        return tau_g

    def _apply_stable_pd_force_control(self) -> None:
        self.control.joint_f.zero_()
        tau_g = self._compute_gravity_feedforward()
        self._H_buf = newton.eval_mass_matrix(self.model, self.state_0, H=self._H_buf)
        if self._H_buf is None:
            raise ValueError("eval_mass_matrix unexpectedly returned None for the Franka articulation")
        ctrl_state = self._act_state.controller_state
        ctrl_state.mass_matrix.assign(self._H_buf)
        # Tan 2011 stable-PD needs gravity on BOTH sides of the implicit solve:
        #   1. bias_forces = C(q,q̇)·q̇ + g(q) at q̈ = 0; without it the -kd·q̈·dt
        #      damping term is wrong and the wrist DOFs explode. eval_inverse_dynamics_passive
        #      returns gravity and Coriolis as separate flat buffers, summed below.
        #   2. feedforward joint_act = tau_g so the static effort holds pose at q̈→0.
        # (eval_inverse_dynamics_passive reads state_0.body_q, kept consistent by UIPC readback.)
        newton.eval_inverse_dynamics_passive(
            self.model, self.state_0, gravity_force=self._id_gravity_force, coriolis_force=self._id_coriolis_force
        )
        n = ctrl_state.bias_forces.shape[1]
        wp.launch(
            _sum_bias_forces_to_worlds,
            dim=(1, n),
            inputs=[self._id_gravity_force, self._id_coriolis_force, n],
            outputs=[ctrl_state.bias_forces],
            device=self.model.device,
        )
        self.control.joint_act.assign(tau_g)
        self.pd_actuator.step(
            sim_state=self.state_0,
            sim_control=self.control,
            current_act_state=self._act_state,
            next_act_state=self._act_state,
            dt=self.sim_dt,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._solve_ik_and_push_control(self.sim_time, self.frame < self.action_delay_frames)
            self._apply_stable_pd_force_control()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1

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
    parser.set_defaults(num_frames=3850)
    parser.add_argument(
        "--cloth-model",
        default="strain_limiting_baraff_witkin",
        choices=("strain_limiting_baraff_witkin", "strain_limiting", "neo_hookean"),
        help="UIPC cloth membrane model.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
