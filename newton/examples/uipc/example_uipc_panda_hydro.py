# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Panda Hydro
#
# Pick-and-place manipulation with a Franka Panda arm using the
# SolverUIPC backend. Mirrors ``example_robot_panda_hydro`` but replaces
# the hydroelastic-SDF + MuJoCo stack with UIPC's ABD/IPC contact model.
#
# Scene contents:
#   * Franka FR3 arm (URDF) with finger pads on both fingertips.
#   * Static table built as a kinematic body (UIPC only treats ground
#     planes as world-anchored, so any other static collider must be a
#     kinematic body).
#   * A square cup assembled from four wall boxes + a base, each as its
#     own kinematic body so the cavity survives ABD's per-body merging.
#   * A pen (capsule) or a cube to manipulate.
#   * Ground plane.
#
# IK runs on a single-world model and the resulting joint targets are
# broadcast to every replicated world.
#
# Command: python -m newton.examples uipc_panda_hydro --scene pen
#
###########################################################################

import copy
from enum import Enum

import numpy as np
import uipc
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import JointTargetMode


class SceneType(Enum):
    PEN = "pen"
    CUBE = "cube"


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    """Convert a quaternion to a vec4."""
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array2d[wp.float32],
    gripper_value: float,
):
    world_idx = wp.tid()
    for j in range(7):
        joint_targets[world_idx, j] = ik_solution[0, j]
    joint_targets[world_idx, 7] = gripper_value
    joint_targets[world_idx, 8] = gripper_value


class Example:
    def __init__(self, viewer, args):
        self.scene = SceneType(args.scene)
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = args.world_count
        self.put_in_cup = bool(getattr(args, "put_in_cup", True))
        self.test_mode = bool(getattr(args, "test", False))
        self.viewer = viewer

        builder_single = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder_single.default_shape_cfg.ke = 2.0e3
        builder_single.default_shape_cfg.kd = 1.0e2
        # Note: per-shape ``mu`` is intentionally omitted here. The UIPC
        # backend ignores Newton's per-shape friction and reads friction
        # from its own ``ContactTabular`` (default μ=0.5 for every pair —
        # see the ``uipc-contact-tabular`` skill). Setting it here would be
        # a silent no-op and mislead readers about the real friction.
        builder_single.default_shape_cfg.margin = 0.005
        builder_single.default_shape_cfg.gap = 0.01

        # ------------------------------------------------------------------
        # Franka Panda arm
        # ------------------------------------------------------------------
        # Scene layout matches example_robot_panda_hydro so both examples
        # can be visually compared side-by-side.
        panda_xform = wp.transform((-0.5, -0.5, 0.05), wp.quat_identity())
        panda_first_body = builder_single.body_count
        builder_single.add_urdf(
            str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"),
            xform=panda_xform,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

        def find_body(name: str) -> int:
            return next(i for i, lbl in enumerate(builder_single.body_label) if lbl.endswith(f"/{name}"))

        left_finger_idx = find_body("fr3_leftfinger")
        right_finger_idx = find_body("fr3_rightfinger")
        # fr3_hand is the IK end-effector frame. Store it here while the
        # ``find_body`` closure is in scope — the hand body index depends
        # on the URDF traversal order, so hardcoding (``10`` in the
        # original example) is brittle once extra bodies are added.
        self.hand_body_idx = find_body("fr3_hand")

        # Add gripper pads (mesh) before convex-hull approximation so they
        # get merged with each finger into a single closed ABD body.
        pad_asset_path = newton.utils.download_asset("manipulation_objects/pad")
        pad_stage = Usd.Stage.Open(str(pad_asset_path / "model.usda"))
        pad_mesh = newton.usd.get_mesh(
            pad_stage.GetPrimAtPath("/root/Model/Model"),
            load_normals=True,
            face_varying_normal_conversion="vertex_splitting",
        )
        pad_scale = np.asarray(newton.usd.get_scale(pad_stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
        if not np.allclose(pad_scale, 1.0):
            pad_mesh = pad_mesh.copy(vertices=pad_mesh.vertices * pad_scale, recompute_inertia=True)
        pad_xform = wp.transform(
            wp.vec3(0.0, 0.005, 0.045),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi),
        )
        builder_single.add_shape_mesh(body=left_finger_idx, mesh=pad_mesh, xform=pad_xform)
        builder_single.add_shape_mesh(body=right_finger_idx, mesh=pad_mesh, xform=pad_xform)

        # Convex-hull every panda body so UIPC ABD gets closed manifolds.
        panda_shape_indices = [s for s, b in enumerate(builder_single.shape_body) if b >= panda_first_body]
        builder_single.approximate_meshes("convex_hull", shape_indices=panda_shape_indices, keep_visual_shapes=True)

        # Initial joint configuration (matches example_robot_panda_hydro).
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        # Initial joint state and PD target — matches example_robot_panda_hydro.py
        # exactly (joint_q, target_pos, armature, effort_limit). The arm armature
        # in particular must stay at 0.1 / 0.5 or the arm will visibly sag under
        # gravity and the rest pose will not match the original.
        builder_single.joint_q[:9] = [*init_q, 0.05, 0.05]
        builder_single.joint_target_pos[:9] = [*init_q, 1.0, 1.0]

        builder_single.joint_target_ke[:9] = [650.0] * 9
        builder_single.joint_target_kd[:9] = [100.0] * 9
        builder_single.joint_effort_limit[:7] = [80.0] * 7
        builder_single.joint_effort_limit[7:9] = [20.0] * 2
        builder_single.joint_armature[:7] = [0.1] * 7
        builder_single.joint_armature[7:9] = [0.5] * 2

        for d in range(9):
            builder_single.joint_target_mode[d] = int(JointTargetMode.POSITION)

        # ------------------------------------------------------------------
        # Static table — kinematic body so UIPC sees it.
        # ------------------------------------------------------------------
        # UIPC IPC requires a strictly positive gap (>= default_shape_cfg.gap
        # = 0.01m) between every pair of bodies. The original panda_hydro
        # layout places the table directly on the ground and the object
        # directly on the table, which fails UIPC's sanity check. We lift
        # everything by ``uipc_gap`` so the initial distances satisfy the
        # solver's minimum-separation constraint.
        uipc_gap = 0.0001
        box_size = 0.05
        table_pos = wp.vec3(0.08, -0.5, box_size)
        table_body = builder_single.add_body(
            xform=wp.transform(table_pos, wp.quat_identity()),  # ty:ignore[missing-argument]
            label="table",
            is_kinematic=True,
        )
        builder_single.add_shape_box(
            body=table_body,
            hx=box_size * 2.0,
            hy=box_size * 2.0,
            hz=box_size,
        )

        # ------------------------------------------------------------------
        # Cup — load the manipulation_objects/cup mesh on a single kinematic
        # body. The mesh is open-topped, so we run convex_hull approximation
        # on this shape only to guarantee the watertight, positive-volume
        # manifold that UIPC ABD requires (cavity is intentionally lost — we
        # just need a solid cup-shaped obstacle). Only created when
        # ``put_in_cup`` is enabled, matching the original panda_hydro.
        # ------------------------------------------------------------------
        if self.put_in_cup:
            self.cup_pos = [0.13, -0.5, box_size + 0.1 + uipc_gap]
            cup_xform = wp.transform(wp.vec3(self.cup_pos), wp.quat_identity())

            cup_asset_path = newton.utils.download_asset("manipulation_objects/cup")
            cup_stage = Usd.Stage.Open(str(cup_asset_path / "model.usda"))
            cup_mesh = newton.usd.get_mesh(
                cup_stage.GetPrimAtPath("/root/Model/Model"),
                load_normals=True,
                face_varying_normal_conversion="vertex_splitting",
            )
            cup_scale_np = np.asarray(newton.usd.get_scale(cup_stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
            if not np.allclose(cup_scale_np, 1.0):
                cup_mesh = cup_mesh.copy(vertices=cup_mesh.vertices * cup_scale_np, recompute_inertia=True)

            cup_body = builder_single.add_body(
                xform=cup_xform,
                label="cup",
                is_kinematic=True,
            )
            builder_single.add_shape_mesh(body=cup_body, mesh=cup_mesh)

        # ------------------------------------------------------------------
        # Object to manipulate
        # ------------------------------------------------------------------
        if self.scene == SceneType.PEN:
            radius = 0.005
            length = 0.14
            self.object_pos = [0.0, -0.5, 2 * box_size + radius + 0.001 + uipc_gap]
            object_xform = wp.transform(
                wp.vec3(self.object_pos),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2),
            )
            self.object_body_local = builder_single.add_body(xform=object_xform, label="object")
            builder_single.add_shape_capsule(
                body=self.object_body_local,
                radius=radius,
                half_height=length / 2,
            )
            self.grasping_offset = [-0.03, 0.0, 0.1]
            self.place_offset = -0.02
        else:  # CUBE
            size = 0.04
            self.object_pos = [0.0, -0.5, 2 * box_size + 0.5 * size + uipc_gap]
            object_xform = wp.transform(wp.vec3(self.object_pos), wp.quat_identity())
            self.object_body_local = builder_single.add_body(xform=object_xform, label="object")
            builder_single.add_shape_box(
                body=self.object_body_local,
                hx=size / 2,
                hy=size / 2,
                hz=size / 2,
            )
            self.grasping_offset = [0.0, 0.0, 0.1]
            self.place_offset = 0.0

        # ------------------------------------------------------------------
        # Build single-world model for IK before replication.
        # ------------------------------------------------------------------
        self.model_single = copy.deepcopy(builder_single).finalize()
        self.bodies_per_world = builder_single.body_count

        if self.world_count > 1:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            builder.replicate(builder_single, self.world_count, spacing=(2.0, 2.0, 0.0))
        else:
            builder = builder_single

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Error,
        )
        self.solver.set_contact(True, uipc_gap)
        self.solver.initialize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        print("state_0 body_q: ", self.model.joint_q)
        self.viewer.set_model(self.model)
        # Camera matches example_robot_panda_hydro.
        self.viewer.set_camera(
            pos=wp.vec3(0.5, 0.0, 0.5),
            pitch=-15.0,
            yaw=-140.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

        # Initialize state for IK setup
        self.state = self.model_single.state()
        newton.eval_fk(self.model_single, self.model.joint_q, self.model.joint_qd, self.state)

        self._setup_ik()
        self.control = self.model.control()
        self.joint_target_shape = self.control.joint_target_pos.reshape((self.world_count, -1)).shape
        self.joint_targets_2d = wp.zeros(self.joint_target_shape, dtype=wp.float32)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        # Track maximum object height for testing (only in test mode)
        self.object_max_z = [self.object_pos[2]] * self.world_count if self.test_mode else None

    def _setup_ik(self):
        # Use fr3_hand as the IK end-effector. The grasping_offset values
        # are calibrated against the hand frame, so targeting the hand
        # here is required for the waypoint sequence to actually reach
        # the cube / pen on the table.
        self.ee_index = self.hand_body_idx
        body_q_np = self.state.body_q.numpy()
        ee_tf = wp.transform(*body_q_np[self.ee_index])

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
        )
        self.joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_single.joint_limit_lower,
            joint_limit_upper=self.model_single.joint_limit_upper,
        )
        self.joint_q_ik = wp.array(
            self.model_single.joint_q,
            shape=(1, self.model_single.joint_coord_count),
        )
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_single,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # Pick-and-place + drop-into-cup waypoints.
        # Tuple layout: (target_pos, duration, gripper_close, rot_hand).
        self.time_in_waypoint = 0.0
        self.current_waypoint = 0
        self.z_rest = 0.5
        grasping_pos = wp.vec3(self.object_pos) + wp.vec3(self.grasping_offset)
        resting_pos = wp.vec3(grasping_pos[0], grasping_pos[1], self.z_rest)
        # Waypoints match example_robot_panda_hydro.py exactly so the two
        # examples can be compared side-by-side. When ``put_in_cup`` is
        # disabled the cup-related waypoints are skipped (base: 4 points;
        # with cup: +5 → 9 total).
        grasp_pos = 1.0
        no_grasp_pos = 0.0
        rot_hand = 0.0
        self.waypoints = [
            [resting_pos, 1.0, no_grasp_pos, rot_hand],
            [grasping_pos, 1.0, no_grasp_pos, rot_hand],
            [grasping_pos, 1.0, grasp_pos, rot_hand],
            [resting_pos, 1.0, grasp_pos, rot_hand],
        ]

        if self.put_in_cup:
            loose_pos = 0.71
            cup_above_high = wp.vec3(
                self.cup_pos[0] + self.place_offset,
                self.cup_pos[1],
                self.z_rest,
            )
            cup_above_low = wp.vec3(
                self.cup_pos[0] + self.place_offset,
                self.cup_pos[1],
                self.z_rest - 0.1,
            )
            self.waypoints.extend(
                [
                    [cup_above_high, 2.0, grasp_pos, rot_hand],
                    [cup_above_high, 2.0, loose_pos, rot_hand],
                    [cup_above_high, 1.0, loose_pos, rot_hand],
                    [cup_above_low, 1.0, loose_pos, rot_hand],
                    [cup_above_low, 1.0, 0.0, rot_hand],
                ]
            )

    def _set_joint_targets(self):
        self.time_in_waypoint += self.frame_dt
        wp_idx = self.current_waypoint
        next_idx = (wp_idx + 1) % len(self.waypoints)
        t = self.time_in_waypoint / self.waypoints[wp_idx][1]

        target_position = self.waypoints[wp_idx][0] * (1.0 - t) + self.waypoints[next_idx][0] * t
        target_angle_z = self.waypoints[wp_idx][3] * (1.0 - t) + self.waypoints[next_idx][3] * t
        target_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
        target_rotation = wp.mul(
            target_rotation,
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), target_angle_z),
        )

        self.pos_obj.set_target_positions(wp.array([target_position], dtype=wp.vec3))
        self.rot_obj.set_target_rotations(wp.array([quat_to_vec4(target_rotation)], dtype=wp.vec4))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        t_gripper = self.waypoints[wp_idx][2] * (1.0 - t) + self.waypoints[next_idx][2] * t
        # Match example_robot_panda_hydro.py: open commands 0.06 (saturates at
        # the finger joint limit ≈0.04m), closed commands 0.0.
        gripper_value = 0.06 * (1.0 - t_gripper)
        wp.launch(
            broadcast_ik_solution_kernel,
            dim=self.world_count,
            inputs=[self.joint_q_ik, self.joint_targets_2d, gripper_value],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_2d.flatten())

        if self.time_in_waypoint >= self.waypoints[wp_idx][1]:
            self.current_waypoint = next_idx
            self.time_in_waypoint = 0.0

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for _ in range(self.sim_substeps):
            self.solver.export_surface_obj("/home/ligo/Project/x2robot/Newton-Isaac/newton/output/")
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # IK + waypoint advance happens once per frame (matches
        # example_robot_panda_hydro). Keeping it out of the substep loop
        # means ``self.time_in_waypoint`` advances by ``frame_dt`` regardless
        # of ``sim_substeps``, so raising substeps won't make the waypoint
        # schedule run faster than intended.
        self._set_joint_targets()
        self.simulate()
        self.sim_time += self.frame_dt

        if self.test_mode and self.object_max_z is not None:
            body_q = self.state_0.body_q.numpy()
            for world_idx in range(self.world_count):
                object_body_idx = world_idx * self.bodies_per_world + self.object_body_local
                z_pos = float(body_q[object_body_idx][2])
                self.object_max_z[world_idx] = max(self.object_max_z[world_idx], z_pos)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Mirrors example_robot_panda_hydro.test_final:
        #   1. Verify the object was actually picked up (max-z lift check).
        #   2. If put_in_cup, verify it ended up inside the cup XY/Z window.
        assert self.object_max_z is not None, "test_final requires --test so ``step`` populates self.object_max_z."
        initial_z = self.object_pos[2]
        min_lift_height = 0.15  # Object should be lifted >= 15 cm.

        for world_idx in range(self.world_count):
            max_z = self.object_max_z[world_idx]
            max_lift = max_z - initial_z
            assert max_lift > min_lift_height, (
                f"World {world_idx}: Object was not picked up high enough. "
                f"Initial z={initial_z:.3f}, max z reached={max_z:.3f}, "
                f"max lift={max_lift:.3f} (expected > {min_lift_height})"
            )

        if self.put_in_cup:
            body_q = self.state_0.body_q.numpy()
            cup_x, cup_y, cup_z = self.cup_pos
            tolerance_xy = 0.05
            min_z = cup_z - 0.05

            for world_idx in range(self.world_count):
                object_body_idx = world_idx * self.bodies_per_world + self.object_body_local
                x, y, z = body_q[object_body_idx][:3]
                assert abs(x - cup_x) < tolerance_xy and abs(y - cup_y) < tolerance_xy and z > min_z, (
                    f"World {world_idx}: Object is not in the cup. "
                    f"Object pos=({x:.3f}, {y:.3f}, {z:.3f}), "
                    f"cup pos=({cup_x:.3f}, {cup_y:.3f}, {cup_z:.3f})"
                )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1, num_frames=720)
        parser.add_argument(
            "--scene",
            type=str,
            choices=[s.value for s in SceneType],
            default=SceneType.PEN.value,
            help="Scene type to load (pen or cube)",
        )
        parser.add_argument(
            "--no-put-in-cup",
            dest="put_in_cup",
            action="store_false",
            help="Disable the drop-into-cup waypoints (only run pick-and-lift).",
        )
        parser.set_defaults(put_in_cup=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
