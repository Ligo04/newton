# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Brick Stacking
#
# UIPC port of ``example_brick_stacking``. A Franka FR3 arm picks up
# colored LEGO bricks from a table and stacks them using IPC-based
# contact. Bricks use the full LEGO mesh geometry (shell + studs +
# interior tubes). The arm is controlled with IK and a Warp-kernel
# finite-state machine that sequences approach, grasp, lift, move, place
# and release for each brick.
#
# Command: python -m newton.examples uipc_brick_stacking
#
###########################################################################

import argparse
import enum

import numpy as np
import uipc
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode

PITCH = 0.008
BRICK_HEIGHT = 0.0096

BRICK_SCALE = 1.0
BRICK_DENSITY = 565.0  # ABS plastic [kg/m³]
BRICK_ABD_KAPPA = 10.0 * uipc.unit.GPa


BRICK_HX = 2.0 * PITCH
BRICK_HY = PITCH
BRICK_HZ = 0.5 * BRICK_HEIGHT

STUD_RADIUS = 0.0024
STUD_HEIGHT = 0.0017
WALL_THICKNESS = 0.0012
TOP_THICKNESS = 0.001
TUBE_OUTER_RADIUS = 0.002755
TUBE_HEIGHT = BRICK_HEIGHT - TOP_THICKNESS
CYLINDER_SEGMENTS = 48

# Gripper finger positions [m]
GRIPPER_OPEN = 0.5 * (2 * PITCH * BRICK_SCALE + 0.004)
GRIPPER_RELEASE = 0.5 * (2 * PITCH * BRICK_SCALE * 2.0)
GRIPPER_CLOSED: int | float = 0.5 * (2 * PITCH * BRICK_SCALE - 0.003)


def _cylinder_mesh(radius, height, segments, cx=0.0, cy=0.0, cz=0.0, bottom_cap=True):
    n = segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    ring_x = cx + radius * cos_a
    ring_y = cy + radius * sin_a

    verts, faces = [], []

    side_bot = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
    side_top = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    verts.append(side_bot)
    verts.append(side_top)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + j, n + i])
        faces.append([i, j, n + j])

    off_top = 2 * n
    cap_top_ring = np.column_stack([ring_x, ring_y, np.full(n, cz + height)]).astype(np.float32)
    cap_top_center = np.array([[cx, cy, cz + height]], dtype=np.float32)
    verts.append(cap_top_ring)
    verts.append(cap_top_center)
    tc = off_top + n
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc, off_top + i, off_top + j])

    if bottom_cap:
        off_bot = off_top + n + 1
        cap_bot_ring = np.column_stack([ring_x, ring_y, np.full(n, cz)]).astype(np.float32)
        cap_bot_center = np.array([[cx, cy, cz]], dtype=np.float32)
        verts.append(cap_bot_ring)
        verts.append(cap_bot_center)
        bc = off_bot + n
        for i in range(n):
            j = (i + 1) % n
            faces.append([bc, off_bot + j, off_bot + i])

    return np.vstack(verts), np.array(faces, dtype=np.int32)


def _combine_meshes(mesh_list):
    all_v, all_f, off = [], [], 0
    for v, f in mesh_list:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v).astype(np.float32), np.vstack(all_f).astype(np.int32)


def _make_shell_mesh(nx, ny):
    ox = nx * PITCH / 2.0
    oy = ny * PITCH / 2.0
    inx = ox - WALL_THICKNESS
    iny = oy - WALL_THICKNESS
    H = BRICK_HEIGHT
    T = TOP_THICKNESS

    v = np.array(
        [
            [-ox, -oy, 0],
            [+ox, -oy, 0],
            [+ox, +oy, 0],
            [-ox, +oy, 0],
            [-ox, -oy, H],
            [+ox, -oy, H],
            [+ox, +oy, H],
            [-ox, +oy, H],
            [-inx, -iny, 0],
            [+inx, -iny, 0],
            [+inx, +iny, 0],
            [-inx, +iny, 0],
            [-inx, -iny, H - T],
            [+inx, -iny, H - T],
            [+inx, +iny, H - T],
            [-inx, +iny, H - T],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [4, 5, 6],
            [4, 6, 7],
            [0, 9, 1],
            [0, 8, 9],
            [1, 10, 2],
            [1, 9, 10],
            [2, 11, 3],
            [2, 10, 11],
            [3, 8, 0],
            [3, 11, 8],
            [9, 8, 12],
            [9, 12, 13],
            [11, 10, 14],
            [11, 14, 15],
            [8, 11, 15],
            [8, 15, 12],
            [10, 9, 13],
            [10, 13, 14],
            [12, 15, 14],
            [12, 14, 13],
        ],
        dtype=np.int32,
    )
    return v, f


def _make_brick_mesh(nx=4, ny=2, tube_outer_radius=TUBE_OUTER_RADIUS):
    shell_v, shell_f = _make_shell_mesh(nx, ny)
    stud_meshes = []
    for i in range(nx):
        for j in range(ny):
            sx = (i - (nx - 1) / 2.0) * PITCH
            sy = (j - (ny - 1) / 2.0) * PITCH
            stud_meshes.append(
                _cylinder_mesh(
                    STUD_RADIUS, STUD_HEIGHT, CYLINDER_SEGMENTS, cx=sx, cy=sy, cz=BRICK_HEIGHT, bottom_cap=True
                )
            )
    tube_meshes = []
    if ny == 2:
        for i in range(nx - 1):
            tx = (i - (nx - 2) / 2.0) * PITCH
            tube_meshes.append(_cylinder_mesh(tube_outer_radius, TUBE_HEIGHT, CYLINDER_SEGMENTS, cx=tx, cy=0.0, cz=0.0))
    return _combine_meshes([(shell_v, shell_f), *stud_meshes, *tube_meshes])


class TaskType(enum.IntEnum):
    APPROACH = 0
    REFINE_APPROACH = 1
    GRASP = 2
    LIFT = 3
    MOVE_TO_DROP_OFF = 4
    REFINE_DROP_OFF = 5
    RELEASE = 6
    HOME = 7


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_limits: wp.array[float],
    task_pick_body: wp.array[int],
    task_drop_body: wp.array[int],
    task_drop_layer: wp.array[int],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    offset_approach: wp.vec3,
    offset_lift: wp.vec3,
    grasp_z_offset: wp.vec3,
    drop_z_offset: wp.vec3,
    brick_stack_height: float,
    home_pos: wp.vec3,
    task_init_body_q: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ee_index: int,
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_interp: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_interp: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    tid = wp.tid()

    idx = task_idx[tid]
    task = task_schedule[idx]
    time_limit = task_time_limits[idx]
    pick_body = task_pick_body[idx]
    drop_body = task_drop_body[idx]
    drop_layer = task_drop_layer[idx]

    task_time_elapsed[tid] += task_dt
    t_lin = wp.min(1.0, task_time_elapsed[tid] / time_limit)
    # Smoothstep easing.
    t = t_lin * t_lin * (3.0 - 2.0 * t_lin)

    ee_pos_prev = wp.transform_get_translation(task_init_body_q[ee_index])
    ee_quat_prev = wp.transform_get_rotation(task_init_body_q[ee_index])
    ee_quat_down = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)

    pick_pos = wp.transform_get_translation(task_init_body_q[pick_body])
    pick_quat = wp.transform_get_rotation(task_init_body_q[pick_body])

    drop_pos = wp.transform_get_translation(task_init_body_q[drop_body])
    drop_quat = wp.transform_get_rotation(task_init_body_q[drop_body])
    layer_offset = wp.float32(drop_layer) * brick_stack_height * wp.vec3(0.0, 0.0, 1.0)
    ee_quat_drop = ee_quat_down * wp.quat_inverse(drop_quat)

    t_gripper = 0.0
    target_pos = home_pos
    target_quat = ee_quat_down

    if task == TaskType.APPROACH.value:
        target_pos = pick_pos + offset_approach
        target_quat = ee_quat_down * wp.quat_inverse(pick_quat)
    elif task == TaskType.REFINE_APPROACH.value:
        target_pos = pick_pos + grasp_z_offset
        target_quat = ee_quat_prev
    elif task == TaskType.GRASP.value:
        target_pos = ee_pos_prev
        target_quat = ee_quat_prev
        t_gripper = t
    elif task == TaskType.LIFT.value:
        target_pos = ee_pos_prev + offset_lift
        target_quat = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.MOVE_TO_DROP_OFF.value:
        target_pos = drop_pos + layer_offset + offset_approach
        target_quat = ee_quat_drop
        t_gripper = 1.0
    elif task == TaskType.REFINE_DROP_OFF.value:
        target_pos = drop_pos + layer_offset + grasp_z_offset + drop_z_offset
        target_quat = ee_quat_drop
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        target_pos = drop_pos + layer_offset + grasp_z_offset + drop_z_offset
        target_quat = ee_quat_drop
        t_gripper = 1.0 - t
    elif task == TaskType.HOME.value:
        target_pos = home_pos
        target_quat = ee_quat_down

    ee_pos_target[tid] = target_pos
    interp_pos = ee_pos_prev * (1.0 - t) + target_pos * t

    # XY alignment correction for IK convergence.
    ee_pos_actual = wp.transform_get_translation(body_q[ee_index])
    xy_err = wp.vec3(
        interp_pos[0] - ee_pos_actual[0],
        interp_pos[1] - ee_pos_actual[1],
        0.0,
    )
    use_align = 1.0
    if task == TaskType.APPROACH.value or task == TaskType.HOME.value:
        use_align = 0.0
    ee_pos_interp[tid] = interp_pos + use_align * xy_err

    ee_rot_target[tid] = target_quat[:4]
    ee_rot_interp[tid] = wp.quat_slerp(ee_quat_prev, target_quat, t)[:4]

    gripper_open = GRIPPER_OPEN
    if task == TaskType.RELEASE.value or task == TaskType.HOME.value:
        gripper_open = GRIPPER_RELEASE
    gripper_pos = gripper_open * (1.0 - t_gripper) + GRIPPER_CLOSED * t_gripper
    gripper_target[tid, 0] = gripper_pos
    gripper_target[tid, 1] = gripper_pos


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    ee_index: int,
    # outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_init_body_q: wp.array[wp.transform],
):
    tid = wp.tid()
    idx = task_idx[tid]
    time_limit = task_time_limits[idx]

    ee_pos_current = wp.transform_get_translation(body_q[ee_index])
    ee_quat_current = wp.transform_get_rotation(body_q[ee_index])

    pos_err = wp.length(ee_pos_target[tid] - ee_pos_current)

    ee_quat_tgt = wp.quaternion(ee_rot_target[tid][:3], ee_rot_target[tid][3])
    quat_rel = ee_quat_current * wp.quat_inverse(ee_quat_tgt)
    rot_err = wp.degrees(2.0 * wp.acos(wp.clamp(wp.abs(quat_rel[3]), 0.0, 1.0)))

    if (
        task_time_elapsed[tid] >= time_limit
        and pos_err < 0.006
        and rot_err < 3.0
        and task_idx[tid] < wp.len(task_time_limits) - 1
    ):
        task_idx[tid] += 1
        task_time_elapsed[tid] = 0.0
        num_bodies = wp.len(body_q)
        for i in range(num_bodies):
            task_init_body_q[i] = body_q[i]


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)
        self.enable_contact = getattr(args, "enable_contact", True)

        self.uipc_gap = 0.001
        self.ee_index = -1
        self.brick_count = 3

        self.table_height = 0.1
        self.table_hx = 0.4
        self.table_hy = 0.4
        self.table_hz = 0.5 * self.table_height
        self.table_pos = wp.vec3(0.0, -0.5, self.table_hz)
        self.table_top_center = self.table_pos + wp.vec3(0.0, 0.0, 0.5 * self.table_height)
        self.table_top_z = float(self.table_top_center[2])
        self.robot_base_pos = self.table_top_center + wp.vec3(-0.5, 0.0, 0.0)

        self.brick_height_scaled = BRICK_HEIGHT * BRICK_SCALE
        self.brick_stack_height = self.brick_height_scaled + self.uipc_gap
        self.brick_width_scaled = 2 * PITCH * BRICK_SCALE
        self.brick_length_scaled = 4 * PITCH * BRICK_SCALE
        # Targets are specified at the Franka hand link. The finger pads sit
        # roughly 58 mm below that link, so the UIPC port uses taller hand-link
        # offsets than the MuJoCo example to keep the initial pose intersection-free.
        self.offset_approach = wp.vec3(0.0, 0.0, 0.025)
        self.offset_lift = wp.vec3(0.0, -0.001, 0.042)
        self.grasp_z_offset = wp.vec3(0.0, 0.0, 0.012)
        self.drop_z_offset = wp.vec3(0.0, 0.0, 0.0)

        self._setup_brick_layout()

        # Build an arm-only IK model first so the simulation starts with the
        # gripper already above the red brick, matching the original example.
        ik_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self._build_franka(ik_builder)
        self.model_ik = ik_builder.finalize()
        self.ee_index = self._find_body(ik_builder, "fr3_hand_tcp")
        self.home_pos = self._get_home_pos(self.model_ik)
        init_joints = self._solve_approach_ik()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        self._build_franka(builder)
        self._configure_franka(builder, init_joints)
        self.ee_index = self._find_body(builder, "fr3_hand_tcp")
        self._add_table(builder)
        self._add_board_floor(builder)
        self._add_bricks(builder)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/brick_stacking",
            dump_enable=True,
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
        )
        self.solver.set_contact(enable=self.enable_contact, d_hat=self.uipc_gap)
        self.solver.configure_contact_tabular(self._configure_contact_tabular)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.solver.initialize(self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        self._setup_ik()
        self._setup_tasks()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "picking"):
            ps = self.viewer.picking.pick_state.numpy()
            ps[0]["pick_stiffness"] = 0.1
            ps[0]["pick_damping"] = 0.01
            self.viewer.picking.pick_state.assign(ps)
        cam_pos = wp.vec3(0.22, -0.5 - 0.18, self.table_top_z + 0.15)
        self.viewer.set_camera(pos=cam_pos, pitch=-30.0, yaw=135.0)
        self.viewer._paused = True

    @classmethod
    def create_parser(cls):
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1800)
        parser.add_argument(
            "--enable-contact",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable UIPC contact handling. Use --no-enable-contact to disable contact.",
        )
        return parser

    def _setup_brick_layout(self):
        sqrt2_2 = float(np.sqrt(2.0) / 2.0)
        self.rot_90z = wp.quat(0.0, 0.0, sqrt2_2, sqrt2_2)
        bh = 0.5 * self.brick_height_scaled

        blue_x = self.table_top_center[0] - 0.05
        blue_y = self.table_top_center[1] - 0.04
        self.board_floor_z = self.table_top_center[2] - 0.8 * self.brick_height_scaled

        # Match the original example's XY layout and position expressions.
        # UIPC attaches the bottom-origin mesh at ``-bh`` so body transforms
        # remain brick centers; add ``bh`` plus ``uipc_gap`` where needed to
        # preserve the original bottom-origin placement.
        self.brick_positions = [
            self.table_top_center + wp.vec3(0.0, 0.06, bh + self.uipc_gap),
            self.table_top_center + wp.vec3(0.05, -0.04, bh + self.uipc_gap),
            wp.vec3(
                blue_x,
                blue_y,
                self.table_top_center[2] + 0.2 * self.brick_height_scaled + bh + self.uipc_gap,
            ),
        ]
        self.brick_rotations = [self.rot_90z, self.rot_90z, wp.quat_identity()]
        self.brick_colors = [(0.8, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.2, 0.8)]
        self.brick_labels = ["brick_red", "brick_green", "brick_blue"]

    def _find_body(self, builder: newton.ModelBuilder, name: str) -> int:
        return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(f"/{name}"))

    def _build_franka(self, builder: newton.ModelBuilder) -> None:
        builder.add_urdf(
            str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"),
            xform=wp.transform(self.robot_base_pos, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )

    def _configure_franka(self, builder: newton.ModelBuilder, arm_q: np.ndarray | list[float]) -> None:
        builder.joint_q[:9] = [*np.asarray(arm_q, dtype=np.float32)[:7].tolist(), GRIPPER_OPEN, GRIPPER_OPEN]
        builder.joint_target_pos[:9] = builder.joint_q[:9]
        builder.joint_target_ke[:9] = [650.0] * 9
        builder.joint_target_kd[:9] = [100.0] * 9
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [20.0] * 2
        builder.joint_armature[:7] = [0.1] * 7
        builder.joint_armature[7:9] = [0.5] * 2

        for d in range(9):
            builder.joint_target_mode[d] = int(JointTargetMode.POSITION)

        non_finger_body_indices = [
            self._find_body(builder, "fr3_leftfinger"),
            self._find_body(builder, "fr3_rightfinger"),
            self._find_body(builder, "fr3_hand"),
        ]
        non_finger_shape_indices = []
        for shape_idx, body_idx in enumerate(builder.shape_body):
            if body_idx not in non_finger_body_indices and builder.shape_type[shape_idx] != newton.GeoType.CONVEX_MESH:
                non_finger_shape_indices.append(shape_idx)

        builder.approximate_meshes(
            method="convex_hull", shape_indices=non_finger_shape_indices, keep_visual_shapes=True
        )

    def _add_table(self, builder: newton.ModelBuilder) -> None:
        self.table_body = builder.add_body(
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            label="table",
            is_kinematic=True,
        )
        builder.add_shape_box(
            body=self.table_body,
            hx=self.table_hx,
            hy=self.table_hy,
            hz=self.table_hz,
            label="table",
            color=(0.8, 0.6, 0.2),
        )

    @staticmethod
    def _build_brick_mesh(color):
        v, f = _make_brick_mesh()
        return newton.Mesh(v, f.flatten(), color=color)

    _brick_mesh_xform = wp.transform(wp.vec3(0.0, 0.0, -BRICK_HZ), wp.quat_identity())

    def _add_board_floor(self, builder: newton.ModelBuilder) -> None:
        """Add eight gray kinematic bricks as a board floor under the stack."""
        board_center = self.brick_positions[2]
        floor_z = self.board_floor_z
        # ``board_floor_z`` follows the original example and denotes the mesh
        # bottom. UIPC bricks attach the bottom-origin mesh with ``-BRICK_HZ``
        # so dynamic body poses stay centered; lift the board body by BRICK_HZ
        # to keep the visible board top and studs above the table.
        floor_center_z = floor_z + BRICK_HZ
        floor_rot = self.rot_90z
        bw = self.brick_width_scaled
        bl = self.brick_length_scaled
        gray_mesh = self._build_brick_mesh((0.35, 0.35, 0.35))
        self.board_floor_bodies = []
        for ix, dx in enumerate((-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw)):
            for iy, dy in enumerate((-0.5 * bl, 0.5 * bl)):
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(float(board_center[0]) + dx, float(board_center[1]) + dy, floor_center_z),
                        floor_rot,
                    ),
                    label=f"board_floor_{ix}_{iy}",
                    is_kinematic=True,
                )
                self.board_floor_bodies.append(body)
                builder.add_shape_mesh(
                    body=body,
                    mesh=gray_mesh,
                    xform=self._brick_mesh_xform,
                    label=f"board_floor_{ix}_{iy}",
                )

    def _add_bricks(self, builder: newton.ModelBuilder) -> None:
        self.brick_bodies = []
        for i in range(self.brick_count):
            body = builder.add_body(
                xform=wp.transform(self.brick_positions[i], self.brick_rotations[i]),
                label=self.brick_labels[i],
                custom_attributes={"uipc:abd_kappa": BRICK_ABD_KAPPA},
            )
            brick_mesh = self._build_brick_mesh(self.brick_colors[i])
            builder.add_shape_mesh(
                body=body,
                mesh=brick_mesh,
                xform=self._brick_mesh_xform,
                cfg=newton.ModelBuilder.ShapeConfig(density=BRICK_DENSITY),
                label=self.brick_labels[i],
            )
            self.brick_bodies.append(body)

    def _configure_contact_tabular(self, contact_tabular, _world_index, ground_elem, env_elem, robo_elem, actor_elem):
        GPa = 1.0e9
        floor_elem = contact_tabular.create("board_floor")
        contact_tabular.insert(floor_elem, actor_elem, 0.5, GPa, True)
        contact_tabular.insert(floor_elem, robo_elem, 0.5, GPa, True)
        contact_tabular.insert(floor_elem, ground_elem, 0.5, GPa, False)
        contact_tabular.insert(floor_elem, env_elem, 0.5, GPa, False)
        contact_tabular.insert(floor_elem, floor_elem, 0.5, GPa, False)
        contact_tabular.insert(env_elem, ground_elem, 0.5, GPa, False)
        return dict.fromkeys(self.board_floor_bodies, floor_elem)

    def _get_home_pos(self, model: newton.Model) -> wp.vec3:
        state_tmp = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_tmp)
        return wp.vec3(*state_tmp.body_q.numpy()[self.ee_index][:3])

    def _solve_approach_ik(self) -> np.ndarray:
        """Solve IK for the approach pose above the red brick."""
        # Use a taller offset for the initial pose to avoid UIPC penetration.
        bh = 0.5 * self.brick_height_scaled
        sqrt2_2 = np.sqrt(2.0) / 2.0

        red_pos = np.array(
            [
                float(self.table_top_center[0]),
                float(self.table_top_center[1]) + 0.06,
                float(self.table_top_center[2]) + bh,
            ]
        )
        target_pos = red_pos + np.array([0.0, 0.0, float(self.offset_approach[2])])

        down = np.array([1.0, 0.0, 0.0, 0.0])
        inv_pick = np.array([0.0, 0.0, -sqrt2_2, sqrt2_2])
        x1, y1, z1, w1 = down
        x2, y2, z2, w2 = inv_pick
        target_quat = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

        ik_dofs = self.model_ik.joint_coord_count
        seed = np.zeros(ik_dofs, dtype=np.float32)
        seed[:7] = [0.0, 0.5, 0.0, -1.5, 0.0, 2.0, 0.78]
        joint_q = wp.array(seed.reshape(1, -1), dtype=wp.float32)

        solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[
                ik.IKObjectivePosition(
                    link_index=self.ee_index,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([target_pos], dtype=wp.vec3),
                ),
                ik.IKObjectiveRotation(
                    link_index=self.ee_index,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([quat_to_vec4(target_quat)], dtype=wp.vec4),
                ),
                ik.IKObjectiveJointLimit(
                    joint_limit_lower=self.model_ik.joint_limit_lower,
                    joint_limit_upper=self.model_ik.joint_limit_upper,
                ),
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        for _ in range(30):
            solver.step(joint_q, joint_q, iterations=24)

        return joint_q.numpy()[0, :7]

    def _setup_ik(self):
        state_tmp = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state_tmp)
        body_q_np = state_tmp.body_q.numpy()

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([self.home_pos], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([body_q_np[self.ee_index][3:][:4]], dtype=wp.vec4),
        )
        self.joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model_ik.joint_limit_lower,
            joint_limit_upper=self.model_ik.joint_limit_upper,
        )
        self.joint_q_ik = wp.clone(self.model.joint_q[: self.model_ik.joint_coord_count].reshape((1, -1)))
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model_ik,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _setup_tasks(self):
        # Round 1: pick red, place on green, release.
        round_1 = [
            TaskType.APPROACH,
            TaskType.REFINE_APPROACH,
            TaskType.GRASP,
            TaskType.LIFT,
            TaskType.MOVE_TO_DROP_OFF,
            TaskType.REFINE_DROP_OFF,
            TaskType.RELEASE,
        ]
        round_1_times = [2.0, 1.0, 0.5, 1.0, 1.5, 1.5, 1.0]

        # Round 2: grip red+green pair, place on blue, release, go home.
        round_2 = [
            TaskType.GRASP,
            TaskType.LIFT,
            TaskType.MOVE_TO_DROP_OFF,
            TaskType.REFINE_DROP_OFF,
            TaskType.RELEASE,
            TaskType.HOME,
        ]
        round_2_times = [0.5, 1.0, 1.5, 1.5, 1.0, 2.5]

        red, green, blue = self.brick_bodies

        task_schedule, task_pick, task_drop, task_layer, time_limits = [], [], [], [], []
        for tasks, times, pick, drop, layer in [
            (round_1, round_1_times, red, green, 1),
            (round_2, round_2_times, red, blue, 2),
        ]:
            task_schedule.extend(tasks)
            task_pick.extend([pick] * len(tasks))
            task_drop.extend([drop] * len(tasks))
            task_layer.extend([layer] * len(tasks))
            time_limits.extend(times)

        self.task_schedule = wp.array(task_schedule, dtype=wp.int32)
        self.task_time_limits = wp.array(time_limits, dtype=float)
        self.task_pick_body = wp.array(task_pick, dtype=wp.int32)
        self.task_drop_body = wp.array(task_drop, dtype=wp.int32)
        self.task_drop_layer = wp.array(task_layer, dtype=wp.int32)

        self.task_idx = wp.zeros(1, dtype=wp.int32)
        self.task_time_elapsed = wp.zeros(1, dtype=wp.float32)
        self.task_init_body_q = wp.clone(self.state_0.body_q)

        self.ee_pos_target = wp.zeros(1, dtype=wp.vec3)
        self.ee_pos_interp = wp.zeros(1, dtype=wp.vec3)
        self.ee_rot_target = wp.zeros(1, dtype=wp.vec4)
        self.ee_rot_interp = wp.zeros(1, dtype=wp.vec4)
        self.gripper_target = wp.zeros(shape=(1, 2), dtype=wp.float32)

    def set_joint_targets(self):
        wp.launch(
            set_target_pose_kernel,
            dim=1,
            inputs=[
                self.task_schedule,
                self.task_time_limits,
                self.task_pick_body,
                self.task_drop_body,
                self.task_drop_layer,
                self.task_idx,
                self.task_time_elapsed,
                self.frame_dt,
                self.offset_approach,
                self.offset_lift,
                self.grasp_z_offset,
                self.drop_z_offset,
                self.brick_stack_height,
                self.home_pos,
                self.task_init_body_q,
                self.state_0.body_q,
                self.ee_index,
            ],
            outputs=[
                self.ee_pos_target,
                self.ee_pos_interp,
                self.ee_rot_target,
                self.ee_rot_interp,
                self.gripper_target,
            ],
        )

        self.pos_obj.set_target_positions(self.ee_pos_interp)
        self.rot_obj.set_target_rotations(self.ee_rot_interp)
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        wp.copy(dest=self.control.joint_target_pos[:7], src=self.joint_q_ik.flatten()[:7])
        wp.copy(dest=self.control.joint_target_pos[7:9], src=self.gripper_target.flatten()[:2])

        wp.launch(
            advance_task_kernel,
            dim=1,
            inputs=[
                self.task_time_limits,
                self.ee_pos_target,
                self.ee_rot_target,
                self.state_0.body_q,
                self.ee_index,
            ],
            outputs=[self.task_idx, self.task_time_elapsed, self.task_init_body_q],
        )

    def simulate(self):
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        for _ in range(self.sim_substeps):
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self):
        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])
        self.joint_q_ik = wp.clone(self.model.joint_q[: self.model_ik.joint_coord_count].reshape((1, -1)))
        self._setup_tasks()

    def step(self):
        self.set_joint_targets()
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        task_idx = int(self.task_idx.numpy()[0])
        body_q = self.state_0.body_q.numpy()
        ee_tf = body_q[self.ee_index]
        ee_pos = ee_tf[:3]
        ee_quat = ee_tf[3:7]
        tgt_pos = self.ee_pos_target.numpy()[0]
        tgt_rot = self.ee_rot_target.numpy()[0]
        tgt_quat = np.array([tgt_rot[0], tgt_rot[1], tgt_rot[2], tgt_rot[3]])
        pos_err = np.linalg.norm(tgt_pos - ee_pos)
        quat_rel = self._quat_mul(ee_quat, self._quat_inv(tgt_quat))
        rot_err = np.degrees(2.0 * np.arccos(np.clip(abs(quat_rel[3]), 0.0, 1.0)))
        elapsed = self.task_time_elapsed.numpy()[0]
        task_name = TaskType(int(self.task_schedule.numpy()[task_idx])).name
        print(
            f"t={self.sim_time:.3f} task[{task_idx}]={task_name} elapsed={elapsed:.3f} pos_err={pos_err:.4f} rot_err={rot_err:.2f}"
        )

    @staticmethod
    def _quat_mul(q, r):
        x = q[3] * r[0] + q[0] * r[3] + q[1] * r[2] - q[2] * r[1]
        y = q[3] * r[1] - q[0] * r[2] + q[1] * r[3] + q[2] * r[0]
        z = q[3] * r[2] + q[0] * r[1] - q[1] * r[0] + q[2] * r[3]
        w = q[3] * r[3] - q[0] * r[0] - q[1] * r[1] - q[2] * r[2]
        return np.array([x, y, z, w])

    @staticmethod
    def _quat_inv(q):
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        task_idx = self.task_idx.numpy()[0]
        total_tasks = len(self.task_schedule)
        if task_idx < total_tasks - 1:
            raise ValueError(f"Task sequence incomplete: reached step {task_idx}/{total_tasks - 1}")

        body_q = self.state_0.body_q.numpy()
        bh = self.brick_height_scaled
        stack_height = self.brick_stack_height
        red, green, blue = self.brick_bodies

        blue_z = body_q[blue][2]
        green_z = body_q[green][2]
        red_z = body_q[red][2]

        blue_xy = body_q[blue][:2]
        green_xy = body_q[green][:2]
        red_xy = body_q[red][:2]

        errors = []
        tol_z = 0.15 * bh

        for name, idx in [("Blue", blue), ("Green", green), ("Red", red)]:
            if not np.all(np.isfinite(body_q[idx])):
                errors.append(f"{name} brick (body {idx}) has non-finite transform")

        if np.linalg.norm(green_xy - blue_xy) > 0.01:
            errors.append(f"Green brick XY offset from blue: {np.linalg.norm(green_xy - blue_xy):.4f} m (max 0.01)")
        if np.linalg.norm(red_xy - blue_xy) > 0.01:
            errors.append(f"Red brick XY offset from blue: {np.linalg.norm(red_xy - blue_xy):.4f} m (max 0.01)")

        dz_green = green_z - blue_z
        if abs(dz_green - stack_height) > tol_z:
            errors.append(f"Green-Blue height gap: {dz_green:.4f} m, expected ~{stack_height:.4f} m")

        dz_red = red_z - blue_z
        if abs(dz_red - 2.0 * stack_height) > tol_z:
            errors.append(f"Red-Blue height gap: {dz_red:.4f} m, expected ~{2.0 * stack_height:.4f} m")

        if errors:
            raise ValueError("Brick stacking verification failed:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
