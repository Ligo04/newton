# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Box Pyramid
#
# UIPC port of ``example_pyramid``. Builds one or more pyramids of rigid
# cubes and optionally drops a wrecking ball down a ramp into them. This
# stresses SolverUIPC's rigid-rigid contact path (IPC barrier + ABD) rather
# than the narrow-phase contact generation exercised by the XPBD version,
# so defaults are tuned much smaller — UIPC contact is O(n_pairs) per
# Newton iteration and explodes quickly with stack height.
#
# Command: python -m newton.examples uipc_pyramid
#
###########################################################################

import numpy as np
import uipc
import warp as wp

import newton
import newton.examples

DEFAULT_NUM_PYRAMIDS = 20
DEFAULT_PYRAMID_SIZE = 20
CUBE_HALF = 0.4
CUBE_SPACING = 2.1 * CUBE_HALF
PYRAMID_SPACING = 2.0 * CUBE_SPACING
Y_STACK = 15.0

WRECKING_BALL_RADIUS = 2.0
WRECKING_BALL_DENSITY_MULT = 100.0
RAMP_LENGTH = 20.0
RAMP_WIDTH = 5.0
RAMP_THICKNESS = 0.5


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)

        num_pyramids = args.num_pyramids
        pyramid_size = args.pyramid_size

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()

        box_count = 0
        top_body_indices: list[int] = []
        pyramid_height = pyramid_size * CUBE_SPACING

        # Small vertical gap between the stacked cubes so UIPC's initial
        # IPC barrier does not see them in contact at t=0 (``d_hat`` is
        # 1 mm on this solver and stacked boxes sitting flush would land
        # inside that shell).
        uipc_gap = 0.02

        for pyramid in range(num_pyramids):
            y_offset = pyramid * PYRAMID_SPACING
            for level in range(pyramid_size):
                num_cubes_in_row = pyramid_size - level
                row_width = (num_cubes_in_row - 1) * CUBE_SPACING
                for i in range(num_cubes_in_row):
                    x_pos = -row_width / 2 + i * CUBE_SPACING
                    z_pos = level * (2.0 * CUBE_HALF + uipc_gap) + CUBE_HALF + uipc_gap
                    y_pos = Y_STACK - y_offset
                    body = builder.add_body(
                        xform=wp.transform(
                            p=wp.vec3(x_pos, y_pos, z_pos),
                            q=wp.quat_identity(),
                        ),
                        label=f"cube_p{pyramid}_l{level}_i{i}",
                    )
                    builder.add_shape_box(body, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF)
                    if level == pyramid_size - 1:
                        top_body_indices.append(body)
                    box_count += 1

        self.box_count = box_count
        self.top_body_indices = top_body_indices
        print(f"Built {num_pyramids} pyramids x {pyramid_size} rows = {box_count} boxes")

        if not self.test_mode:
            # Wrecking ball on a ramp. The ball rolls down and smashes the
            # stack — same idea as the XPBD version, just shorter because
            # UIPC handles far fewer contacts per step.
            ramp_height = 8.4
            ramp_angle = float(np.arctan2(ramp_height, RAMP_LENGTH))
            ball_x = 0.0
            ball_y = Y_STACK + RAMP_LENGTH * 0.9
            ball_z = ramp_height + WRECKING_BALL_RADIUS + 0.1

            body_ball = builder.add_body(
                xform=wp.transform(p=wp.vec3(ball_x, ball_y, ball_z), q=wp.quat_identity()),
                label="wrecking_ball",
            )
            ball_cfg = newton.ModelBuilder.ShapeConfig()
            ball_cfg.density = builder.default_shape_cfg.density * WRECKING_BALL_DENSITY_MULT
            builder.add_shape_sphere(body_ball, radius=WRECKING_BALL_RADIUS, cfg=ball_cfg)

            # Static ramp. SolverUIPC only promotes ``body == -1`` shapes
            # to scene geometry when they are ``GeoType.PLANE`` (see
            # ``rigid_body.py::build_ground_planes``), so a world-attached
            # box ramp would be silently dropped and the wrecking ball
            # would fall straight through. Instead attach the ramp to a
            # kinematic body so it becomes a fixed ABD collider.
            ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(ramp_angle))
            ramp_body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(ball_x, Y_STACK + RAMP_LENGTH / 2, ramp_height / 2),
                    q=ramp_quat,
                ),
                is_kinematic=True,
                label="ramp",
            )
            builder.add_shape_box(
                ramp_body,
                hx=RAMP_WIDTH / 2,
                hy=RAMP_LENGTH / 2,
                hz=RAMP_THICKNESS / 2,
            )

        builder.color()
        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Info,
        )
        self.solver.configure_scene({"line_search": {"report_energy": True}})
        self.solver.set_contact(True, 0.01)
        self.solver.initialize()

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.top_initial_positions = self.state_0.body_q.numpy()[:, :3].copy()

        self.viewer.set_model(self.model)

        cam_dist = max(pyramid_height, num_pyramids * PYRAMID_SPACING * 0.5) + 6.0
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.5),
            pitch=-15.0,
            yaw=135.0,
        )
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
        """Verify pyramid top cubes remain near their initial positions.

        In test mode the wrecking ball and ramp are omitted so the
        pyramids settle under gravity without toppling. Each top cube
        must stay within ``max_displacement`` of its initial position.
        """
        body_q = self.state_0.body_q.numpy()
        max_displacement = 0.5  # [m]
        for idx in self.top_body_indices:
            current_pos = body_q[idx, :3]
            initial_pos = self.top_initial_positions[idx]
            displacement = float(np.linalg.norm(current_pos - initial_pos))
            assert displacement < max_displacement, (
                f"Top cube body {idx}: displaced {displacement:.4f} m (max allowed {max_displacement:.4f} m)"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--num-pyramids",
            type=int,
            default=DEFAULT_NUM_PYRAMIDS,
            help="Number of pyramids to build.",
        )
        parser.add_argument(
            "--pyramid-size",
            type=int,
            default=DEFAULT_PYRAMID_SIZE,
            help="Number of rows in each pyramid base.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
