# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Two Brick Stack
#
# Builds a minimal scene with two LEGO bricks already stacked before
# SolverUIPC initializes. This isolates UIPC's rigid-rigid initialization
# and sanity-check path for interlocking brick geometry without a robot
# manipulation sequence.
#
# Command: python -m newton.examples uipc_two_brick_stack
#
###########################################################################


import numpy as np
import uipc
import warp as wp

import newton
import newton.examples
from newton.examples.uipc.contacts.example_uipc_brick_stacking import (
    BRICK_ABD_KAPPA,
    BRICK_DENSITY,
    BRICK_HEIGHT,
    BRICK_HZ,
    PITCH,
    STUD_RADIUS,
    _make_brick_mesh,
)

UIPC_GAP = 0.0001
DEFAULT_NUM_FRAMES = 60
WORKSPACE_ROOT = "/tmp/newton_uipc/two_brick_stack"
# Tighten the underside tubes to the top studs for the two-brick
# initialization scene. The shared robot stacking example keeps looser tubes
# for manipulation robustness, while this fixture needs real interlock
# geometry at the 0.1 mm UIPC barrier clearance.
INTERLOCK_TUBE_OUTER_RADIUS = float(np.hypot(0.5 * PITCH, 0.5 * PITCH) - STUD_RADIUS - UIPC_GAP)


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)
        self.uipc_gap = float(getattr(args, "uipc_gap", UIPC_GAP))
        if self.uipc_gap <= 0.0:
            raise ValueError("--uipc-gap must be positive so UIPC initialization starts outside the contact barrier")

        self.brick_stack_height = BRICK_HEIGHT + self.uipc_gap
        self.brick_mesh_xform = wp.transform(wp.vec3(0.0, 0.0, -BRICK_HZ), wp.quat_identity())
        self.brick_bodies: list[int] = []
        self.interlock_tube_outer_radius = float(np.hypot(0.5 * PITCH, 0.5 * PITCH) - STUD_RADIUS - self.uipc_gap)
        if self.interlock_tube_outer_radius <= 0.0:
            raise ValueError("--uipc-gap leaves no positive-radius underside tube for interlocking brick geometry")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        self._add_bricks(builder)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.initial_body_q = self.state_0.body_q.numpy()[self.brick_bodies].copy()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            workspace=f"{WORKSPACE_ROOT}",
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            dump_enable=True,
        )
        self.solver.configure_scene(
            {
                "line_search": {"max_iter": 64},
                "newton": {
                    "translation_tol": 1.0e-5,
                    "velocity_tol": 1.0e-5,
                },
            }
        )
        self.solver.set_contact(True, self.uipc_gap)
        self.solver.initialize(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.08, -0.12, 0.08),
            pitch=-25.0,
            yaw=135.0,
        )
        self.viewer._paused = True

    def _add_bricks(self, builder: newton.ModelBuilder) -> None:
        v, f = _make_brick_mesh(tube_outer_radius=self.interlock_tube_outer_radius)
        brick_meshes = [
            newton.Mesh(v, f.flatten(), color=(0.1, 0.2, 0.8)),
            newton.Mesh(v, f.flatten(), color=(0.8, 0.1, 0.1)),
        ]
        brick_cfg = newton.ModelBuilder.ShapeConfig(density=BRICK_DENSITY)
        lower_center_z = BRICK_HZ + self.uipc_gap
        brick_positions = [
            wp.vec3(0.0, 0.0, lower_center_z),
            wp.vec3(0.0, 0.0, lower_center_z + self.brick_stack_height),
        ]
        brick_labels = ["brick_bottom", "brick_top"]

        for label, position, mesh in zip(brick_labels, brick_positions, brick_meshes, strict=True):
            body = builder.add_body(
                xform=wp.transform(position, wp.quat_identity()),
                label=label,
                custom_attributes={"uipc:abd_kappa": BRICK_ABD_KAPPA},
            )
            builder.add_shape_mesh(
                body=body,
                mesh=mesh,
                xform=self.brick_mesh_xform,
                cfg=brick_cfg,
                label=label,
            )
            self.brick_bodies.append(body)

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        bottom, top = self.brick_bodies
        bottom_q = body_q[bottom]
        top_q = body_q[top]

        errors = []
        for name, body in [("Bottom", bottom), ("Top", top)]:
            if not np.all(np.isfinite(body_q[body])):
                errors.append(f"{name} brick (body {body}) has non-finite transform")

        xy_offset = float(np.linalg.norm(top_q[:2] - bottom_q[:2]))
        if xy_offset > 0.005:
            errors.append(f"Top brick XY offset from bottom: {xy_offset:.4f} m (max 0.005)")

        dz = float(top_q[2] - bottom_q[2])
        height_tol = 0.15 * BRICK_HEIGHT
        if abs(dz - self.brick_stack_height) > height_tol:
            errors.append(f"Brick height gap: {dz:.4f} m, expected ~{self.brick_stack_height:.4f} m")

        initial_z = self.initial_body_q[:, 2]
        current_z = body_q[self.brick_bodies, 2]
        if np.any(current_z < initial_z - 0.25 * BRICK_HEIGHT):
            errors.append("A brick dropped more than 25% of a brick height after UIPC initialization")

        if errors:
            raise ValueError("Two-brick UIPC stack verification failed:\n  " + "\n  ".join(errors))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument(
            "--uipc-gap",
            type=float,
            default=UIPC_GAP,
            help="Initial UIPC barrier clearance [m] between the two bricks and the ground plane.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
