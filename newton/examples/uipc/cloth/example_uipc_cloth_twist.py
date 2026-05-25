# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cloth Twist
#
# Twist a square cloth with SolverUIPC.  The cloth uses the UIPC default
# StrainLimitingBaraffWitkinShell + DiscreteShellBending material; pass
# ``--cloth-model neo_hookean`` to set every ``model.uipc.cloth_model`` entry to
# NeoHookeanShell instead.  Edge
# position control is applied through UIPC SoftPositionConstraint targets.
#
# Command: python -m newton.examples uipc_cloth_twist
#
###########################################################################

from __future__ import annotations

import math
import os

import numpy as np
import uipc
import warp as wp
import warp.examples
from pxr import Usd

import newton
import newton.examples
import newton.usd


def _rotate_about_axes(points: np.ndarray, axes: np.ndarray, angular_velocity: float, time: float) -> np.ndarray:
    """Rotate points around per-point axes through the origin."""
    theta = angular_velocity * time
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    roots = np.sum(points * axes, axis=1, keepdims=True) * axes
    rel = points - roots
    return (
        roots
        + rel * cos_t
        + np.cross(axes, rel) * sin_t
        + axes * np.sum(axes * rel, axis=1, keepdims=True) * (1.0 - cos_t)
    )


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.rot_angular_velocity = math.pi / 3.0
        self.rot_end_time = 10.0
        self.soft_strength = float(args.soft_strength)
        self.viewer = viewer

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")
        cloth_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = cloth_mesh.vertices
        mesh_indices = cloth_mesh.indices
        vertices = [wp.vec3(v) for v in mesh_points]

        builder = newton.ModelBuilder(gravity=0.0)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0),
            scale=0.01,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-7,
            edge_ke=1.0e-3,
            edge_kd=1.0e-4,
        )
        builder.color()

        self.model = builder.finalize()
        self.model.uipc.cloth_model[:] = [args.cloth_model] * len(self.model.cloth_ranges)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        cloth_size = int(round(math.sqrt(len(mesh_points))))
        if cloth_size * cloth_size != len(mesh_points):
            raise ValueError(f"Expected square_cloth.usd to contain a square grid, got {len(mesh_points)} vertices")

        left_side = np.array([cloth_size - 1 + i * cloth_size for i in range(cloth_size)], dtype=np.int32)
        right_side = np.array([i * cloth_size for i in range(cloth_size)], dtype=np.int32)
        self.rot_point_indices = np.concatenate((left_side, right_side))
        self.rot_axes = np.vstack(
            (
                np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), (left_side.size, 1)),
                np.tile(np.array([0.0, -1.0, 0.0], dtype=np.float64), (right_side.size, 1)),
            )
        )
        self.rot_rest_positions = self.state_0.particle_q.numpy()[self.rot_point_indices].astype(np.float64)

        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/uipc_cloth_twist",
            dump_enable=True,
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            cloth_soft_position_strength_ratio=self.soft_strength,
        )
        self.solver.set_contact(True, d_hat=0.001)
        self.solver.initialize(self.state_0)
        self._push_soft_targets(0.0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0)

    def _soft_targets(self, time: float) -> np.ndarray:
        return _rotate_about_axes(
            self.rot_rest_positions,
            self.rot_axes,
            self.rot_angular_velocity,
            min(time, self.rot_end_time),
        )

    def _push_soft_targets(self, time: float) -> None:
        self.solver.set_cloth_soft_position_constraints(
            self.rot_point_indices,
            self._soft_targets(time),
            strength_ratio=self.soft_strength,
        )

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self._push_soft_targets(t)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
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
        p_lower = wp.vec3(-0.8, -1.0, -0.8)
        p_upper = wp.vec3(0.8, 1.0, 0.8)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 5.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--cloth-model",
        default="strain_limiting_baraff_witkin",
        choices=("strain_limiting_baraff_witkin", "strain_limiting", "neo_hookean"),
        help="UIPC cloth membrane model.",
    )
    parser.add_argument("--soft-strength", type=float, default=100.0, help="SoftPositionConstraint strength ratio.")
    parser.set_defaults(num_frames=300)

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
