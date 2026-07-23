# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Deformable Body
#
# A deformable rubber duck falls onto a table under gravity using
# SolverUIPC and Stable Neo-Hookean tetrahedra.
#
# Command: python -m newton.examples deformablebody
#
###########################################################################

from __future__ import annotations

import uipc
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.utils
from newton import ModelBuilder
from newton.solvers import SolverUIPC


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()

        table_body = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, -0.5, 0.1), q=wp.quat_identity()),
            is_kinematic=True,
            label="deformablebody_table",
        )
        table_cfg = ModelBuilder.ShapeConfig(mu=0.8, ke=1.0e4, kd=1.0e1)
        builder.add_shape_box(table_body, hx=0.4, hy=0.4, hz=0.1, cfg=table_cfg, label="table")

        duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
        usd_stage = Usd.Stage.Open(str(duck_path / "model.usda"))
        prim = usd_stage.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh = newton.TetMesh.create_from_usd(prim)

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, -0.5, 0.32),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            density=100.0,
            k_mu=1.0e6,
            k_lambda=1.0e6,
            k_damp=1.0e-6,
            particle_radius=0.005,
        )

        builder.color()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        self.solver = SolverUIPC(
            workspace="/tmp/newton_uipc/deformablebody",
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            auto_sync_inertia=False,
        )
        self.solver.set_contact(True, d_hat=0.005)
        self.solver.initialize(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.1), -35.0, -58.0)
        self.viewer._paused = True

    def simulate(self):
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-1.0, -1.5, -0.1)
        p_upper = wp.vec3(1.0, 0.5, 1.2)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=120)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
