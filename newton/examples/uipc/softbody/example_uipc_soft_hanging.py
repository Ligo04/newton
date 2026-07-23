# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Soft Hanging
#
# Hanging deformable grids simulated with SolverUIPC and Stable Neo-Hookean
# deformable bodies. The grids use different Lamé parameters so the effect of
# material stiffness is visible under gravity.
#
# Command: python -m newton.examples uipc_soft_hanging
#
###########################################################################

from __future__ import annotations

import uipc
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverUIPC


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()

        dim_x = 10
        dim_y = 4
        dim_z = 4
        cell_size = 0.1
        spacing = 0.6
        stiffness_values = [1.0e5, 2.0e5, 3.0e5, 4.0e5]

        for i, k_mu in enumerate(stiffness_values):
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 1.0 + i * spacing, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell_size,
                cell_y=cell_size,
                cell_z=cell_size,
                density=1.0e3,
                k_mu=k_mu,
                k_lambda=k_mu,
                k_damp=1.0e-6,
                fix_left=True,
            )

        builder.color()
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        self.solver = SolverUIPC(
            workspace="/tmp/newton_uipc/soft_hanging",
            dump_enable=True,
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            auto_sync_inertia=False,
        )
        self.solver.set_contact(False, d_hat=0.01)
        self.solver.initialize(self.state_0)

        self.viewer.set_model(self.model)
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
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
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
