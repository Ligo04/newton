# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Softbody Dropping to Cloth
#
# This simulation demonstrates a volumetric soft body (tetrahedral grid)
# dropping onto a cloth sheet. The soft body uses Neo-Hookean elasticity
# and deforms on impact with the cloth.
#
# Command: python -m newton.examples uipc_softbody_dropping_to_cloth
#
###########################################################################

import numpy as np
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

        builder = newton.ModelBuilder()
        SolverUIPC.register_custom_attributes(builder)
        builder.add_ground_plane()

        # Add soft body (tetrahedral grid) at elevated position
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=6,
            dim_y=6,
            dim_z=3,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=500,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1e-3,
            particle_radius=0.005,
        )

        # Add cloth grid below the soft body
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, -1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=True,
            fix_right=True,
            dim_x=40,
            dim_y=40,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.0005,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=5,
            edge_kd=1e-2,
            particle_radius=0.001,
        )

        # Color the mesh for the viewer.
        builder.color()

        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.8, -2.8, 1.6), pitch=-18.0, yaw=-45.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 58.0

        self.solver = SolverUIPC(
            workspace="/tmp/newton_uipc/uipc_softbody_dropping_to_cloth",
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            auto_sync_inertia=False,
            enable_soft_position_constraint=False,
        )
        self.solver.set_contact(enable=True, d_hat=0.001)
        self.solver.configure_scene(
            {
                "linear_system": {"fem_preconditioner": "mas"},
            }
        )
        self.solver.initialize(self.state_0)
        self.graph = None
        self.viewer._paused = True

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that bounding box size is reasonable (not exploding)
        particle_q = self.state_0.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)

        # Check bbox size is reasonable (cloth stretches as soft body deforms it)
        assert bbox_size < 20.0, f"Bounding box exploded: size={bbox_size:.2f}"

        # Check no excessive penetration
        assert min_pos[2] > -0.5, f"Excessive penetration: z_min={min_pos[2]:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=120)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
