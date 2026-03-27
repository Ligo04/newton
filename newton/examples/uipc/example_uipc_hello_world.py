# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Hello World
#
# Simplest SolverUIPC demo: a single cube falls from height under gravity.
#
# Command: python -m newton.examples uipc_hello_world
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer:newton.viewer.ViewerBase, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane()

        # A single cube placed at height Y=3
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 5.0, 0.0), q=wp.quat_identity()),
        )
        builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            dt=self.sim_dt,
        )

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.0, 5.0, 10.0),
            pitch=0.0,
            yaw=-180.0,
        )
        self.viewer._paused = True 

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # After falling, the cube should be near the ground (Y < 1) and not exploded
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cube fell toward ground",
            lambda q, qd: float(q[1]) < 3.0 and float(q[1]) > -0.5,
            [0],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
