# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Sensor Contact
#
# Demonstrates contact force sensing using the SensorContact API.
#
# Scene: two boxes stacked on a ground plane. A SensorContact on the
# bottom box measures the total contact force (ground + top box).
#
# Multiple solver backends are supported via ``--solver``:
#   python -m newton.examples uipc_sensor_contact
#   python -m newton.examples uipc_sensor_contact --solver mujoco
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton import Contacts
from newton.sensors import SensorContact

BOX_HALF = 0.5
STACK_GAP = 0.02


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)

        solver_name = getattr(args, "solver", "uipc")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if solver_name == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane()

        z0 = BOX_HALF + STACK_GAP
        self.body_bottom = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z0), q=wp.quat_identity()),
            label="box_bottom",
        )
        builder.add_shape_box(
            self.body_bottom,
            hx=BOX_HALF,
            hy=BOX_HALF,
            hz=BOX_HALF,
            label="shape_bottom",
        )

        z1 = z0 + 2.0 * BOX_HALF + STACK_GAP
        self.body_top = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, z1), q=wp.quat_identity()),
            label="box_top",
        )
        builder.add_shape_box(
            self.body_top,
            hx=BOX_HALF,
            hy=BOX_HALF,
            hz=BOX_HALF,
            label="shape_top",
        )

        builder.color()
        self.model = builder.finalize()

        self.solver, self._use_two_states = self._build_solver(solver_name)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state() if self._use_two_states else None
        self.control = self.model.control()

        self.contact_sensor = SensorContact(
            self.model,
            sensing_obj_shapes=["shape_bottom"],
            counterpart_shapes=["ground_plane", "shape_top"],
            verbose=True,
        )

        rigid_contact_max = (
            self.solver.get_max_contact_count() if hasattr(self.solver, "get_max_contact_count") else 256
        )
        self.contacts = Contacts(
            rigid_contact_max,
            soft_contact_max=0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(4.0, -4.0, 3.0),
            pitch=-20.0,
            yaw=135.0,
        )
        self.viewer._paused = True

    def _build_solver(self, name: str) -> tuple[object, bool]:
        """Construct the requested solver.

        Returns (solver, use_two_states).
        """
        if name == "uipc":
            import uipc  # noqa: PLC0415

            solver = newton.solvers.SolverUIPC(
                self.model,
                dt=self.sim_dt,
                logger_level=uipc.Logger.Warn,
            )
            solver.set_contact(True, 0.001)
            solver.initialize()
            return solver, True
        if name == "mujoco":
            solver = newton.solvers.SolverMuJoCo(
                self.model,
                njmax=100,
                nconmax=100,
            )
            return solver, False
        raise ValueError(f"unsupported --solver: {name!r}")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self._use_two_states:
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            else:
                self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def step(self):
        self.simulate()
        self.solver.update_contacts(self.contacts, self.state_0)

        self.contact_sensor.update(self.state_0, self.contacts)
        sensor_force = self.contact_sensor.total_force.numpy()[0]
        force_matrix = self.contact_sensor.force_matrix.numpy()[0]

        if int(self.sim_time * self.fps) % 60 == 0:
            print(f"t={self.sim_time:.2f}s | SensorContact bottom Fz={sensor_force[2]:.2f} N")
            print(f"  force_matrix: ground Fz={force_matrix[0][2]:.2f} N, top Fz={force_matrix[1][2]:.2f} N")

        self.viewer.log_scalar("Bottom Contact Fz (sensor)", float(sensor_force[2]))

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify contact forces are non-zero after settling."""
        self.solver.update_contacts(self.contacts, self.state_0)
        self.contact_sensor.update(self.state_0, self.contacts)
        sensor_force = self.contact_sensor.total_force.numpy()[0]
        assert abs(sensor_force[2]) > 0.1, (
            f"Expected non-zero contact force on bottom box, got Fz={sensor_force[2]:.4f}"
        )

        body_q = self.state_0.body_q.numpy()
        for i in range(self.model.body_count):
            assert body_q[i, 2] > 0.0, f"Body {i} fell below ground: z={body_q[i, 2]:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            choices=("uipc", "mujoco"),
            default="uipc",
            help="Newton solver backend.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
