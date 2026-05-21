# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Sensor Contact Scene
#
# UIPC version of the sensors/example_sensor_contact example.
# Shows how to use the SensorContact class with the UIPC solver to evaluate
# contact forces, including per-counterpart breakdowns.
#
# The flap has a contact sensor registering the total contact force of
# the objects on top. The plates' sensors register per-counterpart forces
# for the cube and the ball to detect which object touched which plate.
#
# Command: python -m newton.examples uipc_sensor_contact_scene
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Contacts
from newton.sensors import SensorContact
from newton.tests.unittest_utils import find_nonfinite_members


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt
        self.reset_interval = 8.0

        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_usd(newton.examples.get_asset("sensor_contact_scene.usda"))
        builder.add_ground_plane()

        # Raise Cube and Sphere to avoid initial mesh intersection with Flap
        for i, label in enumerate(builder.body_label):
            if "Cube" in label or "Sphere" in label:
                t = builder.body_q[i]
                new_p = wp.vec3(float(t.p[0]), float(t.p[1]), float(t.p[2]) + 0.3)
                builder.body_q[i] = wp.transform(p=new_p, q=t.q)
                # Also update joint_q for free joints (set at joint creation time)
                for j in range(len(builder.joint_child)):
                    if builder.joint_child[j] == i:
                        q_start = builder.joint_q_start[j]
                        builder.joint_q[q_start : q_start + 7] = list(builder.body_q[i])

        builder.color()

        self.model = builder.finalize()

        self.flap_contact_sensor = SensorContact(self.model, sensing_obj_shapes="*Flap", verbose=True)

        plate_labels = ["*Plate1", "*Plate2"]
        counterpart_labels = ["*Cube*", "*Sphere*"]
        self.plate_contact_sensor = SensorContact(
            self.model,
            sensing_obj_shapes=plate_labels,
            counterpart_shapes=counterpart_labels,
            measure_total=False,
            verbose=True,
        )

        import uipc  # noqa: PLC0415

        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/uipc_sensor_contact_scene_0.01",
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            require_profile=True,
            dump_enable=True,
        )
        self.solver.set_contact(True, d_hat=0.01)
        self.solver.initialize()

        self.contacts = Contacts(
            256,
            soft_contact_max=0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        self.viewer.set_model(self.model)
        self.viewer._paused = True  # start paused to avoid stepping before everything is initialized

        self.shape_map = {key: s for s, key in enumerate(self.model.shape_label)}
        self.plates_touched = 2 * [False]
        cube_shape = self.shape_map["/env/Cube"]
        sphere_shape = self.shape_map["/env/Sphere"]
        self.counterpart_col = [
            self.plate_contact_sensor.counterpart_indices[0].index(cube_shape),
            self.plate_contact_sensor.counterpart_indices[1].index(sphere_shape),
        ]
        self.shape_colors = {
            "/env/Plate1": 3 * [0.4],
            "/env/Plate2": 3 * [0.4],
            "/env/Sphere": [1.0, 0.4, 0.2],
            "/env/Cube": [0.2, 0.4, 0.8],
            "/env/Flap": 3 * [0.8],
        }

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        hinge_joint_idx = self.model.joint_label.index("/env/Hinge")
        self.hinge_joint_q_start = int(self.model.joint_q_start.numpy()[hinge_joint_idx])

        self.next_reset = 0.0

        self.initial_joint_q = wp.clone(self.state_0.joint_q)
        self.initial_joint_qd = wp.clone(self.state_0.joint_qd)

    def _set_shape_colors(self, shape_colors: dict[int, list[float] | tuple[float, float, float]]):
        for shape_idx, color in shape_colors.items():
            self.model.shape_color[shape_idx : shape_idx + 1].fill_(wp.vec3(color))

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.update_contacts(self.contacts, self.state_1)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset()

        hinge_angle = min(self.sim_time / 3, 1.6)
        self.control.joint_target_pos[self.hinge_joint_q_start : self.hinge_joint_q_start + 1].fill_(hinge_angle)

        self.simulate()

        self.plate_contact_sensor.update(self.state_0, self.contacts)

        net_force = self.plate_contact_sensor.force_matrix.numpy()
        for i in range(2):
            if self.plates_touched[i]:
                continue
            if np.abs(net_force[i, self.counterpart_col[i]]).max() == 0:
                continue
            plate_shape = self.plate_contact_sensor.sensing_obj_idx[i]
            counterpart_shape = self.plate_contact_sensor.counterpart_indices[i][self.counterpart_col[i]]
            self.plates_touched[i] = True
            plate_label = self.model.shape_label[plate_shape]
            counterpart_label = self.model.shape_label[counterpart_shape]
            print(f"Plate {plate_label} was touched by counterpart {counterpart_label}")
            self._set_shape_colors({plate_shape: self.shape_colors[counterpart_label]})

        self.flap_contact_sensor.update(self.state_0, self.contacts)
        self.viewer.log_scalar(
            "Flap Contact Force",
            np.abs(self.flap_contact_sensor.total_force.numpy()[0, 2]),
            smoothing=10,
        )
        self.sim_time += self.frame_dt

    def reset(self):
        self.sim_time = 0
        self.next_reset = self.sim_time + self.reset_interval
        self._set_shape_colors({self.shape_map[s]: v for s, v in self.shape_colors.items()})
        self.plates_touched = 2 * [False]
        self.viewer.log_scalar("Flap Contact Force", 0.0, clear=True)

        print("Resetting")
        self.state_0.joint_q.assign(self.initial_joint_q)
        self.state_0.joint_qd.assign(self.initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        assert not self.plates_touched[1] or self.plates_touched[0]
        assert len(find_nonfinite_members(self.flap_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor)) == 0

    def test_final(self):
        self.test_post_step()
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        assert len(find_nonfinite_members(self.flap_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor)) == 0
        assert self.model.shape_label[self.plate_contact_sensor.sensing_obj_idx[0]] == "/env/Plate1"
        assert self.model.shape_label[self.plate_contact_sensor.sensing_obj_idx[1]] == "/env/Plate2"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
