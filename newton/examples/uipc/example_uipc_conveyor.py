# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Conveyor
#
# Baggage-claim style conveyor using the SolverUIPC backend. A rotating
# belt mesh is driven by a revolute joint position target, two static
# annular rail meshes keep dynamic "bags" on the belt.
#
# Command: python -m newton.examples uipc_conveyor
#
###########################################################################

import math

import numpy as np
import uipc
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode

BELT_CENTER_Z = 0.55
BELT_RING_RADIUS = 1.8
BELT_HALF_WIDTH = 0.24
BELT_HALF_THICKNESS = 0.04
BELT_MESH_SEGMENTS = 96
RAIL_WALL_THICKNESS = 0.035
RAIL_HEIGHT = 0.16
RAIL_BASE_OVERLAP = -0.002  # small gap above belt to avoid zero-distance contact
BAG_COUNT = 18
BAG_LANE_OFFSETS = (-0.12, 0.0, 0.12)
BAG_DROP_CLEARANCE = 0.035
BELT_SPEED = 0.75  # tangential belt speed [m/s]


def create_annular_prism_mesh(
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    segments: int,
    *,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> newton.Mesh:
    """Create a closed ring prism mesh centered at the origin."""
    if segments < 3:
        raise ValueError("segments must be >= 3")
    if inner_radius <= 0.0 or outer_radius <= inner_radius:
        raise ValueError("Expected 0 < inner_radius < outer_radius")
    if z_max <= z_min:
        raise ValueError("Expected z_max > z_min")

    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    inner_top = np.stack(
        (
            inner_radius * cos_theta,
            inner_radius * sin_theta,
            np.full(segments, z_max, dtype=np.float32),
        ),
        axis=1,
    )
    outer_top = np.stack(
        (
            outer_radius * cos_theta,
            outer_radius * sin_theta,
            np.full(segments, z_max, dtype=np.float32),
        ),
        axis=1,
    )
    inner_bottom = np.stack(
        (
            inner_radius * cos_theta,
            inner_radius * sin_theta,
            np.full(segments, z_min, dtype=np.float32),
        ),
        axis=1,
    )
    outer_bottom = np.stack(
        (
            outer_radius * cos_theta,
            outer_radius * sin_theta,
            np.full(segments, z_min, dtype=np.float32),
        ),
        axis=1,
    )

    vertices = np.vstack((inner_top, outer_top, inner_bottom, outer_bottom)).astype(np.float32)

    it_offset = 0
    outer_top_offset = segments
    ib_offset = 2 * segments
    ob_offset = 3 * segments

    indices: list[int] = []
    for i in range(segments):
        j = (i + 1) % segments

        it_i = it_offset + i
        it_j = it_offset + j
        outer_top_i = outer_top_offset + i
        outer_top_j = outer_top_offset + j
        ib_i = ib_offset + i
        ib_j = ib_offset + j
        ob_i = ob_offset + i
        ob_j = ob_offset + j

        # Top face (+Z)
        indices.extend((it_i, outer_top_i, outer_top_j, it_i, outer_top_j, it_j))
        # Bottom face (-Z)
        indices.extend((ib_i, ib_j, ob_j, ib_i, ob_j, ob_i))
        # Outer face (+radial)
        indices.extend((ob_i, ob_j, outer_top_j, ob_i, outer_top_j, outer_top_i))
        # Inner face (-radial)
        indices.extend((ib_i, it_i, it_j, ib_i, it_j, ib_j))

    mesh = newton.Mesh(vertices=vertices, indices=np.asarray(indices, dtype=np.int32), compute_inertia=False)
    mesh.color = color
    mesh.roughness = roughness
    mesh.metallic = metallic
    return mesh


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.viewer = viewer
        belt_speed = float(args.belt_speed) if args is not None and hasattr(args, "belt_speed") else BELT_SPEED
        self.belt_angular_speed = belt_speed / BELT_RING_RADIUS

        builder = newton.ModelBuilder()

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2

        builder.add_ground_plane()

        # ---- Belt --------------------------------------------------------
        belt_inner_radius = BELT_RING_RADIUS - BELT_HALF_WIDTH
        belt_outer_radius = BELT_RING_RADIUS + BELT_HALF_WIDTH

        belt_mesh = create_annular_prism_mesh(
            inner_radius=belt_inner_radius,
            outer_radius=belt_outer_radius,
            z_min=-BELT_HALF_THICKNESS,
            z_max=BELT_HALF_THICKNESS,
            segments=BELT_MESH_SEGMENTS,
            color=(0.09, 0.09, 0.09),
            roughness=0.94,
            metallic=0.02,
        )

        belt_cfg = newton.ModelBuilder.ShapeConfig(mu=1.2, ke=1.0e3, kd=1.0e-1)

        # Dynamic body driven by a high-stiffness revolute joint target.
        # (UIPC marks kinematic bodies as fixed, so the belt must be dynamic.)
        self.belt_body = builder.add_link(mass=15.0, label="conveyor_belt")
        builder.add_shape_mesh(self.belt_body, mesh=belt_mesh, cfg=belt_cfg, label="conveyor_belt_mesh")
        self.belt_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.belt_body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, BELT_CENTER_Z), q=wp.quat_identity()),
            label="conveyor_belt_joint",
        )

        belt_qd_start = builder.joint_qd_start[self.belt_joint]
        builder.joint_target_ke[belt_qd_start] = 1e5
        builder.joint_target_kd[belt_qd_start] = 1e3
        builder.joint_target_pos[belt_qd_start] = 0.0
        builder.joint_target_mode[belt_qd_start] = int(JointTargetMode.POSITION)
        builder.joint_qd[belt_qd_start] = self.belt_angular_speed
        builder.add_articulation([self.belt_joint], label="conveyor_belt")

        # ---- Rails -------------------------------------------------------
        rail_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=1.0e3, kd=1.0e-1)

        rail_inner_mesh = create_annular_prism_mesh(
            inner_radius=belt_inner_radius - RAIL_WALL_THICKNESS,
            outer_radius=belt_inner_radius,
            z_min=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP,
            z_max=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP + RAIL_HEIGHT,
            segments=BELT_MESH_SEGMENTS,
            color=(0.66, 0.69, 0.74),
            roughness=0.24,
            metallic=0.9,
        )
        rail_outer_mesh = create_annular_prism_mesh(
            inner_radius=belt_outer_radius,
            outer_radius=belt_outer_radius + RAIL_WALL_THICKNESS,
            z_min=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP,
            z_max=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP + RAIL_HEIGHT,
            segments=BELT_MESH_SEGMENTS,
            color=(0.66, 0.69, 0.74),
            roughness=0.24,
            metallic=0.9,
        )

        # Rails are kinematic (fixed in UIPC) since they don't move.
        # Shapes on body=-1 other than PLANE are not processed by UIPC,
        # so we create dedicated bodies.
        for rail_mesh, rail_label in (
            (rail_inner_mesh, "conveyor_rail_inner"),
            (rail_outer_mesh, "conveyor_rail_outer"),
        ):
            rail_body = builder.add_link(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, BELT_CENTER_Z), q=wp.quat_identity()),
                is_kinematic=True,
                label=rail_label,
            )
            builder.add_shape_mesh(rail_body, mesh=rail_mesh, cfg=rail_cfg, label=rail_label)

        # ---- Bags --------------------------------------------------------
        bag_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0, ke=1.0e3, kd=1.0e-1, restitution=0.0)

        self.bag_bodies = []
        belt_top_z = BELT_CENTER_Z + BELT_HALF_THICKNESS
        bag_angles = np.linspace(0.0, 2.0 * math.pi, BAG_COUNT, endpoint=False, dtype=np.float32)

        for i, angle in enumerate(bag_angles):
            lane_idx = i % len(BAG_LANE_OFFSETS)
            radial_offset = BAG_LANE_OFFSETS[lane_idx]
            radius = BELT_RING_RADIUS + radial_offset
            bag_x = radius * math.cos(angle)
            bag_y = radius * math.sin(angle)
            bag_yaw = angle + 0.5 * math.pi

            shape_type = i % 3
            if shape_type == 0:
                bag_vertical_extent = 0.08  # box hz
            elif shape_type == 1:
                bag_vertical_extent = 0.08  # horizontal capsule radius
            else:
                bag_vertical_extent = 0.11  # sphere radius

            bag_z = belt_top_z + bag_vertical_extent + BAG_DROP_CLEARANCE
            bag_body = builder.add_link(
                xform=wp.transform(
                    p=wp.vec3(bag_x, bag_y, bag_z),
                    q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), bag_yaw),
                ),
                mass=2.8 + 0.1 * i,
                label=f"bag_{i}",
            )

            if shape_type == 0:
                builder.add_shape_box(bag_body, hx=0.18, hy=0.12, hz=0.08, cfg=bag_cfg)
            elif shape_type == 1:
                builder.add_shape_capsule(
                    bag_body,
                    radius=0.08,
                    half_height=0.15,
                    xform=wp.transform(q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)),
                    cfg=bag_cfg,
                )
            else:
                builder.add_shape_sphere(bag_body, radius=0.11, cfg=bag_cfg)

            builder.add_articulation([builder.add_joint_free(bag_body)], label=f"bag_{i}")
            self.bag_bodies.append(bag_body)

        builder.color()
        self.model = builder.finalize()
        self.state_0 = self.model.state()

        belt_body_idx = self.belt_body

        def _contact_tabular_fn(contact_tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem):
            GPa = 1e9
            belt_elem = contact_tabular.create("belt")
            contact_tabular.insert(belt_elem, actor_elem, 1.2, 1.0 * GPa, True)
            contact_tabular.insert(belt_elem, env_elem, 0.5, 1.0 * GPa, False)
            contact_tabular.insert(belt_elem, ground_elem, 0.5, 1.0 * GPa, False)
            contact_tabular.insert(belt_elem, robo_elem, 0.5, 1.0 * GPa, False)
            contact_tabular.insert(belt_elem, belt_elem, 0.5, 1.0 * GPa, False)
            return {belt_body_idx: belt_elem}

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Info,
            auto_init=False,
        )
        self.solver.configure_contact_tabular(_contact_tabular_fn)
        self.solver.initialize()

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Cache belt DOF index for runtime target updates
        qd_starts = self.model.joint_qd_start.numpy()
        self.belt_qd_start = int(qd_starts[self.belt_joint])

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.7, -1.3, 5.0), -60.0, -200.0)
        self.viewer._paused = True

    def _update_belt_target(self):
        """Set belt revolute joint target angle for constant-speed rotation."""
        target_angle = self.belt_angular_speed * self.sim_time
        target_pos = self.control.joint_target_pos.numpy()
        target_pos[self.belt_qd_start] = target_angle
        self.control.joint_target_pos.assign(target_pos)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._update_belt_target()
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
        body_q = self.state_0.body_q.numpy()

        # Belt should remain near its initial height
        belt_z = float(body_q[self.belt_body][2])
        assert abs(belt_z - BELT_CENTER_Z) < 0.15, f"Belt body drifted off the conveyor plane: z={belt_z:.4f}"

        # All bags should be alive (finite, above floor, within bounds)
        for body_idx in self.bag_bodies:
            x = float(body_q[body_idx][0])
            y = float(body_q[body_idx][1])
            z = float(body_q[body_idx][2])
            assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z), f"Bag {body_idx} has non-finite pose values."
            assert z > -0.5, f"Bag body {body_idx} fell through the floor: z={z:.4f}"
            assert abs(x) < 4.0 and abs(y) < 4.0, f"Bag body {body_idx} left the scene bounds: ({x:.3f}, {y:.3f})"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--belt-speed",
        type=float,
        default=BELT_SPEED,
        help="Conveyor tangential speed [m/s].",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
