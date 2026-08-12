# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# UIPC Cloth Poker Cards
#
# This simulation demonstrates poker-card cloth patches dropping
# and stacking on a cube with SolverUIPC. Cards use high bending stiffness
# to approximate rigid cards while still using UIPC cloth contact.
#
# Standard poker card dimensions:
# - Width: 6.35 cm (2.5 inches) = 0.0635 m
# - Height: 8.89 cm (3.5 inches) = 0.0889 m
# - Resolution: 4x6 cells per card
#
# Command: uv run -m newton.examples uipc_cloth_poker_cards
#
###########################################################################

import numpy as np
import uipc
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.sim_time = 0.0

        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Standard poker card dimensions in meters
        self.card_width = 0.0635  # m (6.35 cm / 2.5 inches)
        self.card_height = 0.0889  # m (8.89 cm / 3.5 inches)

        # Card resolution: 4x6 cells
        self.dim_x = 4  # cells along width
        self.dim_y = 6  # cells along height
        self.cell_x = self.card_width / self.dim_x  # ~0.0159 m
        self.cell_y = self.card_height / self.dim_y  # ~0.0148 m

        # Number of cards. The default is kept modest because UIPC cloth self-contact is expensive.
        self.num_cards = int(args.num_cards)

        # Cube (table/platform) parameters in meters
        self.cube_size = 0.1  # m (10 cm) - half-size of the cube
        self.cube_height = 0.11

        # Card drop parameters in meters
        # Cards drop onto the cube surface (cube_height + cube_size = top of cube)
        self.drop_height_base = self.cube_height + self.cube_size + 0.05  # m
        self.card_spacing_z = 0.002  # m - vertical spacing between cards for UIPC shell thickness
        self.random_offset_xy = 0.001  # m (0.5 cm) - random XY offset

        # Build the model (using meters)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))  # m/s²
        newton.solvers.SolverUIPC.register_custom_attributes(builder)

        # Add a static cube for cards to stack on
        body_cube = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.cube_height),
                q=wp.quat_identity(),
            ),
            label="cube",
            is_kinematic=True,
        )
        cube_cfg = newton.ModelBuilder.ShapeConfig()
        cube_cfg.density = 0.0  # Static body (infinite mass)
        cube_cfg.ke = 5.0e6  # Contact stiffness
        cube_cfg.kd = 1.0e-4  # Contact damping
        cube_cfg.mu = 0.1  # Friction
        builder.add_shape_box(
            body_cube,
            hx=self.cube_size,
            hy=self.cube_size,
            hz=self.cube_size,
            cfg=cube_cfg,
        )

        # Add a dynamic sphere to knock off the cards
        # Sphere starts to the side and moves toward the card pile
        self.sphere_radius = 0.02  # m (2 cm radius)
        self.sphere_start_x = -0.35  # m - start position to the left
        # Position sphere at card pile height (top of cube + some offset)
        # cube top is at cube_height + cube_size = 0.1 + 0.1 = 0.2m
        self.sphere_height = 0.23  # m - at card pile level
        self.sphere_velocity_x = 0.5  # m/s - velocity toward cards
        self.sphere_current_x = self.sphere_start_x

        body_sphere = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(self.sphere_start_x, 0.0, self.sphere_height),
                q=wp.quat_identity(),
            ),
            label="sphere",
            is_kinematic=False,
        )
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 1000.0
        sphere_cfg.ke = 1.0e5  # Contact stiffness
        sphere_cfg.kd = 1.0e-4  # Contact damping
        sphere_cfg.mu = 0.3  # Friction
        builder.add_shape_sphere(body_sphere, radius=self.sphere_radius, cfg=sphere_cfg)
        sphere_joint = builder.add_joint_free(child=body_sphere, label="sphere_free_joint")
        builder.add_articulation([sphere_joint], label="sphere_articulation")

        # Sphere body index for free-joint SoftTransformConstraint aim-target animation.
        self.sphere_body_index = body_sphere

        # Random generator for reproducible random offsets
        rng = np.random.default_rng(42)

        # Card mass properties
        # Real card: ~1.8g = 0.0018 kg
        # For a 4x6 grid, there are 5x7 = 35 particles
        card_mass_total = 1.8e-3  # kg (1.8 grams)
        num_particles_per_card = (self.dim_x + 1) * (self.dim_y + 1)  # 5 * 7 = 35
        card_mass_per_particle = card_mass_total / num_particles_per_card

        # High bending stiffness for stiff cards
        # tri_ke/tri_ka: in-plane stretch stiffness
        # edge_ke: bending stiffness (key for card rigidity)
        tri_ke = 1.0e4  # High stretch stiffness
        tri_ka = 1.0e4  # High shear stiffness
        tri_kd = 1.0e-4  # Small damping
        edge_ke = 1.0e2  # High bending stiffness for rigid cards
        edge_kd = 1.0e-2  # Bending damping

        # Particle radius for collision (in meters)
        particle_radius = 0.0004

        # Add cards
        for i in range(self.num_cards):
            # Calculate drop position with slight random offset
            offset_x = rng.uniform(-self.random_offset_xy, self.random_offset_xy)
            offset_y = rng.uniform(-self.random_offset_xy, self.random_offset_xy)

            # Cards drop from different heights
            drop_z = self.drop_height_base + i * self.card_spacing_z

            # Random rotation around Z-axis for natural stacking
            random_angle = rng.uniform(-0.1, 0.1)  # Small random rotation

            # Card center position (offset so card center is at origin)
            pos_x = -self.card_width / 2 + offset_x
            pos_y = -self.card_height / 2 + offset_y
            pos_z = drop_z

            builder.add_cloth_grid(
                pos=wp.vec3(pos_x, pos_y, pos_z),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), random_angle),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.dim_x,
                dim_y=self.dim_y,
                cell_x=self.cell_x,
                cell_y=self.cell_y,
                mass=card_mass_per_particle,
                fix_left=False,
                fix_right=False,
                fix_top=False,
                fix_bottom=False,
                tri_ke=tri_ke,
                tri_ka=tri_ka,
                tri_kd=tri_kd,
                edge_ke=edge_ke,
                edge_kd=edge_kd,
                particle_radius=particle_radius,
            )

        # Add ground plane
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.ke = 1.0e5  # Contact stiffness
        ground_cfg.kd = 1.0e-4  # Contact damping
        ground_cfg.mu = 0.3  #
        builder.add_ground_plane(cfg=ground_cfg)

        # Color the mesh for viewer rendering.
        builder.color(include_bending=True)

        # Finalize model
        self.model = builder.finalize()
        self.model.uipc.cloth_model[:] = [args.cloth_model] * len(self.model.cloth_ranges)

        # Create UIPC solver with cloth self-contact enabled through actor-actor contact.
        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/uipc_cloth_poker_cards",
            dump_enable=bool(args.dump),
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            enable_soft_position_constraint=False,
            auto_sync_inertia=False,
        )
        self.solver.set_contact(enable=True, d_hat=0.0005)
        self.solver.configure_scene(
            {
                "newton": {"velocity_tol": 1.0e-3, "translation_tol": 1.0e-3},
                "line_search": {"max_iter": 8},
            }
        )
        self.solver.initialize()
        self.sphere_slot = self.solver.mapping.body_geo_slots[body_sphere]
        self.sphere_instance_id = self.solver.mapping.body_instance_ids[body_sphere]
        sphere_instances = self.sphere_slot.geometry().instances()
        self.sphere_aim_tf = sphere_instances.find("aim_transform")
        if self.sphere_aim_tf is None:
            self.sphere_aim_tf = sphere_instances.find("aim_tf")
        self.sphere_is_constrained = sphere_instances.find("is_constrained")
        if self.sphere_aim_tf is None or self.sphere_is_constrained is None:
            raise RuntimeError("UIPC free-joint sphere is missing soft-transform aim/is_constrained attributes.")
        uipc.view(self.sphere_is_constrained)[self.sphere_instance_id] = 1
        self._write_sphere_aim_transform()

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            wp.copy(self.model.body_q, self.state_0.body_q)
            wp.copy(self.model.body_qd, self.state_0.body_qd)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer._paused = True  # Start paused to inspect initial setup

        # Set camera to view the stacking
        self.viewer.set_camera(
            pos=wp.vec3(0.5, -0.5, 0.3),
            pitch=-15.0,
            yaw=140.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 70.0

        self.capture()

    def capture(self):
        # Disable CUDA graph capture because UIPC aim-target animation
        # updates a host-side geometry attribute each frame.
        self.graph = None

    def _write_sphere_aim_transform(self):
        """Move the dynamic sphere through UIPC's aim transform target."""
        aim_view = uipc.view(self.sphere_aim_tf)
        aim_view[self.sphere_instance_id] = np.array(
            [
                [1.0, 0.0, 0.0, self.sphere_current_x],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, self.sphere_height],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _animate_sphere(self):
        """Advance the dynamic sphere's UIPC aim transform target."""
        self.sphere_current_x += self.sphere_velocity_x * self.sim_dt
        self._write_sphere_aim_transform()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply viewer forces (for interactive manipulation)
            self.viewer.apply_forces(self.state_0)

            # Animate dynamic sphere aim target (move it toward the cards)
            self._animate_sphere()

            # Solver step
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify simulation reached a valid end state."""
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        # Check velocity remains finite and bounded during the UIPC drop.
        max_vel = np.max(np.linalg.norm(particle_qd, axis=1))
        assert np.isfinite(max_vel) and max_vel < 10.0, f"Cards moving too fast: max_vel={max_vel:.4f} m/s"

        # Check bbox size is reasonable (not exploding)
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 2.0, f"Bounding box exploded: size={bbox_size:.2f}"

        # Check no excessive penetration
        assert min_pos[2] > -0.1, f"Excessive penetration: z_min={min_pos[2]:.4f}"

        expected_min_x = self.sphere_start_x + 0.5 * self.sphere_velocity_x * self.sim_time
        assert self.sphere_current_x > expected_min_x, (
            f"Free-joint sphere target did not move toward cards: "
            f"x={self.sphere_current_x:.4f}, expected > {expected_min_x:.4f}"
        )


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()
    parser.add_argument("--num-cards", type=int, default=52, help="Number of cards to drop.")
    parser.add_argument(
        "--cloth-model",
        default="strain_limiting_baraff_witkin",
        choices=("strain_limiting_baraff_witkin", "strain_limiting", "neo_hookean"),
        help="UIPC cloth membrane model.",
    )
    parser.add_argument("--dump", action="store_true", help="Dump UIPC surface OBJ files.")
    parser.set_defaults(num_frames=180)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
