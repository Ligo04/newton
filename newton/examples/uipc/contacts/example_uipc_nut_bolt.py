# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Nut Bolt
#
# UIPC version of ``nut_bolt_hydro``. The original example uses Newton's
# hydroelastic SDF contact pipeline with XPBD/MuJoCo. This version can use
# either the original IsaacGym nut/bolt meshes or Autodesk affine-body-dynamics
# screw/nut assets and runs them as closed UIPC affine bodies with IPC contact.
#
# Command: python -m newton.examples uipc_nut_bolt
#
###########################################################################

import numpy as np
import trimesh
import uipc
import warp as wp

import newton
import newton.examples

ABD_REPO_URL = "https://github.com/Autodesk/affine-body-dynamics.git"
ABD_SCREW_NUT_FOLDER = "meshes/screw-and-nut"
SCREW_MESH_NAME = "screw-big.obj"
NUT_MESH_NAME = "nut-big.obj"

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
ORIGINAL_ASSEMBLY_STR = "m20_loose"
ORIGINAL_BOLT_MESH_NAME = f"factory_bolt_{ORIGINAL_ASSEMBLY_STR}.obj"
ORIGINAL_NUT_MESH_NAME = f"factory_nut_{ORIGINAL_ASSEMBLY_STR}_subdiv_3x.obj"

AUTODESK_MESH_SOURCE = "autodesk"
ORIGINAL_MESH_SOURCE = "original"
ORIGIN_MESH_SOURCE_ALIAS = "origin"
MESH_SOURCE_CHOICES = (AUTODESK_MESH_SOURCE, ORIGINAL_MESH_SOURCE, ORIGIN_MESH_SOURCE_ALIAS)

# Autodesk ABD meshes are authored in millimeters with the screw axis along Y.
# Scale the Autodesk screw head radius to the IsaacGym M20 bolt head radius
# and rotate the screw axis to Newton's Z-up world with the screw head below
# the threaded shaft. Use a proper right-handed rotation to preserve handedness.
AUTODESK_MESH_SCALE = 0.00326
ORIGINAL_MESH_SCALE = 1.0
# The original IsaacGym visual meshes are tight enough that UIPC surface IPC
# can wedge at the first thread when d_hat=1e-4. Add a small radial-only
# clearance for the threaded surfaces while preserving the authored Z pitch.
ORIGINAL_BOLT_RADIAL_SCALE = 0.955
ORIGINAL_NUT_RADIAL_SCALE = 1.0
# Use the requested UIPC contact thickness for threaded contact. Nut start
# heights below are measured with this gap to avoid initial mesh intersection.
UIPC_GAP = 0.0001
ASSEMBLY_SPACING = 0.1
AUTODESK_BOLT_START_Z = 0.048
AUTODESK_NUT_START_Z = 0.05676
ORIGINAL_BOLT_START_Z = 0.0
# The SDF original starts the nut at 0.041 m, with slight initial mesh overlap.
# UIPC rejects intersecting surface meshes during sanity checks, so use the
# lowest tested non-intersecting start heights for UIPC_GAP=1e-4.
ORIGINAL_NUT_START_Z = 0.04262
NUT_START_YAW = np.pi / 8.0
MIN_NUT_DROP_BY_MESH_SOURCE = {
    AUTODESK_MESH_SOURCE: 0.005,
    ORIGINAL_MESH_SOURCE: 0.004,
}
MIN_NUT_ROTATION = 1.0
# Threaded UIPC contact needs tighter solve tolerances than the defaults;
# otherwise the first thread contact is accepted as converged after only a tiny
# displacement, making the nut appear stuck instead of sliding/rotating.
UIPC_SOLVE_TOL = 1.0e-5
# UIPC Coulomb friction resists the very small gravity-driven tangent component
# on M20-scale thread flanks. Use frictionless nut/bolt contact so the screw
# geometry, not an external drive, converts gravity into rotation.
THREAD_CONTACT_MU = 0.0
# UIPC ABD stiffness override for the small steel parts. The solver default is
# 1 GPa; the higher value keeps the thread profile closer to rigid while still
# avoiding the very stiff 10 GPa setting used by larger brick examples.
NUT_BOLT_ABD_KAPPA = 2.0 * uipc.unit.GPa

SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=THREAD_CONTACT_MU,
    ke=1.0e7,
    kd=1.0e4,
    gap=UIPC_GAP,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
)


def _canonical_mesh_source(mesh_source: str) -> str:
    """Return the canonical mesh source name for aliases."""
    if mesh_source == ORIGIN_MESH_SOURCE_ALIAS:
        return ORIGINAL_MESH_SOURCE
    return mesh_source


def _load_centered_mesh(
    mesh_file: str,
    scale: float,
    *,
    rotate_y_axis_to_z: bool = False,
    radial_scale: float = 1.0,
) -> tuple[newton.Mesh, wp.vec3]:
    """Load an OBJ mesh and recenter it around its AABB center.

    Args:
        mesh_file: Mesh file path.
        scale: Uniform mesh scale [unitless].
        rotate_y_axis_to_z: Whether to rotate vertices from a Y-axis screw to a
            Z-up screw with the screw head at the lower Z end while preserving
            mesh handedness.
        radial_scale: Scale applied to centered XY coordinates only [unitless].

    Returns:
        Tuple of ``(mesh, center_vec)`` where ``center_vec`` is the local AABB
        center offset [m] that must be added to the body transform.
    """
    mesh_data = trimesh.load(mesh_file, force="mesh", process=True)
    mesh_data.merge_vertices()

    vertices = np.asarray(mesh_data.vertices, dtype=np.float32)
    if rotate_y_axis_to_z:
        vertices = np.ascontiguousarray(np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1])))
    faces = np.asarray(mesh_data.faces, dtype=np.int32)

    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = (vertices - center).astype(np.float32, copy=False)
    vertices[:, 0:2] *= np.float32(radial_scale)
    mesh = newton.Mesh(vertices, faces.reshape(-1))
    return mesh, wp.vec3(center) * float(scale)


def _transform_with_mesh_center(base: wp.transform, center_vec: wp.vec3) -> wp.transform:
    """Shift a body transform so the recentered mesh appears at the authored pose."""
    return wp.transform(base.p + wp.quat_rotate(base.q, center_vec), base.q)


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.world_count = args.world_count
        self.num_per_world = args.num_per_world
        self.mesh_source = _canonical_mesh_source(getattr(args, "mesh_source", AUTODESK_MESH_SOURCE))
        if self.mesh_source not in MESH_SOURCE_CHOICES:
            raise ValueError(f"Unknown mesh source: {self.mesh_source}. Choose from {MESH_SOURCE_CHOICES}.")
        self.test_mode = bool(getattr(args, "test", False))
        self.viewer = viewer

        self.grid_x = int(np.ceil(np.sqrt(self.num_per_world)))
        self.grid_y = int(np.ceil(self.num_per_world / self.grid_x))
        self.spacing = ASSEMBLY_SPACING

        world_builder = self._build_nut_bolt_scene()
        self.bodies_per_world = world_builder.body_count

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(world_builder, world_count=self.world_count, spacing=(0.15, 0.15, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            workspace=f"/tmp/newton_uipc/uipc_nut_bolt_{self.mesh_source}",
            dump_enable=True,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
        )
        self.solver.configure_scene(
            {
                "linear_system": {"tol_rate": UIPC_SOLVE_TOL},
                "newton": {
                    "transrate_tol": UIPC_SOLVE_TOL,
                    "velocity_tol": UIPC_SOLVE_TOL,
                },
            }
        )
        self.solver.set_contact(True, UIPC_GAP)
        self.solver.configure_contact_tabular(self._configure_contact_tabular)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._init_test_tracking()

        self.solver.initialize(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0.15, 0.15, 0.0))
        camera_offset = max(1.0, np.sqrt(self.world_count) * 0.15 * 1.5)
        self.viewer.set_camera(
            pos=wp.vec3(camera_offset, -camera_offset, 0.5 * camera_offset),
            pitch=-15.0,
            yaw=135.0,
        )
        self.viewer._paused = True

    def _load_mesh_source(self):
        if self.mesh_source == AUTODESK_MESH_SOURCE:
            print("Downloading Autodesk screw/nut assets...")
            asset_path = newton.examples.download_external_git_folder(ABD_REPO_URL, ABD_SCREW_NUT_FOLDER)
            print(f"Assets downloaded to: {asset_path}")

            bolt_file = str(asset_path / SCREW_MESH_NAME)
            nut_file = str(asset_path / NUT_MESH_NAME)
            bolt_mesh, bolt_center = _load_centered_mesh(
                bolt_file,
                AUTODESK_MESH_SCALE,
                rotate_y_axis_to_z=True,
            )
            nut_mesh, nut_center = _load_centered_mesh(
                nut_file,
                AUTODESK_MESH_SCALE,
                rotate_y_axis_to_z=True,
            )
            return (
                bolt_mesh,
                bolt_center,
                nut_mesh,
                nut_center,
                AUTODESK_MESH_SCALE,
                AUTODESK_BOLT_START_Z,
                AUTODESK_NUT_START_Z,
            )

        print("Downloading original IsaacGym nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        bolt_file = str(asset_path / ORIGINAL_BOLT_MESH_NAME)
        nut_file = str(asset_path / ORIGINAL_NUT_MESH_NAME)
        bolt_mesh, bolt_center = _load_centered_mesh(
            bolt_file,
            ORIGINAL_MESH_SCALE,
            radial_scale=ORIGINAL_BOLT_RADIAL_SCALE,
        )
        nut_mesh, nut_center = _load_centered_mesh(
            nut_file,
            ORIGINAL_MESH_SCALE,
            radial_scale=ORIGINAL_NUT_RADIAL_SCALE,
        )
        return (
            bolt_mesh,
            bolt_center,
            nut_mesh,
            nut_center,
            ORIGINAL_MESH_SCALE,
            ORIGINAL_BOLT_START_Z,
            ORIGINAL_NUT_START_Z,
        )

    def _build_nut_bolt_scene(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)

        bolt_mesh, bolt_center, nut_mesh, nut_center, mesh_scale, bolt_start_z, nut_start_z = self._load_mesh_source()

        count = 0
        for i in range(self.grid_x):
            if count >= self.num_per_world:
                break
            for j in range(self.grid_y):
                if count >= self.num_per_world:
                    break

                x_offset = (i - (self.grid_x - 1) / 2.0) * self.spacing
                y_offset = (j - (self.grid_y - 1) / 2.0) * self.spacing

                bolt_pose = wp.transform(wp.vec3(x_offset, y_offset, bolt_start_z + UIPC_GAP), wp.quat_identity())
                bolt_body = builder.add_body(
                    xform=_transform_with_mesh_center(bolt_pose, bolt_center),
                    label=f"bolt_{i}_{j}",
                    is_kinematic=True,
                    custom_attributes={"uipc:abd_kappa": NUT_BOLT_ABD_KAPPA},
                )
                builder.add_shape_mesh(
                    body=bolt_body,
                    mesh=bolt_mesh,
                    scale=(mesh_scale, mesh_scale, mesh_scale),
                    cfg=SHAPE_CFG,
                    label=f"bolt_{i}_{j}",
                )

                nut_pose = wp.transform(
                    wp.vec3(x_offset, y_offset, nut_start_z + UIPC_GAP),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(NUT_START_YAW)),
                )
                nut_body = builder.add_body(
                    xform=_transform_with_mesh_center(nut_pose, nut_center),
                    label=f"nut_{i}_{j}",
                    custom_attributes={"uipc:abd_kappa": NUT_BOLT_ABD_KAPPA},
                )
                builder.add_shape_mesh(
                    body=nut_body,
                    mesh=nut_mesh,
                    scale=(mesh_scale, mesh_scale, mesh_scale),
                    cfg=SHAPE_CFG,
                    label=f"nut_{i}_{j}",
                )

                count += 1

        return builder

    def _configure_contact_tabular(self, contact_tabular, _world_index, ground_elem, env_elem, _robo_elem, actor_elem):
        # Bolts are kinematic ``env`` bodies, nuts are free-joint ``actor`` bodies.
        # Keep ground/nut contact enabled for safety but disable bolt/bolt and
        # environment self-contact to avoid unnecessary IPC pairs.
        GPa = 1.0e9
        contact_tabular.insert(env_elem, env_elem, 0.5, GPa, False)
        contact_tabular.insert(env_elem, actor_elem, SHAPE_CFG.mu, GPa, True)
        contact_tabular.insert(ground_elem, env_elem, 0.5, GPa, False)
        contact_tabular.insert(ground_elem, actor_elem, 0.5, GPa, True)
        contact_tabular.insert(actor_elem, actor_elem, SHAPE_CFG.mu, GPa, False)
        return None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._track_test_data()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _init_test_tracking(self):
        body_q = self.state_0.body_q.numpy()
        nut_body_indices = [i for i, label in enumerate(self.model.body_label) if label.startswith("nut_")]
        self.nut_initial_by_body = {idx: body_q[idx].copy() for idx in nut_body_indices}

        if not self.test_mode:
            self.bolt_body_indices = None
            self.nut_body_indices = None
            return

        self.bolt_body_indices = [i for i, label in enumerate(self.model.body_label) if label.startswith("bolt_")]
        self.nut_body_indices = nut_body_indices
        self.bolt_initial_transforms = [body_q[idx].copy() for idx in self.bolt_body_indices]
        self.nut_initial_transforms = [body_q[idx].copy() for idx in self.nut_body_indices]
        self.nut_max_rotation_change = [0.0] * len(self.nut_body_indices)
        self.nut_min_z = [float(body_q[idx][2]) for idx in self.nut_body_indices]

    def _track_test_data(self):
        if not self.test_mode:
            return

        body_q = self.state_0.body_q.numpy()
        for i, nut_idx in enumerate(self.nut_body_indices):
            current_q = body_q[nut_idx]
            initial_q = self.nut_initial_transforms[i]
            q_current = current_q[3:7]
            q_initial = initial_q[3:7]
            dot = min(abs(float(np.dot(q_current, q_initial))), 1.0)
            rotation_angle = 2.0 * np.arccos(dot)
            self.nut_max_rotation_change[i] = max(self.nut_max_rotation_change[i], rotation_angle)
            self.nut_min_z[i] = min(self.nut_min_z[i], float(current_q[2]))

    def test_final(self):
        assert self.bolt_body_indices is not None and self.nut_body_indices is not None
        assert len(self.bolt_body_indices) == self.world_count * self.num_per_world
        assert len(self.nut_body_indices) == self.world_count * self.num_per_world

        body_q = self.state_0.body_q.numpy()
        max_bolt_displacement = 0.005
        for i, bolt_idx in enumerate(self.bolt_body_indices):
            displacement = np.linalg.norm(body_q[bolt_idx][:3] - self.bolt_initial_transforms[i][:3])
            assert displacement < max_bolt_displacement, (
                f"Bolt {i}: displaced {displacement:.4f} m (max allowed {max_bolt_displacement:.4f} m)"
            )

        min_drop = MIN_NUT_DROP_BY_MESH_SOURCE[self.mesh_source]
        for i, nut_idx in enumerate(self.nut_body_indices):
            assert np.all(np.isfinite(body_q[nut_idx])), f"Nut {i}: non-finite transform {body_q[nut_idx]}"

            drop = self.nut_initial_transforms[i][2] - self.nut_min_z[i]
            assert drop > min_drop, f"Nut {i}: dropped {drop:.4f} m (expected > {min_drop:.4f} m)"
            rotation = self.nut_max_rotation_change[i]
            assert rotation > MIN_NUT_ROTATION, (
                f"Nut {i}: rotated {np.degrees(rotation):.2f} degrees "
                f"(expected > {np.degrees(MIN_NUT_ROTATION):.2f} degrees)"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1, num_frames=120)
        parser.add_argument(
            "--num-per-world",
            type=int,
            default=1,
            help="Number of nut/bolt assemblies per world.",
        )
        parser.add_argument(
            "--mesh-source",
            choices=MESH_SOURCE_CHOICES,
            default=AUTODESK_MESH_SOURCE,
            help=(
                "Mesh source: 'autodesk' uses Autodesk ABD screw/nut meshes; "
                "'original' uses the IsaacGym factory meshes from the SDF example "
                "('origin' is accepted as an alias)."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
