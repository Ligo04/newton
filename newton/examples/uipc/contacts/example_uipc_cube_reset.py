# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cube Reset
#
# Several cubes are dropped onto a static table in each of several parallel
# worlds, with random per-world layouts. Every reset interval a subset of
# worlds is re-randomized and pushed back into the live UIPC scene via
# SolverUIPC.reset(state, world_mask) -- without rebuilding the solver --
# while the remaining worlds keep simulating.
#
# Command: python -m newton.examples uipc_cube_reset
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

NUM_CUBES = 5
CUBE_HALF = 0.03  # m
TABLE_HX = 0.30
TABLE_HY = 0.30
TABLE_HZ = 0.05
TABLE_TOP_Z = 2.0 * TABLE_HZ  # table body sits with base at z=0
UIPC_GAP = 0.001
RESET_INTERVAL = 60  # frames
MIN_SEPARATION = 3.0 * CUBE_HALF
DEFAULT_NUM_FRAMES = 240


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.test_mode = getattr(args, "test", False)
        self.world_count = int(getattr(args, "world_count", 4))
        self.num_cubes = int(getattr(args, "num_cubes", NUM_CUBES))
        self.reset_interval = int(getattr(args, "reset_interval", RESET_INTERVAL))
        self.uipc_gap = float(getattr(args, "uipc_gap", UIPC_GAP))
        self.rng = np.random.default_rng(int(getattr(args, "seed", 0)))
        self.frame = 0

        world_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(world_builder)
        self._add_table(world_builder)
        self.cube_bodies: list[int] = []
        self._add_cubes(world_builder)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverUIPC.register_custom_attributes(builder)
        builder.replicate(world_builder, world_count=self.world_count, spacing=(1.0, 1.0, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Per-body world index + kinematic flag, for masked reset writes.
        self.body_world = self.model.body_world.numpy()
        self.body_flags = self.model.body_flags.numpy()

        # body_q is in the global frame: builder.replicate offsets each world by
        # a per-world translation. Recover each world's offset from its static
        # table (authored local origin (0, 0, TABLE_HZ)) so reset() can place
        # cubes in the correct world rather than at the local origin.
        init_body_q = self.state_0.body_q.numpy()
        kinematic = int(newton.BodyFlags.KINEMATIC)
        table_local = np.array([0.0, 0.0, TABLE_HZ])
        self.world_offset = np.zeros((self.world_count, 3))
        for w in range(self.world_count):
            tables = [b for b in np.where(self.body_world == w)[0] if int(self.body_flags[b]) & kinematic]
            if tables:
                self.world_offset[w] = init_body_q[tables[0]][:3] - table_local

        self.solver = newton.solvers.SolverUIPC(self.model, dt=self.sim_dt)
        self.solver.set_contact(True, self.uipc_gap)
        self.solver.initialize(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.2, -1.2, 1.0), pitch=-30.0, yaw=135.0)

    def _add_table(self, builder: newton.ModelBuilder) -> None:
        table = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, TABLE_HZ), wp.quat_identity()),
            label="table",
            is_kinematic=True,
        )
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6)
        builder.add_shape_box(table, hx=TABLE_HX, hy=TABLE_HY, hz=TABLE_HZ, cfg=cfg, label="table")

    def _sample_layout(self) -> np.ndarray:
        """Return (num_cubes, 3) of x, y, yaw on the tabletop with min separation."""
        out = np.zeros((self.num_cubes, 3))
        placed: list[np.ndarray] = []
        margin = CUBE_HALF + 0.02
        for i in range(self.num_cubes):
            xy = np.zeros(2)
            for _ in range(200):
                xy = self.rng.uniform(
                    [-TABLE_HX + margin, -TABLE_HY + margin],
                    [TABLE_HX - margin, TABLE_HY - margin],
                )
                if all(np.linalg.norm(xy - p) >= MIN_SEPARATION for p in placed):
                    break
            placed.append(xy)
            yaw = self.rng.uniform(-np.pi, np.pi)
            out[i] = [xy[0], xy[1], yaw]
        return out

    def _layout_to_transform(self, layout_row: np.ndarray) -> wp.transform:
        z = TABLE_TOP_Z + CUBE_HALF + self.uipc_gap
        yaw = float(layout_row[2])
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
        return wp.transform(wp.vec3(float(layout_row[0]), float(layout_row[1]), z), q)

    def _add_cubes(self, builder: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.6)
        layout = self._sample_layout()
        for i in range(self.num_cubes):
            xf = self._layout_to_transform(layout[i])
            body = builder.add_body(xform=xf, label=f"cube_{i}")
            builder.add_shape_box(body, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF, cfg=cfg, label=f"cube_{i}")
            self.cube_bodies.append(body)

    def _dynamic_bodies_in_world(self, w: int) -> list[int]:
        kinematic = int(newton.BodyFlags.KINEMATIC)
        return [int(b) for b in np.where(self.body_world == w)[0] if not (int(self.body_flags[b]) & kinematic)]

    def _reset_worlds(self, world_ids: list[int]) -> None:
        body_q = self.state_0.body_q.numpy()
        for w in world_ids:
            layout = self._sample_layout()
            offset = self.world_offset[w]
            for k, b in enumerate(self._dynamic_bodies_in_world(w)[: self.num_cubes]):
                xf = self._layout_to_transform(layout[k])
                body_q[b] = [xf.p[0] + offset[0], xf.p[1] + offset[1], xf.p[2] + offset[2], *xf.q]
        self.state_0.body_q = wp.array(body_q, dtype=wp.transform, device=self.model.device)
        self.state_0.body_qd.zero_()
        mask = np.zeros(self.world_count, dtype=bool)
        mask[world_ids] = True
        self.solver.reset(self.state_0, world_mask=wp.array(mask, dtype=wp.bool, device=self.model.device))

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.frame += 1
        if self.frame % self.reset_interval == 0:
            parity = (self.frame // self.reset_interval) % 2
            world_ids = [w for w in range(self.world_count) if w % 2 == parity]
            if world_ids:
                self._reset_worlds(world_ids)
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        kinematic = int(newton.BodyFlags.KINEMATIC)
        dyn = [b for b in range(self.model.body_count) if not (int(self.body_flags[b]) & kinematic)]
        q = body_q[dyn]
        if not np.all(np.isfinite(q)):
            raise ValueError("Non-finite cube transform after simulation")
        # Cubes must rest on the static table (center ~TABLE_TOP_Z + CUBE_HALF),
        # not collapse with the table to the ground (~CUBE_HALF). The threshold
        # below the tabletop catches both fall-through and a non-static table.
        min_resting_z = TABLE_TOP_Z - CUBE_HALF
        if np.any(q[:, 2] < min_resting_z):
            raise ValueError(
                f"A cube settled below the table top (min z {float(q[:, 2].min()):.4f} m < {min_resting_z:.4f} m); "
                "the table may not be static"
            )

    def test_post_step(self) -> None:
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("Non-finite body transform during stepping")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument("--world-count", type=int, default=4)
        parser.add_argument("--num-cubes", type=int, default=NUM_CUBES)
        parser.add_argument("--reset-interval", type=int, default=RESET_INTERVAL)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--uipc-gap", type=float, default=UIPC_GAP)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
