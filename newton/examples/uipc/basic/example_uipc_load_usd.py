# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Load USD
#
# Generic ``USD -> SolverUIPC`` loader driven from the command line.  Given
# a ``--usd-path``, the example builds a :class:`~newton.ModelBuilder` via
# :meth:`~newton.ModelBuilder.add_usd`, steps the articulation under
# :class:`~newton.solvers.SolverUIPC`, and verifies the resulting body state
# stays finite in :meth:`test_final`.
#
# Command:
#   python -m newton.examples uipc_load_usd --usd-path <path-to-usd>
#   python -m newton.examples uipc_load_usd  # uses the default fixture
#
###########################################################################

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import uipc
import warp as wp

import newton
import newton.examples

_DEFAULT_USD = str(Path(newton.__file__).parent / "tests" / "assets" / "four_link_chain_articulation.usda")


class Example:
    def __init__(self, viewer, args: argparse.Namespace):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.viewer = viewer

        usd_path = str(args.usd_path)
        if not Path(usd_path).exists():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_usd(
            usd_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
            collapse_fixed_joints=bool(args.collapse_fixed_joints),
            floating=bool(args.floating),
            hide_collision_shapes=True,
        )
        if args.add_ground:
            builder.add_ground_plane()

        self.model = builder.finalize()

        # FK to sync body_q from the authored joint_q -- guard the optional
        # arrays so type checkers understand they are populated after
        # ``finalize()`` on a model that has bodies.
        assert self.model.joint_q is not None and self.model.joint_qd is not None
        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)  # ty:ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        self.solver = newton.solvers.SolverUIPC(
            workspace="/tmp/newton_uipc/load_usd",
            model=self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Info,
            dump_enable=True,
        )
        self.solver.initialize(self.state_0)

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(3.0, 3.0, 2.0),
            pitch=-15.0,
            yaw=-135.0,
        )
        self.viewer._paused = True

        print(
            f"[uipc_load_usd] loaded '{usd_path}': "
            f"{self.model.body_count} bodies, {self.model.joint_count} joints, "
            f"{self.model.joint_dof_count} DOFs"
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
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
        assert self.state_0.body_q is not None
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(body_q)), f"body_q became non-finite: {body_q}"
        max_pos = float(np.abs(body_q[:, :3]).max()) if body_q.size else 0.0
        assert max_pos < 1.0e3, f"body positions exploded: max={max_pos:.3f}"

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--usd-path",
            type=str,
            default=_DEFAULT_USD,
            help=(
                "Path to the USD/USDA/USDC file to load. Defaults to the "
                "four-link-chain articulation fixture shipped with Newton."
            ),
        )
        parser.add_argument(
            "--floating",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Add a FREE (D6) root joint if the USD has none. Disabled by "
                "default because SolverUIPC does not currently support D6 "
                "joints -- use --no-floating (default) to get a fixed root."
            ),
        )
        parser.add_argument(
            "--collapse-fixed-joints",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Collapse fixed joints during USD import.",
        )
        parser.add_argument(
            "--add-ground",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Add a ground plane underneath the imported articulation.",
        )
        parser.add_argument(
            "--height",
            type=float,
            default=1.0,
            help="Z offset applied to the imported articulation root [m].",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
