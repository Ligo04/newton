# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC UR10
#
# Shows how to set up a simulation of a UR10 robot arm
# from a USD file using the SolverUIPC backend, and applies a sinusoidal
# trajectory to the joint targets.
#
# The task (home pose, trajectory, and PD gains) is deliberately identical
# to ``example_uipc_ur10_force.py`` so the drive flavours compare directly:
# aim drive / --implicit-pd here versus EFFORT + ControllerPD / --stable-pd
# there.
#
# --implicit-pd switches the joint drives from the gain-agnostic aim drive
# (strength from ``drive_strength_ratio``) to implicit PD with physical
# gain semantics: ``joint_target_ke`` / ``joint_target_kd`` act as
# stiffness and damping springs inside UIPC's incremental potential
# (gravity sag = tau/ke, kd damps transients).
#
# Command: python -m newton.examples uipc_ur10 --world-count 4
#          python -m newton.examples uipc_ur10 --implicit-pd [--hold]
#
###########################################################################

import numpy as np
import uipc
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode


class Example:
    # Shared task spec — keep in sync with example_uipc_ur10_force.py.
    # Gains match the original example_robot_ur10 (uniform 500/50).
    HOME_POSE = np.array([0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, np.pi / 2, 0.0], dtype=np.float32)
    KP = np.array([500.0] * 6, dtype=np.float32)
    KD = np.array([50.0] * 6, dtype=np.float32)
    TRAJ_AMP = 0.4  # [rad]
    TRAJ_OMEGA = 1.2  # [rad/s]
    TRAJ_PHASE = 0.8  # [rad] per DOF index

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.world_count = args.world_count
        self.implicit_pd = bool(args.implicit_pd)
        self.hold = bool(args.hold)
        self.viewer = viewer
        self._frame_count = 0

        ur10 = newton.ModelBuilder()

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        # ``floating=False`` welds the base to the world via an explicit FIXED
        # joint. The USD default is a D6 with all axes locked, which SolverUIPC
        # silently skips — and this example runs with contact disabled, so a
        # skipped base joint would leave the whole arm in free fall.
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        # Create a pedestal
        ur10.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0, 0, height / 2)), half_height=height / 2, radius=0.08)

        rev = 0
        for j in range(len(ur10.joint_type)):
            if ur10.joint_type[j] != newton.JointType.REVOLUTE:
                continue
            dof = ur10.joint_qd_start[j]
            ur10.joint_target_ke[dof] = float(self.KP[rev])
            ur10.joint_target_kd[dof] = float(self.KD[rev])
            ur10.joint_target_mode[dof] = int(JointTargetMode.POSITION)
            ur10.joint_armature[dof] = 1e-2
            rev += 1

        if self.world_count > 1:
            builder = newton.ModelBuilder()
            builder.replicate(ur10, self.world_count, spacing=(2.0, 2.0, 0.0))
        else:
            builder = ur10

        # Start every world from the shared home pose.
        builder.joint_q = np.tile(self.HOME_POSE, self.world_count).tolist()

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
            implicit_pd=self.implicit_pd,
        )
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.solver.initialize(self.state_0)

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        self.dof_per_world = self.model.joint_dof_count // self.world_count if self.world_count > 0 else 0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(5.0, 5.0, 3.0),
            pitch=-20.0,
            yaw=-135.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

    def _update_targets(self):
        """Apply the shared home-centered sinusoidal trajectory to all worlds."""
        if self.hold:
            target = self.HOME_POSE
        else:
            phases = self.TRAJ_PHASE * np.arange(self.dof_per_world, dtype=np.float32)
            target = self.HOME_POSE + self.TRAJ_AMP * np.sin(self.TRAJ_OMEGA * self.sim_time + phases)
        target_pos = np.tile(target.astype(np.float32), self.world_count)
        self.control.joint_target_q.assign(target_pos)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._update_targets()
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
        if self._frame_count % self.fps == 0:
            self._log_tracking()
        self._frame_count += 1
        self.sim_time += self.frame_dt

    def _log_tracking(self):
        """Print world-0 joint positions, velocities, and tracking error once per second."""
        n = self.dof_per_world
        q = self.state_0.joint_q.numpy()[:n]
        qd = self.state_0.joint_qd.numpy()[:n]
        target = self.control.joint_target_q.numpy()[:n]
        err = target - q
        q_str = " ".join(f"{v:+.4f}" for v in q)
        qd_str = " ".join(f"{v:+.4f}" for v in qd)
        err_str = " ".join(f"{e:+.3f}" for e in err)
        print(f"[t={self.sim_time:6.3f}s] q(rad)=[{q_str}]  qd(rad/s)=[{qd_str}]\n              err(rad)=[{err_str}]")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.joint_q.numpy()
        assert np.all(np.isfinite(q)), "joint_q went non-finite (divergence)"
        assert np.all(np.abs(q) < 20.0), f"joint_q blew up: {q.tolist()}"
        # The sinusoidal trajectory is slow relative to the drive stiffness,
        # so both drive flavours must stay within phase-lag distance of the
        # commanded target on every DOF.
        target = self.control.joint_target_q.numpy()
        err = np.abs(q - target)
        assert np.all(err < 0.5), f"tracking error too large: {err.tolist()}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--implicit-pd",
            action="store_true",
            help="Drive joints with implicit PD (physical joint_target_ke/kd semantics) instead of the aim drive.",
        )
        parser.add_argument(
            "--hold",
            action="store_true",
            help="Hold the home pose instead of tracking the shared sinusoid (settling / steady-state comparison).",
        )
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
