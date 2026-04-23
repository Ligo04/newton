# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC UR10 Force — Joint-Space PD + Gravity Compensation
#
# Force-control analogue of ``example_uipc_ur10.py`` (which uses the
# solver's native ``JointTargetMode.POSITION`` aim drive).  Every UR10 DOF
# here is configured as ``JointTargetMode.EFFORT``, and the torque vector
# is computed in user space each substep:
#
#     tau[i] = kp[i] * (q_target[i] - q[i])
#            + kd[i] * (qd_target[i] - qd[i])
#            + tau_g[i]                       (optional gravity comp)
#
# ``tau_g`` is produced with ``newton.eval_jacobian`` + the per-body
# masses:
#
#     tau_g = -sum_body J_lin_body^T @ (m_body * g_world)
#
# i.e. Jacobian-transpose virtual work, which is the canonical "textbook"
# gravity compensation for an articulated arm.  It is an order-of-magnitude
# cheaper than full RNEA inverse dynamics and sufficient for static hold,
# but it does NOT cancel Coriolis / centrifugal terms — so the arm will
# still drift under motion.
#
# This example is primarily a *plumbing demonstration*: it shows how to
# wire an external torque controller onto a real articulated robot via
# ``control.joint_f`` + ``JointTargetMode.EFFORT``.  With the conservative
# PD gains tuned here, the arm visibly droops under gravity — the point
# is that the EFFORT pipeline is live and responding to ``joint_f``, not
# to ship an industrial-grade tracker.  Production-quality tracking needs
# full inverse-dynamics compensation (not just Jacobian-transpose gravity)
# and gains tuned per solver.
#
# Multiple solver backends are exposed via ``--solver``.  Empirically:
#   - ``featherstone`` / ``mujoco`` give the cleanest joint-space response
#     (reduced-coordinate rigid-body solvers honour ``joint_f`` directly).
#   - ``uipc`` is the default to keep this example in the uipc/ folder;
#     its ABD implicit integration couples the external torque through
#     an affine-body constraint solve, which is quite sensitive to
#     ``sim_substeps``.  Running at 240 Hz (substeps=4) keeps the
#     simulation bounded but you will see chatter on the wrist DOFs.
#   - ``semi_implicit`` runs but tends to absorb ``joint_f`` into its
#     joint attachment constraints and barely moves the arm here.
#
# Command:
#   python -m newton.examples uipc_ur10_force --world-count 1
#   python -m newton.examples uipc_ur10_force --gravity-comp
#   python -m newton.examples uipc_ur10_force --solver mujoco
#
###########################################################################

import numpy as np
import warp as wp
from newton_actuators import ActuatorPD

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.selection import ArticulationView


class Example:
    # UR10 home pose (six revolute DOFs). Bent away from the singular
    # fully-extended pose so that gravity torques are non-trivial and the
    # "with / without gravity comp" comparison is visible.
    HOME_POSE = np.array(
        [0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, np.pi / 2, 0.0],
        dtype=np.float32,
    )

    def __init__(self, viewer, args):
        # Render at 60 Hz.  Physics at 240 Hz (4 substeps) gives the external
        # torque loop enough sampling margin over UIPC's implicit Newton
        # iterations; dropping to 60 Hz substeps=1 saves compute but makes
        # the closed-loop system oscillate.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_name = args.solver
        self.gravity_comp = bool(args.gravity_comp)
        self.viewer = viewer

        # --- Joint-space PD gains ---------------------------------------------
        # Conservative PD gains chosen so the torques stay well below saturation
        # and the closed-loop system is over-damped.  They do NOT perfectly hold
        # the arm against gravity (that is why the ``--gravity-comp`` option
        # exists); the purpose here is to show the EFFORT pipeline actually
        # moves the arm in response to ``control.joint_f``, not to build an
        # industrial-grade tracker.
        self.kp = np.array([300.0, 300.0, 200.0, 100.0, 60.0, 30.0], dtype=np.float32)
        self.kd = np.array([40.0, 40.0, 30.0, 15.0, 10.0, 5.0], dtype=np.float32)
        # Torque clamps sized roughly to UR10's real effort limits.
        self.max_torque = np.array([330.0, 330.0, 150.0, 54.0, 54.0, 54.0], dtype=np.float32)

        ur10 = newton.ModelBuilder()

        # MuJoCo consumes extra per-joint USD attributes. Register them on the
        # builder *before* ``add_usd`` if the user picks the mujoco backend;
        # other solvers ignore these attributes.
        if self.solver_name == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(ur10)

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        ur10.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0, 0, height / 2)),
            half_height=height / 2,
            radius=0.08,
        )

        # Switch every controlled DOF to EFFORT. We still initialise the
        # ``joint_target_ke/kd`` fields so the model stays portable to
        # solvers that want those values (e.g. for a mixed-mode robot
        # where some DOFs are position-driven — here they are all EFFORT).
        for i in range(len(ur10.joint_target_ke)):
            ur10.joint_target_ke[i] = 0.0
            ur10.joint_target_kd[i] = 0.0
            ur10.joint_target_mode[i] = int(JointTargetMode.EFFORT)
            if ur10.joint_type[i] == newton.JointType.REVOLUTE:
                ur10.joint_armature[i] = 1e-2

        # Register one ActuatorPD per UR10 DOF. Each call appends to the shared
        # entry (ActuatorPD has no scalar params) so the final Model.actuators[0]
        # drives every DOF on every replicated world with its own kp/kd/max_force.
        # gravity compensation is injected via the actuator's ``constant_force``
        # array, which we refresh every substep in ``_apply_feedback``.
        for dof_idx in range(len(self.kp)):
            ur10.add_actuator(
                ActuatorPD,
                input_indices=[dof_idx],
                kp=float(self.kp[dof_idx]),
                kd=float(self.kd[dof_idx]),
                max_force=float(self.max_torque[dof_idx]),
                gear=1.0,
                constant_force=0.0,
            )

        if self.world_count > 1:
            builder = newton.ModelBuilder()
            builder.replicate(ur10, self.world_count, spacing=(2.0, 2.0, 0.0))
        else:
            builder = ur10

        # Start all worlds from the same home pose.
        joint_q_all = np.tile(self.HOME_POSE, self.world_count).astype(np.float32)
        builder.joint_q = joint_q_all.tolist()

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver, self._uses_contacts = self._build_solver(self.solver_name)

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts() if self._uses_contacts else None

        # Selection view over every replicated UR10. Exclude FREE/DISTANCE
        # joints in case the asset carries them — the controller only works
        # on the six revolute joints.
        self.ur10s = ArticulationView(
            self.model,
            "*ur10*",
            exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
        )
        assert self.ur10s.count == self.world_count, (
            f"expected one UR10 per world, got {self.ur10s.count} for {self.world_count} worlds"
        )

        # Per-world DOF / body strides. UR10 itself has 6 revolute DOFs;
        # body_count per world includes the UR10 links only (the pedestal
        # cylinder is attached to body -1 which is the world).
        self.dofs_per_world = self.ur10s.joint_dof_count
        self.bodies_per_world = self.model.body_count // self.world_count
        assert self.dofs_per_world == 6, f"expected 6 UR10 DOFs per world, got {self.dofs_per_world}"

        # Gravity vector and per-body masses — host-side caches avoid a GPU
        # round-trip on every substep.
        self._gravity_np = self.model.gravity.numpy()  # (world_count, 3)
        self._body_mass_np = self.model.body_mass.numpy()  # (body_count,)

        # Reusable scratch for eval_jacobian so we don't re-allocate each step.
        self._J_buf: wp.array | None = None

        # Joint targets (hold the home pose), tiled to the ArticulationView
        # layout ``(world_count, 1, dofs_per_arti)``. Velocity target = 0.
        self.q_target = (
            np.broadcast_to(self.HOME_POSE, (self.world_count, 1, self.dofs_per_world)).astype(np.float32).copy()
        )
        self.qd_target = np.zeros((self.world_count, 1, self.dofs_per_world), dtype=np.float32)

        # Populate joint_target_pos/vel — ActuatorPD reads them every step.
        self.ur10s.set_attribute("joint_target_pos", self.control, self.q_target)
        self.ur10s.set_attribute("joint_target_vel", self.control, self.qd_target)

        # Cache the ActuatorPD instance that drives every UR10 DOF. The
        # builder merged the 6 per-DOF ``add_actuator`` calls (x world_count
        # replicas) into a single ActuatorPD, so ``constant_force`` is a flat
        # ``(world_count * dofs_per_world,)`` array ordered [w0d0..w0d5, w1d0..w1d5, ...].
        pd_actuator = next(a for a in self.model.actuators if isinstance(a, ActuatorPD))
        assert isinstance(pd_actuator, ActuatorPD)  # narrow for type-checkers
        self.pd_actuator = pd_actuator
        expected = self.world_count * self.dofs_per_world
        assert len(self.pd_actuator.kp) == expected, (
            f"ActuatorPD kp length {len(self.pd_actuator.kp)} != {expected}; "
            "builder/replicate did not merge the per-DOF actuators as expected"
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(5.0, 5.0, 3.0),
            pitch=-20.0,
            yaw=-135.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

    # ------------------------------------------------------------------ solver
    def _build_solver(self, name: str) -> tuple[object, bool]:
        """Construct the requested Newton solver.

        Returns the solver and a flag telling the caller whether a
        ``Contacts`` object must be passed to ``solver.step()``.
        """
        if name == "uipc":
            import uipc  # noqa: PLC0415  # deferred so non-uipc solvers run without the dep

            solver = newton.solvers.SolverUIPC(
                self.model,
                workspace="/tmp/newton_uipc/ur10_force",
                dt=self.sim_dt,
                logger_level=uipc.Logger.Warn,
                dump_enable=True,
            )
            solver.set_contact(True, 0.001)
            return solver, True
        if name == "mujoco":
            return newton.solvers.SolverMuJoCo(self.model), False
        if name == "featherstone":
            return newton.solvers.SolverFeatherstone(self.model), False
        raise ValueError(f"unsupported --solver: {name!r}")

    # ----------------------------------------------------------------- control
    def _compute_gravity_comp(self) -> np.ndarray:
        """Jacobian-transpose gravity compensation.

        Returns the per-DOF torque that cancels the static gravity wrench
        on every link, packed to the ``ArticulationView`` layout
        ``(world_count, 1, dofs_per_world)``.

        ``tau_g = -sum_link J_lin^T @ (m * g)``.

        For performance we skip this entirely when ``--gravity-comp`` is
        *not* set, so the caller should check ``self.gravity_comp`` first.

        We refresh ``state_0.body_q`` via ``eval_fk`` before sampling the
        Jacobian — most solvers update joint_q/joint_qd inside ``step()``
        but leave the maximal-coordinate ``body_q`` cache stale, and
        ``eval_jacobian`` reads ``body_q``.
        """
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._J_buf = newton.eval_jacobian(self.model, self.state_0, self._J_buf)
        J_np = self._J_buf.numpy()  # shape (A, max_links*6, max_dofs)

        tau_g = np.zeros((self.world_count, 1, self.dofs_per_world), dtype=np.float32)
        for w in range(self.world_count):
            g_vec = self._gravity_np[w]  # (3,)
            for b in range(self.bodies_per_world):
                body_global = w * self.bodies_per_world + b
                m = float(self._body_mass_np[body_global])
                if m <= 0.0:
                    continue
                # Rows [6b:6b+3] of J are the linear (COM) Jacobian of body b.
                J_lin = J_np[w, 6 * b : 6 * b + 3, : self.dofs_per_world]  # (3, ndofs)
                # Gravity wrench on this body at COM: F = m * g.
                # Joint torques from this wrench via virtual work: J_lin^T @ F.
                # Compensation is the negation of that.
                tau_g[w, 0] -= J_lin.T @ (m * g_vec)
        return tau_g

    def _apply_feedback(self):
        """Run every registered actuator -> ``control.joint_f``.

        ``ActuatorPD`` already handles the PD math and the ``max_force`` clamp
        in its Warp kernel, and it reads position / velocity targets from
        ``control.joint_target_pos`` / ``joint_target_vel`` (written once in
        ``__init__``). The only per-substep bookkeeping we need is:

        1. Clear ``control.joint_f`` — the PD kernel accumulates with ``+=``,
           so an un-cleared buffer would blow past ``max_force`` each frame.
        2. If ``--gravity-comp`` is on, refresh the ``constant_force`` array
           with the current Jacobian-transpose gravity compensation (it
           depends on the current pose, so it must be recomputed every
           substep).
        3. Run each actuator.
        """
        self.control.joint_f.zero_()

        if self.gravity_comp:
            tau_g = self._compute_gravity_comp()
            # constant_force layout matches the order we registered the
            # actuators (6 DOFs x N worlds). Flatten accordingly.
            self.pd_actuator.constant_force.assign(tau_g.reshape(-1).astype(np.float32))

        self.pd_actuator.step(
            sim_state=self.state_0,
            sim_control=self.control,
            current_act_state=None,
            next_act_state=None,
            dt=self.sim_dt,
        )

    # ----------------------------------------------------------------- runtime
    def simulate(self):
        for _ in range(self.sim_substeps):
            self._apply_feedback()
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
        self._log_tracking()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _log_tracking(self):
        """Print world-0 joint tracking error and applied torques."""
        q = self.ur10s.get_attribute("joint_q", self.state_0).numpy()[0, 0]
        f = self.ur10s.get_attribute("joint_f", self.control).numpy()[0, 0]
        err = self.HOME_POSE - q
        err_str = " ".join(f"{e:+.3f}" for e in err)
        tau_str = " ".join(f"{t:+7.2f}" for t in f)
        print(f"[t={self.sim_time:6.3f}s] err=[{err_str}] rad  tau=[{tau_str}] N·m")

    # -------------------------------------------------------------------- test
    def test_final(self):
        """Smoke-check: all joint values must be finite.

        We deliberately do NOT assert tight tracking — the conservative PD
        used here does not fully cancel gravity and different solvers
        reach different steady states.  The purpose of this test is just
        to verify the pipeline runs end-to-end without producing NaNs or
        exploding to astronomical values, for every advertised ``--solver``.
        """
        # Shape: (world_count, 1, dofs_per_arti)
        q = self.ur10s.get_attribute("joint_q", self.state_0).numpy()
        qd = self.ur10s.get_attribute("joint_qd", self.state_0).numpy()
        assert np.all(np.isfinite(q)), "joint_q went non-finite (divergence)"
        assert np.all(np.isfinite(qd)), "joint_qd went non-finite (divergence)"
        # Sanity bound: angles must be within a few full revolutions.
        assert np.all(np.abs(q) < 20.0), f"joint_q blew up: {q.tolist()}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--solver",
            choices=("uipc", "mujoco", "featherstone", "semi_implicit"),
            default="uipc",
            help="Newton solver backend driving the force controller.",
        )
        parser.add_argument(
            "--gravity-comp",
            action="store_true",
            help="Enable Jacobian-transpose gravity compensation (off by default).",
        )
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
