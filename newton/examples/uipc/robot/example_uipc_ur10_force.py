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
#     an affine-body constraint solve.  The light wrist DOFs limit-cycle
#     under plain PD unless reflected rotor inertia is added, so this
#     example sets ``Model.joint_armature`` (``self.armature`` below) —
#     SolverUIPC folds it into each child link's ABD inertia internally.
#   - ``semi_implicit`` runs but tends to absorb ``joint_f`` into its
#     joint attachment constraints and barely moves the arm here.
#
# --stable-pd swaps every actuator's controller to Tan 2011 stable PD.
# The implicit rewrite
#
#     (M + diag(Kd)·Δt)·qddot = Kp·(q* - q - q̇·Δt) + Kd·(q̇* - q̇) - C
#     τ = const + Kp·(q* - q - q̇·Δt) + Kd·(q̇* - q̇) - Kd·qddot·Δt
#
# needs the articulation mass matrix ``M(q)`` and bias forces
# ``C(q, q̇)`` fed into the controller state each substep.  Here we wire:
#
#     ctrl_state.mass_matrix = newton.eval_mass_matrix(model, state)
#     ctrl_state.bias_forces = newton.eval_inverse_dynamics(model, state)  # C(q,q̇)q̇ + g(q)
#
# ``eval_inverse_dynamics`` returns the exact RNEA bias force (gravity plus
# Coriolis/centrifugal) at q̈ = 0.  Since gravity is already baked into
# ``bias_forces``, --stable-pd is mutually exclusive with --gravity-comp.
#
# Command:
#   python -m newton.examples uipc_ur10_force --world-count 1
#   python -m newton.examples uipc_ur10_force --gravity-comp
#   python -m newton.examples uipc_ur10_force --stable-pd
#   python -m newton.examples uipc_ur10_force --solver mujoco
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.actuators import Actuator as _NewtonActuator
from newton.actuators import ClampingMaxEffort, ControllerPD, ControllerStablePD
from newton.selection import ArticulationView


class Example:
    # UR10 home pose (six revolute DOFs). Bent away from the singular
    # fully-extended pose so that gravity torques are non-trivial and the
    # "with / without gravity comp" comparison is visible.
    HOME_POSE = np.array(
        [0.0, -np.pi / 3, np.pi / 2, -np.pi / 6, np.pi / 2, 0.0],
        dtype=np.float32,
    )
    # Shared task spec — keep in sync with example_uipc_ur10.py, so the
    # EFFORT drive flavours here (ControllerPD / --stable-pd) compare
    # directly against that example's aim drive / --implicit-pd.
    TRAJ_AMP = 0.4  # [rad]
    TRAJ_OMEGA = 1.2  # [rad/s]
    TRAJ_PHASE = 0.8  # [rad] per DOF index

    def __init__(self, viewer, args):
        # Render at 60 Hz.  Physics at 240 Hz (4 substeps) gives the external
        # torque loop enough sampling margin over UIPC's implicit Newton
        # iterations; dropping to 60 Hz substeps=1 saves compute but makes
        # the closed-loop system oscillate.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_name = args.solver
        self.stable_pd = bool(args.stable_pd)
        self.hold = bool(args.hold)
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
        # Reflected rotor inertia per revolute DOF (I_rotor * gear_ratio^2,
        # ballpark for UR10-class harmonic drives). The heavier wrist values
        # are what keeps plain PD free of the high-gain limit cycle on the
        # light wrist links under the uipc backend.
        self.armature = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2], dtype=np.float32)

        ur10 = newton.ModelBuilder()

        # MuJoCo consumes extra per-joint USD attributes. Register them on the
        # builder *before* ``add_usd`` if the user picks the mujoco backend;
        # other solvers ignore these attributes.
        if self.solver_name == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(ur10)

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        # ``floating=False`` welds the UR10 base to the world via an explicit
        # FIXED joint instead of the USD default D6-with-all-axes-locked.
        # Both resolve to "0 DOF for the base" in intent, but SolverUIPC
        # silently SKIPS D6 joints (see ArticulationBuilder.build), which
        # would leave base_link as a free 12-DOF ABD body held up only by
        # IPC ground contact — the arm would free-fall from ``height=1.2m``
        # before PD could do anything. FIXED joint is supported by every
        # backend here (UIPC / MuJoCo / Featherstone) and anchors base_link
        # rigidly, so the 6 revolute DOFs really are the full control space.
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            floating=False,
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

        # ``joint_armature`` is the portable cross-solver interface:
        # Featherstone/MuJoCo add it to their joint-space mass matrix, and
        # SolverUIPC folds it into the child link's ABD inertia about the
        # joint axis (see the armature notes in ``docs/integrations/uipc.md``).
        # ``newton.eval_mass_matrix`` includes it by default, keeping the
        # Stable-PD plant model consistent on every backend.
        rev_dof = 0
        for j in range(len(ur10.joint_type)):
            if ur10.joint_type[j] != newton.JointType.REVOLUTE:
                continue
            ur10.joint_armature[ur10.joint_qd_start[j]] = float(self.armature[rev_dof])
            rev_dof += 1
        assert rev_dof == len(self.armature), f"expected {len(self.armature)} revolute joints, found {rev_dof}"

        # Register one PD actuator per UR10 DOF. ``--stable-pd`` swaps plain
        # ``ControllerPD`` for ``ControllerStablePD`` (Tan et al. 2011): the
        # stable-PD control law predicts the next-step position via ``-qd*dt``
        # before the PD error, which removes the high-gain oscillation that
        # plain PD hits under UIPC's implicit integrator. Both controllers
        # are composed under ``newton.actuators.Actuator`` with a per-DOF
        # ``ClampingMaxEffort`` layer handling torque limits — clamping
        # lives outside the controller kernel in the new architecture.
        controller_cls = ControllerStablePD if self.stable_pd else ControllerPD
        for dof_idx in range(len(self.kp)):
            # num_worlds drives the per-world block-diagonal Cholesky in
            # ControllerStablePD; ControllerPD doesn't accept it.
            extra_kwargs = {"num_worlds": self.world_count} if self.stable_pd else {}
            ur10.add_actuator(
                controller_cls,
                index=dof_idx,
                kp=float(self.kp[dof_idx]),
                kd=float(self.kd[dof_idx]),
                clamping=[
                    (ClampingMaxEffort, {"max_effort": float(self.max_torque[dof_idx])}),
                ],
                **extra_kwargs,
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

        # Joint targets (home-centered shared trajectory, refreshed per frame
        # in ``_update_targets``), tiled to the ArticulationView layout
        # ``(world_count, 1, dofs_per_arti)``. Velocity target = 0.
        self.q_target = (
            np.broadcast_to(self.HOME_POSE, (self.world_count, 1, self.dofs_per_world)).astype(np.float32).copy()
        )
        self.qd_target = np.zeros((self.world_count, 1, self.dofs_per_world), dtype=np.float32)

        # Populate joint_target_q/qd — ActuatorPD reads them every step.
        self.ur10s.set_attribute("joint_target_q", self.control, self.q_target)
        self.ur10s.set_attribute("joint_target_qd", self.control, self.qd_target)

        # Cache the composed Actuator that drives every UR10 DOF. ``.kp``
        # lives on ``pd_actuator.controller``; effort limits on the
        # ClampingMaxEffort entry, not on the actuator itself. The builder
        # merged the 6 per-DOF ``add_actuator`` calls (x world_count replicas)
        # into one flat ``(world_count*6,)`` parameter vector, ordered
        # [w0d0..w0d5, w1d0..w1d5, ...]. Under ``--stable-pd`` the
        # controller class is ``ControllerStablePD``; otherwise ``ControllerPD``.
        expected_controller_cls = ControllerStablePD if self.stable_pd else ControllerPD
        pd_actuator = next(
            a
            for a in self.model.actuators
            if isinstance(a, _NewtonActuator) and isinstance(a.controller, expected_controller_cls)
        )
        self.pd_actuator = pd_actuator
        expected = self.world_count * self.dofs_per_world
        kp_len = len(pd_actuator.controller.kp)  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        assert kp_len == expected, (
            f"PD actuator kp length {kp_len} != {expected}; "
            "builder/replicate did not merge the per-DOF actuators as expected"
        )

        # ControllerStablePD needs ``mass_matrix`` + ``bias_forces`` in
        # its State each step; populated below in ``_apply_feedback``.
        self._act_state: _NewtonActuator.State | None = None
        if self.stable_pd:
            self._act_state = self.pd_actuator.state()

            # Pre-allocated scratch reused across every eval_mass_matrix /
            # eval_jacobian call this substep (P1: no per-call cudaMalloc/Free).
            W = self.world_count
            max_dofs = self.model.max_dofs_per_articulation
            max_links = self.model.max_joints_per_articulation
            dev = self.model.device
            self._mm_J = wp.zeros((W, max_links * 6, max_dofs), dtype=float, device=dev)
            self._mm_body_I_s = wp.zeros(self.model.body_count, dtype=wp.spatial_matrix, device=dev)
            self._mm_joint_S_s = wp.zeros(self.model.joint_dof_count, dtype=wp.spatial_vector, device=dev)
            self._H_buf = wp.empty((W, max_dofs, max_dofs), dtype=float, device=dev)

        # CUDA-graph state for the plain-PD actuator step (see
        # _capture_actuator_graphs). None until captured / when capture is
        # unavailable (stable-PD path or non-CUDA device).
        self._actuator_graphs: list | None = None
        self._state_parity = 0

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(5.0, 5.0, 3.0),
            pitch=-20.0,
            yaw=-135.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

        self._capture_actuator_graphs()

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
                workspace="/tmp/newton_uipc",
                dt=self.sim_dt,
                logger_level=uipc.Logger.Warn,
                dump_enable=True,
            )
            solver.set_contact(True, 0.001)
            solver.sync_uipc_inertia_with_model()
            solver.initialize()
            return solver, True
        if name == "mujoco":
            return newton.solvers.SolverMuJoCo(self.model), False
        if name == "featherstone":
            return newton.solvers.SolverFeatherstone(self.model), False
        raise ValueError(f"unsupported --solver: {name!r}")

    def _apply_feedback(self, state):
        """Run every registered actuator -> ``control.joint_f``.

        ``ActuatorPD`` already handles the PD math and the ``max_force`` clamp
        in its Warp kernel, and it reads position / velocity targets from
        ``control.joint_target_q`` / ``joint_target_qd`` (written once in
        ``__init__``). The only per-substep bookkeeping we need is:

        1. Clear ``control.joint_f`` — the PD kernel accumulates with ``+=``,
           so an un-cleared buffer would blow past ``max_force`` each frame.
        2. If ``--gravity-comp`` is on, refresh the ``constant_force`` array
           with the current Jacobian-transpose gravity compensation (it
           depends on the current pose, so it must be recomputed every
           substep).
        3. Run each actuator.

        Args:
            state: Simulation state the actuator reads ``joint_q`` / ``joint_qd``
                from. Passed explicitly (rather than ``self.state_0``) so the
                plain-PD path can be captured into a CUDA graph per physical
                state buffer — see :meth:`_capture_actuator_graphs`.
        """
        self.control.joint_f.zero_()  # pyright: ignore[reportOptionalMemberAccess]  # ty:ignore[unresolved-attribute]

        if self.stable_pd:
            if self._act_state is None:
                raise ValueError("ControllerStablePD state was not initialized")
            ctrl_state = self._act_state.controller_state

            # Mass matrix at the current pose. eval_mass_matrix consumes the
            # supplied J (and reuses body_I_s) instead of allocating per call.
            newton.eval_jacobian(self.model, state, self._mm_J, joint_S_s=self._mm_joint_S_s)
            newton.eval_mass_matrix(self.model, state, H=self._H_buf, J=self._mm_J, body_I_s=self._mm_body_I_s)
            # H is (W, max_dofs, max_dofs); matches State.mass_matrix's
            # (W, n_per_world, n_per_world) for max_dofs == n_per_world == 6.
            ctrl_state.mass_matrix.assign(self._H_buf)  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

            # Bias forces C(q, q̇) q̇ + g(q) via the exact RNEA inverse-dynamics
            # API (q̈ = 0). Writes straight into the controller's (W, n) buffer;
            # the physical-dynamics sign is exactly what the Stable-PD rhs
            # subtracts.
            newton.eval_inverse_dynamics(self.model, state, tau=ctrl_state.bias_forces)  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        # ControllerStablePD's update_state is a no-op (per-step scratch, no
        # cross-step information), so passing the same State object as both
        # current and next is safe; the composed Actuator just needs a
        # non-None next_act_state for its stateful-actuator validation.
        next_state = self._act_state if self.stable_pd else None
        self.pd_actuator.step(
            sim_state=state,
            sim_control=self.control,
            current_act_state=self._act_state,
            next_act_state=next_state,
            dt=self.sim_dt,
        )

    def _capture_actuator_graphs(self):
        """Capture the plain-PD actuator step into a CUDA graph per state buffer.

        The actuator pipeline (zero ``joint_f`` -> ``ControllerPD`` ->
        ``ClampingMaxEffort`` -> scatter-add) is pure ``wp.launch`` with no
        host<->device transfers, so it is CUDA-graph capturable. The
        ``--stable-pd`` path is left eager and this method is skipped: its bias
        assembly calls :func:`newton.eval_inverse_dynamics`, which allocates
        scratch per call, and CUDA-graph capture forbids allocation. Preallocating
        that scratch to enable capture is a follow-up (report P3).

        ``simulate`` ping-pongs ``state_0``/``state_1`` every substep, and a
        captured graph bakes in the array pointers it was recorded against.
        We therefore capture one graph per *physical* state buffer and replay
        whichever one currently holds the live state (tracked by
        ``_state_parity``). Falls back to eager execution on non-CUDA devices.
        """
        if self.stable_pd or not wp.get_device().is_cuda:
            self._actuator_graphs = None
            return
        # Warmup so kernel modules are loaded before capture (capture forbids
        # the lazy module-load allocations the first launch would trigger).
        self._apply_feedback(self.state_0)
        graphs = []
        for state in (self.state_0, self.state_1):
            with wp.ScopedCapture() as capture:
                self._apply_feedback(state)
            graphs.append(capture.graph)
        self._actuator_graphs = graphs
        # state_0 is the live buffer at capture time -> graphs[0].
        self._state_parity = 0

    # ----------------------------------------------------------------- runtime
    def simulate(self):
        for _ in range(self.sim_substeps):
            if self._actuator_graphs is not None:
                wp.capture_launch(self._actuator_graphs[self._state_parity])
            else:
                self._apply_feedback(self.state_0)
            self.state_0.clear_forces()
            self.solver.step(  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            if self._actuator_graphs is not None:
                # Track which physical buffer now holds the live state so the
                # next replay reads the buffer it was captured against.
                self._state_parity ^= 1

    def _update_targets(self):
        """Apply the shared home-centered sinusoidal trajectory to all worlds.

        Writes into the existing ``control.joint_target_q`` buffer, so the
        captured plain-PD CUDA graphs keep reading fresh targets.
        """
        if self.hold:
            target = self.HOME_POSE
        else:
            phases = self.TRAJ_PHASE * np.arange(self.dofs_per_world, dtype=np.float32)
            target = self.HOME_POSE + self.TRAJ_AMP * np.sin(self.TRAJ_OMEGA * self.sim_time + phases)
        self.q_target[:] = target.astype(np.float32)
        self.ur10s.set_attribute("joint_target_q", self.control, self.q_target)

    def step(self):
        self._update_targets()
        self.simulate()
        self._log_tracking()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _log_tracking(self):
        """Print world-0 joint positions, velocities, tracking error, and torques."""
        q = self.ur10s.get_attribute("joint_q", self.state_0).numpy()[0, 0]
        qd = self.ur10s.get_attribute("joint_qd", self.state_0).numpy()[0, 0]
        f = self.ur10s.get_attribute("joint_f", self.control).numpy()[0, 0]
        err = self.q_target[0, 0] - q
        q_str = " ".join(f"{p:+.4f}" for p in q)
        qd_str = " ".join(f"{v:+.4f}" for v in qd)
        err_str = " ".join(f"{e:+.3f}" for e in err)
        tau_str = " ".join(f"{t:+7.2f}" for t in f)
        print(
            f"[t={self.sim_time:6.3f}s] q(rad)=[{q_str}]  qd(rad/s)=[{qd_str}]\n"
            f"              err(rad)=[{err_str}]  tau(N·m)=[{tau_str}]"
        )

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
            "--stable-pd",
            action="store_true",
            help=(
                "Use ControllerStablePD (Tan et al. 2011) instead of ControllerPD. "
                "The controller runs an on-device (M + diag(Kd)·Δt)·qddot = b "
                "solve each substep, so the example populates its State with "
                "M = newton.eval_mass_matrix and bias_forces = Jacobian-T "
                "gravity plus finite-difference Coriolis every substep. "
                "Mutually exclusive with --gravity-comp (bias_forces "
                "already contains gravity). Requires --world-count 1."
            ),
        )
        parser.add_argument(
            "--hold",
            action="store_true",
            help="Hold the home pose instead of tracking the shared sinusoid (settling / steady-state comparison).",
        )
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
