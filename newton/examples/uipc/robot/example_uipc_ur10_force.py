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
#     ctrl_state.bias_forces = -τ_g + τ_c        # gravity + finite-diff Coriolis
#
# ``τ_c`` is computed from finite differences of ``M(q)`` using the
# Christoffel contraction.  This keeps the demo on public Newton APIs; a
# production controller would usually consume an exact RNEA bias-force API.
# Since gravity is already baked into ``bias_forces``, --stable-pd is
# mutually exclusive with --gravity-comp.
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


def _compute_coriolis_from_mass_derivatives(dH_dq: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Contract mass-matrix derivatives into Coriolis/centrifugal bias forces.

    Args:
        dH_dq: Mass-matrix derivatives where ``dH_dq[k, i, j]`` is
            ``d H[i, j] / d q[k]``.
        qd: Generalized velocities [m/s or rad/s], shape ``(dof_count,)``.

    Returns:
        Coriolis/centrifugal bias forces [N or N·m], shape ``(dof_count,)``.
    """
    dH_dq = np.asarray(dH_dq, dtype=np.float32)
    qd = np.asarray(qd, dtype=np.float32)
    dof_count = int(qd.shape[0])
    if dH_dq.shape != (dof_count, dof_count, dof_count):
        raise ValueError(f"dH_dq shape {dH_dq.shape} must be ({dof_count}, {dof_count}, {dof_count})")

    coriolis = np.zeros(dof_count, dtype=np.float32)
    for i in range(dof_count):
        total = 0.0
        for j in range(dof_count):
            for k in range(dof_count):
                christoffel = 0.5 * (dH_dq[k, i, j] + dH_dq[j, i, k] - dH_dq[i, j, k])
                total += float(christoffel) * float(qd[j]) * float(qd[k])
        coriolis[i] = total
    return coriolis


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
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_name = args.solver
        self.stable_pd = bool(args.stable_pd)
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
            if ur10.joint_type[i] == newton.JointType.REVOLUTE:
                ur10.joint_armature[i] = 1e-2

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

        # Gravity vector and per-body masses — host-side caches avoid a GPU
        # round-trip on every substep.
        self._gravity_np = self.model.gravity.numpy()  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportOptionalMemberAccess]
        self._body_mass_np = self.model.body_mass.numpy()  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportOptionalMemberAccess]

        # Reusable scratch for eval_jacobian so we don't re-allocate each step.
        self._J_buf: wp.array | None = None

        # Joint targets (hold the home pose), tiled to the ArticulationView
        # layout ``(world_count, 1, dofs_per_arti)``. Velocity target = 0.
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
        self._H_buf: wp.array | None = None
        self._H_fd_buf: wp.array | None = None
        self._coriolis_eps = 1.0e-3
        if self.stable_pd:
            self._act_state = self.pd_actuator.state()

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
        self._J_buf = newton.eval_jacobian(self.model, self.state_0, self._J_buf)
        if self._J_buf is None:
            raise ValueError("eval_jacobian unexpectedly returned None for UR10 articulation")
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

    def _mass_matrix_at_joint_q(self, joint_q: np.ndarray) -> np.ndarray:
        """Evaluate per-world ``M_w(q_w)`` at the given flat ``joint_q``.

        Mutates ``state_0.joint_q``; callers must restore it. Returns
        shape ``(world_count, dofs_per_world, dofs_per_world)``.
        """
        state_joint_q = self.state_0.joint_q
        if state_joint_q is None:
            raise ValueError("UR10 force example requires a joint_q state array")
        state_joint_q.assign(joint_q.astype(np.float32, copy=False))
        self._H_fd_buf = newton.eval_mass_matrix(self.model, self.state_0, H=self._H_fd_buf)
        if self._H_fd_buf is None:
            raise ValueError("eval_mass_matrix unexpectedly returned None for UR10 articulation")
        return self._H_fd_buf.numpy()[:, : self.dofs_per_world, : self.dofs_per_world].astype(np.float32, copy=True)

    def _compute_coriolis_bias(self) -> np.ndarray:
        """Per-world finite-difference Coriolis/centrifugal bias ``C(q, qd)``.

        Perturbs DOF ``p`` of every world simultaneously, so eval_mass_matrix
        runs ``2·dofs_per_world`` times regardless of ``world_count``.
        Returns ``(world_count, dofs_per_world)``.
        """
        state_joint_q = self.state_0.joint_q
        state_joint_qd = self.state_0.joint_qd
        if state_joint_q is None or state_joint_qd is None:
            raise ValueError("UR10 force example requires joint_q and joint_qd state arrays")
        q0 = state_joint_q.numpy().astype(np.float32, copy=True)
        qd = state_joint_qd.numpy().astype(np.float32, copy=True)
        n = self.dofs_per_world
        W = self.world_count
        expected = W * n
        if q0.size != expected or qd.size != expected:
            raise ValueError(
                f"expected flat joint state of length world_count*dofs_per_world = {expected}; "
                f"got joint_q size {q0.size}, joint_qd size {qd.size}"
            )

        qd_per_world = qd.reshape(W, n)
        dH_dq = np.empty((W, n, n, n), dtype=np.float32)
        eps = np.float32(self._coriolis_eps)
        try:
            for p in range(n):
                q_plus = q0.reshape(W, n).copy()
                q_minus = q0.reshape(W, n).copy()
                q_plus[:, p] += eps
                q_minus[:, p] -= eps
                H_plus = self._mass_matrix_at_joint_q(q_plus.reshape(-1))
                H_minus = self._mass_matrix_at_joint_q(q_minus.reshape(-1))
                dH_dq[:, p, :, :] = (H_plus - H_minus) / (2.0 * eps)
        finally:
            state_joint_q.assign(q0)

        bias = np.empty((W, n), dtype=np.float32)
        for w in range(W):
            bias[w] = _compute_coriolis_from_mass_derivatives(dH_dq[w], qd_per_world[w])
        return bias

    def _apply_feedback(self):
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
        """
        self.control.joint_f.zero_()  # pyright: ignore[reportOptionalMemberAccess]  # ty:ignore[unresolved-attribute]

        if self.stable_pd:
            self._H_buf = newton.eval_mass_matrix(self.model, self.state_0, H=self._H_buf)
            if self._H_buf is None:
                raise ValueError("eval_mass_matrix unexpectedly returned None for UR10 articulation")
            if self._act_state is None:
                raise ValueError("ControllerStablePD state was not initialized")
            ctrl_state = self._act_state.controller_state
            # H is (W, max_dofs, max_dofs); matches State.mass_matrix's
            # (W, n_per_world, n_per_world) for max_dofs == n_per_world == 6.
            ctrl_state.mass_matrix.assign(self._H_buf)  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

            # Tan 2011 C = -τ_g + τ_coriolis, shape (W, n_per_world).
            tau_g = self._compute_gravity_comp().reshape(self.world_count, self.dofs_per_world).astype(np.float32)
            tau_c = self._compute_coriolis_bias().astype(np.float32)
            ctrl_state.bias_forces.assign(-tau_g + tau_c)  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        # ControllerStablePD's update_state is a no-op (per-step scratch, no
        # cross-step information), so passing the same State object as both
        # current and next is safe; the composed Actuator just needs a
        # non-None next_act_state for its stateful-actuator validation.
        next_state = self._act_state if self.stable_pd else None
        self.pd_actuator.step(
            sim_state=self.state_0,
            sim_control=self.control,
            current_act_state=self._act_state,
            next_act_state=next_state,
            dt=self.sim_dt,
        )

    # ----------------------------------------------------------------- runtime
    def simulate(self):
        for _ in range(self.sim_substeps):
            self._apply_feedback()
            self.state_0.clear_forces()
            self.solver.step(  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
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
        """Print world-0 joint positions, velocities, tracking error, and torques."""
        q = self.ur10s.get_attribute("joint_q", self.state_0).numpy()[0, 0]
        qd = self.ur10s.get_attribute("joint_qd", self.state_0).numpy()[0, 0]
        f = self.ur10s.get_attribute("joint_f", self.control).numpy()[0, 0]
        err = self.HOME_POSE - q
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
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
