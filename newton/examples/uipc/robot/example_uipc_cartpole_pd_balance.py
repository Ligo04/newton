# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Cartpole PD — Classic Inverted-Pendulum Balance (Force)
#
# Textbook cartpole stabilisation problem:
#
#     "Given an inverted pendulum on a force-controlled cart, keep the pole
#      upright (theta ~= 0) while regulating the cart back to the origin."
#
# Unlike ``example_uipc_cartpole_pd_force`` (cart *tracks a sinusoid* while
# the poles hang passively) and ``example_uipc_cartpole_pd_position`` (same
# tracking task via UIPC's native aim drive), this example implements the
# classical balancing controller studied in every control-theory course:
# a 4-state linear feedback on cart position and pole angle.
#
# Textbook form:
#     F_cart = -(K_theta * theta + K_theta_d * theta_dot
#                + K_x * x       + K_x_d     * x_dot)         # F = -K * state
#
# Implementation form (see "Empirical sign note" in ``_apply_feedback``):
#     f[cart_dof] = k_pole  * theta     + k_poled * theta_dot
#                 + k_cart  * x         + k_cartd * x_dot
#
# The sign difference comes from this particular USD/UIPC import: a positive
# value in ``joint_f[cart_dof]`` accelerates the cart in the direction that
# catches a positive-theta lean, so the state-feedback sign flips relative
# to the textbook convention. This is cross-DOF feedback (the cart force
# depends on pole state), which the single-DOF ``newton.actuators.ControllerPD``
# cannot express, so we compute the force in Python and write it directly
# into ``control.joint_f``. The cart DOF is set to ``JointTargetMode.EFFORT``
# so the UIPC solver consumes that force as a generalised effort on the
# prismatic joint.
#
# The default ``cartpole.usda`` has a *double* pole (pole1 -> pole2). A
# single cart actuator with four scalar gains cannot stabilise a double
# inverted pendulum, so we also inject a stiff joint-lock PD on the pole2
# DOF that drives the relative angle theta_2 to zero. The articulation
# then behaves like a single rigid inverted pendulum — the setup used in
# the textbook analysis.
#
# DOF layout after collapse_fixed_joints:
#   d=0 : prismatic cart slider  (EFFORT — textbook F_cart)
#   d=1 : revolute  pole1        (NONE   — passive, balanced via the cart)
#   d=2 : revolute  pole2        (EFFORT — locked to pole1 by stiff PD)
#
# Multiple solver backends are supported via ``--solver {uipc,mujoco,
# featherstone,semi_implicit}``. All four consume ``control.joint_f`` via
# the EFFORT target mode, so the same balancing controller drives each of
# them; only the constructor kwargs and contact setup differ.
#
# --stable-pd switches the pole2 joint-lock from a hand-rolled scalar PD
# to ``ControllerStablePD`` (Tan et al. 2011).  Tan's implicit-in-position
# rewrite
#
#     (M + Kd·Δt)·θ̈ = Kp·(0 - θ - θ̇·Δt) + Kd·(0 - θ̇) - C
#     τ = Kp·(0 - θ - θ̇·Δt) + Kd·(0 - θ̇) - Kd·θ̈·Δt
#
# cancels the high-frequency numerical oscillation that plain explicit PD
# exhibits at the stiff ``k_lock = 400 N·m/rad`` used here.  The cart
# controller is *not* touched — its state feedback is cross-DOF (F_cart
# depends on pole1 state) and ControllerStablePD only handles per-DOF PD.
# For the StablePD actuator we wire:
#
#     ctrl_state.mass_matrix = H[pole2, pole2]              # eval_mass_matrix slice
#     ctrl_state.bias_forces = (C(q,q̇)q̇ + g(q))[pole2]      # eval_inverse_dynamics slice
#
# The bias is the exact RNEA inverse-dynamics term (gravity plus Coriolis)
# projected onto the pole2 DOF.
#
# Command: python -m newton.examples uipc_cartpole_pd_balance --world-count 1
#          python -m newton.examples uipc_cartpole_pd_balance --stable-pd
#          python -m newton.examples uipc_cartpole_pd_balance --solver mujoco
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.actuators import Actuator as _NewtonActuator
from newton.actuators import ClampingMaxEffort, ControllerStablePD
from newton.selection import ArticulationView


class Example:
    def __init__(self, viewer, args):
        # The unstable pole mode has time constant tau = sqrt(L/g) ~ 0.32 s.
        # Render at 60 Hz but step physics at 240 Hz so the controller is
        # well-oversampled relative to tau.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_name = args.solver
        self.stable_pd = bool(args.stable_pd)
        self.viewer = viewer

        # ControllerStablePD batches the implicit solve block-diagonally
        # over worlds via num_worlds; --world-count > 1 is supported.

        # --- Cart state-feedback gains (all positive) -------------------------
        # These are the magnitudes of the four classical LQR gains; see the
        # "Empirical sign note" in ``_apply_feedback`` for why all four enter
        # the force expression with a + sign in this particular USD/UIPC
        # import rather than the textbook - sign.
        #
        # Order-of-magnitude check (M+m ~ 1.5 kg, g ~ 9.81, L ~ 1 m):
        #   k_pole must exceed (M+m)*g*L ~ 14.7 to beat gravity.
        #   We pick ~20x that for a comfortable stability margin.
        self.k_pole = 300.0  # [N / rad]
        self.k_poled = 40.0  # [N s / rad]
        self.k_cart = 8.0  # [N / m]
        self.k_cartd = 10.0  # [N s / m]
        self.max_force = 500.0  # [N] symmetric clamp on F_cart

        # --- Pole2 joint-lock gains (turns the double-pole into a single rod) -
        # Stiff PD driving theta_2 -> 0. Makes the pole2 joint *effectively*
        # rigid so the system matches the textbook single-pendulum model.
        self.k_lock = 400.0  # [N m / rad]
        self.k_lock_d = 20.0  # [N m s / rad]
        self.max_torque_lock = 200.0  # [N m]

        cartpole = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1
        cartpole.default_body_armature = 0.1

        # MuJoCo consumes extra per-joint/per-shape USD attributes. Register
        # them on the builder before ``add_usd`` so the importer can copy
        # them across; the other solvers ignore these attributes.
        if self.solver_name == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)

        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # Pole1 starts 0.05 rad off vertical — small enough that the linear
        # state feedback is valid, large enough that the controller has to
        # act immediately.
        cartpole.joint_q[-3:] = [0.0, 0.05, 0.0]

        cart_dof = len(cartpole.joint_target_mode) - 3
        cartpole.joint_target_mode[cart_dof] = int(JointTargetMode.EFFORT)
        cartpole.joint_target_mode[cart_dof + 1] = int(JointTargetMode.NONE)
        cartpole.joint_target_mode[cart_dof + 2] = int(JointTargetMode.EFFORT)

        # Register a ControllerStablePD on pole2 so ``--stable-pd`` can drive
        # the joint-lock via Tan 2011 instead of the hand-rolled scalar PD
        # in ``_apply_feedback``.  The composed Actuator scatter-adds into
        # ``control.joint_f`` (``+=``), so we zero joint_f before writing
        # the cart force and running the actuator so no residual leaks
        # across substeps.  Effort clamping is handled by the dedicated
        # ``ClampingMaxEffort`` layer (the controller kernel itself no
        # longer clamps).  ``target_pos`` / ``target_vel`` both default to
        # 0 (``Model.control()`` zero-inits them), which is the pole2
        # lock's set-point, so no per-step target update is needed.
        if self.stable_pd:
            cartpole.add_actuator(
                ControllerStablePD,
                index=cart_dof + 2,
                kp=self.k_lock,
                kd=self.k_lock_d,
                clamping=[(ClampingMaxEffort, {"max_effort": self.max_torque_lock})],
                num_worlds=self.world_count,
            )

        if self.world_count > 1:
            builder = newton.ModelBuilder(newton.Axis.Z)
            builder.replicate(cartpole, self.world_count, spacing=(1.0, 2.0, 0.0))
        else:
            builder = cartpole

        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver, self._uses_contacts = self._build_solver(self.solver_name)

        self.state_1 = self.model.state()
        self.control = self.model.control()
        # UIPC needs a Contacts object even for collision-free runs; the
        # maximal-/minimal-coordinate articulation solvers don't.
        self.contacts = self.model.contacts() if self._uses_contacts else None

        # Selection view over every replicated cartpole — read/write
        # (world_count, 1, dofs_per_arti) tensors through a single handle
        # instead of stride-mangled flat indices.
        self.cartpoles = ArticulationView(self.model, "/cartPole")
        assert self.cartpoles.count == self.world_count, (
            f"expected one /cartPole per world, got {self.cartpoles.count} for {self.world_count} worlds"
        )
        self.dofs_per_world = self.cartpoles.joint_dof_count
        # DOF layout after collapse_fixed_joints: [cart_slider, pole1, pole2].
        self.cart_dof = 0
        self.pole1_dof = 1
        self.pole2_dof = 2

        # ControllerStablePD state + scratch for the per-substep Tan 2011
        # solve.  The composed ``Actuator.State`` wraps the controller's
        # ``State`` under ``.controller_state``.  ``eval_mass_matrix`` wants a
        # reusable output buffer so we don't re-allocate every substep.
        self._pole2_actuator: _NewtonActuator | None = None
        self._act_state: _NewtonActuator.State | None = None
        self._H_buf: wp.array | None = None
        if self.stable_pd:
            pole2_actuator = next(
                (
                    a
                    for a in self.model.actuators
                    if isinstance(a, _NewtonActuator) and isinstance(a.controller, ControllerStablePD)
                ),
                None,
            )
            if pole2_actuator is None:
                raise RuntimeError("--stable-pd set but ControllerStablePD missing from model.actuators")
            kp_len = len(pole2_actuator.controller.kp)
            assert kp_len == self.world_count, (
                f"ControllerStablePD kp length {kp_len} != world_count {self.world_count}"
            )
            self._pole2_actuator = pole2_actuator
            self._act_state = pole2_actuator.state()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(9.5, 5, 3.5),
            pitch=-10.0,
            yaw=-160.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer._paused = True

    def _build_solver(self, name: str) -> tuple[object, bool]:
        """Construct the requested Newton solver.

        Returns the solver and a flag telling the caller whether a
        ``Contacts`` object must be passed to ``solver.step()`` — UIPC
        requires one even in the collision-free balance task; the others
        don't use collisions for this articulation.
        """
        if name == "uipc":
            import uipc  # noqa: PLC0415  # deferred so non-uipc solvers run without the dep

            solver = newton.solvers.SolverUIPC(
                self.model,
                workspace="/tmp/newton_uipc/cartpole_pd_balance",
                dt=self.sim_dt,
                logger_level=uipc.Logger.Warn,
            )
            return solver, True
        if name == "mujoco":
            return newton.solvers.SolverMuJoCo(self.model), False
        if name == "featherstone":
            return newton.solvers.SolverFeatherstone(self.model), False
        if name == "semi_implicit":
            # joint_attach_ke/kd keep the articulation attachments stiff
            # enough that the balance controller sees textbook dynamics.
            return (
                newton.solvers.SolverSemiImplicit(
                    self.model,
                    joint_attach_ke=1.6e3,
                    joint_attach_kd=2.0e1,
                ),
                False,
            )
        raise ValueError(f"unsupported --solver: {name!r}")

    def _apply_feedback(self):
        """Cross-DOF state feedback -> ``control.joint_f``.

        Reads (x, x_dot, theta1, theta1_dot, theta2, theta2_dot) for every
        world through :class:`ArticulationView`, computes the classical
        cartpole balancing force plus the pole2 joint lock, and writes the
        (world_count, 1, 3) force tensor back in one shot.

        We rebuild the force vector from ``np.zeros`` every frame, which
        implicitly zeros the DOFs we don't touch — the same pattern the
        other cartpole examples use to avoid the ``+=`` accumulation
        behaviour of the per-DOF ``ActuatorPD`` kernel.

        When ``--stable-pd`` is set, the pole2 slot is left at zero and
        the Tan 2011 ``ControllerStablePD`` then accumulates its implicit
        torque on top of the freshly written cart force.
        """
        # Shape: (world_count, 1, dofs_per_arti)
        q = self.cartpoles.get_attribute("joint_q", self.state_0).numpy()
        qd = self.cartpoles.get_attribute("joint_qd", self.state_0).numpy()

        x = q[:, 0, self.cart_dof]
        xd = qd[:, 0, self.cart_dof]
        th1 = q[:, 0, self.pole1_dof]
        thd1 = qd[:, 0, self.pole1_dof]
        th2 = q[:, 0, self.pole2_dof]
        thd2 = qd[:, 0, self.pole2_dof]

        # Textbook single-pendulum state feedback on the cart.
        #
        # Empirical sign note: in this USD/UIPC import the prismatic DOF
        # is effectively oriented such that a *positive* ``joint_f[cart]``
        # accelerates the cart in the direction that counteracts a
        # positive-theta lean of pole1 (pole tip moving -Y). All four
        # gain coefficients therefore enter with a *positive* sign; the
        # cart-regulation terms still produce a restoring force because
        # the sign convention above also flips the cart-position term.
        f_cart = self.k_pole * th1 + self.k_poled * thd1 + self.k_cart * x + self.k_cartd * xd
        np.clip(f_cart, -self.max_force, self.max_force, out=f_cart)

        f = np.zeros((self.world_count, 1, self.dofs_per_world), dtype=np.float32)
        f[:, 0, self.cart_dof] = f_cart
        if not self.stable_pd:
            # Hand-rolled scalar PD locking pole2 to pole1. Same torque law
            # ControllerStablePD would produce with C=0 under explicit integration.
            tau2 = -self.k_lock * th2 - self.k_lock_d * thd2
            np.clip(tau2, -self.max_torque_lock, self.max_torque_lock, out=tau2)
            f[:, 0, self.pole2_dof] = tau2
        self.cartpoles.set_attribute("joint_f", self.control, f)

        if self.stable_pd:
            # (pole2, pole2) entry of the per-world inertia, shape (W, 1, 1).
            self._H_buf = newton.eval_mass_matrix(self.model, self.state_0, H=self._H_buf)
            p2 = self.pole2_dof
            pole2_m = np.ascontiguousarray(self._H_buf.numpy()[:, p2 : p2 + 1, p2 : p2 + 1], dtype=np.float32)
            ctrl_state = self._act_state.controller_state
            ctrl_state.mass_matrix.assign(pole2_m)
            # bias_forces = pole2 component of the exact RNEA bias C(q,q̇)q̇ + g(q)
            # (previously Jacobian-transpose gravity only, Coriolis neglected).
            bias_full = newton.eval_inverse_dynamics(self.model, self.state_0)
            ctrl_state.bias_forces.assign(np.ascontiguousarray(bias_full.numpy()[:, p2 : p2 + 1], dtype=np.float32))

            # ControllerStablePD.update_state is a no-op (per-step scratch,
            # no cross-step information), so passing the same State as both
            # current and next is safe; the composed Actuator just needs a
            # non-None next_act_state for its stateful-actuator validation.
            self._pole2_actuator.step(
                sim_state=self.state_0,
                sim_control=self.control,
                current_act_state=self._act_state,
                next_act_state=self._act_state,
                dt=self.sim_dt,
            )

    def simulate(self):
        # The pole's unstable mode is fast (tau ~ 0.32 s); running the
        # feedback at the outer 60 Hz frame rate lets the closed-loop
        # system oscillate. We therefore recompute F_cart inside the
        # substep loop so the controller effectively runs at 240 Hz.
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
        self._log_balance()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _log_balance(self):
        """Print world-0 controller telemetry — cart, pole1, and applied force."""
        q = self.cartpoles.get_attribute("joint_q", self.state_0).numpy()
        qd = self.cartpoles.get_attribute("joint_qd", self.state_0).numpy()
        f = self.cartpoles.get_attribute("joint_f", self.control).numpy()
        print(
            f"[t={self.sim_time:6.3f}s] "
            f"x={float(q[0, 0, self.cart_dof]):+.3f} m  "
            f"theta1={float(q[0, 0, self.pole1_dof]):+.3f} rad  "
            f"theta1_dot={float(qd[0, 0, self.pole1_dof]):+.3f} rad/s  "
            f"F_cart={float(f[0, 0, self.cart_dof]):+8.2f} N"
        )

    def test_final(self):
        """After simulation the pole should remain upright and the cart near zero.

        Tolerances are generous: we only verify the system did not diverge
        (classical textbook success criterion — the linear controller is
        only guaranteed within the region of attraction of the upright
        equilibrium, not arbitrary precision).
        """
        # Shape: (world_count, 1, dofs_per_arti)
        q = self.cartpoles.get_attribute("joint_q", self.state_0).numpy()
        qd = self.cartpoles.get_attribute("joint_qd", self.state_0).numpy()

        theta1 = q[:, 0, self.pole1_dof]
        theta1_dot = qd[:, 0, self.pole1_dof]
        x = q[:, 0, self.cart_dof]

        assert np.all(np.abs(theta1) < 0.2), f"pole fell over: theta1={theta1.tolist()} rad"
        assert np.all(np.abs(theta1_dot) < 2.0), f"pole still oscillating hard: theta1_dot={theta1_dot.tolist()} rad/s"
        assert np.all(np.abs(x) < 2.0), f"cart drifted off track: x={x.tolist()} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--solver",
            choices=("uipc", "mujoco", "featherstone", "semi_implicit"),
            default="uipc",
            help="Newton solver to drive the balance controller.",
        )
        parser.add_argument(
            "--stable-pd",
            action="store_true",
            help=(
                "Drive the stiff pole2 joint-lock with ControllerStablePD "
                "(Tan et al. 2011) instead of the hand-rolled scalar PD. "
                "The per-substep State is populated with pole2's diagonal "
                "entry from newton.eval_mass_matrix and its bias force from "
                "newton.eval_inverse_dynamics (gravity plus Coriolis). Cart "
                "state feedback is untouched. Multi-world is supported via the "
                "controller's block-diagonal batched Cholesky."
            ),
        )
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
