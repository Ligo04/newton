# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""UIPC physics engine solver backend for Newton."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import uipc
import uipc.adapter.warp
import warp as wp
from uipc import Logger as ULogger
from uipc.core import AffineBodyStateAccessorFeature
from uipc.core import Scene as UScene
from uipc.unit import GPa, MPa

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ...sim.enums import JointType
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .articulation_builder import ArticulationBuilder
from .cloth import ClothBuilder
from .converter import (
    UIpcMappingInfo,
    _read_from_backend_kernel,
    _spatial_to_vel_mat44_kernel,
    _transform_to_mat44_kernel,
    populate_backend_offsets,
)
from .deformable_body import DeformableBodyBuilder
from .rigid_body import RigidBodyBuilder


class SolverUIPC(SolverBase):
    """Solver backend that wraps the `UIPC <https://github.com/spiriMirror/libuipc>`_ physics engine.

    UIPC provides implicit simulation of rigid bodies (via AffineBody), deformable objects,
    and cloth. This solver converts Newton's :class:`~newton.Model` into UIPC scene objects
    and synchronizes state between Newton and UIPC each step using GPU warp kernels.

    Joint targets are driven via UIPC's native **Animator** mechanism: animation callbacks
    registered during construction fire inside ``world.advance()`` before each physics solve,
    reading the cached control values and writing ``aim_angle`` / ``aim_position`` to the
    joint geometry.

    The solver supports a **deferred initialization** workflow so that users can
    configure the UIPC scene and contact tabular before the world is initialized:

    .. code-block:: python

        solver = newton.solvers.SolverUIPC(model, dt=1.0 / 60.0, auto_init=False)

        # Customize scene config
        solver.configure_scene({"newton_tol": 1e-3, "line_search": {"max_iter": 8}})


        # Customize contact tabular (called once per world with ground/env/robot/free elements)
        def setup_contacts(tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem):
            gripper_elem = tabular.create(f"gripper_{world_index}")
            tabular.insert(gripper_elem, env_elem, 0.8, 1e9, False)
            tabular.insert(gripper_elem, ground_elem, 0.8, 1e9, False)


        solver.configure_contact_tabular(setup_contacts)

        # Build scene objects and initialize the UIPC world
        solver.initialize()

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    When ``auto_init=True`` (the default), the constructor calls
    :meth:`initialize` automatically, preserving full backward compatibility.

    For multi-world models produced by :meth:`~newton.ModelBuilder.replicate`,
    the solver uses UIPC's ``subscene_tabular`` to configure contact isolation
    between Newton worlds within a single UIPC scene. By default, bodies in
    different Newton worlds do not contact each other. Use
    :meth:`configure_subscene_tabular` to customize cross-world contact.

    .. note::

        - This solver requires ``libuipc`` (the ``uipc`` Python package) to be installed.
        - Supports rigid bodies (AffineBody), cloth (NeoHookeanShell), and deformable bodies
          (StableNeoHookean).
        - Joint types: REVOLUTE, PRISMATIC, FIXED, FREE.
        - BALL, DISTANCE, D6, and CABLE joints are not supported.
    """

    _uipc = None

    @classmethod
    def import_uipc(cls):
        """Import the UIPC dependencies and cache them as a class variable."""
        if cls._uipc is None:
            try:
                import uipc

                cls._uipc = uipc
            except ImportError as e:
                raise ImportError(
                    "UIPC backend not installed. Please install libuipc: "
                    "see https://github.com/spiriMirror/libuipc for instructions."
                ) from e
        return cls._uipc

    def __init__(
        self,
        model: Model,
        backend: str = "cuda",
        workspace: str = "/tmp/newton_uipc",
        dt: float = 1.0 / 60.0,
        scene_config: dict[str, Any] | None = None,  # pyright: ignore[reportRedeclaration]
        kappa: float = 100 * MPa,
        default_mass_density: float = 1000.0,
        logger_level=ULogger.Error,
        auto_init: bool = False,
    ):
        """Create a UIPC solver instance from a Newton model.

        Args:
            model: The Newton model to simulate.
            backend: UIPC backend name (default: ``"cuda"``).
            workspace: Working directory for UIPC engine output.
            dt: Time step [s]. UIPC uses a fixed time step configured here.
            scene_config: Optional UIPC scene configuration dict passed directly
                to ``uipc.Scene()``. If ``None``, uses ``Scene.default_config()``
                with ``dt`` and ``gravity`` overridden from the Newton model.
            kappa: AffineBody stiffness parameter [Pa].
            default_mass_density: Default mass density [kg/m^3] for bodies.
            logger_level: UIPC logger verbosity. Use ``uipc.Logger.Critical``,
                ``uipc.Logger.Error``, ``uipc.Logger.Warn``, ``uipc.Logger.Info``,
                ``uipc.Logger.Debug``, or ``uipc.Logger.Trace``.
                Defaults to ``uipc.Logger.Critical`` to suppress UIPC console spam.
            auto_init: If ``True`` (default), call :meth:`initialize` at the end
                of the constructor. Set to ``False`` to configure the scene and
                contact tabular before initialization via :meth:`configure_scene`
                and :meth:`configure_contact_tabular`.
        """
        super().__init__(model=model)
        self.import_uipc()

        ULogger.set_level(logger_level)

        self._dt = dt
        self._step_count = 0
        self._initialized = False

        # Store construction parameters for deferred init
        self._backend = backend
        self._workspace = workspace
        self._kappa = kappa
        self._default_mass_density = default_mass_density

        # Scene config: start from default, apply Newton model overrides
        if scene_config is None:
            scene_config: dict[str, Any] = UScene.default_config()
            scene_config["dt"] = dt
        scene_config["d_hat"] = 0.001
        scene_config["contact"]["enable"] = False
        scene_config["newton"]["velocity_tol"] = 0.001
        scene_config["newton"]["translation_tol"] = 0.01
        print(scene_config)
        if model.gravity is not None:
            gravity_np = model.gravity.numpy().flatten()
            scene_config["gravity"] = [[float(gravity_np[0])], [float(gravity_np[1])], [float(gravity_np[2])]]
        self._scene_config = scene_config

        # User-registered callbacks (set via configure_* methods)
        self._contact_tabular_fn: Callable | None = None
        self._subscene_tabular_fn: Callable | None = None

        # Builders (populated during initialize)
        self._rigid_body_builder: RigidBodyBuilder
        self._articulation_builder: ArticulationBuilder
        self._cloth_builder: ClothBuilder
        self._deformable_builder: DeformableBodyBuilder

        if auto_init:
            self.initialize()

    # ------------------------------------------------------------------
    # Pre-initialization configuration
    # ------------------------------------------------------------------

    def configure_scene(self, config: dict[str, Any]) -> None:
        """Update UIPC scene configuration before initialization.

        Merges the provided key-value pairs into the scene config dict. Must be
        called **before** :meth:`initialize` (i.e. with ``auto_init=False``).

        Args:
            config: Dictionary of UIPC scene configuration overrides. These are
                merged (shallow update) into the base config. Common keys include
                ``"dt"``, ``"gravity"``, ``"newton_tol"``, ``"line_search"``,
                ``"cfl"``, ``"friction"``, etc. Refer to the UIPC documentation
                for the full list.

        Raises:
            RuntimeError: If the solver has already been initialized.

        Example
        -------

        .. code-block:: python

            solver = SolverUIPC(model, auto_init=False)
            solver.configure_scene({
                "newton_tol": 1e-3,
                "line_search": {"max_iter": 8},
            })
            solver.initialize()
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot configure scene after initialization. Pass auto_init=False to defer initialization."
            )
        self._scene_config.update(config)

    def set_contact(self, enable: bool, d_hat: float = 0.001) -> None:
        """Enable/disable global contact handling and optionally tune ``d_hat``.

        Toggles the ``contact.enable`` flag on the underlying UIPC scene config,
        and (optionally) updates the IPC barrier distance ``contact.d_hat``
        [m]. ``d_hat`` is the thickness of the contact "safety layer": pairs
        whose surface distance drops below it receive a barrier force that
        repels them before an actual penetration occurs. Smaller values allow
        tighter contacts (e.g. a gripper closing on a thin object) but demand
        more Newton iterations and smaller time steps to stay stable. The UIPC
        default is ``0.01`` m.

        Safe to call either before or after :meth:`initialize`:

        - **Before init:** updates the cached scene config that will be passed
          to ``uipc.Scene`` on :meth:`initialize`.
        - **After init:** updates ``scene.config()`` in place. Takes effect on
          the next ``scene.update()`` / ``world.advance()`` call.

        Args:
            enable: ``True`` to enable contact, ``False`` to disable.
            d_hat: Optional IPC barrier distance [m]. When provided, updates
                ``contact.d_hat``. When ``None`` (default), the current value
                is left untouched.
        """
        flag = bool(enable)

        if self._initialized:
            # Mutate the live scene config. UIPC exposes scene.config() as a
            # mutable view onto the underlying JSON-like config object.
            scene_cfg = self.scene.config()
            scene_cfg["contact"]["enable"] = flag  # ty:ignore[not-subscriptable]
            if d_hat is not None:
                scene_cfg["contact"]["d_hat"] = float(d_hat)  # ty:ignore[not-subscriptable]
        else:
            contact_cfg = self._scene_config.setdefault("contact", {})
            contact_cfg["enable"] = flag
            if d_hat is not None:
                contact_cfg["d_hat"] = float(d_hat)

    def configure_contact_tabular(self, fn: Callable) -> None:
        """Register a callback to configure the UIPC contact tabular before initialization.

        The solver creates a shared **ground_elem** and, for each Newton world,
        three additional contact elements:

        - **ground_elem** - applied to ground planes, shared across all worlds.
        - **env_elem** - applied to non-articulated rigid bodies, kinematic
          bodies, cloth, and deformable objects.
        - **robo_elem** - applied to articulated robot links (non-free joints).
        - **actor_elem** - applied to bodies attached via free joints.

        Default contact pairs (friction ``0.5``, stiffness ``1 GPa``) are
        inserted for all combinations except ``robo-robo``.  The callback is
        invoked once per world so that users can create additional elements,
        insert custom contact pairs, or modify the defaults.

        Must be called **before** :meth:`initialize` (i.e. with ``auto_init=False``).

        Args:
            fn: A callable with signature
                ``fn(tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem) -> None``.
                ``tabular`` is the UIPC ``ContactTabular`` obtained from
                ``scene.contact_tabular()``.  ``world_index`` is the Newton
                world index (``0`` for single-world models).  ``ground_elem``
                is the shared ground element.  ``env_elem``, ``robo_elem``,
                and ``actor_elem`` are the pre-created contact elements for
                that world.

        Raises:
            RuntimeError: If the solver has already been initialized.

        Example
        -------

        .. code-block:: python

            def setup_contacts(tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem):
                gripper_elem = tabular.create(f"gripper_{world_index}")
                tabular.insert(gripper_elem, env_elem, 0.8, 1e9, False)
                tabular.insert(gripper_elem, ground_elem, 0.8, 1e9, False)


            solver = SolverUIPC(model, auto_init=False)
            solver.configure_contact_tabular(setup_contacts)
            solver.initialize()
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot configure contact tabular after initialization. Pass auto_init=False to defer initialization."
            )
        self._contact_tabular_fn = fn

    def configure_subscene_tabular(self, fn: Callable) -> None:
        """Register a callback to customize subscene contact configuration.

        For multi-world models, the solver creates one UIPC subscene per Newton
        world. By default, bodies in different worlds do **not** contact each
        other (replicating the old ``separate_worlds`` behavior). This callback
        lets you override the default subscene contact table.

        Must be called **before** :meth:`initialize` (i.e. with ``auto_init=False``).

        Args:
            fn: A callable with signature
                ``fn(tabular, world_subscenes, default_element) -> None``.
                ``tabular`` is the UIPC ``SubsceneTabular``; ``world_subscenes``
                is a list of ``SubsceneElement`` (one per Newton world);
                ``default_element`` is the default subscene element (used by
                ground planes and global objects).

        Raises:
            RuntimeError: If the solver has already been initialized.

        Example
        -------

        .. code-block:: python

            def setup_subscenes(tabular, world_subscenes, default_elem):
                # Enable contact between world 0 and world 1
                tabular.insert(world_subscenes[0], world_subscenes[1], True)


            solver = SolverUIPC(model, auto_init=False)
            solver.configure_subscene_tabular(setup_subscenes)
            solver.initialize()
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot configure subscene tabular after initialization. Pass auto_init=False to defer initialization."
            )
        self._subscene_tabular_fn = fn

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Build UIPC scene objects from the Newton model and initialize the world.

        Creates a single UIPC Engine, World, and Scene. For multi-world models,
        configures ``subscene_tabular`` to isolate contact between Newton worlds.
        Builds rigid body / articulation / cloth / deformable geometries and
        calls ``world.init(scene)``.

        This method is called automatically when ``auto_init=True`` (the
        default). When ``auto_init=False``, call this explicitly after
        :meth:`configure_scene`, :meth:`configure_contact_tabular`, and
        :meth:`configure_subscene_tabular`.

        Raises:
            RuntimeError: If already initialized.
        """
        if self._initialized:
            raise RuntimeError("SolverUIPC is already initialized.")

        model = self.model

        # Create a single UIPC Engine / World / Scene
        self.engine = uipc.Engine(backend_name=self._backend, workspace=self._workspace)
        self.world = uipc.World(self.engine)
        self.scene = uipc.Scene(self._scene_config)

        # Contact tabular — shared ground + per-world env / robot element pairs
        contact_tabular = self.scene.contact_tabular()

        # Ground element is shared across all worlds
        ground_elem = contact_tabular.create("ground")
        contact_tabular.insert(ground_elem, ground_elem, 0.5, 1.0 * GPa, False)

        env_elems: list[Any] = []
        robo_elems: list[Any] = []
        actor_elems: list[Any] = []
        body_element_overrides: dict[int, Any] = {}

        for world_index in range(model.world_count):
            suffix = f"_{world_index}" if model.world_count > 1 else ""
            env_elem = contact_tabular.create(f"env{suffix}")
            robo_elem = contact_tabular.create(f"robot{suffix}")
            actor_elem = contact_tabular.create(f"actor{suffix}")
            contact_tabular.insert(env_elem, env_elem, 0.5, 1.0 * GPa, False)
            contact_tabular.insert(env_elem, robo_elem, 0.5, 1.0 * GPa, True)
            contact_tabular.insert(env_elem, actor_elem, 0.5, 1.0 * GPa, True)
            contact_tabular.insert(ground_elem, env_elem, 0.5, 1.0 * GPa, False)
            contact_tabular.insert(ground_elem, robo_elem, 0.5, 1.0 * GPa, True)
            contact_tabular.insert(ground_elem, actor_elem, 0.5, 1.0 * GPa, True)
            contact_tabular.insert(robo_elem, robo_elem, 0.5, 1.0 * GPa, False)
            contact_tabular.insert(robo_elem, actor_elem, 0.5, 1.0 * GPa, True)
            contact_tabular.insert(actor_elem, actor_elem, 0.5, 1.0 * GPa, True)

            if self._contact_tabular_fn is not None:
                overrides = self._contact_tabular_fn(
                    contact_tabular, world_index, ground_elem, env_elem, robo_elem, actor_elem
                )
                if overrides is not None:
                    body_element_overrides.update(overrides)

            env_elems.append(env_elem)
            robo_elems.append(robo_elem)
            actor_elems.append(actor_elem)

        # Subscene tabular for multi-world contact isolation
        subscene_elements: list[Any] = []
        if model.world_count > 1:
            tabular = self.scene.subscene_tabular()
            default_subscene_elem = tabular.default_element()

            for world_index in range(model.world_count):
                se = tabular.create(f"world_{world_index}")
                subscene_elements.append(se)

            # Default: no contact between different worlds,
            # enable contact between each world and default (ground)
            for i in range(model.world_count):
                for j in range(i + 1, model.world_count):
                    tabular.insert(subscene_elements[i], subscene_elements[j], False)
                tabular.insert(default_subscene_elem, subscene_elements[i], True)

            # Let user override subscene configuration
            if self._subscene_tabular_fn is not None:
                self._subscene_tabular_fn(tabular, subscene_elements, default_subscene_elem)

        self.mapping = UIpcMappingInfo()
        scene = self.scene

        # Create one builder per type (reused across worlds)
        rb = RigidBodyBuilder(model, scene, self.mapping, self._kappa, self._default_mass_density)
        ab = ArticulationBuilder(model, scene, self.mapping, self._dt, kappa=self._kappa)
        cb = ClothBuilder(model, scene, self.mapping)
        db = DeformableBodyBuilder(model, scene, self.mapping, default_mass_density=self._default_mass_density)

        rb.build_ground_planes(ground_elem)

        # Build set of body indices that belong to articulations (robot links),
        # a separate set for bodies attached via free joints, and a set of
        # bodies that must not be instanced (children of ball joints).
        articulation_bodies: set[int] = set()
        free_joint_bodies: set[int] = set()
        ball_joint_bodies: set[int] = set()
        if model.joint_child is not None:
            joint_child_np = model.joint_child.numpy()
            joint_type_np = model.joint_type.numpy() if model.joint_type is not None else None
            for j in range(model.joint_count):
                child = int(joint_child_np[j])
                if child < 0:
                    continue
                jtype = int(joint_type_np[j]) if joint_type_np is not None else -1
                if jtype == int(JointType.FREE):
                    free_joint_bodies.add(child)
                elif jtype == int(JointType.BALL):
                    articulation_bodies.add(child)
                    ball_joint_bodies.add(child)
                else:
                    articulation_bodies.add(child)
            # Also include parent bodies that are part of articulations
            if model.joint_parent is not None:
                joint_parent_np = model.joint_parent.numpy()
                for j in range(model.joint_count):
                    parent = int(joint_parent_np[j])
                    if parent >= 0:
                        articulation_bodies.add(parent)

        # Host-side indexing for per-world ranges (multi-world only)
        if model.world_count > 1:
            body_ws = model.body_world_start.numpy()  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportOptionalMemberAccess]
            joint_ws = model.joint_world_start.numpy()  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportOptionalMemberAccess]
            particle_ws = model.particle_world_start.numpy() if model.particle_world_start is not None else None
        else:
            body_ws = None
            joint_ws = None
            particle_ws = None

        for world_index in range(model.world_count):
            if body_ws is not None:
                body_range = (int(body_ws[world_index]), int(body_ws[world_index + 1]))
                joint_range = (int(joint_ws[world_index]), int(joint_ws[world_index + 1]))  # ty:ignore[not-subscriptable]  # pyright: ignore[reportOptionalSubscript]
                particle_range = (
                    (int(particle_ws[world_index]), int(particle_ws[world_index + 1]))
                    if particle_ws is not None
                    else None
                )
            else:
                body_range = (0, model.body_count)
                joint_range = (0, model.joint_count)
                particle_range = (0, model.particle_count)
            se = subscene_elements[world_index] if subscene_elements else None
            rb.build_body_shape_mapping(body_range)
            ab.compute_fk(joint_range)
            rb.build_affine_bodies(
                env_elems[world_index],
                robo_elems[world_index],
                actor_elems[world_index],
                articulation_bodies,
                free_joint_bodies,
                body_range,
                se,
                ab._body_transforms,
                body_element_overrides,
                no_instance_bodies=ball_joint_bodies,
            )
            ab.build_joints(robo_elems[world_index], joint_range, se)
            if cb.has_cloth:
                cb.build(env_elems[world_index], particle_range, se)
            if db.has_deformable:
                db.build(env_elems[world_index], particle_range, se)

        self._rigid_body_builder = rb
        self._articulation_builder = ab
        self._cloth_builder = cb
        self._deformable_builder = db

        # Initialize UIPC world and set up state accessors
        self.world.init(scene)
        if not self.world.is_valid():
            raise RuntimeError(
                "UIPC world initialization failed (world is not valid). Check the UIPC log above for details."
            )

        populate_backend_offsets(self.mapping, model.device)

        # Pre-allocate GPU buffers for reading state back from UIPC (from-UIPC direction).
        # Uses uipc.adapter.warp.buffer() so copy_transform_to/copy_velocity_to can write
        # directly into device memory owned by us.
        self._abd_accessor: AffineBodyStateAccessorFeature = self.world.features().find(AffineBodyStateAccessorFeature)  # ty:ignore[invalid-assignment]
        n = self.mapping.num_mapped_bodies
        if n > 0:
            self._abd_transform_buf = uipc.adapter.warp.buffer(n, dtype=wp.mat44d, device=model.device)
            self._abd_velocity_buf = uipc.adapter.warp.buffer(n, dtype=wp.mat44d, device=model.device)
        else:
            self._abd_transform_buf = None
            self._abd_velocity_buf = None

        self._initialized = True

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Simulate one time step using UIPC.

        If the solver was created with ``auto_init=False`` and
        :meth:`initialize` has not been called yet, it is called
        automatically before the first step.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input. ``None`` uses model defaults.
            contacts: Unused -- UIPC handles contacts internally.
            dt: The time step [s].
        """
        if not self._initialized:
            self.initialize()

        if abs(dt - self._dt) > 1e-10 and self._step_count == 0:
            warnings.warn(
                f"SolverUIPC: step dt={dt} differs from configured dt={self._dt}. "
                "UIPC uses a fixed time step set at construction.",
                stacklevel=2,
            )

        if control is None:
            control = self.model.control(clone_variables=False)

        # Phase 1: Cache joint control
        self._articulation_builder.cache_joint_control(control)

        # Phase 2: Advance UIPC (animator callbacks fire here)
        self.world.advance()
        self.world.retrieve()

        # Phase 3: Read back results
        self._sync_body_state_from_uipc(state_out)
        self._articulation_builder.write_joint_readback(state_out)

        if state_out.body_f is not None:
            state_out.body_f.zero_()

        self._step_count += 1
        self._articulation_builder.increment_step()

    @override
    def notify_model_changed(self, flags: int) -> None:
        """Notify the solver that parts of the model were modified.

        Supported flags:
            - :attr:`~newton.SolverNotifyFlags.BODY_PROPERTIES`: push
              ``model.body_q`` and ``model.body_qd`` into the UIPC backend
              for the mapped affine bodies (state reset).
            - :attr:`~newton.SolverNotifyFlags.JOINT_PROPERTIES`: recompute
              forward kinematics from ``model.joint_q`` and push the
              resulting body state into UIPC.

        Other flags (``BODY_INERTIAL_PROPERTIES``, ``SHAPE_PROPERTIES``,
        etc.) are not currently supported and trigger a warning; recreating
        the solver is the recommended workaround for those.

        .. note::

            After ``JOINT_PROPERTIES`` the rigid-body state (``body_q`` /
            ``body_qd``) is reset consistently, but UIPC's internal
            revolute/prismatic joint angle tracker does **not** reflect the
            reset until the simulation has taken at least one step that
            drives the joint through the new configuration. If your
            downstream code relies on ``state.joint_q`` immediately after a
            reset, prefer reading from ``model.joint_q`` (which is updated
            by the internal FK call) until the next step completes.

        Args:
            flags: Bit-mask of model-update flags.
        """
        if not self._initialized:
            # Nothing to push yet -- :meth:`initialize` will read the
            # model's current state when it runs.
            return

        state_mask = SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.JOINT_PROPERTIES
        unsupported_mask = (
            SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            | SolverNotifyFlags.SHAPE_PROPERTIES
            | SolverNotifyFlags.JOINT_DOF_PROPERTIES
        )

        if flags & unsupported_mask:
            warnings.warn(
                "SolverUIPC.notify_model_changed: inertial / shape / joint-axis updates are "
                "not supported. Recreate the solver if these properties have changed.",
                stacklevel=2,
            )

        if flags & state_mask == 0:
            return

        # Recompute FK so that model.body_q / self._articulation_builder._body_transforms
        # reflect the new model.joint_q before we push state into UIPC.
        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            # Full-model FK sweep (joint_range=None) updates all child body
            # transforms and flushes them back to model.body_q.
            self._articulation_builder.compute_fk()

        self._push_body_state_to_uipc()

    def _push_body_state_to_uipc(self) -> None:
        """Push ``model.body_q`` / ``model.body_qd`` into the UIPC backend.

        Uses Warp kernels to build the 4x4 transform / velocity matrices
        on-device in one batched launch, then performs a single
        device→host copy and scatters the rows into each geometry slot's
        per-instance ``transform`` / ``velocity`` attributes before calling
        ``AffineBodyStateAccessorFeature.copy_from`` to synchronise the
        UIPC backend.

        UIPC's Python API only exposes ``copy_from(SimplicialComplex)``
        for the write direction, so a single host copy is unavoidable —
        but the math and the 128-bit element construction are done on
        GPU instead of per-body Python, which is a large speedup for
        models with many affine bodies.
        """
        mapping = self.mapping
        if mapping.num_mapped_bodies == 0 or not mapping.body_geo_slots:
            return

        model = self.model
        if model.body_q is None:
            return
        assert mapping.body_indices_wp is not None
        assert self._abd_transform_buf is not None
        assert self._abd_velocity_buf is not None

        n = mapping.num_mapped_bodies
        device = model.device

        # Batch-convert every mapped body's transform (and velocity, if
        # available) into UIPC's 4x4 mat64 layout on-device. The same
        # ``_abd_transform_buf`` / ``_abd_velocity_buf`` pool used by the
        # read-back path is re-purposed for this write direction to
        # avoid an extra allocation.
        wp.launch(
            _transform_to_mat44_kernel,
            dim=n,
            inputs=[model.body_q, mapping.body_indices_wp, self._abd_transform_buf.warp()],
            device=device,
        )
        if model.body_qd is not None:
            wp.launch(
                _spatial_to_vel_mat44_kernel,
                dim=n,
                inputs=[model.body_qd, mapping.body_indices_wp, self._abd_velocity_buf.warp()],
                device=device,
            )
        else:
            self._abd_velocity_buf.warp().zero_()

        # Single device→host sync: both buffers now hold per-mapped-body
        # mat44d rows aligned with ``mapping.body_indices_wp``.
        transforms_host = self._abd_transform_buf.warp().numpy()
        velocities_host = self._abd_velocity_buf.warp().numpy()

        # Build / reuse a body_idx → row-in-host-array lookup. This is a
        # function of ``body_indices_wp``, which is immutable after
        # ``populate_backend_offsets``, so cache it on the solver.
        if getattr(self, "_mapped_body_row", None) is None:
            body_indices_np = mapping.body_indices_wp.numpy()
            self._mapped_body_row = {int(b): i for i, b in enumerate(body_indices_np)}
        body_row = self._mapped_body_row

        # Group mapped bodies by the geometry slot they share (instanced
        # bodies share a slot with distinct instance ids).
        slot_bodies: dict[int, list[int]] = {}
        slot_objs: dict[int, Any] = {}
        for body_idx, slot in mapping.body_geo_slots.items():
            sid = slot.id()
            slot_bodies.setdefault(sid, []).append(body_idx)
            slot_objs[sid] = slot

        for sid, body_list in slot_bodies.items():
            slot = slot_objs[sid]
            geo = slot.geometry()

            # ``copy_to`` ensures ``transform`` and ``velocity`` instance
            # attributes exist and are sized to the slot's instance
            # count; we then selectively overwrite the rows we own.
            self._abd_accessor.copy_to(geo)

            transform_attr = geo.instances().find("transform")
            velocity_attr = geo.instances().find("velocity")
            if transform_attr is None:
                warnings.warn(
                    f"SolverUIPC.notify_model_changed: geometry slot {sid} has no 'transform' "
                    "instance attribute after copy_to; skipping.",
                    stacklevel=3,
                )
                continue
            transform_view = transform_attr.view()
            velocity_view = velocity_attr.view() if velocity_attr is not None else None

            for body_idx in body_list:
                row = body_row.get(body_idx)
                if row is None:
                    continue  # unmapped body (e.g. shapeless kinematic)
                inst_id = mapping.body_instance_ids.get(body_idx, 0)
                transform_view[inst_id] = transforms_host[row]
                if velocity_view is not None:
                    velocity_view[inst_id] = velocities_host[row]

            # Push the (now-updated) per-instance attributes into UIPC.
            self._abd_accessor.copy_from(geo)

    @override
    def update_contacts(self, contacts: Contacts) -> None:  # ty:ignore[invalid-method-override]  # pyright: ignore[reportIncompatibleMethodOverride]
        """Update a Contacts object. No-op -- UIPC handles contacts internally."""
        pass

    # ------------------------------------------------------------------
    # GPU batch sync methods
    # ------------------------------------------------------------------

    def _sync_body_state_from_uipc(self, state_out: State) -> None:
        """Read UIPC body state back into Newton state arrays via pre-allocated GPU buffers.

        Uses ``copy_transform_to`` / ``copy_velocity_to`` to let UIPC copy its
        internal state into our pre-allocated :class:`uipc.adapter.warp` buffers.
        The kernel accounts for Eigen's column-major layout by swapping row/column
        indices.
        """
        model = self.model
        n = self.mapping.num_mapped_bodies
        if n > 0 and state_out.body_q is not None:
            assert self.mapping.backend_offsets_wp is not None
            assert self._abd_transform_buf is not None
            assert self._abd_velocity_buf is not None

            # Copy UIPC backend state into our pre-allocated device buffers.
            self._abd_accessor.copy_transform_to(self._abd_transform_buf.buffer_view(), 0, n)
            self._abd_accessor.copy_velocity_to(self._abd_velocity_buf.buffer_view(), 0, n)

            wp.launch(
                _read_from_backend_kernel,
                dim=n,
                inputs=[
                    self.mapping.backend_offsets_wp,
                    self._abd_transform_buf.warp(),
                    self._abd_velocity_buf.warp(),
                    self.mapping.body_indices_wp,
                    state_out.body_q,
                    state_out.body_qd,
                ],
                device=model.device,
            )
