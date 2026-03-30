# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""UIPC physics engine solver backend for Newton."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import uipc
import uipc.builtin as uipc_builtin
import warp as wp
from uipc import Logger as ULogger
from uipc import Matrix4x4, view
from uipc.core import AffineBodyStateAccessorFeature
from uipc.core import Scene as UScene
from uipc.unit import GPa, MPa

from ...core.types import override
from ...sim import Contacts, Control, Model, State
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


        # Customize contact tabular (receives the scene's ContactTabular object)
        def setup_contacts(tabular):
            elem_a = tabular.create("rubber")
            elem_b = tabular.create("steel")
            tabular.insert(elem_a, elem_b, 0.8, 1e9, False)
            return elem_a  # returned as default contact element


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
        auto_init: bool = True,
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
        scene_config["contact"]["enable"] = False
        if model.gravity is not None:
            gravity_np = model.gravity.numpy().flatten()
            scene_config["gravity"] = [[float(gravity_np[0])], [float(gravity_np[1])], [float(gravity_np[2])]]
        self._scene_config = scene_config

        # User-registered callbacks (set via configure_* methods)
        self._contact_tabular_fn: Callable | None = None
        self._subscene_tabular_fn: Callable | None = None

        # Builders (populated during initialize)
        self._rigid_body_builder: RigidBodyBuilder | None = None
        self._articulation_builder: ArticulationBuilder | None = None
        self._cloth_builder: ClothBuilder | None = None
        self._deformable_builder: DeformableBodyBuilder | None = None

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

    def configure_contact_tabular(self, fn: Callable) -> None:
        """Register a callback to configure the UIPC contact tabular before initialization.

        The callback receives the UIPC ``ContactTabular`` object and should
        create contact elements, insert contact pairs, and return the **default**
        contact element that will be used by the rigid body, cloth, and deformable
        builders. If the callback returns ``None``, a default element with
        ``friction=0.5`` and ``stiffness=1 GPa`` is created automatically.

        Must be called **before** :meth:`initialize` (i.e. with ``auto_init=False``).

        Args:
            fn: A callable with signature ``fn(tabular) -> contact_element | None``.
                ``tabular`` is the UIPC ``ContactTabular`` object obtained from
                ``scene.contact_tabular()``.

        Raises:
            RuntimeError: If the solver has already been initialized.

        Example
        -------

        .. code-block:: python

            def setup_contacts(tabular):
                rubber = tabular.create("rubber")
                steel = tabular.create("steel")
                tabular.insert(rubber, steel, 0.8, 1e9, False)
                tabular.insert(rubber, rubber, 0.6, 1e8, False)
                return rubber  # default element for body/cloth/deformable builders


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

        # Contact tabular
        if self._contact_tabular_fn is not None:
            result = self._contact_tabular_fn(self.scene.contact_tabular())
            if result is not None:
                contact_elem = result
            else:
                contact_elem = self.scene.contact_tabular().create("default")
                self.scene.contact_tabular().insert(contact_elem, contact_elem, 0.5, 1.0 * GPa, False)
        else:
            contact_elem = self.scene.contact_tabular().create("default")
            self.scene.contact_tabular().insert(contact_elem, contact_elem, 0.5, 1.0 * GPa, False)

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
        rb = RigidBodyBuilder(model, scene, contact_elem, self.mapping, self._kappa, self._default_mass_density)
        ab = ArticulationBuilder(model, scene, self.mapping, self._dt, contact_elem=contact_elem, kappa=self._kappa)
        cb = ClothBuilder(model, scene, contact_elem, self.mapping)
        db = DeformableBodyBuilder(
            model, scene, contact_elem, self.mapping, default_mass_density=self._default_mass_density
        )

        rb.build_ground_planes()

        if model.world_count > 1:
            assert model.body_world_start is not None
            assert model.joint_world_start is not None
            # Host-side indexing needed to compute per-world ranges (init only)
            body_ws = model.body_world_start.numpy() if model.body_world_start is not None else None
            joint_ws = model.joint_world_start.numpy() if model.joint_world_start is not None else None
            particle_ws = model.particle_world_start.numpy() if model.particle_world_start is not None else None

            for world_index in range(model.world_count):
                body_range = (int(body_ws[world_index]), int(body_ws[world_index + 1])) if body_ws is not None else None
                joint_range = (
                    (int(joint_ws[world_index]), int(joint_ws[world_index + 1])) if joint_ws is not None else None
                )
                particle_range = (
                    (int(particle_ws[world_index]), int(particle_ws[world_index + 1]))
                    if particle_ws is not None
                    else None
                )
                se = subscene_elements[world_index] if subscene_elements is not None else None

                rb.build_body_shape_mapping(body_range)
                rb.build_affine_bodies(body_range, se)
                ab.build_joints(rb.body_transforms, joint_range, se)
                if cb.has_cloth:
                    cb.build(particle_range, se)
                if db.has_deformable:
                    db.build(particle_range, se)
        else:
            rb.build_body_shape_mapping()
            rb.build_affine_bodies()
            ab.build_joints(rb.body_transforms)
            if cb.has_cloth:
                cb.build()
            if db.has_deformable:
                db.build()

        self._rigid_body_builder = rb
        self._articulation_builder = ab
        self._cloth_builder = cb
        self._deformable_builder = db

        # Initialize UIPC world and set up state accessors
        self.world.init(scene)
        populate_backend_offsets(self.mapping, model.device)

        self._abd_accessor: AffineBodyStateAccessorFeature = self.world.features().find(AffineBodyStateAccessorFeature)  # ty:ignore[invalid-assignment]
        self._abd_state_geo = self._abd_accessor.create_geometry()
        self._abd_state_geo.instances().create(uipc_builtin.transform, Matrix4x4.Zero())
        self._abd_state_geo.instances().create(uipc_builtin.velocity, Matrix4x4.Zero())

        # Pre-allocate GPU buffers for batch transform sync
        n = self.mapping.num_mapped_bodies
        if n > 0:
            self._transforms_wp = wp.zeros(n, dtype=wp.mat44d, device=model.device)
            self._velocities_wp = wp.zeros(n, dtype=wp.mat44d, device=model.device)
        else:
            self._transforms_wp = None
            self._velocities_wp = None

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

        Args:
            flags: Bit-mask of model-update flags.
        """
        warnings.warn(
            "SolverUIPC.notify_model_changed: incremental model updates are not yet supported. "
            "Consider recreating the solver if the model has changed significantly.",
            stacklevel=2,
        )

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        """Update a Contacts object. No-op -- UIPC handles contacts internally."""
        pass

    # ------------------------------------------------------------------
    # GPU batch sync methods
    # ------------------------------------------------------------------

    def _sync_body_state_to_uipc(self, state_in: State) -> None:
        """Write Newton body transforms and velocities into UIPC."""
        model = self.model
        n = self.mapping.num_mapped_bodies

        if n > 0 and state_in.body_q is not None:
            wp.launch(
                _transform_to_mat44_kernel,
                dim=n,
                inputs=[state_in.body_q, self.mapping.body_indices_wp, self._transforms_wp],
                device=model.device,
            )
            if state_in.body_qd is not None:
                wp.launch(
                    _spatial_to_vel_mat44_kernel,
                    dim=n,
                    inputs=[state_in.body_qd, self.mapping.body_indices_wp, self._velocities_wp],
                    device=model.device,
                )

            self._abd_accessor.copy_to(self._abd_state_geo)
            transform_view = view(self._abd_state_geo.transforms())
            velocity_view = view(self._abd_state_geo.instances().find(uipc_builtin.velocity))  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]

            # UIPC view is numpy-based; vectorised scatter via advanced indexing
            assert self._transforms_wp is not None
            assert self._velocities_wp is not None
            assert self.mapping.backend_offsets_wp is not None
            offsets = self.mapping.backend_offsets_wp.numpy()
            transform_view[offsets] = self._transforms_wp.numpy()
            velocity_view[offsets] = self._velocities_wp.numpy()

            self._abd_accessor.copy_from(self._abd_state_geo)

    def _sync_body_state_from_uipc(self, state_out: State) -> None:
        """Read UIPC body state back into Newton state arrays using GPU kernels."""
        model = self.model
        n = self.mapping.num_mapped_bodies
        if n > 0 and state_out.body_q is not None:
            self._abd_accessor.copy_to(self._abd_state_geo)
            transform_view = view(self._abd_state_geo.transforms())
            velocity_view = view(self._abd_state_geo.instances().find(uipc_builtin.velocity))  # ty:ignore[no-matching-overload]  # pyright: ignore[reportArgumentType]

            # Pass full UIPC view to kernel; offsets index into it directly
            assert self.mapping.backend_offsets_wp is not None
            transforms_wp = wp.from_numpy(np.ascontiguousarray(transform_view), dtype=wp.mat44d, device=model.device)
            velocities_wp = wp.from_numpy(np.ascontiguousarray(velocity_view), dtype=wp.mat44d, device=model.device)

            wp.launch(
                _read_from_backend_kernel,
                dim=n,
                inputs=[
                    self.mapping.backend_offsets_wp,
                    transforms_wp,
                    velocities_wp,
                    self.mapping.body_indices_wp,
                    state_out.body_q,
                    state_out.body_qd,
                ],
                device=model.device,
            )
