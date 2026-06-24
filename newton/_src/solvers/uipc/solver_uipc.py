# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""UIPC physics engine solver backend for Newton."""

from __future__ import annotations

import os
import warnings
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import uipc
import uipc.adapter.warp
import warp as wp
from uipc import Logger as ULogger
from uipc.core import (
    AffineBodyStateAccessorFeature,
    ContactElement,
    ContactSystemFeature,
    ContactTabular,
    FiniteElementStateAccessorFeature,
    SanityCheckResult,
    SceneIO,
    SubsceneElement,
)
from uipc.core import Scene as UScene
from uipc.stats import SimulationStats as USimulationStats
from uipc.unit import GPa

import newton
from newton import BodyFlags, Contacts, Control, JointType, Model, ModelBuilder, State, StateFlags

from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .utils import _view_attr

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override
from .articulation_builder import ArticulationBuilder
from .cloth import ClothBuilder
from .contact_forces import (
    ContactForceReadback,
    _populate_contact_pairs_kernel,
    _scatter_contact_forces_kernel,
    build_gpu_vertex_maps,
    prepare_contact_gpu_data,
    retrieve_contact_forces,
)
from .converter import (
    UIpcMappingInfo,
    _read_fem_particle_positions_from_backend_kernel,
    _read_fem_particles_from_backend_kernel,
    _read_from_backend_kernel,
    _spatial_to_vel_mat44_kernel,
    _transform_to_mat44_kernel,
    _write_fem_particle_positions_to_backend_kernel,
    _write_fem_particles_to_backend_kernel,
    populate_backend_offsets,
)
from .deformable_body import DeformableBodyBuilder
from .rigid_body import RigidBodyBuilder

# ---------------------------------------------------------------------------
# UIPC ABD meta-attribute names (see libuipc AffineBodyConstitution).
# Every AffineBody geometry carries six per-group meta attributes that
# together describe its rigid-body mass properties:
#
#   "mass"                 Float   - scalar mass m                [kg]
#   "mass_center"          Vec3    - COM c in body frame           [m]
#   "inertia"              Mat3x3  - standard inertia at COM       [kg*m^2]
#                                    (= integral rho*(|r|^2*E - r*r^T) dV)
#   "abd_mass"             Float   - same m (cached)
#   "abd_mass_x_bar"       Vec3    - integral rho*x dV      = m*c
#   "abd_mass_x_bar_x_bar" Mat3x3  - integral rho*x*x^T dV  (about body-frame origin)
#
# The "friendly" triplet matches Newton's ``body_mass`` / ``body_com`` /
# ``body_inertia`` conventions directly (inertia is about the COM, in
# body-local axes, units kg*m^2), so reading back is a plain copy.
# ---------------------------------------------------------------------------
_UIPC_MASS_ATTR: str = "mass"
_UIPC_COM_ATTR: str = "mass_center"
_UIPC_INERTIA_ATTR: str = "inertia"
_UIPC_ABD_MASS_ATTR: str = "abd_mass"
_UIPC_ABD_MX_ATTR: str = "abd_mass_x_bar"
_UIPC_ABD_MXX_ATTR: str = "abd_mass_x_bar_x_bar"


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

        solver = newton.solvers.SolverUIPC(model, dt=1.0 / 60.0)

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

    For multi-world models produced by :meth:`~newton.ModelBuilder.replicate`,
    the solver uses UIPC's ``subscene_tabular`` to configure contact isolation
    between Newton worlds within a single UIPC scene. By default, bodies in
    different Newton worlds do not contact each other. Use
    :meth:`configure_subscene_tabular` to customize cross-world contact.

    .. note::

        - This solver requires ``libuipc`` (the ``uipc`` Python package) to be installed.
        - Supports rigid bodies (AffineBody), cloth (NeoHookeanShell), and deformable bodies
          (StableNeoHookean by default).
        - Joint types: REVOLUTE, PRISMATIC, FIXED, FREE.
        - BALL, DISTANCE, D6, and CABLE joints are not supported.
    """

    _uipc = None

    CONTACTS_PER_ENV: int = 1024
    """Per-environment contact budget for :meth:`get_max_contact_count`.

    Used to estimate the reported :class:`~newton.Contacts` capacity when
    ``rigid_contact_max`` is not given: the estimate is this value times
    ``Model.num_envs`` (mirroring MuJoCo's ``nconmax`` times the world count).
    """

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register UIPC-specific custom Model attributes.

        ``uipc:abd_kappa`` is a per-body override for the
        :class:`~uipc.constitution.AffineBodyConstitution` stiffness parameter
        [Pa].  A negative value leaves the solver-level ``kappa`` default in
        effect for that body. ``uipc:cloth_model`` and
        ``uipc:deformable_model`` select constitutions per authored cloth or
        deformable range.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="abd_kappa",
                frequency=Model.AttributeFrequency.BODY,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=-1.0,
                namespace="uipc",
            )
        )
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="cloth", namespace="uipc"))
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="deformable_body", namespace="uipc"))
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="deformable_model",
                frequency="uipc:deformable_body",
                assignment=Model.AttributeAssignment.MODEL,
                dtype=str,
                default=DeformableBodyBuilder.DEFORMABLE_MODEL_STABLE_NEO_HOOKEAN,
                namespace="uipc",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="cloth_model",
                frequency="uipc:cloth",
                assignment=Model.AttributeAssignment.MODEL,
                dtype=str,
                default=ClothBuilder.CLOTH_MODEL_STRAIN_LIMITING_BARAFF_WITKIN,
                namespace="uipc",
            )
        )
        builder._sync_uipc_range_custom_frequencies()

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

    @staticmethod
    def _body_kappa_from_model(model: Model) -> np.ndarray | None:
        """Return optional per-body ABD stiffness overrides from ``model.uipc``.

        The ``uipc:abd_kappa`` custom attribute uses negative values as the
        "inherit solver default" sentinel. Positive values override the
        solver-level ``kappa`` for the corresponding body.
        """
        uipc_attrs = getattr(model, "uipc", None)
        if uipc_attrs is None or not hasattr(uipc_attrs, "abd_kappa"):
            return None

        abd_kappa = uipc_attrs.abd_kappa
        if isinstance(abd_kappa, wp.array):
            values = abd_kappa.numpy()
        else:
            values = np.asarray(abd_kappa)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape[0] != model.body_count:
            raise RuntimeError(
                f"uipc:abd_kappa must have one value per body; got {values.shape[0]} for {model.body_count} bodies."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("uipc:abd_kappa values must be finite.")
        return values

    def __init__(
        self,
        model: Model,
        backend: str = "cuda",
        workspace: str = "/tmp/newton_uipc",
        dt: float = 1.0 / 60.0,
        scene_config: dict[str, Any] | None = None,  # pyright: ignore[reportRedeclaration]
        kappa: float = 1 * GPa,
        default_mass_density: float = 1000.0,
        logger_level=ULogger.Warn,
        dump_enable: bool = False,
        require_profile: bool = False,
        auto_sync_inertia: bool = True,
        cloth_soft_position_strength_ratio: float = 100.0,
        enable_soft_position_constraint: bool = True,
        rigid_contact_max: int | None = None,
    ):
        """Create a UIPC solver instance from a Newton model.

        Args:
            model: The Newton model to simulate.
            backend: UIPC backend name (default: ``"cuda"``).
            workspace: Working directory for UIPC engine output. Also used
                as the destination for surface-mesh dumps when
                ``dump_enable=True`` and for performance reports written by
                :meth:`save_performance_report`.
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
            require_profile: Enable UIPC timer collection for performance
                reports. When ``True``, each :meth:`step` records timer data
                that can later be exported via :meth:`save_performance_report`.
                A weak-reference destructor hook automatically invokes
                :meth:`save_performance_report` when the solver is
                garbage-collected or at interpreter shutdown, so users do
                not need to call it explicitly.
            auto_sync_inertia: If ``True`` (default), :meth:`initialize`
                finishes by calling :meth:`sync_model_inertia_from_uipc`
                so the Newton model's ``body_mass`` / ``body_com`` /
                ``body_inertia`` (and their inverses) mirror the finalised
                UIPC ABD values.  Set to ``False`` to preserve the exact
                authored values from ``ModelBuilder`` — useful when a
                downstream consumer (e.g. Featherstone-derived mass matrix
                for stable PD) has already been tuned against those values
                and must not see UIPC's mesh-volume-derived drift.  Bodies
                flagged via :meth:`sync_uipc_inertia_with_model` are
                always pushed into UIPC regardless of this flag.
            cloth_soft_position_strength_ratio: Default UIPC
                ``SoftPositionConstraint`` strength ratio added to cloth
                vertices.  Vertices are unconstrained until
                :meth:`set_cloth_soft_position_constraints` enables them.
            enable_soft_position_constraint: Whether to add dormant UIPC
                ``SoftPositionConstraint`` attributes to cloth and deformable
                vertices.
            rigid_contact_max: Capacity of the :class:`~newton.Contacts` buffer
                reported by :meth:`get_max_contact_count` for contact-sensor
                reporting. UIPC detects collisions internally and has no fixed
                contact-buffer size, so when ``None`` (default) the capacity is
                estimated as :attr:`CONTACTS_PER_ENV` times ``Model.num_envs``.
                Set an explicit value to override for dense-contact scenes.
        """
        super().__init__(model=model)
        self.import_uipc()

        ULogger.set_level(logger_level)

        self._dt = dt
        self._step_count = 0
        self._initialized = False

        # Lazily filled per-mapped-world host caches used by reset().
        self._mapped_body_world = None
        self._mapped_particle_world = None

        # Store construction parameters for deferred init
        self._backend = backend
        self._workspace = workspace
        self._kappa = kappa
        self._default_mass_density = default_mass_density
        self._dump_enable = dump_enable
        self._cloth_soft_position_strength_ratio = cloth_soft_position_strength_ratio
        self._enable_soft_position_constraint = enable_soft_position_constraint
        self._rigid_contact_max = rigid_contact_max

        # Scene config: start from UIPC defaults, apply Newton model overrides.
        if scene_config is None:
            scene_config: dict[str, Any] = UScene.default_config()
        scene_config["dt"] = dt
        scene_config["contact"]["d_hat"] = 0.001
        scene_config["contact"]["enable"] = False
        scene_config["newton"]["velocity_tol"] = 0.001
        scene_config["newton"]["transrate_tol"] = 0.01
        if model.gravity is not None:
            gravity_np = model.gravity.numpy().flatten()
            scene_config["gravity"] = [
                [float(gravity_np[0])],
                [float(gravity_np[1])],
                [float(gravity_np[2])],
            ]
        self._scene_config = scene_config

        # Performance statistics collector (only when enabled)
        self._stats: USimulationStats | None = USimulationStats() if require_profile else None
        self._auto_report_saved: bool = False

        # weakref.finalize fires when the solver is GC'd (e.g. viewer closed
        # → example freed → solver freed) OR at interpreter shutdown —
        # whichever comes first.  The atexit approach failed because the
        # solver is often a temporary whose ref dies before atexit fires.
        if require_profile:

            def _auto_save(stats: USimulationStats, workspace: str) -> None:
                if stats.num_frames == 0:
                    return
                try:
                    output_dir = os.path.join(workspace, "perf_report")
                    result = stats.summary_report(output_dir=output_dir, workspace=workspace)
                    if result is not None:
                        print(f"[SolverUIPC] Performance report saved to: {result}", flush=True)
                except Exception as exc:
                    warnings.warn(
                        f"SolverUIPC: save_performance_report() failed: {exc}",
                        stacklevel=1,
                    )

            self._finalizer = weakref.finalize(self, _auto_save, self._stats, self._workspace)  # ty:ignore[invalid-argument-type]

        # User-registered callbacks (set via configure_* methods)
        self._contact_tabular_fn: Callable | None = None
        self._subscene_tabular_fn: Callable | None = None

        # Bodies whose Newton-authored (mass, com, inertia) must be pushed
        # into the UIPC ABD geometry instead of letting UIPC re-derive them
        # from ``mass_density * mesh_volume``.  Populated via
        # :meth:`sync_uipc_inertia_with_model` before :meth:`initialize`.
        self._custom_inertia_bodies: set[int] = set()

        # Bodies that must not share UIPC AffineBody instancing.  Add indices
        # before :meth:`initialize`; passed through to
        # :meth:`~RigidBodyBuilder.build_affine_bodies` (merged with custom
        # inertia there).
        self._no_instance_bodies: set[int] = set()

        # Whether :meth:`initialize` should auto-call
        # :meth:`sync_model_inertia_from_uipc` at the end so the Newton
        # model mirrors UIPC's ABD-finalised mass properties.  Disable to
        # keep the authored ``ModelBuilder`` values untouched.
        self._auto_sync_inertia: bool = auto_sync_inertia

        # Builders (populated during initialize)
        self._rigid_body_builder: RigidBodyBuilder
        self._articulation_builder: ArticulationBuilder
        self._cloth_builder: ClothBuilder
        self._deformable_builder: DeformableBodyBuilder

        # Body → ContactElement mapping (populated during initialize)
        self._body_contact_elem: dict[int, ContactElement] = {}
        self._contact_tabular_ref: ContactTabular | None = None

    # ------------------------------------------------------------------
    # Pre-initialization configuration
    # ------------------------------------------------------------------

    def configure_scene(self, config: dict[str, Any]) -> None:
        """Update UIPC scene configuration before initialization.

        Performs a recursive deep merge of the provided overrides into the
        existing scene config.  For nested dicts the merge descends into
        sub-keys so that unmentioned siblings are preserved.  Non-dict
        values (scalars, lists) are replaced outright.

        Must be called **before** :meth:`initialize`.

        Args:
            config: Dictionary of UIPC scene configuration overrides.
                Common keys include ``"dt"``, ``"gravity"``,
                ``"newton"``, ``"line_search"``, ``"cfl"``, ``"friction"``,
                etc.  Refer to the UIPC documentation for the full list.

        Raises:
            RuntimeError: If the solver has already been initialized.

        Example
        -------

        .. code-block:: python

            solver = SolverUIPC(model)
            solver.configure_scene(
                {
                    "newton": {"velocity_tol": 1e-3},
                    "line_search": {"max_iter": 8},
                }
            )
            solver.initialize()
        """
        if self._initialized:
            raise RuntimeError("Cannot configure scene after initialization.")

        def merge(base: dict, override: dict) -> None:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge(base[key], value)
                else:
                    base[key] = value

        merge(self._scene_config, config)

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
            self._scene_config["contact"]["enable"] = flag
            if d_hat is not None:
                self._scene_config["contact"]["d_hat"] = float(d_hat)

    def set_animator_substep(self, substep: int) -> None:
        """Set the number of animator substeps per simulation step.

        Controls how many times the UIPC animator callbacks fire within a
        single ``world.advance()`` call.  Higher values give smoother
        kinematic target interpolation at the cost of more callback
        invocations.

        Must be called **after** :meth:`initialize`.

        Args:
            substep: Number of animator substeps (must be >= 1).

        Raises:
            RuntimeError: If the solver has not been initialized yet.
            ValueError: If *substep* < 1.
        """
        if not self._initialized:
            raise RuntimeError("Cannot set animator substep before initialization. Call initialize() first.")
        if substep < 1:
            raise ValueError(f"substep must be >= 1, got {substep}")
        self.scene.animator().substep(substep)

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

        Must be called **before** :meth:`initialize`.

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


            solver = SolverUIPC(model)
            solver.configure_contact_tabular(setup_contacts)
            solver.initialize()
        """
        if self._initialized:
            raise RuntimeError("Cannot configure contact tabular after initialization.")
        self._contact_tabular_fn = fn

    def is_contact_enabled(self, body_a: int, body_b: int) -> bool:
        """Query whether contact is enabled between two bodies.

        Must be called **after** :meth:`initialize`.

        Use ``-1`` for the ground plane.

        Args:
            body_a: Index of the first body, or ``-1`` for ground plane.
            body_b: Index of the second body, or ``-1`` for ground plane.

        Returns:
            ``True`` if contact is enabled between the two bodies.

        Raises:
            RuntimeError: If the solver has not been initialized.
            KeyError: If a body index was not assigned a contact element.
        """
        if not self._initialized:
            raise RuntimeError("Solver must be initialized before querying contact state.")
        elem_a = self._ground_contact_elem if body_a == -1 else self._body_contact_elem[body_a]
        elem_b = self._ground_contact_elem if body_b == -1 else self._body_contact_elem[body_b]
        model = self._contact_tabular_ref.at(elem_a.id(), elem_b.id())  # pyright: ignore[reportOptionalMemberAccess]  # ty:ignore[unresolved-attribute]
        return model.is_enabled()

    def configure_subscene_tabular(self, fn: Callable) -> None:
        """Register a callback to customize subscene contact configuration.

        For multi-world models, the solver creates one UIPC subscene per Newton
        world. By default, bodies in different worlds do **not** contact each
        other (replicating the old ``separate_worlds`` behavior). This callback
        lets you override the default subscene contact table.

        Must be called **before** :meth:`initialize`.

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


            solver = SolverUIPC(model)
            solver.configure_subscene_tabular(setup_subscenes)
            solver.initialize()
        """
        if self._initialized:
            raise RuntimeError("Cannot configure subscene tabular after initialization.")
        self._subscene_tabular_fn = fn

    # ------------------------------------------------------------------
    # Mass / inertia bridge: read ABD-derived values back into Newton
    # ------------------------------------------------------------------

    def sync_uipc_inertia_with_model(
        self,
        body_indices: list[int] | None = None,
    ) -> list[int]:
        """Mark bodies whose UIPC ABD mass properties must follow Newton.

        By default :class:`~uipc.constitution.AffineBodyConstitution` recomputes
        each body's mass, COM, and inertia from ``mass_density * mesh_volume``
        and the mesh's spatial moments — ignoring any hand-authored values in
        ``ModelBuilder`` (e.g. :attr:`Model.body_com`, :attr:`Model.body_inertia`
        set from URDF ``<inertial>`` or direct assignment).

        Calling this before :meth:`initialize` flags a set of body indices
        whose geometry should instead be built through the explicit
        ``apply_to(sc, kappa, mass_matrix, volume)`` overload, with the 12x12
        ABD mass matrix produced by
        :func:`uipc.geometry.affine_body.from_rigid_body` from Newton's
        :attr:`Model.body_mass` / :attr:`Model.body_com` /
        :attr:`Model.body_inertia`.

        Because a single UIPC ``SimplicialComplex`` carries one shared set of
        ABD meta attributes, each flagged body is removed from the per-shape
        instance grouping in :meth:`RigidBodyBuilder.build_affine_bodies` and
        placed into its own geometry.

        Must be called **before** :meth:`initialize`.

        Args:
            body_indices: Bodies whose authored mass properties should be
                pushed into UIPC.  ``None`` = every body in the model.

        Returns:
            The full list of body indices currently flagged (cumulative
            across calls).

        Raises:
            RuntimeError: If the solver has already been initialized.
            IndexError: If any entry in ``body_indices`` is out of range.
        """
        if self._initialized:
            raise RuntimeError(
                "sync_uipc_inertia_with_model must be called before "
                "initialize() — the UIPC ABD geometry is built there and "
                "cannot be rewritten after world.init()."
            )
        model = self.model
        if body_indices is None:
            indices: list[int] = list(range(model.body_count))
        else:
            indices = [int(b) for b in body_indices]
            for b in indices:
                if not (0 <= b < model.body_count):
                    raise IndexError(f"body index {b} out of range [0, {model.body_count})")
        self._custom_inertia_bodies.update(indices)
        return sorted(self._custom_inertia_bodies)

    def read_uipc_body_inertia(self, body_idx: int) -> dict[str, Any]:
        """Read UIPC's ABD mass properties for a mapped Newton body.

        Must be called **after** :meth:`initialize` (which runs
        ``AffineBodyConstitution.apply_to`` and ``world.init(scene)``;
        the latter is when UIPC finalises the ABD integrals on each
        geometry).

        The six ``SimplicialComplex.meta()`` attributes of the body's
        UIPC geometry are returned:

        - ``mass``                 — ``float``, scalar mass [kg]
        - ``mass_center``          — ``(3,) float64``, COM in body frame [m]
        - ``inertia``              — ``(3, 3) float64``, standard inertia at COM [kg·m²]
        - ``abd_mass``             — ``float``, same value as ``mass``
        - ``abd_mass_x_bar``       — ``(3,) float64``, ``m·c``
        - ``abd_mass_x_bar_x_bar`` — ``(3, 3) float64``, second moment integral at origin  # noqa: RUF002

        Missing attributes map to ``None`` (happens e.g. for proxy
        bodies that were not built via ``AffineBodyConstitution``).

        Args:
            body_idx: Newton body index.

        Raises:
            RuntimeError: If the solver has not been initialized.
            KeyError: If ``body_idx`` has no mapped UIPC geometry.
        """
        if not self._initialized:
            raise RuntimeError(
                "read_uipc_body_inertia requires the solver to be initialized; "
                "call initialize() (or let step() do it) first."
            )
        geo_slot = self.mapping.body_geo_slots.get(body_idx)
        if geo_slot is None:
            raise KeyError(f"body {body_idx} has no mapped UIPC geometry")

        meta = geo_slot.geometry().meta()
        # Mirror the C++ pattern from the reference test:
        #   auto mass_attr = mesh1.meta().find<Float>("mass");
        #   Float mass_val = mass_attr->view()[0];
        # Python binding: meta.find(name) -> Attribute | None, then view(attr)[0].
        out: dict[str, Any] = {}
        for name, shape in (
            (_UIPC_MASS_ATTR, None),
            (_UIPC_ABD_MASS_ATTR, None),
            (_UIPC_COM_ATTR, (3,)),
            (_UIPC_ABD_MX_ATTR, (3,)),
            (_UIPC_INERTIA_ATTR, (3, 3)),
            (_UIPC_ABD_MXX_ATTR, (3, 3)),
        ):
            attr = meta.find(name)
            if attr is None:
                out[name] = None
                continue
            v = np.asarray(_view_attr(attr)[0], dtype=np.float64)
            out[name] = float(v) if shape is None else v.reshape(shape).copy()
        return out

    def sync_model_inertia_from_uipc(
        self,
        body_indices: list[int] | None = None,
    ) -> list[int]:
        """Overwrite Newton ``model.body_{mass,com,inertia}`` with UIPC ABD values.

        UIPC's :class:`AffineBodyConstitution` derives each body's mass,
        center-of-mass, and inertia from ``mass_density * mesh_volume``
        and the mesh's spatial moments, which can diverge from the
        URDF / USD-authored values stored in the Newton model when the
        collision geometry is simplified (hulls, boxes, etc.).

        This method reads the finalised UIPC meta attributes
        (``mass`` / ``mass_center`` / ``inertia``) and writes them back
        into ``model.body_mass`` / ``model.body_com`` /
        ``model.body_inertia``.  ``model.body_inv_mass`` and
        ``model.body_inv_inertia`` are refreshed to stay consistent.

        Must be called **after** ``world.init(scene)`` has run (i.e.
        after :meth:`initialize`); otherwise the ABD meta has not been
        finalised yet.

        Args:
            body_indices: Bodies to synchronise.  ``None`` = every body
                that has a UIPC geometry slot.

        Returns:
            The list of body indices that were actually written
            (skips unmapped bodies, bodies whose geometry lacks ABD
            attributes such as proxy bodies, and zero-mass bodies).
        """
        if not self._initialized:
            raise RuntimeError(
                "sync_model_inertia_from_uipc must be called after "
                "initialize()/world.init(); UIPC has not finalised the ABD "
                "meta attributes yet."
            )
        model = self.model
        if model.body_mass is None or model.body_com is None or model.body_inertia is None:
            return []

        if body_indices is None:
            body_indices = sorted(self.mapping.body_geo_slots.keys())

        # Pull once, mutate host arrays, push back in one shot — avoids
        # body_count round-trips to the device.
        body_mass_np = model.body_mass.numpy().copy()
        body_com_np = model.body_com.numpy().copy()
        body_inertia_np = model.body_inertia.numpy().copy()

        inv_mass_np = model.body_inv_mass.numpy().copy() if model.body_inv_mass is not None else None
        inv_inertia_np = model.body_inv_inertia.numpy().copy() if model.body_inv_inertia is not None else None

        written: list[int] = []
        for b in body_indices:
            props = self.read_uipc_body_inertia(b)
            m = props[_UIPC_MASS_ATTR]
            c = props[_UIPC_COM_ATTR]
            i_cm = props[_UIPC_INERTIA_ATTR]
            if m is None or c is None or i_cm is None:
                continue  # proxy or otherwise missing ABD metadata
            if m <= 0.0:
                continue

            body_mass_np[b] = np.float32(m)
            body_com_np[b] = np.asarray(c, dtype=np.float32)
            body_inertia_np[b] = np.asarray(i_cm, dtype=np.float32).reshape(3, 3)
            if inv_mass_np is not None:
                inv_mass_np[b] = np.float32(1.0 / m)
            if inv_inertia_np is not None:
                # Symmetric 3x3 invert via numpy; falls back to pseudo-inverse
                # if inertia is singular (e.g. a degenerate ABD proxy).
                try:
                    inv_i = np.linalg.inv(i_cm)
                except np.linalg.LinAlgError:
                    inv_i = np.linalg.pinv(i_cm)
                inv_inertia_np[b] = inv_i.astype(np.float32).reshape(3, 3)
            written.append(b)

        if written:
            model.body_mass.assign(body_mass_np)
            model.body_com.assign(body_com_np)
            model.body_inertia.assign(body_inertia_np)
            if inv_mass_np is not None and model.body_inv_mass is not None:
                model.body_inv_mass.assign(inv_mass_np)
            if inv_inertia_np is not None and model.body_inv_inertia is not None:
                model.body_inv_inertia.assign(inv_inertia_np)
        return written

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, state: State | None = None) -> None:  # pyright: ignore[reportRedeclaration]
        """Build UIPC scene objects from the Newton model and initialize the world.

        Creates a single UIPC Engine, World, and Scene. For multi-world models,
        configures ``subscene_tabular`` to isolate contact between Newton worlds.
        Builds rigid body / articulation / cloth / deformable geometries and
        calls ``world.init(scene)``.

        Call this explicitly after any :meth:`configure_scene`,
        :meth:`configure_contact_tabular`,
        :meth:`configure_subscene_tabular`, and
        :meth:`sync_uipc_inertia_with_model` calls.

        After ``world.init(scene)`` returns, the Newton model's
        :attr:`Model.body_mass` / :attr:`Model.body_com` /
        :attr:`Model.body_inertia` (and their inverses) are **auto-synced** to
        the finalised UIPC ABD values via
        :meth:`sync_model_inertia_from_uipc` so both sides agree on rigid
        body dynamics.  Bodies flagged through
        :meth:`sync_uipc_inertia_with_model` round-trip their authored
        values unchanged; all other bodies adopt UIPC's mesh-volume-derived
        triplet.  Shapeless articulation proxies (which carry sentinel ABD
        meta) are excluded from the sync.  Pass
        ``auto_sync_inertia=False`` to the solver constructor to skip this
        final sync and keep the authored ``ModelBuilder`` values verbatim.

        Args:
            state: Optional initial :class:`State` whose ``body_q`` /
                ``body_qd`` are pushed to UIPC after world init.  Typically
                the state the user has populated via :func:`newton.eval_fk`.
                If ``None``, falls back to running IK from ``model.body_q``
                followed by FK to sync body transforms.

        Raises:
            RuntimeError: If already initialized.
        """
        if self._initialized:
            raise RuntimeError("SolverUIPC is already initialized.")

        model = self.model

        os.makedirs(self._workspace, exist_ok=True)

        # Create a single UIPC Engine / World / Scene
        self.engine = uipc.Engine(backend_name=self._backend, workspace=self._workspace)
        self.world = uipc.World(self.engine)
        self.scene = uipc.Scene(self._scene_config)
        print(f"scene_config:{self._scene_config}")
        body_kappa = self._body_kappa_from_model(model)

        # Subscene tabular for multi-world contact isolation — set up BEFORE
        # contact elements so that elements can be created within subscenes.
        subscene_elements: list[SubsceneElement] = []
        tabular = self.scene.subscene_tabular()
        default_subscene_elem = tabular.default_element()
        for world_index in range(model.world_count):
            se = tabular.create(f"world_{world_index}")
            subscene_elements.append(se)

        # Cross-subscene contact is disabled by default in UIPC;
        # only enable each world ↔ default (ground).
        for se in subscene_elements:
            tabular.insert(default_subscene_elem, se, True)

        # Let user override subscene configuration (called once with all
        # subscenes available).
        if self._subscene_tabular_fn is not None:
            self._subscene_tabular_fn(tabular, subscene_elements, default_subscene_elem)

        # Contact tabular — shared ground + per-world env / robot element pairs
        contact_tabular: ContactTabular = self.scene.contact_tabular()
        self._contact_tabular_ref = contact_tabular
        # Ground element is shared across all worlds
        ground_elem: ContactElement = contact_tabular.default_element()
        self._ground_contact_elem = ground_elem
        env_elems: list[ContactElement] = []
        robo_elems: list[ContactElement] = []
        actor_elems: list[ContactElement] = []
        body_element_overrides: dict[int, ContactElement] = {}

        for world_index in range(model.world_count):
            suffix = f"_{world_index}"
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

        self.mapping = UIpcMappingInfo()
        scene: UScene = self.scene

        # Create one builder per type (reused across worlds)
        self._rigid_body_builder = RigidBodyBuilder(model, scene, self.mapping, self._kappa, self._default_mass_density)
        self._articulation_builder = ArticulationBuilder(
            model, scene, self.mapping, self._dt, kappa=self._kappa, body_kappa=body_kappa
        )
        self._cloth_builder = ClothBuilder(
            model,
            scene,
            self.mapping,
            enable_soft_position_constraint=self._enable_soft_position_constraint,
            soft_position_strength_ratio=self._cloth_soft_position_strength_ratio,
        )
        self._deformable_builder = DeformableBodyBuilder(
            model,
            scene,
            self.mapping,
            default_mass_density=self._default_mass_density,
            enable_soft_position_constraint=self._enable_soft_position_constraint,
        )

        self._rigid_body_builder.build_ground_planes(ground_elem)

        # Build set of body indices that belong to articulations (robot links),
        # a separate set for bodies attached via free joints, and auto-flag
        # ball-joint children so they don't share AffineBody instances with
        # shape-key siblings.  User-supplied entries in
        # ``self._no_instance_bodies`` are merged with the BALL set below.
        articulation_bodies: set[int] = set()
        free_joint_bodies: set[int] = set()
        ball_joint_bodies: set[int] = set()
        joint_child = model.joint_child
        joint_type = model.joint_type
        if joint_child is not None:
            joint_child_np = joint_child.numpy()
            joint_type_np = joint_type.numpy() if joint_type is not None else None
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
            joint_parent = model.joint_parent
            if joint_parent is not None:
                joint_parent_np = joint_parent.numpy()
                for j in range(model.joint_count):
                    parent = int(joint_parent_np[j])
                    if parent >= 0:
                        articulation_bodies.add(parent)

        # Kinematic free-joint bodies are environment, not actors
        if model.body_flags is not None and free_joint_bodies:
            body_flags_np = model.body_flags.numpy()
            free_joint_bodies -= {b for b in free_joint_bodies if int(body_flags_np[b]) & int(BodyFlags.KINEMATIC)}

        # Host-side indexing for per-world ranges (multi-world only)
        if model.world_count > 1:
            body_world_start = model.body_world_start
            joint_world_start = model.joint_world_start
            particle_world_start = model.particle_world_start
            if body_world_start is None or joint_world_start is None:
                raise RuntimeError("Multi-world UIPC initialization requires body and joint world ranges.")

            body_ws = body_world_start.numpy()
            joint_ws = joint_world_start.numpy()
            particle_ws = particle_world_start.numpy() if particle_world_start is not None else None
        else:
            body_ws = None
            joint_ws = None
            particle_ws = None

        if state is None:
            joint_q = model.joint_q
            joint_qd = model.joint_qd
            if model.joint_count > 0:
                if joint_q is None or joint_qd is None:
                    raise RuntimeError("UIPC initialization requires joint_q and joint_qd for articulated models.")
            state: State = model.state()
            if model.joint_count > 0:
                newton.eval_fk(model, joint_q, joint_qd, state)  # ty:ignore[invalid-argument-type]
        if model.body_q is not None and state.body_q is not None:
            wp.copy(model.body_q, state.body_q)
        if state.body_qd is not None and model.body_qd is not None:
            wp.copy(model.body_qd, state.body_qd)

        for world_index in range(model.world_count):
            if body_ws is not None:
                body_range = (int(body_ws[world_index]), int(body_ws[world_index + 1]))
                joint_range = (int(joint_ws[world_index]), int(joint_ws[world_index + 1]))  # ty:ignore[not-subscriptable]  # pyright: ignore[reportOptionalSubscript]
                particle_range = (
                    (int(particle_ws[world_index]), int(particle_ws[world_index + 1]))
                    if particle_ws is not None
                    else (0, model.particle_count)
                )
            else:
                body_range = (0, model.body_count)
                joint_range = (0, model.joint_count)
                particle_range: tuple[int, int] = (0, model.particle_count)
            se = subscene_elements[world_index]
            self._rigid_body_builder.build_body_shape_mapping(body_range)
            self._rigid_body_builder.build_affine_bodies(
                env_elems[world_index],
                robo_elems[world_index],
                actor_elems[world_index],
                articulation_bodies,
                free_joint_bodies,
                body_range,
                se,
                body_element_overrides,
                no_instance_bodies=ball_joint_bodies | self._no_instance_bodies,
                custom_inertia_bodies=self._custom_inertia_bodies,
                body_kappa=body_kappa,
            )
            for b in range(body_range[0], body_range[1]):
                self._body_contact_elem[b] = self._rigid_body_builder._resolve_contact_elem(
                    b,
                    env_elems[world_index],
                    robo_elems[world_index],
                    actor_elems[world_index],
                    articulation_bodies,
                    free_joint_bodies,
                    body_element_overrides,
                )
            self._rigid_body_builder.build_static_colliders(env_elems[world_index], se)
            self._articulation_builder.build_joints(robo_elems[world_index], joint_range, se)
            if self._cloth_builder.has_cloth:
                self._cloth_builder.build(actor_elems[world_index], particle_range, se)
            if self._deformable_builder.has_deformable:
                self._deformable_builder.build(actor_elems[world_index], particle_range, se)

        # Resolve mimic joint couplings now that every world's joints are
        # registered. Mimic followers are driven from their leaders each
        # step in :meth:`step` via ``apply_mimic_targets``.
        self._articulation_builder.setup_mimic_constraints()

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
        # Allocate buffers large enough to cover the highest backend index,
        # which may exceed num_mapped_bodies when UIPC assigns non-contiguous
        # backend offsets across worlds.
        buf_count = self.mapping.max_backend_count
        if n > 0:
            self._abd_transform_buf = uipc.adapter.warp.buffer(buf_count, dtype=wp.mat44d, device=model.device)
            self._abd_velocity_buf = uipc.adapter.warp.buffer(buf_count, dtype=wp.mat44d, device=model.device)
        else:
            self._abd_transform_buf = None
            self._abd_velocity_buf = None

        self._fem_accessor: FiniteElementStateAccessorFeature = self.world.features().find(
            FiniteElementStateAccessorFeature
        )  # ty:ignore[invalid-assignment]
        self._fem_position_buf = None
        self._fem_velocity_buf = None
        self._fem_backend_offsets_wp = None
        self._fem_particle_indices_wp = None
        self._fem_mapped_vertex_count = 0
        self._fem_backend_vertex_count = 0
        if self._fem_accessor is not None:
            fem_backend_offsets: list[int] = []
            fem_particle_indices: list[int] = []
            for geo_slot, particle_indices in [
                *zip(self.mapping.cloth_geo_slots, self.mapping.cloth_particle_indices, strict=False),
                *zip(self.mapping.deformable_geo_slots, self.mapping.deformable_particle_indices, strict=False),
            ]:
                geo = geo_slot.geometry()
                offset_attr = geo.meta().find("backend_fem_vertex_offset") or geo.meta().find("global_vertex_offset")
                if offset_attr is None:
                    continue
                backend_offset = int(_view_attr(offset_attr)[0])
                if backend_offset < 0:
                    continue
                particle_indices_np = np.asarray(particle_indices, dtype=np.int32)
                fem_backend_offsets.extend(backend_offset + i for i in range(particle_indices_np.size))
                fem_particle_indices.extend(int(i) for i in particle_indices_np)

            if fem_backend_offsets:
                self._fem_backend_vertex_count = int(self._fem_accessor.vertex_count())
                self._fem_mapped_vertex_count = len(fem_backend_offsets)
                self._fem_backend_offsets_wp = wp.array(
                    fem_backend_offsets,
                    dtype=wp.uint32,
                    device=model.device,
                )
                self._fem_particle_indices_wp = wp.array(
                    fem_particle_indices,
                    dtype=wp.int32,
                    device=model.device,
                )
                self._fem_position_buf = uipc.adapter.warp.buffer(
                    self._fem_backend_vertex_count,
                    dtype=wp.vec3d,
                    device=model.device,
                )
                self._fem_velocity_buf = uipc.adapter.warp.buffer(
                    self._fem_backend_vertex_count,
                    dtype=wp.vec3d,
                    device=model.device,
                )

        self._csf: ContactSystemFeature | None = self.world.features().find(ContactSystemFeature)  # ty:ignore[invalid-assignment]

        if self._csf is not None:
            build_gpu_vertex_maps(self.mapping, model.body_count, model.device)

        self._initialized = True

        # Sync model mass/inertia from UIPC ABD (after world init) so host-side
        # dynamics (e.g. eval_mass_matrix) match the UIPC body properties.
        # Skip shapeless articulation proxies; set auto_sync_inertia=False to
        # leave model inertias as authored.
        if self._auto_sync_inertia:
            shape_backed_bodies = [b for b in self.mapping.body_geo_slots if self.mapping.body_shapes.get(b)]
            if shape_backed_bodies:
                self.sync_model_inertia_from_uipc(shape_backed_bodies)

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None = None,
        dt: float | None = None,
    ) -> None:
        """Simulate one time step using UIPC.

        If :meth:`initialize` has not been called yet, it is called
        automatically before the first step.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input. ``None`` uses model defaults.
            contacts: Unused. UIPC computes contacts internally; call
                :meth:`update_contacts` after stepping to read them back.
            dt: Time step [s]. When omitted, the fourth positional argument must
                be the time step.
        """
        if dt is None:
            raise TypeError("SolverUIPC.step() missing required time step 'dt'.")

        if not self._initialized:
            self.initialize(state_in)

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

        # Snapshot pre-advance joint angles/distances so the post-retrieve
        # finite difference yields a true (q_{t+dt} - q_t) / dt velocity.
        self._articulation_builder.read_joint_state_pre_advance()

        # Drive mimic followers from their leaders. Runs after control
        # caching (target arrays populated, device synced) and after the
        # pre-advance snapshot (leader fallback value available), but
        # before the animator fires in world.advance().
        self._articulation_builder.apply_mimic_targets()

        # Dump surface geometry before physics advance
        if self._dump_enable:
            self.export_surface_obj(self._workspace)

        # Phase 2: Advance UIPC (animator callbacks fire here)
        self.world.advance()
        self.world.retrieve()
        if self._stats is not None:
            self._stats.collect()

        # Phase 3: Read back results
        self._sync_body_state_from_uipc(state_out)
        self._sync_particle_state_from_uipc(state_out)
        self._articulation_builder.read_joint_state_post_retrieve()
        self._articulation_builder.write_joint_readback(state_out)

        if state_out.body_f is not None:
            state_out.body_f.zero_()
        if state_out.particle_f is not None:
            state_out.particle_f.zero_()
        self._current_state_out = state_out

        self._step_count += 1
        self._articulation_builder.increment_step()

    def _get_contact_forces(self) -> ContactForceReadback:
        """Retrieve per-body, per-primitive contact forces from UIPC (diagnostic).

        Must be called after :meth:`step` (i.e. after ``world.retrieve()``).
        Returns a :class:`ContactForceReadback` with per-body, per-primitive,
        per-channel (normal/friction) force data.

        Body keys:
            - Non-negative int: rigid body index (ABD).
            - ``-1 - mesh_idx``: cloth mesh.
            - ``-(10000 + mesh_idx)``: deformable mesh.

        Returns:
            ContactForceReadback with ``data[body_key][prim_type][channel]``.
        """
        return self._retrieve_contact_data()

    def _retrieve_contact_data(self) -> ContactForceReadback:
        """Retrieve contact forces via CPU diagnostic path."""
        if not self._initialized:
            raise RuntimeError("_get_contact_forces() requires initialize() first.")

        if self._csf is None:
            return ContactForceReadback()

        model = self.model
        body_q_np = model.body_q.numpy() if model.body_q is not None else None
        body_com_np = model.body_com.numpy() if model.body_com is not None else None

        return retrieve_contact_forces(
            csf=self._csf,
            mapping=self.mapping,
            body_q_np=body_q_np,
            body_com_np=body_com_np,
            dt=self._dt,
        )

    def _ground_shape_index(self) -> int:
        """Return the first ground shape index, or -1 when no ground shape exists."""
        model = self.model
        shape_body = model.shape_body.numpy()  # ty:ignore[unresolved-attribute]
        for s in range(model.shape_count):
            if shape_body[s] == -1:
                return s
        return -1

    def export_surface_obj(self, path: str) -> None:
        """Export the current scene surface geometry as a Wavefront OBJ file.

        Writes the surface mesh of all bodies in the scene to a single
        ``.obj`` file using UIPC's built-in :class:`~uipc.core.SceneIO`.

        Must be called after :meth:`initialize` (or after the first
        :meth:`step`).

        Args:
            path: Directory to write the OBJ file into (created if needed).
        """
        if not self._initialized:
            raise RuntimeError(
                "SolverUIPC.export_surface_obj() requires the solver to be "
                "initialized. Call step() or initialize() first."
            )
        os.makedirs(path, exist_ok=True)
        sio = SceneIO(self.scene)
        sio.write_surface(os.path.join(path, f"scene_surface_{self.world.frame():06d}.obj"))

    def save_performance_report(
        self,
        output_dir: str | None = None,
        keys: list[str] | None = None,
    ) -> str | None:
        """Generate a UIPC performance summary report to disk.

        Produces a folder containing ``report.md``, per-timer SVG charts,
        a profiler heatmap, and (when available) a system dependency graph.
        Requires at least one :meth:`step` call so that timer data has been
        collected.

        Args:
            output_dir: Directory to write the report into.  Defaults to
                ``<workspace>/perf_report``.
            keys: Timer keys (or alias keys) for per-frame panels.
                Defaults to the UIPC built-in set (Newton iteration,
                global linear system, line search, DCD, SPMV).

        Returns:
            Path to the generated ``report.md``, or ``None`` if no frames
            have been collected.
        """
        if self._stats is None:
            warnings.warn(
                "Time report not enabled — set require_profile=True when constructing SolverUIPC.",
                stacklevel=2,
            )
            return None

        if self._stats.num_frames == 0:
            warnings.warn(
                "No simulation frames collected yet — call step() first.",
                stacklevel=2,
            )
            return None

        if output_dir is None:
            output_dir = os.path.join(self._workspace, "perf_report")

        kwargs: dict[str, object] = {"output_dir": output_dir}
        if keys is not None:
            kwargs["keys"] = keys
        kwargs["workspace"] = self._workspace

        result = self._stats.summary_report(**kwargs)  # ty:ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        self._auto_report_saved = True
        if hasattr(self, "_finalizer"):
            self._finalizer.detach()
        result_str = str(result) if result is not None else None
        if result_str is not None:
            print(f"[SolverUIPC] Performance report saved to: {result_str}", flush=True)
        return result_str

    @override
    def notify_model_changed(self, flags: int) -> None:
        """Notify the solver that parts of the model were modified.

        Dispatches supported flag bits to dedicated ``_notify_*`` handlers.
        Unsupported flags trigger a single aggregated warning -- the UIPC
        backend bakes those properties into scene objects at build time,
        so the user must recreate the solver to apply them.

        Supported flags:
            - :attr:`~newton.SolverNotifyFlags.BODY_PROPERTIES`: push
              ``model.body_q`` and ``model.body_qd`` into the UIPC backend
              for the mapped affine bodies (state reset).
            - :attr:`~newton.SolverNotifyFlags.JOINT_PROPERTIES`: recompute
              forward kinematics from ``model.joint_q`` / ``joint_qd`` via
              :func:`newton.eval_fk` and push the resulting ``body_q`` /
              ``body_qd`` into UIPC.
            - :attr:`~newton.SolverNotifyFlags.MODEL_PROPERTIES`: propagate
              ``model.gravity`` into the live UIPC ``scene.config()``; the
              new gravity takes effect on the next ``world.advance()``.

        Unsupported flags (aggregated into a single warning):
            ``JOINT_DOF_PROPERTIES``, ``BODY_INERTIAL_PROPERTIES``,
            ``SHAPE_PROPERTIES``, ``CONSTRAINT_PROPERTIES``,
            ``TENDON_PROPERTIES``, ``ACTUATOR_PROPERTIES``.

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

        # Unsupported flags: emit one aggregated warning. UIPC bakes these
        # properties into scene objects at build time (or has no
        # equivalent concept), so a runtime update is not possible -- the
        # user must recreate the solver.
        unsupported_mask = (
            SolverNotifyFlags.JOINT_DOF_PROPERTIES
            | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            | SolverNotifyFlags.SHAPE_PROPERTIES
            | SolverNotifyFlags.CONSTRAINT_PROPERTIES
            | SolverNotifyFlags.TENDON_PROPERTIES
            | SolverNotifyFlags.ACTUATOR_PROPERTIES
        )
        if flags & unsupported_mask:
            warnings.warn(
                "SolverUIPC.notify_model_changed: joint-DOF, body-inertial, shape, "
                "constraint, tendon, and actuator property updates are not supported "
                "by the UIPC backend. Recreate the solver if these properties changed.",
                stacklevel=2,
            )

        # Supported flags: dispatch each to its dedicated handler.
        # JOINT_PROPERTIES and BODY_PROPERTIES cooperate via
        # ``_state_dirty`` so only a single state push to UIPC runs even
        # when both flags are set together.
        self._state_dirty = False

        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self._notify_joint_properties()
        if flags & SolverNotifyFlags.BODY_PROPERTIES:
            self._notify_body_properties()
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self._notify_model_properties()

        if self._state_dirty:
            self._sync_state_to_uipc()
        self._state_dirty = False

    # ------------------------------------------------------------------
    # Per-flag notify_model_changed handlers (supported flags only)
    # ------------------------------------------------------------------

    def _notify_joint_properties(self) -> None:
        """Handle :attr:`~newton.SolverNotifyFlags.JOINT_PROPERTIES`.

        Recomputes forward kinematics with :func:`newton.eval_fk` so that
        ``model.body_q`` / ``model.body_qd`` reflect the updated
        ``model.joint_q`` / ``joint_qd`` / ``joint_X_p`` / ``joint_X_c``,
        then flags the state buffers for a push into UIPC.

        Uses a single on-device Warp launch over all articulations, which
        refreshes ``body_q`` and ``body_qd`` together from the new joint
        coordinates and velocities.
        """
        model = self.model
        joint_q = model.joint_q
        joint_qd = model.joint_qd
        if model.joint_count > 0:
            if joint_q is None or joint_qd is None:
                raise RuntimeError("UIPC joint property notification requires joint_q and joint_qd.")
        state = self.model.state()
        if model.joint_count > 0:
            newton.eval_fk(model, joint_q, joint_qd, state)  # ty:ignore[invalid-argument-type]
        self._state_dirty = True

    def _notify_body_properties(self) -> None:
        """Handle :attr:`~newton.SolverNotifyFlags.BODY_PROPERTIES`.
        Flags Newton-owned state buffers for a push into UIPC so that
        ``model.body_q`` / ``model.body_qd`` and FEM particle state are
        mirrored by the backend.
        """
        self._state_dirty = True

    def _notify_model_properties(self) -> None:
        """Handle :attr:`~newton.SolverNotifyFlags.MODEL_PROPERTIES`.

        Propagates ``model.gravity`` into the live UIPC ``scene.config()``.
        UIPC expects gravity as a ``3x1`` column-vector-of-lists, matching
        the format used during :meth:`initialize`. The update takes effect
        on the next ``world.advance()``.

        Other global model properties (e.g. time step, solver tolerances)
        are deliberately not forwarded here -- changing ``dt`` mid-run is
        unsafe for IPC line-search tuning, and UIPC tolerances are exposed
        through :meth:`configure_scene` at construction time.
        """
        model = self.model
        if model.gravity is None:
            return

        gravity_np = model.gravity.numpy().flatten()
        scene_cfg = self.scene.config()
        scene_cfg["gravity"] = [  # ty:ignore[invalid-assignment]
            [float(gravity_np[0])],
            [float(gravity_np[1])],
            [float(gravity_np[2])],
        ]

    def reset(
        self,
        state: State,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Re-push masked-world state into the live UIPC scene without rebuild.

        Overwrites only the bodies/particles belonging to the worlds selected
        by *world_mask* with the poses in *state*, leaving every other world at
        its current simulated configuration.  See
        :meth:`~newton.solvers.SolverBase.reset` for argument semantics.

        Note:
            After a body push UIPC's internal revolute/prismatic angle tracker
            lags by one step; read :attr:`Model.joint_q` rather than
            ``state.joint_q`` for articulated bodies until the next
            :meth:`step`.
        """
        if not self._initialized:
            return
        model = self.model
        # StateFlags is an IntEnum (not IntFlag), so combinations like
        # PARTICLE_Q | PARTICLE_QD (48) are not canonical members; keep flags as
        # a plain int and rely on bitwise tests below.
        flags = int(StateFlags.ALL) if flags is None else int(flags)

        # Host bool view of the world mask, computed once and reused.
        mask_host = None
        if world_mask is not None:
            mask_host = (world_mask.numpy() if isinstance(world_mask, wp.array) else np.asarray(world_mask)).astype(
                bool
            )

        def _rows_for_world(world_index_host: np.ndarray) -> np.ndarray | None:
            if mask_host is None:
                return None
            return np.nonzero(mask_host[world_index_host])[0].astype(np.int64)

        # --- rigid bodies ---
        mapping = self.mapping
        if mapping.num_mapped_bodies > 0 and mapping.body_geo_slots and model.body_world is not None:
            if self._mapped_body_world is None:
                self._mapped_body_world = model.body_world.numpy()[mapping.body_indices_wp.numpy()]
            rows = _rows_for_world(self._mapped_body_world)
            if rows is None or rows.size > 0:
                # Only run FK from joint coords when joints are the source (a
                # JOINT flag set and BODY_Q not) and both joint arrays exist;
                # otherwise eval_fk would clobber the caller's body_q (or crash
                # on a None joint_qd).
                do_fk = (
                    bool(flags & (StateFlags.JOINT_Q | StateFlags.JOINT_QD))
                    and not bool(flags & StateFlags.BODY_Q)
                    and state.joint_q is not None
                    and state.joint_qd is not None
                )
                if do_fk:
                    # Restrict FK to masked worlds via a per-articulation mask so
                    # non-masked worlds' state.body_q is not silently recomputed.
                    articulation_mask = None
                    if mask_host is not None and model.articulation_world is not None:
                        aw = model.articulation_world.numpy()
                        articulation_sel = np.where(aw >= 0, mask_host[np.clip(aw, 0, None)], False)
                        articulation_mask = wp.array(articulation_sel, dtype=wp.bool, device=model.device)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state, mask=articulation_mask)
                write_transform = bool(flags & StateFlags.BODY_Q) or do_fk
                write_velocity = bool(flags & StateFlags.BODY_QD) or do_fk
                if write_transform or write_velocity:
                    self._sync_body_state_to_uipc(
                        body_q=state.body_q,
                        body_qd=state.body_qd,
                        selected_rows=rows,
                        write_transform=write_transform,
                        write_velocity=write_velocity,
                        check_sanity=True,
                    )

        # --- FEM particles ---
        if (
            self._fem_accessor is not None
            and self._fem_mapped_vertex_count > 0
            and self._fem_particle_indices_wp is not None
            and model.particle_world is not None
            and bool(flags & (StateFlags.PARTICLE_Q | StateFlags.PARTICLE_QD))
        ):
            if self._mapped_particle_world is None:
                self._mapped_particle_world = model.particle_world.numpy()[self._fem_particle_indices_wp.numpy()]
            rows = _rows_for_world(self._mapped_particle_world)
            if rows is None or rows.size > 0:
                self._sync_particle_state_to_uipc(
                    particle_q=state.particle_q,
                    particle_qd=state.particle_qd,
                    selected_rows=rows,
                    write_position=bool(flags & StateFlags.PARTICLE_Q),
                    write_velocity=bool(flags & StateFlags.PARTICLE_QD),
                    check_sanity=True,
                )

    def _sync_state_to_uipc(self) -> None:
        """Push Newton-owned body and FEM particle state into UIPC."""
        self._sync_body_state_to_uipc(check_sanity=False)
        self._sync_particle_state_to_uipc(check_sanity=False)
        self.world.retrieve()
        self._raise_if_sanity_check_failed()

    def _sync_body_state_to_uipc(
        self,
        *,
        body_q: wp.array | None = None,
        body_qd: wp.array | None = None,
        selected_rows: np.ndarray | None = None,
        write_transform: bool = True,
        write_velocity: bool = True,
        check_sanity: bool = True,
    ) -> None:
        """Push body transforms / velocities into the UIPC backend.

        Args default to ``model.body_q`` / ``model.body_qd`` and all mapped
        rows, preserving the original notify behavior.  ``selected_rows`` is an
        array of row indices into the mapped-body arrays (``[0, n)``); only
        those rows are scattered, so non-selected bodies keep their current
        live UIPC pose (the ``copy_to`` seed).  ``write_transform`` /
        ``write_velocity`` gate which attribute is overwritten.
        """
        mapping = self.mapping
        if mapping.num_mapped_bodies == 0 or not mapping.body_geo_slots:
            return

        model = self.model
        src_q = model.body_q if body_q is None else body_q
        src_qd = model.body_qd if body_qd is None else body_qd
        if src_q is None:
            return
        assert mapping.body_indices_wp is not None
        assert mapping.backend_offsets_wp is not None
        assert self._abd_transform_buf is not None
        assert self._abd_velocity_buf is not None

        n = mapping.num_mapped_bodies
        device = model.device

        wp.launch(
            _transform_to_mat44_kernel,
            dim=n,
            inputs=[src_q, mapping.body_indices_wp, self._abd_transform_buf.warp()],
            device=device,
        )
        if write_velocity and src_qd is not None:
            wp.launch(
                _spatial_to_vel_mat44_kernel,
                dim=n,
                inputs=[src_qd, src_q, mapping.body_indices_wp, self._abd_velocity_buf.warp()],
                device=device,
            )
        else:
            self._abd_velocity_buf.warp().zero_()

        transforms_host = self._abd_transform_buf.warp().numpy()[:n]
        velocities_host = self._abd_velocity_buf.warp().numpy()[:n]

        # Lazily build the master state geometry. It spans every ABD body
        # in UIPC, so row ``q_idx`` of its ``transform`` / ``velocity``
        # instance attributes maps directly onto UIPC's flat q array slot
        # ``q_idx``. The geo topology is immutable after scene init, so
        # one allocation suffices for the solver lifetime.
        state_geo = getattr(self, "_master_state_geo", None)
        if state_geo is None:
            state_geo = self._abd_accessor.create_geometry()
            state_geo.instances().create("transform", np.eye(4, dtype=np.float64))
            state_geo.instances().create("velocity", np.zeros((4, 4), dtype=np.float64))
            self._master_state_geo = state_geo

        # Cache host-side backend offsets (= UIPC q indices per mapped
        # body). ``mapping.backend_offsets_wp`` is populated once by
        # ``populate_backend_offsets`` and immutable thereafter.
        if getattr(self, "_backend_offsets_host", None) is None:
            self._backend_offsets_host = mapping.backend_offsets_wp.numpy().astype(np.int64, copy=False)
        offsets_np = self._backend_offsets_host

        # Seed the master geo with the current UIPC state so unmapped
        # bodies round-trip unchanged, then overwrite the rows Newton owns.
        self._abd_accessor.copy_to(state_geo)

        transform_attr = state_geo.instances().find("transform")
        velocity_attr = state_geo.instances().find("velocity")
        assert transform_attr is not None
        transform_view = transform_attr.view()
        velocity_view = velocity_attr.view() if velocity_attr is not None else None

        rows = slice(None) if selected_rows is None else selected_rows
        if write_transform:
            transform_view[offsets_np[rows]] = transforms_host[rows]
        if velocity_view is not None and write_velocity:
            velocity_view[offsets_np[rows]] = velocities_host[rows]

        # Single push into UIPC — triggers one `update_dof_attributes`.
        self._abd_accessor.copy_from(state_geo)
        if check_sanity:
            self.world.retrieve()
            self._raise_if_sanity_check_failed()

    def _sync_particle_state_to_uipc(
        self,
        *,
        particle_q: wp.array | None = None,
        particle_qd: wp.array | None = None,
        selected_rows: np.ndarray | None = None,
        write_position: bool = True,
        write_velocity: bool = True,
        check_sanity: bool = True,
    ) -> None:
        """Push FEM particle state into the UIPC backend.

        Defaults to ``model.particle_q`` / ``model.particle_qd`` and all mapped
        vertices.  ``selected_rows`` selects columns into the mapped-vertex
        arrays; non-selected vertices keep their live UIPC position.
        """
        model = self.model
        src_q = model.particle_q if particle_q is None else particle_q
        src_qd = model.particle_qd if particle_qd is None else particle_qd
        if (
            src_q is None
            or self._fem_accessor is None
            or self._fem_mapped_vertex_count == 0
            or self._fem_backend_vertex_count == 0
            or self._fem_backend_offsets_wp is None
            or self._fem_particle_indices_wp is None
            or self._fem_position_buf is None
        ):
            return

        state_geo = getattr(self, "_master_fem_state_geo", None)
        if state_geo is None:
            state_geo = self._fem_accessor.create_geometry()
            state_geo.vertices().create("position", np.zeros((3, 1), dtype=np.float64))
            if self._fem_velocity_buf is not None:
                state_geo.vertices().create("velocity", np.zeros((3, 1), dtype=np.float64))
            self._master_fem_state_geo = state_geo

        # Seed the master geo from UIPC so unmapped FEM vertices, if any,
        # round-trip unchanged, then overwrite Newton-owned vertices.
        self._fem_accessor.copy_to(state_geo)

        position_attr = state_geo.vertices().find("position")
        assert position_attr is not None
        position_view = _view_attr(position_attr)

        # The FEM kernels scatter `dst[backend_idx] = src[particle_idx]`, so the
        # buffers and the view are backend-vertex ordered with the vertex axis
        # first: `position_view` is `[backend_vertex_count, 3, 1]` and
        # `positions_host` is `[backend_vertex_count, 3]`. `selected_rows` are
        # compact mapped-vertex indices; translate to backend slots via the
        # cached backend offsets (mirroring the body path's `offsets_np[rows]`).
        if getattr(self, "_fem_backend_offsets_host", None) is None:
            self._fem_backend_offsets_host = self._fem_backend_offsets_wp.numpy().astype(np.int64, copy=False)
        rows = slice(None) if selected_rows is None else self._fem_backend_offsets_host[selected_rows]

        if write_velocity and src_qd is not None and self._fem_velocity_buf is not None:
            velocity_attr = state_geo.vertices().find("velocity")
            assert velocity_attr is not None
            velocity_view = _view_attr(velocity_attr)
            wp.launch(
                _write_fem_particles_to_backend_kernel,
                dim=self._fem_mapped_vertex_count,
                inputs=[
                    self._fem_backend_offsets_wp,
                    self._fem_particle_indices_wp,
                    src_q,
                    src_qd,
                    self._fem_position_buf.warp(),
                    self._fem_velocity_buf.warp(),
                ],
                device=model.device,
            )
            positions_host = self._fem_position_buf.warp().numpy()
            velocities_host = self._fem_velocity_buf.warp().numpy()
            if write_position:
                position_view[rows, :, 0] = positions_host[rows]
            velocity_view[rows, :, 0] = velocities_host[rows]
        else:
            wp.launch(
                _write_fem_particle_positions_to_backend_kernel,
                dim=self._fem_mapped_vertex_count,
                inputs=[
                    self._fem_backend_offsets_wp,
                    self._fem_particle_indices_wp,
                    src_q,
                    self._fem_position_buf.warp(),
                ],
                device=model.device,
            )
            positions_host = self._fem_position_buf.warp().numpy()
            if write_position:
                position_view[rows, :, 0] = positions_host[rows]

        self._fem_accessor.copy_from(state_geo)
        if check_sanity:
            self.world.retrieve()
            self._raise_if_sanity_check_failed()

    def _raise_if_sanity_check_failed(self) -> None:
        """Raise when UIPC reports an invalid world after a state push."""
        checker = self.world.sanity_checker()
        result = checker.check()
        if result == SanityCheckResult.Success:
            return

        report = checker.report()
        raise RuntimeError(
            f"SolverUIPC: UIPC sanity check reported {result.name} after pushing state into UIPC: {report}"
        )

    def get_max_contact_count(self) -> int:
        """Return the :class:`~newton.Contacts` capacity for contact-sensor reporting.

        UIPC detects collisions internally, so there is no fixed contact-buffer
        size as there is for solvers driving Newton's collision pipeline. The
        returned value is the ``rigid_contact_max`` passed at construction, or
        :attr:`CONTACTS_PER_ENV` times ``Model.num_envs`` when that was ``None``.
        Contacts reported beyond this capacity by :meth:`update_contacts` are
        dropped.
        """
        if self._rigid_contact_max is not None:
            return self._rigid_contact_max
        num_envs = max(int(getattr(self.model, "num_envs", 0) or 0), 1)
        return self.CONTACTS_PER_ENV * num_envs

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Write UIPC contact forces into Newton state and Contacts.

        Populates ``state.body_f`` with per-rigid-body total contact wrench
        and ``state.particle_f`` with per-particle contact forces for
        cloth/deformable bodies. Per-body spatial forces are also written
        into ``contacts.force`` if allocated.
        """
        state_out: State | None = state if state is not None else getattr(self, "_current_state_out", None)

        if self._csf is None:
            return
        if self.mapping.vertex_to_body_wp is None:
            return

        gpu_data = prepare_contact_gpu_data(self._csf, dt=self._dt)
        if gpu_data is None:
            if hasattr(contacts, "rigid_contact_count"):
                contacts.rigid_contact_count.zero_()
            return

        n_verts = gpu_data.vertex_indices.shape[0]
        n_inst = gpu_data.vert_a.shape[0]
        if n_verts == 0 and n_inst == 0:
            if hasattr(contacts, "rigid_contact_count"):
                contacts.rigid_contact_count.zero_()
            return

        device = self.model.device
        mapping = self.mapping

        if n_verts > 0 and state_out is not None:
            vi_wp = wp.from_numpy(gpu_data.vertex_indices, dtype=wp.int32, device=device)
            f_wp = wp.from_numpy(gpu_data.forces, dtype=wp.vec3f, device=device)

            body_f = state_out.body_f
            particle_f = state_out.particle_f
            body_q = state_out.body_q if state_out.body_q is not None else self.model.body_q

            if body_f is not None and particle_f is not None and body_q is not None:
                wp.launch(
                    kernel=_scatter_contact_forces_kernel,
                    dim=n_verts,
                    inputs=[
                        vi_wp,
                        f_wp,
                        mapping.vertex_to_body_wp,
                        mapping.vertex_to_particle_wp,
                        body_q,
                        self.model.body_com,
                        self.model.body_count,
                        mapping.max_global_vertex,
                    ],
                    outputs=[body_f, particle_f],
                    device=device,
                )

        if n_inst > 0 and hasattr(contacts, "rigid_contact_shape0") and contacts.force is not None:
            fn_wp = wp.from_numpy(gpu_data.forces_n, dtype=wp.vec3f, device=device)
            ff_wp = wp.from_numpy(gpu_data.forces_f, dtype=wp.vec3f, device=device)
            va_wp = wp.from_numpy(gpu_data.vert_a, dtype=wp.int32, device=device)
            vb_wp = wp.from_numpy(gpu_data.vert_b, dtype=wp.int32, device=device)

            contacts.rigid_contact_count.zero_()
            ground_shape = self._ground_shape_index()

            wp.launch(
                kernel=_populate_contact_pairs_kernel,
                dim=n_inst,
                inputs=[
                    va_wp,
                    vb_wp,
                    fn_wp,
                    ff_wp,
                    mapping.vertex_to_body_wp,
                    mapping.body_to_first_shape_wp,
                    self.model.body_count,
                    mapping.max_global_vertex,
                    ground_shape,
                ],
                outputs=[
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_normal,
                    contacts.force,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_max,
                ],
                device=device,
            )
        elif hasattr(contacts, "rigid_contact_count"):
            contacts.rigid_contact_count.zero_()

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
            # Read the full backend range so that non-contiguous offsets
            # (common with multi-world replicate) are all covered.
            buf_count = self.mapping.max_backend_count
            self._abd_accessor.copy_transform_to(self._abd_transform_buf.buffer_view(), 0, buf_count)
            self._abd_accessor.copy_velocity_to(self._abd_velocity_buf.buffer_view(), 0, buf_count)

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

    def _sync_particle_state_from_uipc(self, state_out: State) -> None:
        """Read UIPC FEM vertex state back into Newton particles."""
        if (
            state_out.particle_q is None
            or self._fem_accessor is None
            or self._fem_mapped_vertex_count == 0
            or self._fem_backend_vertex_count == 0
            or self._fem_backend_offsets_wp is None
            or self._fem_particle_indices_wp is None
            or self._fem_position_buf is None
        ):
            return

        self._fem_accessor.copy_position_to(
            self._fem_position_buf.buffer_view(),
            0,
            self._fem_backend_vertex_count,
        )

        if state_out.particle_qd is not None and self._fem_velocity_buf is not None:
            self._fem_accessor.copy_velocity_to(
                self._fem_velocity_buf.buffer_view(),
                0,
                self._fem_backend_vertex_count,
            )
            wp.launch(
                _read_fem_particles_from_backend_kernel,
                dim=self._fem_mapped_vertex_count,
                inputs=[
                    self._fem_backend_offsets_wp,
                    self._fem_position_buf.warp(),
                    self._fem_velocity_buf.warp(),
                    self._fem_particle_indices_wp,
                    state_out.particle_q,
                    state_out.particle_qd,
                ],
                device=self.model.device,
            )
        else:
            wp.launch(
                _read_fem_particle_positions_from_backend_kernel,
                dim=self._fem_mapped_vertex_count,
                inputs=[
                    self._fem_backend_offsets_wp,
                    self._fem_position_buf.warp(),
                    self._fem_particle_indices_wp,
                    state_out.particle_q,
                ],
                device=self.model.device,
            )

    def set_cloth_soft_position_constraints(
        self,
        particle_indices: np.ndarray | list[int],
        aim_positions: np.ndarray | list[tuple[float, float, float]],
        strength_ratio: float | None = None,
        enabled: bool = True,
    ) -> None:
        """Enable UIPC soft-position control for selected cloth particles.

        The UIPC cloth builder adds a dormant
        ``SoftPositionConstraint`` to every cloth mesh.  This method marks
        selected vertices as constrained and writes their target
        ``aim_position`` values.  It is intended for kinematic cloth handles
        such as twisting edges or robot-gripper attachments.

        Args:
            particle_indices: Newton particle indices to constrain.
            aim_positions: Target world positions [m], shape ``(N, 3)``.
            strength_ratio: Optional per-call UIPC ``strength_ratio``.  If
                ``None``, keeps the value created by the builder.
            enabled: Whether the selected vertices are constrained.

        Raises:
            RuntimeError: If the solver has not been initialized or cloth
                soft-position attributes are unavailable.
            ValueError: If the index and target arrays have incompatible
                shapes.
        """
        self._set_soft_position_constraints(
            self.mapping.cloth_geo_slots,
            self.mapping.cloth_particle_indices,
            "cloth",
            particle_indices,
            aim_positions,
            strength_ratio,
            enabled,
        )

    def set_deformable_soft_position_constraints(
        self,
        particle_indices: np.ndarray | list[int],
        aim_positions: np.ndarray | list[tuple[float, float, float]],
        strength_ratio: float | None = None,
        enabled: bool = True,
    ) -> None:
        """Enable UIPC soft-position control for selected deformable particles.

        Args:
            particle_indices: Newton particle indices to constrain.
            aim_positions: Target world positions [m], shape ``(N, 3)``.
            strength_ratio: Optional per-call UIPC ``strength_ratio``.  If
                ``None``, keeps the value created by the builder.
            enabled: Whether the selected vertices are constrained.
        """
        self._set_soft_position_constraints(
            self.mapping.deformable_geo_slots,
            self.mapping.deformable_particle_indices,
            "deformable",
            particle_indices,
            aim_positions,
            strength_ratio,
            enabled,
        )

    def _set_soft_position_constraints(
        self,
        geo_slots: list[Any],
        particle_index_sets: list[Any],
        geometry_name: str,
        particle_indices: np.ndarray | list[int],
        aim_positions: np.ndarray | list[tuple[float, float, float]],
        strength_ratio: float | None,
        enabled: bool,
    ) -> None:
        """Enable UIPC soft-position control for selected FEM vertices."""
        if not self._initialized:
            raise RuntimeError(f"set_{geometry_name}_soft_position_constraints requires an initialized SolverUIPC.")

        indices = np.asarray(particle_indices, dtype=np.int32).reshape(-1)
        targets = np.asarray(aim_positions, dtype=np.float64)
        if targets.shape == (3,) and indices.size == 1:
            targets = targets.reshape(1, 3)
        if targets.shape != (indices.size, 3):
            raise ValueError(f"aim_positions must have shape ({indices.size}, 3), got {targets.shape}")

        found_any = False
        for geo_slot, mesh_particle_indices in zip(geo_slots, particle_index_sets, strict=False):
            local_by_global = {int(global_idx): local for local, global_idx in enumerate(mesh_particle_indices)}
            local_indices: list[int] = []
            local_targets: list[np.ndarray] = []
            for particle_index, target in zip(indices, targets, strict=True):
                local = local_by_global.get(int(particle_index))
                if local is None:
                    continue
                local_indices.append(local)
                local_targets.append(target)

            if not local_indices:
                continue

            geo = geo_slot.geometry()
            constrained_attr = geo.vertices().find("is_constrained")
            aim_attr = geo.vertices().find("aim_position")
            if constrained_attr is None or aim_attr is None:
                raise RuntimeError(
                    f"UIPC {geometry_name} soft-position attributes are missing. "
                    "Recreate SolverUIPC with soft-position constraints enabled."
                )

            local_indices_np = np.asarray(local_indices, dtype=np.int64)
            _view_attr(constrained_attr)[local_indices_np] = int(enabled)
            _view_attr(aim_attr)[local_indices_np] = np.asarray(local_targets, dtype=np.float64).reshape(-1, 3, 1)

            if strength_ratio is not None:
                strength_attr = geo.vertices().find("strength_ratio")
                if strength_attr is None:
                    raise RuntimeError(f"UIPC {geometry_name} soft-position strength_ratio attribute is missing.")
                _view_attr(strength_attr)[local_indices_np] = float(strength_ratio)

            found_any = True

        if not found_any:
            raise ValueError(f"None of the requested particle_indices belong to UIPC {geometry_name} geometry.")

    def clear_cloth_soft_position_constraints(
        self,
        particle_indices: np.ndarray | list[int] | None = None,
    ) -> None:
        """Disable UIPC soft-position constraints on cloth particles.

        Args:
            particle_indices: Optional Newton particle indices to disable.
                ``None`` disables all UIPC cloth soft-position constraints.
        """
        self._clear_soft_position_constraints(
            self.mapping.cloth_geo_slots,
            self.mapping.cloth_particle_indices,
            "cloth",
            particle_indices,
        )

    def clear_deformable_soft_position_constraints(
        self,
        particle_indices: np.ndarray | list[int] | None = None,
    ) -> None:
        """Disable UIPC soft-position constraints on deformable particles.

        Args:
            particle_indices: Optional Newton particle indices to disable.
                ``None`` disables all UIPC deformable soft-position constraints.
        """
        self._clear_soft_position_constraints(
            self.mapping.deformable_geo_slots,
            self.mapping.deformable_particle_indices,
            "deformable",
            particle_indices,
        )

    def _clear_soft_position_constraints(
        self,
        geo_slots: list[Any],
        particle_index_sets: list[Any],
        geometry_name: str,
        particle_indices: np.ndarray | list[int] | None,
    ) -> None:
        """Disable UIPC soft-position constraints on FEM vertices."""
        if not self._initialized:
            raise RuntimeError(f"clear_{geometry_name}_soft_position_constraints requires an initialized SolverUIPC.")

        requested = None if particle_indices is None else set(np.asarray(particle_indices, dtype=np.int32).reshape(-1))
        for geo_slot, mesh_particle_indices in zip(geo_slots, particle_index_sets, strict=False):
            constrained_attr = geo_slot.geometry().vertices().find("is_constrained")
            if constrained_attr is None:
                continue
            constrained = _view_attr(constrained_attr)
            if requested is None:
                constrained[:] = 0
                continue
            local_indices = [
                local for local, global_idx in enumerate(mesh_particle_indices) if int(global_idx) in requested
            ]
            if local_indices:
                constrained[np.asarray(local_indices, dtype=np.int64)] = 0
