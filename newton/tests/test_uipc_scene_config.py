# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for UIPC scene configuration defaults."""

import importlib.util
import unittest

import newton

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
if _HAS_UIPC:
    import uipc


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCSceneConfig(unittest.TestCase):
    def test_default_scene_config_uses_full_gpu_fused_pcg(self):
        """Use full-GPU FusedPCG CUDA Graph defaults."""
        model = newton.ModelBuilder().finalize()

        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)

        self.assertEqual(solver._scene_config["linear_system"]["solver"], "fused_pcg")
        self.assertEqual(solver._scene_config["linear_system"]["use_cuda_graph"], 2)
        self.assertEqual(solver._scene_config["linear_system"]["fem_preconditioner"], "mas")
        self.assertEqual(solver._scene_config["contact"]["constitution"], "ipc")

    def test_explicit_scene_config_preserves_linear_solver_options(self):
        """Preserve explicit scene configuration overrides."""
        model = newton.ModelBuilder().finalize()
        scene_config = uipc.Scene.default_config()
        scene_config["linear_system"]["solver"] = "linear_pcg"
        scene_config["linear_system"]["use_cuda_graph"] = 0
        scene_config["linear_system"]["fem_preconditioner"] = "diag"
        scene_config["contact"]["constitution"] = "al-ipc"

        solver = newton.solvers.SolverUIPC(
            model,
            backend="none",
            scene_config=scene_config,
            logger_level=uipc.Logger.Error,
        )

        self.assertEqual(solver._scene_config["linear_system"]["solver"], "linear_pcg")
        self.assertEqual(solver._scene_config["linear_system"]["use_cuda_graph"], 0)
        self.assertEqual(solver._scene_config["linear_system"]["fem_preconditioner"], "diag")
        self.assertEqual(solver._scene_config["contact"]["constitution"], "al-ipc")

    def test_default_contact_pairs_use_adaptive_kappa(self):
        """Use UIPC's adaptive-kappa sentinel for built-in contact pairs."""
        model = newton.ModelBuilder().finalize()
        observed_kappas = []

        def inspect_defaults(tabular, _world_index, ground_elem, env_elem, robo_elem, actor_elem):
            pairs = (
                (env_elem, env_elem),
                (env_elem, robo_elem),
                (env_elem, actor_elem),
                (ground_elem, env_elem),
                (ground_elem, robo_elem),
                (ground_elem, actor_elem),
                (robo_elem, robo_elem),
                (robo_elem, actor_elem),
                (actor_elem, actor_elem),
            )
            observed_kappas.extend(tabular.at(lhs.id(), rhs.id()).resistance() for lhs, rhs in pairs)

        solver = newton.solvers.SolverUIPC(model, backend="none", logger_level=uipc.Logger.Error)
        solver.configure_contact_tabular(inspect_defaults)
        solver.initialize()

        self.assertEqual(observed_kappas, [-1.0] * 9)
