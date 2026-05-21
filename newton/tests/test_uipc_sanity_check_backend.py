# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: compare SolverUIPC ``sanity_check.backend`` cuda vs cpu.

Builds the Allegro hand model from :mod:`newton.examples.uipc.robot.example_uipc_allegro_hand`
replicated to ``WORLD_COUNT`` worlds and times only ``SolverUIPC.initialize()``
with the ``sanity_check`` backend set to ``"cuda"`` and ``"cpu"`` respectively.
"""

import importlib.util
import time
import unittest

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode

_HAS_UIPC = importlib.util.find_spec("uipc") is not None

uipc = importlib.import_module("uipc") if _HAS_UIPC else None

WORLD_COUNT = 100


def _build_allegro_hand_model() -> newton.Model:
    """Replicate the model construction from ``example_uipc_allegro_hand``."""
    allegro_hand = newton.ModelBuilder(up_axis=newton.Axis.Z)
    allegro_hand.default_shape_cfg.ke = 1.0e3
    allegro_hand.default_shape_cfg.kd = 1.0e2
    allegro_hand.default_shape_cfg.margin = 0.005
    allegro_hand.default_shape_cfg.gap = 0.015

    asset_path = newton.utils.download_asset("wonik_allegro")
    asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
    allegro_hand.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0, 0, 0.5)),
        enable_self_collisions=False,
        ignore_paths=[".*Dummy", ".*CollisionPlane"],
        hide_collision_shapes=True,
    )

    finger_dof_count = allegro_hand.joint_dof_count - 6
    for i in range(finger_dof_count):
        allegro_hand.joint_target_ke[i] = 150
        allegro_hand.joint_target_kd[i] = 5
        allegro_hand.joint_q[i] = 0.3
        allegro_hand.joint_target_pos[i] = 0.3
        if allegro_hand.joint_label[i][-2:] == "_0":
            allegro_hand.joint_q[i] = 0.6
            allegro_hand.joint_target_pos[i] = 0.6
        allegro_hand.joint_target_mode[i] = int(JointTargetMode.POSITION)
        if allegro_hand.joint_type[i] == newton.JointType.REVOLUTE:
            allegro_hand.joint_armature[i] = 1e-2

    q = np.array(allegro_hand.joint_q)
    q[-7:-4] += np.array([0.0, 0.0, 0.05])
    q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
    allegro_hand.joint_q = q.tolist()

    if WORLD_COUNT > 1:
        builder = newton.ModelBuilder(newton.Axis.Z)
        builder.replicate(allegro_hand, WORLD_COUNT, spacing=(1.0, 2.0, 0.0))
    else:
        builder = allegro_hand

    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()

    return builder.finalize()


def _time_initialize(model: newton.Model, backend: str) -> float:
    """Construct a SolverUIPC and return the elapsed seconds for ``initialize()``."""
    assert uipc is not None  # guarded by @skipUnless on the test class
    GPa = 1e9
    cube_body_idx = model.body_count - 1

    def _contact_tabular_fn(contact_tabular, _world_index, ground_elem, env_elem, robo_elem, _actor_elem):
        cube_elem = contact_tabular.create("cube")
        contact_tabular.insert(cube_elem, robo_elem, 0.5, 1.0 * GPa, True)
        contact_tabular.insert(cube_elem, ground_elem, 0.5, 1.0 * GPa, True)
        contact_tabular.insert(cube_elem, env_elem, 0.5, 1.0 * GPa, True)
        contact_tabular.insert(cube_elem, cube_elem, 0.5, 1.0 * GPa, False)
        return {cube_body_idx: cube_elem}

    solver = newton.solvers.SolverUIPC(
        model,
        dt=1.0 / 60.0,
        logger_level=uipc.Logger.Warn,
        backend=backend,
    )
    solver.set_contact(True)
    solver.configure_contact_tabular(_contact_tabular_fn)

    start = time.perf_counter()
    solver.initialize()
    elapsed = time.perf_counter() - start
    return elapsed


@unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
class TestUIPCSanityCheckBackend(unittest.TestCase):
    """Benchmark SolverUIPC.initialize() across sanity_check backends."""

    @classmethod
    def setUpClass(cls):
        cls.model = _build_allegro_hand_model()

    def test_initialize_cuda_vs_none(self):
        cuda_time = _time_initialize(self.model, "cuda")
        none_time = _time_initialize(self.model, "none")

        print(
            f"\n[uipc_allegro_hand backend benchmark, world_count={WORLD_COUNT}]"
            f"\n  backend='cuda' initialize(): {cuda_time * 1000:.2f} ms"
            f"\n  backend='none' initialize(): {none_time * 1000:.2f} ms"
            f"\n  ratio (none / cuda): {none_time / cuda_time:.2f}x"
        )

        # Both runs must actually complete; guard against silent zero timings.
        self.assertGreater(cuda_time, 0.0)
        self.assertGreater(none_time, 0.0)


if __name__ == "__main__":
    unittest.main()
