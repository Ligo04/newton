# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverUIPC.reset() per-world masked state push."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import get_selected_cuda_test_devices

_HAS_UIPC = importlib.util.find_spec("uipc") is not None
_CUDA_TEST_DEVICES = get_selected_cuda_test_devices(mode="basic")


def _build_two_world_cubes(device):
    """One free cube per world, two worlds, over a ground plane."""
    world = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverUIPC.register_custom_attributes(world)
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0)
    body = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()), label="cube")
    world.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverUIPC.register_custom_attributes(builder)
    builder.replicate(world, world_count=2, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    return builder.finalize(device=device)


class TestUIPCReset(unittest.TestCase):
    @unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
    def test_masked_reset_updates_only_selected_world(self):
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                with wp.ScopedDevice(device):
                    model = _build_two_world_cubes(device)
                    state_0 = model.state()
                    state_1 = model.state()
                    control = model.control()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

                    solver = newton.solvers.SolverUIPC(model, dt=1.0 / 120.0)
                    solver.set_contact(True, 0.001)
                    solver.initialize(state_0)

                    for _ in range(10):
                        state_0.clear_forces()
                        solver.step(state_0, state_1, control, dt=1.0 / 120.0)
                        state_0, state_1 = state_1, state_0

                    body_world = model.body_world.numpy()
                    pre = state_0.body_q.numpy().copy()
                    world1_body = int(np.where(body_world == 1)[0][0])
                    world0_body = int(np.where(body_world == 0)[0][0])

                    # New pose only for world 0.
                    new_q = state_0.body_q.numpy().copy()
                    new_q[world0_body, :3] = [0.3, 0.0, 0.5]
                    state_0.body_q = wp.array(new_q, dtype=wp.transform, device=device)
                    state_0.body_qd.zero_()

                    mask = wp.array([True, False], dtype=wp.bool, device=device)
                    solver.reset(state_0, world_mask=mask)

                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, dt=1.0 / 120.0)
                    state_0, state_1 = state_1, state_0
                    post = state_0.body_q.numpy()

                    # World 0 jumped to the new high pose; world 1 untouched.
                    self.assertGreater(post[world0_body, 2], 0.4)
                    self.assertAlmostEqual(post[world1_body, 0], pre[world1_body, 0], places=3)
                    self.assertAlmostEqual(post[world1_body, 1], pre[world1_body, 1], places=3)
                    self.assertTrue(np.all(np.isfinite(post)))

    @unittest.skipUnless(_HAS_UIPC, "uipc is not installed")
    def test_masked_reset_updates_only_selected_world_particles(self):
        for device in _CUDA_TEST_DEVICES:
            with self.subTest(device=str(device)):
                with wp.ScopedDevice(device):
                    world = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.8)
                    newton.solvers.SolverUIPC.register_custom_attributes(world)
                    world.add_cloth_grid(
                        pos=wp.vec3(0.0, 0.0, 0.5),
                        rot=wp.quat_identity(),
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        dim_x=2,
                        dim_y=2,
                        cell_x=0.1,
                        cell_y=0.1,
                        mass=0.01,
                        tri_ke=1.0e3,
                        tri_ka=1.0e3,
                        tri_kd=0.0,
                    )
                    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.8)
                    newton.solvers.SolverUIPC.register_custom_attributes(builder)
                    builder.replicate(world, world_count=2, spacing=(1.0, 0.0, 0.0))
                    model = builder.finalize(device=device)

                    state_0 = model.state()
                    state_1 = model.state()
                    control = model.control()

                    solver = newton.solvers.SolverUIPC(model, dt=1.0 / 120.0)
                    solver.initialize(state_0)

                    for _ in range(10):
                        state_0.clear_forces()
                        solver.step(state_0, state_1, control, dt=1.0 / 120.0)
                        state_0, state_1 = state_1, state_0

                    particle_world = model.particle_world.numpy()
                    w0 = np.where(particle_world == 0)[0]
                    w1 = np.where(particle_world == 1)[0]
                    pre = state_0.particle_q.numpy().copy()

                    # Lift world 0's particles by 1 m; leave world 1 alone.
                    new_q = state_0.particle_q.numpy().copy()
                    new_q[w0, 2] += 1.0
                    state_0.particle_q = wp.array(new_q, dtype=wp.vec3, device=device)
                    state_0.particle_qd.zero_()

                    mask = wp.array([True, False], dtype=wp.bool, device=device)
                    solver.reset(
                        state_0,
                        world_mask=mask,
                        flags=newton.StateFlags.PARTICLE_Q | newton.StateFlags.PARTICLE_QD,
                    )

                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, dt=1.0 / 120.0)
                    state_0, state_1 = state_1, state_0
                    post = state_0.particle_q.numpy()

                    # World 0 jumped ~1 m higher than before; world 1 was not reset
                    # and only advanced by its own (free-falling) one-step motion.
                    self.assertGreater(post[w0, 2].mean(), pre[w0, 2].mean() + 0.5)
                    self.assertLess(abs(post[w1, 2].mean() - pre[w1, 2].mean()), 0.05)
                    self.assertTrue(np.all(np.isfinite(post)))


if __name__ == "__main__":
    unittest.main()
