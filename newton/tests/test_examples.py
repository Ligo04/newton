# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test examples in the newton.examples package.

Currently, this script mainly checks that the examples can run. It also treats
deprecation warnings as failures by default so examples do not regress onto
deprecated APIs.

The test parameters are typically tuned so that each test can run in 10 seconds
or less, ignoring module compilation time. A notable exception is the robot
manipulating cloth example, which takes approximately 35 seconds to run on a
CUDA device.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from typing import Any

import numpy as np
import warp as wp

import newton.tests.unittest_utils
from newton.tests.unittest_utils import (
    USD_AVAILABLE,
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
    sanitize_identifier,
)

_HAS_UIPC = importlib.util.find_spec("uipc") is not None

if _HAS_UIPC:
    from newton.examples.uipc.cloth.example_uipc_cloth_franka import Example as UIPCClothFrankaExample
    from newton.examples.uipc.cloth.example_uipc_cloth_franka_stable_pd_force import (
        Example as StablePDForceExample,
    )
    from newton.examples.uipc.contacts.example_uipc_brick_stacking import (
        BRICK_ABD_KAPPA as UIPC_BRICK_ABD_KAPPA,
    )
    from newton.examples.uipc.contacts.example_uipc_brick_stacking import (
        STUD_HEIGHT as UIPC_BRICK_STUD_HEIGHT,
    )
    from newton.examples.uipc.contacts.example_uipc_brick_stacking import (
        Example as UIPCBrickStackingExample,
    )
    from newton.examples.uipc.multiphysics.example_uipc_softbody_dropping_to_cloth import (
        Example as UIPCSoftbodyDroppingToClothExample,
    )

_HAS_ONNX = importlib.util.find_spec("onnx") is not None


def _build_command_line_options(test_options: dict[str, Any]) -> list:
    """Helper function to build command-line options from the test options dictionary."""
    additional_options = []

    for key, value in test_options.items():
        if isinstance(value, bool):
            # Default behavior expecting argparse.BooleanOptionalAction support
            additional_options.append(f"--{'no-' if not value else ''}{key.replace('_', '-')}")
        elif isinstance(value, list):
            additional_options.extend([f"--{key.replace('_', '-')}"] + [str(v) for v in value])
        else:
            # Just add --key value
            additional_options.extend(["--" + key.replace("_", "-"), str(value)])

    return additional_options


def _merge_options(base_options: dict[str, Any], device_options: dict[str, Any]) -> dict[str, Any]:
    """Helper function to merge base test options with device-specific test options."""
    merged_options = base_options.copy()

    #  Update options with device-specific dictionary, overwriting existing keys with the more-specific values
    merged_options.update(device_options)
    return merged_options


def add_example_test(
    cls: type,
    name: str,
    devices: list | None = None,
    test_options: dict[str, Any] | None = None,
    test_options_cpu: dict[str, Any] | None = None,
    test_options_cuda: dict[str, Any] | None = None,
    use_viewer: bool = False,
    test_suffix: str | None = None,
):
    """Registers a Newton example to run on ``devices`` as a TestCase."""

    # verify the module exists (use package-relative path so this works from any CWD)
    _examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if not os.path.exists(os.path.join(_examples_dir, f"{name.replace('.', '/')}.py")):
        raise ValueError(f"Example {name} does not exist")

    if test_options is None:
        test_options = {}
    if test_options_cpu is None:
        test_options_cpu = {}
    if test_options_cuda is None:
        test_options_cuda = {}

    def run(test, device):
        if wp.get_device(device).is_cuda:
            options = _merge_options(test_options, test_options_cuda)
        else:
            options = _merge_options(test_options, test_options_cpu)

        # Mark the test as skipped if onnx is not installed but required
        onnx_required = options.pop("onnx_required", False)
        # Legacy alias: torch_required examples have been migrated to ONNX.
        onnx_required = onnx_required or options.pop("torch_required", False)
        if onnx_required and not _HAS_ONNX:
            test.skipTest("onnx not installed")

        # Mark the test as skipped if USD is not installed but required
        usd_required = options.pop("usd_required", False)
        if usd_required and not USD_AVAILABLE:
            test.skipTest("Requires usd-core")

        # Deprecations should fail example tests by default. Opt out only for
        # a known third-party or asset issue that still needs follow-up.
        allow_deprecation_warnings = options.pop("allow_deprecation_warnings", False)

        # Pass the parent dir; the subprocess's init_kernel_cache appends the version.
        warp_cache_path = wp.config.kernel_cache_dir

        env_vars = os.environ.copy()
        if warp_cache_path is not None:
            env_vars["WARP_CACHE_PATH"] = os.path.dirname(warp_cache_path)
        if not allow_deprecation_warnings:
            env_vars["PYTHONWARNINGS"] = "error::DeprecationWarning"
        else:
            env_vars.pop("PYTHONWARNINGS", None)

        if newton.tests.unittest_utils.coverage_enabled:
            # Generate a random coverage data file name - file is deleted along with containing directory
            with tempfile.NamedTemporaryFile(
                dir=newton.tests.unittest_utils.coverage_temp_dir, delete=False
            ) as coverage_file:
                pass

            command = ["coverage", "run", f"--data-file={coverage_file.name}"]

            if newton.tests.unittest_utils.coverage_branch:
                command.append("--branch")

        else:
            command = [sys.executable]

        # Append Warp commands
        command.extend(["-m", f"newton.examples.{name}", "--device", str(device), "--test", "--quiet"])

        if not use_viewer:
            stage_path = (
                options.pop(
                    "stage_path",
                    os.path.join(os.path.dirname(__file__), f"outputs/{name}_{sanitize_identifier(device)}.usd"),
                )
                if USD_AVAILABLE
                else "None"
            )

            if stage_path:
                command.extend(["--stage-path", stage_path])
                try:
                    os.remove(stage_path)
                except OSError:
                    pass
        else:
            # new-style example, use null viewer for tests (no disk I/O needed)
            stage_path = "None"
            command.extend(["--viewer", "null"])
            # Remove viewer/stage_path from options so they can't override the null viewer
            options.pop("viewer", None)
            options.pop("stage_path", None)

        command.extend(_build_command_line_options(options))

        # Set the test timeout in seconds
        test_timeout = options.pop("test_timeout", 600)

        # Can set active=True when tuning the test parameters
        with wp.ScopedTimer(f"{name}_{sanitize_identifier(device)}", active=False):
            # Run the script as a subprocess
            result = subprocess.run(
                command, capture_output=True, text=True, env=env_vars, timeout=test_timeout, check=False
            )

        # print any error messages (e.g.: module not found)
        if result.stderr != "":
            print(result.stderr)

        # Check the return code (0 is standard for success)
        test.assertEqual(
            result.returncode,
            0,
            msg=f"Failed with return code {result.returncode}, command: {' '.join(command)}\n\nOutput:\n{result.stdout}\n{result.stderr}",
        )

        # Clean up output file for old-style examples that may have created one
        if stage_path and stage_path != "None" and result.returncode == 0:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    test_name = f"test_{name}_{test_suffix}" if test_suffix else f"test_{name}"
    add_function_test(cls, test_name, run, devices=devices, check_output=False)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestBasicExamples(unittest.TestCase):
    pass


add_example_test(TestBasicExamples, name="basic.example_basic_pendulum", devices=test_devices, use_viewer=True)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    use_viewer=True,
    test_suffix="xpbd",
)
add_example_test(
    TestBasicExamples,
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200, "solver": "vbd"},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    use_viewer=True,
    test_suffix="vbd",
)

add_example_test(TestBasicExamples, name="basic.example_basic_viewer", devices=test_devices, use_viewer=True)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_joints",
    devices=test_devices,
    use_viewer=True,
    test_suffix="xpbd",
)
add_example_test(
    TestBasicExamples,
    name="basic.example_basic_joints",
    devices=test_devices,
    use_viewer=True,
    test_options={"solver": "vbd"},
    test_suffix="vbd",
)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_shapes",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 150},
)

add_example_test(
    TestBasicExamples,
    name="basic.example_basic_conveyor",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 100},
)


class TestCableExamples(unittest.TestCase):
    pass


add_example_test(
    TestCableExamples,
    name="cable.example_cable_twist",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_y_junction",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_bundle_hysteresis",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_cross_slide_table",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 540},
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_pile",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 20},
)


class TestClothExamples(unittest.TestCase):
    pass


add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_bending",
    devices=test_devices,
    test_options={"num-frames": 400},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    devices=test_devices,
    test_options={},
    test_options_cpu={"width": 32, "height": 16, "num-frames": 10},
    use_viewer=True,
    test_suffix="vbd",
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_hanging",
    devices=test_devices,
    test_options={"solver": "style3d"},
    test_options_cpu={"width": 32, "height": 16, "num-frames": 10},
    use_viewer=True,
    test_suffix="style3d",
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_style3d",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_h1",
    devices=cuda_test_devices,
    test_options={},
    test_options_cuda={"num-frames": 32},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_franka",
    devices=cuda_test_devices,
    test_options={"num-frames": 50},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_twist",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_rollers",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
)


class TestRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotExamples,
    name="robot.example_robot_cartpole",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_c_walk",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500, "onnx_required": True},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_anymal_d",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_g1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_h1",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_ur10",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_allegro_hand",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 500},
    use_viewer=True,
)
add_example_test(
    TestRobotExamples,
    name="robot.example_robot_panda_hydro",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 720},
    use_viewer=True,
)


class TestRobotPolicyExamples(unittest.TestCase):
    pass


add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "onnx_required": True, "robot": "g1_29dof"},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    test_suffix="G1_29dof",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "onnx_required": True, "robot": "g1_23dof"},
    use_viewer=True,
    test_suffix="G1_23dof",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "onnx_required": True, "robot": "g1_23dof", "physx": True},
    use_viewer=True,
    test_suffix="G1_23dof_Physx",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "onnx_required": True, "robot": "anymal"},
    use_viewer=True,
    test_suffix="Anymal",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"num-frames": 500, "onnx_required": True, "robot": "anymal", "physx": True},
    use_viewer=True,
    test_suffix="Anymal_Physx",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"onnx_required": True},
    test_options_cuda={"num-frames": 500, "robot": "go2"},
    use_viewer=True,
    test_suffix="Go2",
)
add_example_test(
    TestRobotPolicyExamples,
    name="robot.example_robot_policy",
    devices=cuda_test_devices,
    test_options={"onnx_required": True},
    test_options_cuda={"num-frames": 500, "robot": "go2", "physx": True},
    use_viewer=True,
    test_suffix="Go2_Physx",
)


class TestAdvancedRobotExamples(unittest.TestCase):
    pass


add_example_test(
    TestAdvancedRobotExamples,
    name="mpm.example_mpm_anymal",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "onnx_required": True},
    use_viewer=True,
)


class TestIKExamples(unittest.TestCase):
    pass


add_example_test(TestIKExamples, name="ik.example_ik_franka", devices=test_devices, use_viewer=True)

add_example_test(TestIKExamples, name="ik.example_ik_h1", devices=test_devices, use_viewer=True)

add_example_test(TestIKExamples, name="ik.example_ik_custom", devices=cuda_test_devices, use_viewer=True)

add_example_test(
    TestIKExamples,
    name="ik.example_ik_cube_stacking",
    test_options_cuda={"world-count": 16, "num-frames": 2000},
    devices=cuda_test_devices,
    use_viewer=True,
)


class TestSelectionAPIExamples(unittest.TestCase):
    pass


add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_articulations",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_cartpole",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_materials",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)
add_example_test(
    TestSelectionAPIExamples,
    name="selection.example_selection_multiple",
    devices=test_devices,
    test_options={"num-frames": 100},
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)


class TestDiffSimExamples(unittest.TestCase):
    pass


add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_ball",
    devices=test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 36},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_cloth",
    devices=test_devices,
    test_options={"num-frames": 4 * 120},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 120},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_drone",
    devices=test_devices,
    test_options={"num-frames": 180},  # sim_steps
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_spring_cage",
    devices=test_devices,
    test_options={"num-frames": 4 * 30},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 30},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_soft_body",
    devices=test_devices,
    test_options={"num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 60},
    use_viewer=True,
)

add_example_test(
    TestDiffSimExamples,
    name="diffsim.example_diffsim_bear",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2, "sim-steps": 10},
    use_viewer=True,
)


class TestSensorExamples(unittest.TestCase):
    pass


add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_contact",
    devices=test_devices,
    test_options={"num-frames": 160},  # required for ball to reach plate
    use_viewer=True,
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_tiled_camera",
    devices=cuda_test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    use_viewer=True,
)

add_example_test(
    TestSensorExamples,
    name="sensors.example_sensor_imu",
    devices=test_devices,
    test_options={"num-frames": 200},  # allow cubes to settle
    use_viewer=True,
)


class TestMPMExamples(unittest.TestCase):
    pass


add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_granular",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_multi_material",
    devices=cuda_test_devices,
    test_options={"num-frames": 10},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_grain_rendering",
    devices=cuda_test_devices,
    test_options={"num-frames": 10},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_twoway_coupling",
    devices=cuda_test_devices,
    test_options={"num-frames": 80},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_beam_twist",
    devices=cuda_test_devices,
    test_options={"num-frames": 100},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_snow_ball",
    devices=cuda_test_devices,
    test_options={"num-frames": 30, "voxel-size": 0.2},
    use_viewer=True,
)

add_example_test(
    TestMPMExamples,
    name="mpm.example_mpm_viscous",
    devices=cuda_test_devices,
    test_options={"num-frames": 30, "voxel-size": 0.01},
    use_viewer=True,
)


add_example_test(
    TestBasicExamples,
    name="basic.example_basic_plotting",
    devices=test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
)


class TestContactsExamples(unittest.TestCase):
    pass


add_example_test(
    TestContactsExamples,
    name="contacts.example_nut_bolt_sdf",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    use_viewer=True,
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_nut_bolt_hydro",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "world-count": 1},
    use_viewer=True,
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_brick_stacking",
    devices=cuda_test_devices,
    test_options={"num-frames": 1200},
    use_viewer=True,
)
add_example_test(
    TestContactsExamples,
    name="contacts.example_pyramid",
    devices=cuda_test_devices,
    test_options={"num-frames": 120, "num-pyramids": 3, "pyramid-size": 5},
    use_viewer=True,
)


class TestMultiphysicsExamples(unittest.TestCase):
    pass


add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_gift",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="cloth.example_cloth_poker_cards",
    devices=cuda_test_devices,
    test_options={"num-frames": 30},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_dropping_to_cloth",
    devices=cuda_test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
)


class TestSoftbodyExamples(unittest.TestCase):
    pass


add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_hanging",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)


class TestUIPCSoftbodyExamples(unittest.TestCase):
    def test_uipc_brick_stacking_matches_core_original_behaviors(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "uipc",
            "contacts",
            "example_uipc_brick_stacking.py",
        )
        with open(example_path) as f:
            source = f.read()

        self.assertIn("class TaskType(enum.IntEnum)", source)
        self.assertIn("set_target_pose_kernel", source)
        self.assertIn("advance_task_kernel", source)
        self.assertIn("Smoothstep easing", source)
        self.assertIn("XY alignment correction", source)
        self.assertIn("(round_1, round_1_times, red, green, 1)", source)
        self.assertIn("(round_2, round_2_times, red, blue, 2)", source)
        self.assertIn("_add_board_floor", source)
        self.assertIn("_solve_approach_ik", source)
        self.assertIn("Task sequence incomplete", source)
        self.assertIn("Green-Blue height gap", source)
        self.assertIn("Red-Blue height gap", source)
        self.assertIn('"--enable-contact"', source)
        self.assertIn("default=True", source)
        self.assertIn('self.enable_contact = getattr(args, "enable_contact", True)', source)
        self.assertIn("self.solver.set_contact(enable=self.enable_contact, d_hat=self.uipc_gap)", source)
        self.assertIn("BRICK_ABD_KAPPA = 10.0 * uipc.unit.GPa", source)
        self.assertIn("newton.solvers.SolverUIPC.register_custom_attributes(builder)", source)
        self.assertIn('custom_attributes={"uipc:abd_kappa": BRICK_ABD_KAPPA}', source)
        self.assertIn("self.brick_stack_height = self.brick_height_scaled + self.uipc_gap", source)
        self.assertIn("self.brick_stack_height,", source)
        self.assertIn("self.drop_z_offset = wp.vec3(0.0, 0.0, 0.0)", source)
        self.assertIn("blue_x = self.table_top_center[0] - 0.05", source)
        self.assertIn("blue_y = self.table_top_center[1] - 0.04", source)
        self.assertIn("self.table_top_center + wp.vec3(0.0, 0.06, bh + self.uipc_gap)", source)
        self.assertIn("self.table_top_center + wp.vec3(0.05, -0.04, bh + self.uipc_gap)", source)
        self.assertIn("self.table_top_center[2] + 0.2 * self.brick_height_scaled + bh + self.uipc_gap", source)
        self.assertIn("floor_center_z = floor_z + BRICK_HZ", source)
        self.assertIn("for ix, dx in enumerate((-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw))", source)
        self.assertIn("for iy, dy in enumerate((-0.5 * bl, 0.5 * bl))", source)
        self.assertIn("parser.set_defaults(num_frames=1800)", source)

    def test_uipc_brick_stacking_contact_flag_defaults_on_and_can_disable(self):
        if not _HAS_UIPC:
            self.skipTest("Requires uipc")

        parser = UIPCBrickStackingExample.create_parser()
        default_args = parser.parse_args([])
        disabled_args = parser.parse_args(["--no-enable-contact"])

        self.assertTrue(default_args.enable_contact)
        self.assertFalse(disabled_args.enable_contact)

    def test_uipc_brick_stacking_initial_pose_is_red_approach(self):
        if not _HAS_UIPC:
            self.skipTest("Requires uipc")
        if not cuda_test_devices:
            self.skipTest("Requires a CUDA test device")

        class DummyViewer:
            def __init__(self):
                self._paused = False

            def set_model(self, model):
                pass

            def set_camera(self, **kwargs):
                pass

            def begin_frame(self, *args, **kwargs):
                pass

            def log_state(self, *args, **kwargs):
                pass

            def log_contacts(self, *args, **kwargs):
                pass

            def end_frame(self):
                pass

        with wp.ScopedDevice(cuda_test_devices[0]):
            example = UIPCBrickStackingExample(DummyViewer(), types.SimpleNamespace(test=True))
            body_q = example.state_0.body_q.numpy()
            red_pos = body_q[example.brick_bodies[0]][:3]
            ee_pos = body_q[example.ee_index][:3]
            abd_kappa = example.model.uipc.abd_kappa.numpy()

        target_pos = red_pos + np.array([0.0, 0.0, float(example.offset_approach[2])], dtype=np.float32)
        self.assertLess(float(np.linalg.norm(ee_pos - target_pos)), 0.025)
        np.testing.assert_allclose(abd_kappa[example.brick_bodies], UIPC_BRICK_ABD_KAPPA)
        np.testing.assert_allclose(abd_kappa[example.board_floor_bodies], -1.0)
        board_floor_labels = [label for label in example.model.body_label if label.startswith("board_floor_")]
        self.assertEqual(len(board_floor_labels), 8)

        board_floor_z = body_q[example.board_floor_bodies[0]][2]
        board_top = board_floor_z + 0.5 * example.brick_height_scaled
        board_stud_top = board_top + UIPC_BRICK_STUD_HEIGHT
        self.assertGreater(board_top, example.table_top_z)
        self.assertGreater(board_stud_top, example.table_top_z)

    def test_uipc_softbody_dropping_to_cloth_uses_uipc_solver(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "uipc",
            "multiphysics",
            "example_uipc_softbody_dropping_to_cloth.py",
        )
        with open(example_path) as f:
            source = f.read()

        self.assertIn("SolverUIPC", source)
        self.assertIn("fix_left=True", source)
        self.assertIn("fix_right=True", source)
        self.assertIn("enable_soft_position_constraint=False", source)
        self.assertIn("self.graph = None", source)
        self.assertIn("parser.set_defaults(num_frames=120)", source)

    def test_uipc_softbody_dropping_to_cloth_smoke(self):
        if not _HAS_UIPC:
            self.skipTest("Requires uipc")
        if not cuda_test_devices:
            self.skipTest("Requires a CUDA test device")

        class DummyViewer:
            def __init__(self):
                self._paused = False

            def set_model(self, model):
                pass

            def set_camera(self, **kwargs):
                pass

            def begin_frame(self, *args, **kwargs):
                pass

            def log_state(self, *args, **kwargs):
                pass

            def log_contacts(self, *args, **kwargs):
                pass

            def end_frame(self):
                pass

            def apply_forces(self, state):
                pass

        with wp.ScopedDevice(cuda_test_devices[0]):
            example = UIPCSoftbodyDroppingToClothExample(DummyViewer(), types.SimpleNamespace())
            example.step()
            particle_q = example.state_0.particle_q.numpy()

        self.assertTrue(np.all(np.isfinite(particle_q)))
        self.assertLess(float(np.linalg.norm(particle_q.max(axis=0) - particle_q.min(axis=0))), 20.0)

    def test_uipc_cloth_franka_uses_robot_contact_and_activation_values(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "uipc",
            "cloth",
            "example_uipc_cloth_franka.py",
        )
        with open(example_path) as f:
            source = f.read()

        self.assertNotIn("_push_cloth_handle", source)
        self.assertNotIn("set_cloth_soft_position_constraints", source)
        self.assertNotIn("--soft-strength", source)
        self.assertIn("enable_soft_position_constraint=False", source)
        self.assertIn("configure_contact_tabular", source)
        self.assertIn("_solve_ik_and_push_control", source)
        self.assertIn("clamp_close_activation_val = 0.1", source)
        self.assertIn("clamp_open_activation_val = 0.8", source)
        self.assertIn("gripper_activation_scale = 0.04", source)
        self.assertIn("set_gripper_q", source)
        self.assertIn("finger_pos_buf", source)
        self.assertIn("joint_label", source)
        self.assertIn("JointTargetMode.POSITION", source)
        self.assertIn("parser.set_defaults(num_frames=3850)", source)

    def test_uipc_cloth_franka_delays_actions_by_25_frames(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "uipc",
            "cloth",
            "example_uipc_cloth_franka.py",
        )
        with open(example_path) as f:
            source = f.read()

        self.assertIn("self.action_delay_frames = 25", source)
        self.assertIn("self.action_delay = self.action_delay_frames * self.frame_dt", source)
        self.assertIn("self.frame < self.action_delay_frames", source)
        self.assertIn("wp.copy(self.control.joint_target_pos, self.model.joint_q)", source)
        self.assertIn("self.state_0.body_q.numpy()[self.ee_index]", source)
        self.assertIn("def _target_at_time(self, time: float, is_delayed: bool)", source)
        self.assertIn("parser.set_defaults(num_frames=3850)", source)

    def test_uipc_cloth_franka_closes_fingers_in_close_segment(self):
        if not _HAS_UIPC:
            self.skipTest("Requires uipc")
        if not cuda_test_devices:
            self.skipTest("Requires a CUDA test device")

        class DummyViewer:
            def __init__(self):
                self._paused = False

            def set_model(self, model):
                pass

            def set_camera(self, **kwargs):
                pass

            def begin_frame(self, *args, **kwargs):
                pass

            def log_state(self, *args, **kwargs):
                pass

            def end_frame(self):
                pass

        with wp.ScopedDevice(cuda_test_devices[0]):
            example = UIPCClothFrankaExample(
                DummyViewer(),
                types.SimpleNamespace(cloth_model="strain_limiting_baraff_witkin"),
            )
            finger_open = example.state_0.joint_q.numpy()[7:9].copy()
            example.sim_time = 7.9
            example.frame = example.action_delay_frames
            example.step()
            finger_closed = example.state_0.joint_q.numpy()[7:9]

        self.assertLess(float(finger_closed.max()), float(finger_open.max()))
        self.assertLess(float(finger_closed.max()), 0.04)

    def test_uipc_cloth_franka_stable_pd_force_holds_initial_pose_with_feedforward(self):
        if not _HAS_UIPC:
            self.skipTest("Requires uipc")
        if not cuda_test_devices:
            self.skipTest("Requires a CUDA test device")

        class DummyViewer:
            def __init__(self):
                self._paused = False

            def set_model(self, model):
                pass

            def set_camera(self, **kwargs):
                pass

            def begin_frame(self, *args, **kwargs):
                pass

            def log_state(self, *args, **kwargs):
                pass

            def end_frame(self):
                pass

        with wp.ScopedDevice(cuda_test_devices[0]):
            example = StablePDForceExample(
                DummyViewer(),
                types.SimpleNamespace(cloth_model="strain_limiting_baraff_witkin"),
            )
            example._solve_ik_and_push_control(example.sim_time, is_delayed=True)
            example._apply_stable_pd_force_control()
            joint_f = example.control.joint_f.numpy()

        self.assertGreater(float(np.max(np.abs(joint_f[:7]))), 1.0e-3)

    def test_uipc_cloth_franka_stable_pd_force_generates_nonzero_joint_force(self):
        if not _HAS_UIPC:
            self.skipTest("Requires uipc")
        if not cuda_test_devices:
            self.skipTest("Requires a CUDA test device")

        class DummyViewer:
            def __init__(self):
                self._paused = False

            def set_model(self, model):
                pass

            def set_camera(self, **kwargs):
                pass

            def begin_frame(self, *args, **kwargs):
                pass

            def log_state(self, *args, **kwargs):
                pass

            def end_frame(self):
                pass

        with wp.ScopedDevice(cuda_test_devices[0]):
            example = StablePDForceExample(
                DummyViewer(),
                types.SimpleNamespace(cloth_model="strain_limiting_baraff_witkin"),
            )
            example.sim_time = 7.9
            example.frame = example.action_delay_frames
            example._solve_ik_and_push_control(example.sim_time, is_delayed=False)
            example._apply_stable_pd_force_control()
            joint_f = example.control.joint_f.numpy()

        self.assertGreater(float(np.max(np.abs(joint_f))), 1.0e-3)

    def test_uipc_cloth_franka_stable_pd_force_uses_effort_actuators(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "uipc",
            "cloth",
            "example_uipc_cloth_franka_stable_pd_force.py",
        )
        with open(example_path) as f:
            source = f.read()

        self.assertIn("ControllerStablePD", source)
        self.assertIn("ClampingMaxEffort", source)
        self.assertIn("JointTargetMode.EFFORT", source)
        self.assertIn("control.joint_f.zero_()", source)
        self.assertIn("eval_mass_matrix", source)
        self.assertIn("bias_forces", source)
        self.assertIn("_solve_ik_and_push_control", source)
        self.assertIn("configure_contact_tabular", source)
        self.assertNotIn("JointTargetMode.POSITION", source)
        self.assertNotIn("set_cloth_soft_position_constraints", source)


if _HAS_UIPC:
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.softbody.example_uipc_soft_hanging",
        devices=cuda_test_devices,
        test_options={"num-frames": 60},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.softbody.example_deformablebody",
        devices=cuda_test_devices,
        test_options={"num-frames": 10},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.softbody.example_uipc_softbody_franka",
        devices=cuda_test_devices,
        test_options={"num-frames": 10},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.multiphysics.example_uipc_softbody_dropping_to_cloth",
        devices=cuda_test_devices,
        test_options={"num-frames": 1},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.cloth.example_uipc_cloth_franka",
        devices=cuda_test_devices,
        test_options={"num-frames": 1},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.cloth.example_uipc_cloth_franka_stable_pd_force",
        devices=cuda_test_devices,
        test_options={"num-frames": 1},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.cloth.example_uipc_cloth_poker_cards",
        devices=cuda_test_devices,
        test_options={"num-frames": 5, "num-cards": 4},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.sensors.example_uipc_sensor_contact",
        devices=cuda_test_devices,
        test_options={"num-frames": 60},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.sensors.example_uipc_sensor_contact",
        devices=cuda_test_devices,
        test_options={"num-frames": 60, "solver": "mujoco"},
        use_viewer=True,
        test_suffix="mujoco",
    )


class TestKaminoExamples(unittest.TestCase):
    pass


add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_basic_fourbar",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_basic_heterogeneous",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_basic_dr_testmech",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_robot_dr_legs",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestKaminoExamples,
    name="kamino.example_kamino_robot_anymal_d",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
