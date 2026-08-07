# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test examples in the newton.examples package.

Currently, this script mainly checks that the examples can run. When the test
runner is invoked with ``--strict-warnings`` (as CI does), example subprocesses
treat deprecation warnings as failures so examples do not regress onto deprecated
APIs; otherwise deprecations are non-fatal. (The broader newton.* escalation of
``--strict-warnings`` applies to the in-process tests, not example subprocesses.)

The test parameters are typically tuned so that each test can run in 10 seconds
or less, ignoring module compilation time. A notable exception is the robot
manipulating cloth example, which takes approximately 35 seconds to run on a
CUDA device.
"""

import importlib.util
import os
import re
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
    NewtonTestCase,
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
        PITCH as UIPC_BRICK_PITCH,
    )
    from newton.examples.uipc.contacts.example_uipc_brick_stacking import (
        STUD_HEIGHT as UIPC_BRICK_STUD_HEIGHT,
    )
    from newton.examples.uipc.contacts.example_uipc_brick_stacking import (
        STUD_RADIUS as UIPC_BRICK_STUD_RADIUS,
    )
    from newton.examples.uipc.contacts.example_uipc_brick_stacking import (
        Example as UIPCBrickStackingExample,
    )
    from newton.examples.uipc.contacts.example_uipc_nut_bolt import (
        Example as UIPCNutBoltExample,
    )
    from newton.examples.uipc.contacts.example_uipc_two_brick_stack import (
        BRICK_HEIGHT as UIPC_TWO_BRICK_HEIGHT,
    )
    from newton.examples.uipc.contacts.example_uipc_two_brick_stack import (
        INTERLOCK_TUBE_OUTER_RADIUS as UIPC_INTERLOCK_TUBE_OUTER_RADIUS,
    )
    from newton.examples.uipc.contacts.example_uipc_two_brick_stack import (
        UIPC_GAP as UIPC_TWO_BRICK_GAP,
    )
    from newton.examples.uipc.contacts.example_uipc_two_brick_stack import (
        Example as UIPCTwoBrickStackExample,
    )
    from newton.examples.uipc.multiphysics.example_uipc_softbody_dropping_to_cloth import (
        Example as UIPCSoftbodyDroppingToClothExample,
    )

_HAS_ONNX_RUNTIME = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("warp_nn") is not None
_PXR_WORK_THREAD_LIMIT_OUTPUT_RE = (
    r"(?s)#+\n#  PXR_WORK_THREAD_LIMIT is overridden to '1'\.  Default is '0'\.  #\n#+\n?"
)
_WARP_CUDA_UNAVAILABLE_OUTPUT_RE = (
    r"(?:"
    r"Warp CUDA warning: Could not find or load the NVIDIA CUDA driver\. "
    r"GPU execution will not be available\."
    r"|"
    r"Warp CUDA error 100: no CUDA-capable device is detected "
    r"\(in function init_cuda_driver, [^\n]*cuda_util\.cpp:\d+\)"
    r")\n?"
)
_NEWTON_ASSET_DOWNLOAD_OUTPUT_RE = (
    r"Cloning https://github\.com/newton-physics/newton-assets\.git "
    r"\(ref: [0-9a-f]{40}\)\.\.\.\n"
    r"Successfully downloaded folder to: [^\n]+\n?"
)
_MATPLOTLIB_FONT_CACHE_OUTPUT_RE = r"Matplotlib is building the font cache; this may take a moment\.\n?"
_DIFFSIM_BALL_GRADIENT_OUTPUT_RE = r"(?:numeric grad: \[[^\n]+\]\nanalytic grad: \[[^\n]+\]\n?){2}"
_DIFFSIM_DRONE_LOSS_LINE_RE = r"\[\s*\d{1,3}/360\] loss=-?\d+\.\d{8}\n?"
_DIFFSIM_DRONE_LOSS_OUTPUT_RE = rf"(?:{_DIFFSIM_DRONE_LOSS_LINE_RE}){{10}}"
_BASIC_PLOTTING_OUTPUT_RE = (
    r"(?:"
    r"Diagnostics plot saved to solver_convergence\.png\n?"
    r"|"
    r"\n?Simulation diagnostics summary \(\d+ steps\):\n"
    r"  Iterations \(max\):   mean=[^\n]*\n"
    r"  Kinetic E \[J\]:    final=[^\n]*\n"
    r"  Potential E \[J\]:  final=[^\n]*\n"
    r"  Constraints:        mean=[^\n]*\n?"
    r")"
)
_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE = (
    r"(?m)"
    r"(?:^.*wp_sdf_contact_write_contact_to_reducer_[^\n]*\.cpp:\d+:\d+: warning: "
    r"implicit conversion from 'long' to 'const wp::int32'.*\n"
    r"^.*\n"
    r"^.*\n"
    r")+"
    r"^\d+ warnings? generated\.\n?"
)
_ANYMAL_TEXTURE_WITHOUT_UVS_WARNING_RE = (
    r"^.*newton[/\\]_src[/\\]utils[/\\]import_urdf\.py:\d+: UserWarning: Warning: mesh "
    r"[^\n]*[/\\]base\.dae has a texture but no UVs; texture will be ignored\.\n"
    r"  parse_shapes\(link, visuals, density=0\.0, just_visual=True, visible=not hide_visuals\)\n?"
)
_EXAMPLE_ALLOW_OUTPUT_REGEXES = [
    (_PXR_WORK_THREAD_LIMIT_OUTPUT_RE, "stderr"),
    (_WARP_CUDA_UNAVAILABLE_OUTPUT_RE, "stderr"),
    (_NEWTON_ASSET_DOWNLOAD_OUTPUT_RE, "stdout"),
]
_OutputRegexSpec = str | tuple[str, str]


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
    expect_output_regexes: list[_OutputRegexSpec] | None = None,
    allow_output_regexes: list[_OutputRegexSpec] | None = None,
):
    """Registers a Newton example to run on ``devices`` as a TestCase."""

    if (expect_output_regexes is not None or allow_output_regexes is not None) and not issubclass(cls, NewtonTestCase):
        raise TypeError("Output regex expectations require a NewtonTestCase subclass")

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

        # Mark the test as skipped if ONNX policy inference is not installed but required.
        onnx_required = options.pop("onnx_required", False)
        torch_required = options.pop("torch_required", False)
        onnx_required = onnx_required or torch_required
        if onnx_required and not _HAS_ONNX_RUNTIME:
            test.skipTest("onnx or warp-nn not installed")

        # Mark the test as skipped if USD is not installed but required
        usd_required = options.pop("usd_required", False)
        if usd_required and not USD_AVAILABLE:
            test.skipTest("Requires usd-core")

        # Escalate deprecations to errors in the example subprocess only when the
        # runner was invoked with --strict-warnings (CI) and the example has not
        # opted out.
        allow_deprecation_warnings = options.pop("allow_deprecation_warnings", False)
        strict_warnings = newton.tests.unittest_utils.strict_warnings and not allow_deprecation_warnings

        # Pass the parent dir; the subprocess's init_kernel_cache appends the version.
        warp_cache_path = wp.config.kernel_cache_dir

        env_vars = os.environ.copy()
        if warp_cache_path is not None:
            env_vars["WARP_CACHE_PATH"] = os.path.dirname(warp_cache_path)
        # Drop any ambient PYTHONWARNINGS so a stray policy in the caller's
        # environment cannot turn a lenient run strict; govern the policy solely
        # through the -W flag below.
        env_vars.pop("PYTHONWARNINGS", None)

        # Escalate deprecations from interpreter startup for strict runs.
        # newton.examples defers to any explicit -W policy (via sys.warnoptions),
        # so this governs instead of the helper's lenient "default" filter.
        warning_args = ["-W", "error::DeprecationWarning"] if strict_warnings else []

        if newton.tests.unittest_utils.coverage_enabled:
            # Generate a random coverage data file name - file is deleted along with containing directory
            with tempfile.NamedTemporaryFile(
                dir=newton.tests.unittest_utils.coverage_temp_dir, delete=False
            ) as coverage_file:
                pass

            command = [sys.executable, *warning_args, "-m", "coverage", "run", f"--data-file={coverage_file.name}"]

            if newton.tests.unittest_utils.coverage_branch:
                command.append("--branch")

        else:
            command = [sys.executable, *warning_args]

        # Append Warp commands
        command.extend(["-m", f"newton.examples.{name}", "--device", str(device), "--test", "--quiet"])

        # Forward any --warp-config overrides from the test runner
        for entry in newton.tests.unittest_utils.warp_config_overrides:
            command.extend(["--warp-config", entry])

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

        if isinstance(test, NewtonTestCase):
            _register_output_regexes(test, expect_output_regexes, required=True)
            _register_output_regexes(test, _EXAMPLE_ALLOW_OUTPUT_REGEXES, required=False)
            _register_output_regexes(test, allow_output_regexes, required=False)
            test.assertSubprocessSuccess(result, command=command)
        else:
            # print any error messages (e.g.: module not found)
            if result.stderr != "":
                print(result.stderr)

            # Check the return code (0 is standard for success)
            test.assertEqual(
                result.returncode,
                0,
                msg=(
                    f"Failed with return code {result.returncode}, command: {' '.join(command)}\n\n"
                    f"Output:\n{result.stdout}\n{result.stderr}"
                ),
            )

        # Clean up output file for old-style examples that may have created one
        if stage_path and stage_path != "None" and result.returncode == 0:
            try:
                os.remove(stage_path)
            except OSError:
                pass

    test_name = f"test_{name}_{test_suffix}" if test_suffix else f"test_{name}"
    add_function_test(cls, test_name, run, devices=devices, check_output=False)


def _register_output_regexes(test: NewtonTestCase, regexes: list[_OutputRegexSpec] | None, *, required: bool):
    add_regex = test.expectOutputRegex if required else test.allowOutputRegex
    for regex_spec in regexes or ():
        if isinstance(regex_spec, tuple):
            regex, stream = regex_spec
        else:
            regex, stream = regex_spec, "any"
        add_regex(regex, stream=stream)


class TestExampleOutputRegexes(unittest.TestCase):
    def test_warp_cuda_unavailable_output_is_allowed(self):
        outputs = (
            "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver. GPU execution will not be available.\n",
            "Warp CUDA error 100: no CUDA-capable device is detected "
            "(in function init_cuda_driver, /builds/omniverse/warp/warp/native/cuda_util.cpp:319)\n",
        )

        for output in outputs:
            with self.subTest(output=output):
                unmatched_output = re.sub(_WARP_CUDA_UNAVAILABLE_OUTPUT_RE, "", output, flags=re.MULTILINE)
                self.assertEqual(unmatched_output, "")

    def test_basic_plotting_output_does_not_consume_trailing_output(self):
        unexpected_output = "unexpected output\n"
        output = (
            "Simulation diagnostics summary (3 steps):\n"
            "  Iterations (max):   mean=1.0, peak=2\n"
            "  Kinetic E [J]:    final=2.0\n"
            "  Potential E [J]:  final=3.0\n"
            "  Constraints:        mean=4.0, peak=5.0\n" + unexpected_output
        )

        unmatched_output = re.sub(_BASIC_PLOTTING_OUTPUT_RE, "", output, flags=re.MULTILINE)

        self.assertEqual(unmatched_output, unexpected_output)


cuda_test_devices = get_selected_cuda_test_devices(mode="basic")  # Don't test on multiple GPUs to save time
test_devices = get_test_devices(mode="basic")


class TestBasicExamples(NewtonTestCase):
    pass


def add_basic_example_test(**kwargs):
    add_example_test(TestBasicExamples, **kwargs)


add_basic_example_test(name="basic.example_basic_pendulum", devices=test_devices, use_viewer=True)

add_basic_example_test(
    name="basic.example_recording",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 120, "world-count": 8},
)

add_basic_example_test(
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    use_viewer=True,
    test_suffix="xpbd",
)
add_basic_example_test(
    name="basic.example_basic_urdf",
    devices=test_devices,
    test_options={"num-frames": 200, "solver": "vbd"},
    test_options_cpu={"world_count": 16},
    test_options_cuda={"world_count": 64},
    use_viewer=True,
    test_suffix="vbd",
)

add_basic_example_test(name="basic.example_basic_viewer", devices=test_devices, use_viewer=True)

add_basic_example_test(
    name="basic.example_basic_joints",
    devices=test_devices,
    use_viewer=True,
    test_suffix="xpbd",
)
add_basic_example_test(
    name="basic.example_basic_joints",
    devices=test_devices,
    use_viewer=True,
    test_options={"solver": "vbd"},
    test_suffix="vbd",
)

add_basic_example_test(
    name="basic.example_basic_shapes",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 150, "solver": "xpbd"},
    test_suffix="xpbd",
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_basic_example_test(
    name="basic.example_basic_shapes",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 150, "solver": "vbd"},
    test_suffix="vbd",
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)

add_basic_example_test(
    name="basic.example_basic_conveyor",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 100},
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_basic_example_test(
    name="basic.example_basic_conveyor_forces",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 100, "solver": "xpbd"},
    test_suffix="xpbd",
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_basic_example_test(
    name="basic.example_basic_conveyor_forces",
    devices=cuda_test_devices,
    use_viewer=True,
    test_options={"num-frames": 100, "solver": "vbd"},
    test_suffix="vbd",
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_basic_example_test(
    name="basic.example_basic_conveyor_forces",
    devices=cuda_test_devices,
    use_viewer=True,
    test_options={"num-frames": 100, "solver": "mujoco"},
    test_suffix="mujoco",
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_basic_example_test(
    name="basic.example_basic_dzhanibekov",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 230, "solver": "vbd"},
    test_suffix="vbd",
)
add_basic_example_test(
    name="basic.example_basic_dzhanibekov",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 230, "solver": "xpbd"},
    test_suffix="xpbd",
)
add_basic_example_test(
    name="basic.example_basic_dzhanibekov",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 230, "solver": "mujoco"},
    test_suffix="mujoco",
)

add_basic_example_test(
    name="basic.example_basic_multi_solver_overlay",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 50},
)


class TestCableExamples(NewtonTestCase):
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
    name="cable.example_cable_bundle_hysteresis",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 150, "eps-max": 2.0, "tau": 0.1},
    test_suffix="dahl_retention",
)
add_example_test(
    TestCableExamples,
    name="cable.example_cable_bundle_hysteresis",
    devices=test_devices,
    use_viewer=True,
    test_options={"num-frames": 150, "no-dahl": True},
    test_suffix="no_dahl_recovery",
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
add_example_test(
    TestCableExamples,
    name="cable.example_cable_plectoneme",
    devices=cuda_test_devices,
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
add_example_test(
    TestClothExamples,
    name="vbd.example_cloth_stiff_material_hanging",
    devices=cuda_test_devices,
    test_options={"usd_required": True, "num-frames": 360},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="vbd.example_cloth_stiff_material_stretch",
    devices=cuda_test_devices,
    test_options={"num-frames": 360},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="vbd.example_vbd_gripper_soft_triangle",
    devices=cuda_test_devices,
    test_options={"num-frames": 360},
    use_viewer=True,
)
add_example_test(
    TestClothExamples,
    name="vbd.example_vbd_gripper_soft_grid",
    devices=cuda_test_devices,
    test_options={"num-frames": 360},
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
    # Deterministic contacts keep the pick-and-place check from flaking.
    test_options={"usd_required": True, "num-frames": 720, "deterministic": True},
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


class TestAdvancedRobotExamples(NewtonTestCase):
    pass


add_example_test(
    TestAdvancedRobotExamples,
    name="mpm.example_mpm_anymal",
    devices=cuda_test_devices,
    test_options={"num-frames": 100, "onnx_required": True},
    allow_output_regexes=[(_ANYMAL_TEXTURE_WITHOUT_UVS_WARNING_RE, "stderr")],
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


class TestMuJoCoExamples(unittest.TestCase):
    pass


add_example_test(
    TestMuJoCoExamples,
    name="mujoco.example_mujoco_sleeping",
    devices=cuda_test_devices,
    test_options={"stack-count": 2, "num-frames": 300},
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


class TestDiffSimExamples(NewtonTestCase):
    pass


def add_diffsim_example_test(**kwargs: Any) -> None:
    extra_allow_output_regexes = kwargs.pop("allow_output_regexes", None) or ()
    allow_output_regexes = [
        (_PXR_WORK_THREAD_LIMIT_OUTPUT_RE, "stderr"),
        (_WARP_CUDA_UNAVAILABLE_OUTPUT_RE, "stderr"),
        *extra_allow_output_regexes,
    ]
    add_example_test(TestDiffSimExamples, allow_output_regexes=allow_output_regexes, **kwargs)


add_diffsim_example_test(
    name="diffsim.example_diffsim_ball",
    devices=test_devices,
    test_options={"num-frames": 4 * 36},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 36},
    use_viewer=True,
    expect_output_regexes=[(_DIFFSIM_BALL_GRADIENT_OUTPUT_RE, "stdout")],
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_cloth",
    devices=test_devices,
    test_options={"num-frames": 4 * 120},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 120},
    use_viewer=True,
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_drone",
    devices=test_devices,
    test_options={"num-frames": 180},  # sim_steps
    test_options_cpu={"num-frames": 10},
    use_viewer=True,
    expect_output_regexes=[(_DIFFSIM_DRONE_LOSS_OUTPUT_RE, "stdout")],
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_spring_cage",
    devices=test_devices,
    test_options={"num-frames": 4 * 30},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 30},
    use_viewer=True,
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_soft_body",
    devices=test_devices,
    test_options={"num-frames": 4 * 60},  # train_iters * sim_steps
    test_options_cpu={"num-frames": 2 * 60},
    use_viewer=True,
)

add_diffsim_example_test(
    name="diffsim.example_diffsim_bear",
    devices=test_devices,
    test_options={"usd_required": True, "num-frames": 4 * 120, "sim-steps": 120},  # train_iters * sim_steps
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


add_basic_example_test(
    name="basic.example_basic_plotting",
    devices=test_devices,
    test_options={"num-frames": 200},
    use_viewer=True,
    expect_output_regexes=[(_BASIC_PLOTTING_OUTPUT_RE, "stdout")],
    allow_output_regexes=[(_MATPLOTLIB_FONT_CACHE_OUTPUT_RE, "stderr")],
)


class TestContactsExamples(unittest.TestCase):
    pass


for example_name in (
    "contacts.example_balance_bird",
    "contacts.example_domino_spiral",
    "contacts.example_newton_cradle",
):
    for solver in ("xpbd", "vbd"):
        add_example_test(
            TestContactsExamples,
            name=example_name,
            devices=cuda_test_devices,
            test_options={"num-frames": 60, "solver": solver},
            use_viewer=True,
            test_suffix=solver,
        )


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


class TestMultiphysicsExamples(NewtonTestCase):
    pass


add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_gift",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="cloth.example_cloth_poker_cards",
    devices=test_devices,
    test_options={"num-frames": 30},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_dropping_to_cloth",
    devices=test_devices,
    test_options={"num-frames": 200},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_softbody_dropping_to_cloth",
    devices=test_devices,
    test_options={"num-frames": 2, "solver": "coupled", "vbd-iterations": 2},
    use_viewer=True,
    test_suffix="coupled",
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_rigid_soft_contact",
    devices=test_devices,
    test_options={"num-frames": 180, "solver": "xpbd"},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
    test_suffix="xpbd",
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_rigid_soft_contact",
    devices=test_devices,
    test_options={"num-frames": 180, "solver": "semi_implicit"},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
    test_suffix="semi_implicit",
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_rigid_soft_contact",
    devices=test_devices,
    test_options={"num-frames": 180, "solver": "vbd"},
    test_options_cpu={"num-frames": 2},
    use_viewer=True,
    test_suffix="vbd",
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_rigid_soft_contact",
    devices=test_devices,
    test_options={"num-frames": 2, "solver": "coupled", "rigid-solver": "mjc", "vbd-iterations": 1},
    use_viewer=True,
    test_suffix="coupled_mjc",
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_vbd_admm_solver",
    devices=test_devices,
    test_options={"num-frames": 30},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_admm_contact_solver",
    devices=test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_kamino_mujoco_admm_solver",
    devices=["cpu"],
    test_options={"num-frames": 30, "world-count": 4},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_xpbd_vbd_coupled_solver",
    devices=test_devices,
    test_options={"num-frames": 5, "xpbd-iterations": 4, "vbd-iterations": 2},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_franka_vbd_cable_admm_solver",
    devices=cuda_test_devices,
    test_options={
        "num-frames": 2,
        "world-count": 1,
        "substeps": 1,
        "admm-iterations": 1,
        "payload-segments": 3,
        "xpbd-iterations": 2,
        "graph-capture": False,
    },
    use_viewer=True,
    allow_output_regexes=[(_WARP_SDF_CONSTANT_CONVERSION_WARNING_RE, "stderr")],
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_mpm_coupled_solver",
    devices=cuda_test_devices,
    test_options={"num-frames": 2, "rigid-substeps": 1, "proxy-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_vbd_coupled_solver",
    devices=test_devices,
    test_options={"num-frames": 2, "proxy-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_mujoco_xpbd_coupled_solver",
    devices=test_devices,
    test_options={"num-frames": 2, "proxy-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_proxy_joint_gripper",
    devices=test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_vbd_mpm_coupled_solver",
    devices=cuda_test_devices,
    test_options={"num-frames": 2, "proxy-iterations": 1, "vbd-iterations": 2, "mpm-iterations": 1},
    use_viewer=True,
)
add_example_test(
    TestMultiphysicsExamples,
    name="multiphysics.example_xpbd_mpm_coupled_solver",
    devices=cuda_test_devices,
    test_options={
        "num-frames": 2,
        "proxy-iterations": 1,
        "xpbd-iterations": 2,
        "xpbd-dim-x": 2,
        "xpbd-dim-y": 2,
        "xpbd-dim-z": 2,
        "mpm-iterations": 1,
        "grid-padding": 8,
        "substeps": 1,
    },
    use_viewer=True,
)


class TestSoftbodyExamples(NewtonTestCase):
    pass


add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_hanging",
    devices=test_devices,
    test_options={"num-frames": 120},
    test_options_cpu={"num-frames": 2},
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
        self.assertIn(
            "self.interlock_tube_outer_radius = self._compute_interlock_tube_outer_radius(self.uipc_gap)", source
        )
        self.assertIn("_make_brick_mesh(tube_outer_radius=tube_outer_radius)", source)
        self.assertIn('"--uipc-gap"', source)
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
        self.assertAlmostEqual(example.uipc_gap, 0.0005)
        self.assertAlmostEqual(
            example.interlock_tube_outer_radius,
            np.hypot(0.5 * UIPC_BRICK_PITCH, 0.5 * UIPC_BRICK_PITCH) - UIPC_BRICK_STUD_RADIUS - example.uipc_gap,
        )
        np.testing.assert_allclose(abd_kappa[example.brick_bodies], UIPC_BRICK_ABD_KAPPA)
        np.testing.assert_allclose(abd_kappa[example.board_floor_bodies], -1.0)
        board_floor_labels = [label for label in example.model.body_label if label.startswith("board_floor_")]
        self.assertEqual(len(board_floor_labels), 8)

        board_floor_z = body_q[example.board_floor_bodies[0]][2]
        board_top = board_floor_z + 0.5 * example.brick_height_scaled
        board_stud_top = board_top + UIPC_BRICK_STUD_HEIGHT
        self.assertGreater(board_top, example.table_top_z)
        self.assertGreater(board_stud_top, example.table_top_z)

    def test_uipc_two_brick_stack_initializes_stacked_bricks(self):
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

        args = types.SimpleNamespace(test=True, uipc_gap=UIPC_TWO_BRICK_GAP)
        with wp.ScopedDevice(cuda_test_devices[0]):
            example = UIPCTwoBrickStackExample(DummyViewer(), args)
            self.assertEqual([example.model.body_label[i] for i in example.brick_bodies], ["brick_bottom", "brick_top"])
            self.assertTrue(example.solver.is_contact_enabled(example.brick_bodies[0], example.brick_bodies[1]))
            self.assertTrue(example.solver.is_contact_enabled(-1, example.brick_bodies[0]))
            body_q = example.state_0.body_q.numpy()[example.brick_bodies]
            initial_body_q = example.initial_body_q.copy()
            abd_kappa = example.model.uipc.abd_kappa.numpy()[example.brick_bodies]

        np.testing.assert_allclose(body_q, initial_body_q)
        np.testing.assert_allclose(abd_kappa, UIPC_BRICK_ABD_KAPPA)
        self.assertAlmostEqual(example.uipc_gap, 1.0e-4)
        self.assertAlmostEqual(example.interlock_tube_outer_radius, UIPC_INTERLOCK_TUBE_OUTER_RADIUS)
        stud_tube_clearance = (
            np.hypot(0.5 * UIPC_BRICK_PITCH, 0.5 * UIPC_BRICK_PITCH)
            - example.interlock_tube_outer_radius
            - UIPC_BRICK_STUD_RADIUS
        )
        self.assertAlmostEqual(stud_tube_clearance, example.uipc_gap)
        self.assertLess(float(np.linalg.norm(body_q[1, :2] - body_q[0, :2])), 1.0e-6)
        self.assertAlmostEqual(
            float(body_q[1, 2] - body_q[0, 2]),
            UIPC_TWO_BRICK_HEIGHT + args.uipc_gap,
            delta=1.0e-6,
        )

    def test_uipc_nut_bolt_uses_contact_instead_of_aim_target_drive(self):
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples",
            "uipc",
            "contacts",
            "example_uipc_nut_bolt.py",
        )
        with open(example_path) as f:
            source = f.read()

        self.assertIn('ABD_REPO_URL = "https://github.com/Autodesk/affine-body-dynamics.git"', source)
        self.assertIn('ABD_SCREW_NUT_FOLDER = "meshes/screw-and-nut"', source)
        self.assertIn('SCREW_MESH_NAME = "screw-big.obj"', source)
        self.assertIn('NUT_MESH_NAME = "nut-big.obj"', source)
        self.assertIn('ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"', source)
        self.assertIn('ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"', source)
        self.assertIn('ORIGINAL_ASSEMBLY_STR = "m20_loose"', source)
        self.assertIn('ORIGIN_MESH_SOURCE_ALIAS = "origin"', source)
        self.assertIn(
            "MESH_SOURCE_CHOICES = (AUTODESK_MESH_SOURCE, ORIGINAL_MESH_SOURCE, ORIGIN_MESH_SOURCE_ALIAS)", source
        )
        self.assertIn("def _canonical_mesh_source(mesh_source: str) -> str:", source)
        self.assertIn("if mesh_source == ORIGIN_MESH_SOURCE_ALIAS:", source)
        self.assertIn("AUTODESK_MESH_SCALE = 0.00326", source)
        self.assertIn("ORIGINAL_MESH_SCALE = 1.0", source)
        self.assertIn("ORIGINAL_BOLT_RADIAL_SCALE = 0.955", source)
        self.assertIn("ORIGINAL_NUT_RADIAL_SCALE = 1.0", source)
        self.assertIn("ASSEMBLY_SPACING = 0.1", source)
        self.assertIn("AUTODESK_BOLT_START_Z = 0.048", source)
        self.assertIn("AUTODESK_NUT_START_Z = 0.05676", source)
        self.assertIn("ORIGINAL_BOLT_START_Z = 0.0", source)
        self.assertIn("ORIGINAL_NUT_START_Z = 0.04262", source)
        self.assertIn("NUT_START_YAW = np.pi / 8.0", source)
        self.assertIn("MIN_NUT_DROP_BY_MESH_SOURCE", source)
        self.assertIn("ORIGINAL_MESH_SOURCE: 0.004", source)
        self.assertIn("MIN_NUT_ROTATION = 1.0", source)
        self.assertIn("UIPC_SOLVE_TOL = 1.0e-5", source)
        self.assertIn("THREAD_CONTACT_MU = 0.0", source)
        self.assertIn("mu=THREAD_CONTACT_MU", source)
        self.assertNotIn("mu=0.12", source)
        self.assertIn("self.solver.configure_scene", source)
        self.assertIn('"translation_tol": UIPC_SOLVE_TOL', source)
        self.assertIn('"velocity_tol": UIPC_SOLVE_TOL', source)
        self.assertIn('"linear_system": {"tol_rate": UIPC_SOLVE_TOL}', source)
        self.assertIn('parser.add_argument(\n            "--mesh-source"', source)
        self.assertIn("default=AUTODESK_MESH_SOURCE", source)
        self.assertIn("'origin' is accepted as an alias", source)
        self.assertIn(
            "vertices = np.ascontiguousarray(np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1])))",
            source,
        )
        self.assertNotIn("vertices = vertices[:, [0, 2, 1]]", source)
        self.assertIn("vertices[:, 0:2] *= np.float32(radial_scale)", source)
        self.assertIn("rotate_y_axis_to_z=True", source)
        self.assertIn("radial_scale=ORIGINAL_BOLT_RADIAL_SCALE", source)
        self.assertIn("radial_scale=ORIGINAL_NUT_RADIAL_SCALE", source)
        self.assertIn("ORIGINAL_NUT_MESH_NAME", source)
        self.assertNotIn("convex_hull", source)
        self.assertIn("configure_contact_tabular", source)
        self.assertNotIn("aim_transform", source)
        self.assertNotIn("aim_tf", source)
        self.assertNotIn("is_constrained", source)
        self.assertNotIn("_write_nut_aim_transforms", source)
        self.assertNotIn("NUT_SPIN_RATE", source)
        self.assertNotIn("NUT_DROP_RATE", source)
        self.assertNotIn("NUT_MAX_DROP", source)
        self.assertNotIn("body_qd[nut_body]", source)
        self.assertNotIn("joint_qd[qd_start + 5]", source)
        self.assertNotIn("if self.mesh_source != AUTODESK_MESH_SOURCE", source)
        self.assertIn("drop > min_drop", source)
        self.assertIn("rotation > MIN_NUT_ROTATION", source)

    def test_uipc_nut_bolt_tracks_initial_nuts_outside_test_mode(self):
        if not _HAS_UIPC:
            self.skipTest("Requires uipc")

        class DummyBodyQ:
            def numpy(self):
                return np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 2.0, 3.0, 0.0, 0.0, np.sin(0.25), np.cos(0.25)],
                    ],
                    dtype=np.float32,
                )

        example = UIPCNutBoltExample.__new__(UIPCNutBoltExample)
        example.test_mode = False
        example.model = types.SimpleNamespace(body_label=["bolt_0_0", "nut_0_0"])
        example.state_0 = types.SimpleNamespace(body_q=DummyBodyQ())

        example._init_test_tracking()

        self.assertEqual(example.bolt_body_indices, None)
        self.assertEqual(example.nut_body_indices, None)
        self.assertIn(1, example.nut_initial_by_body)
        self.assertFalse(hasattr(example, "nut_initial_yaw_by_body"))
        np.testing.assert_allclose(example.nut_initial_by_body[1][:3], [1.0, 2.0, 3.0])

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
        self.assertIn("wp.copy(self.control.joint_target_q, self.model.joint_q)", source)
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
        name="uipc.basic.example_uipc_revolute_multi_turn",
        devices=cuda_test_devices,
        test_options={"num-frames": 240},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.robot.example_uipc_ur10",
        devices=cuda_test_devices,
        test_options={"num-frames": 120},
        use_viewer=True,
    )
    add_example_test(
        TestUIPCSoftbodyExamples,
        name="uipc.robot.example_uipc_ur10",
        devices=cuda_test_devices,
        test_options={"num-frames": 120, "implicit-pd": True},
        use_viewer=True,
        test_suffix="implicit_pd",
    )
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
        name="uipc.contacts.example_uipc_two_brick_stack",
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
    test_options={"num-frames": 500},
    use_viewer=True,
)


class TestControllersExamples(unittest.TestCase):
    pass


add_example_test(
    TestControllersExamples,
    name="controllers.example_controller_joint_impedance_heterogeneous",
    devices=cuda_test_devices,
    test_options={"num-frames": 120},
    use_viewer=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
