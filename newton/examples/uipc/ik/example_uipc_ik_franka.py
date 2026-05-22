# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC IK Franka
#
# Interactive inverse kinematics on a Franka FR3 arm simulated with the
# SolverUIPC backend. A viewer gizmo controls the TCP target; an IK solver
# turns the gizmo pose into joint targets that are fed to the UIPC ABD/IPC
# stack as POSITION-mode actuator commands.
#
# This is essentially ``example_ik_franka`` (gizmo + IK only, no physics)
# upgraded to drive a real UIPC simulation, mirroring the actuator setup
# from ``example_uipc_panda``.
#
# Command: python -m newton.examples uipc_ik_franka
#
###########################################################################

import numpy as np
import uipc
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import JointTargetMode


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def write_joint_targets_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array[wp.float32],
    gripper_value: float,
):
    # Single-world helper: copy 7 arm DOFs from the IK result and stamp the
    # two finger DOFs at a constant gripper opening.
    tid = wp.tid()
    if tid == 0:
        for j in range(7):
            joint_targets[j] = ik_solution[0, j]
        joint_targets[7] = gripper_value
        joint_targets[8] = gripper_value


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        # ------------------------------------------------------------------
        # Build the Franka FR3 arm.
        # ------------------------------------------------------------------
        franka = newton.ModelBuilder(up_axis=newton.Axis.Z)

        franka.add_urdf(
            str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"),
            xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()),
            floating=False,
        )

        def find_body(name: str) -> int:
            return next(i for i, lbl in enumerate(franka.body_label) if lbl.endswith(f"/{name}"))

        left_finger_idx = find_body("fr3_leftfinger")
        right_finger_idx = find_body("fr3_rightfinger")

        pad_asset_path = newton.utils.download_asset("manipulation_objects/pad")
        pad_stage = Usd.Stage.Open(str(pad_asset_path / "model.usda"))
        pad_mesh = newton.usd.get_mesh(
            pad_stage.GetPrimAtPath("/root/Model/Model"),
            load_normals=True,
            face_varying_normal_conversion="vertex_splitting",
        )
        pad_scale = np.asarray(newton.usd.get_scale(pad_stage.GetPrimAtPath("/root/Model")), dtype=np.float32)
        if not np.allclose(pad_scale, 1.0):
            pad_mesh = pad_mesh.copy(vertices=pad_mesh.vertices * pad_scale, recompute_inertia=True)
        pad_xform = wp.transform(
            wp.vec3(0.0, 0.005, 0.045),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi),
        )
        franka.add_shape_mesh(body=left_finger_idx, mesh=pad_mesh, xform=pad_xform)
        franka.add_shape_mesh(body=right_finger_idx, mesh=pad_mesh, xform=pad_xform)

        # Initial joint configuration (matches example_uipc_panda).
        init_q = [
            -3.6802115e-03,
            2.3901723e-02,
            3.6804110e-03,
            -2.3683236e00,
            -1.2918962e-04,
            2.3922248e00,
            7.8549200e-01,
        ]
        franka.joint_q[:9] = [*init_q, 0.04, 0.04]

        for d in range(9):
            franka.joint_target_pos[d] = franka.joint_q[d]
            franka.joint_target_ke[d] = 650.0
            franka.joint_target_kd[d] = 100.0
            franka.joint_target_mode[d] = int(JointTargetMode.POSITION)
            franka.joint_armature[d] = 1e-2 if d < 7 else 5e-2

        franka.add_ground_plane()

        # ------------------------------------------------------------------
        # Finalize model and UIPC solver.
        # ------------------------------------------------------------------
        self.model = franka.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Warn,
        )
        self.solver.set_contact(True)
        self.solver.initialize()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Persistent flat buffer mirroring control.joint_target_pos shape.
        # Slice-assigning into control.joint_target_pos via wp.copy on views
        # is unreliable; we always write the whole buffer instead.
        self.joint_targets_flat = wp.zeros_like(self.control.joint_target_pos)
        wp.copy(self.joint_targets_flat, self.model.joint_q)
        wp.copy(self.control.joint_target_pos, self.joint_targets_flat)

        # ------------------------------------------------------------------
        # Viewer setup.
        # ------------------------------------------------------------------
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(1.2, 1.2, 0.9),
            pitch=-25.0,
            yaw=-135.0,
        )
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        # Start paused so the user can verify the rest pose before stepping.
        self.viewer._paused = True

        # ------------------------------------------------------------------
        # IK setup — single-problem solver targeting ``fr3_hand``, which
        # matches the convention used by ``example_uipc_panda``. The
        # wrist chain (``link7 → link8 → fr3_hand``) now behaves correctly
        # under UIPC because ``articulation_franka._compute_shape_body_anchors``
        # collapses the shapeless ``fr3_link8`` into its nearest ABD-bearing
        # ancestor, so the gizmo-driven IK solution is actually tracked by
        # the physics.
        # ------------------------------------------------------------------
        self.ee_index = next(i for i, lbl in enumerate(self.model.body_label) if lbl.endswith("/fr3_hand"))
        body_q_np = self.state_0.body_q.numpy()
        self.ee_tf = wp.transform(*body_q_np[self.ee_index])

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self.ee_tf)], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(dtype=wp.float32),
            target_rotations=wp.array([quat_to_vec4(wp.transform_get_rotation(self.ee_tf))], dtype=wp.vec4),
        )
        self.joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        # Seed the IK warm-start buffer from the model rest pose via numpy
        # so we are sure the values land in the (1, dofs) buffer — passing a
        # wp.array straight into wp.array(..., shape=...) does not always
        # copy the underlying data on every Warp version.
        joint_q_np = self.model.joint_q.numpy().astype(np.float32).reshape(1, self.model.joint_coord_count)
        self.joint_q_ik = wp.array(joint_q_np, dtype=wp.float32)
        self.ik_iters = 24
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # Persistent gripper opening (kept constant — gizmo only drives TCP).
        self.gripper_value = 0.04

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _push_targets_from_gizmo(self):
        """Read gizmo-updated transform and push into the IK objectives."""
        pos = wp.transform_get_translation(self.ee_tf)
        # Keep the TCP above the table to avoid driving into the ground.
        pos = wp.vec3(pos[0], pos[1], max(pos[2], 0.11))
        self.pos_obj.set_target_position(0, pos)
        q = wp.transform_get_rotation(self.ee_tf)
        self.rot_obj.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    def _solve_ik_and_push_control(self):
        self._push_targets_from_gizmo()
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.ik_iters)

        # Build the full target buffer in one kernel launch and copy the
        # whole thing into control — sliced wp.copy on Warp array views is
        # unreliable, which silently leaves UIPC stuck on the rest pose.
        wp.launch(
            write_joint_targets_kernel,
            dim=1,
            inputs=[self.joint_q_ik, self.joint_targets_flat, self.gripper_value],
        )
        wp.copy(self.control.joint_target_pos, self.joint_targets_flat)

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._solve_ik_and_push_control()
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
        self.sim_time += self.frame_dt
        # Note: ``fr3_link8`` / ``fr3_hand_tcp`` are shapeless frame links in
        # the URDF so they have no ABD body in UIPC, but nothing in this
        # example references their ``body_q`` (no visuals, no IK targets on
        # them) so the stale values are harmless. ``fr3_hand`` itself is
        # now correctly simulated thanks to the shapeless-anchor collapse
        # in ``articulation_franka._compute_shape_body_anchors``.

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Show the gizmo at the TCP target; release snaps to the simulated EE.
        body_q_np = self.state_0.body_q.numpy()
        self.viewer.log_gizmo(
            "target_tcp",
            self.ee_tf,
            snap_to=wp.transform(*body_q_np[self.ee_index]),
        )

        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Sanity check: the simulated TCP should remain above the ground.
        body_q = self.state_0.body_q.numpy()
        z = float(body_q[self.ee_index][2])
        assert z > 0.0, f"End-effector dropped below the ground (z={z:.3f})"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
