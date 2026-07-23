# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example UIPC Joints
#
# Demonstrates REVOLUTE, PRISMATIC, and BALL joints using the SolverUIPC
# backend.  Each joint type uses a fixed-to-world anchor link with a
# swinging, sliding, or freely-rotating child link, mirroring the layout
# in example_basic_joints.
#
# Command: python -m newton.examples uipc_joints
#
###########################################################################

import uipc
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt

        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Common geometry settings
        cuboid_hx = 0.1
        cuboid_hy = 0.1
        cuboid_hz = 0.75
        upper_hz = 0.25 * cuboid_hz

        # Layout positions (Y-rows), above ground
        rows = [-3.0, 0.0, 3.0]
        drop_z = 2.0

        # float64 identity quaternion reused across all transforms (UIPC uses double precision)
        q_id = wp.quat_identity(dtype=wp.float64)  # pyright: ignore[reportArgumentType]

        # ---------------------------------------------------------
        # REVOLUTE (hinge) joint
        # ---------------------------------------------------------
        y = rows[0]

        a_rev = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + upper_hz),
                q=q_id,
            ),
        )
        b_rev = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),  # pyright: ignore[reportArgumentType]
            ),
            label="b_rev",
        )
        builder.add_shape_box(a_rev, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_rev, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        j_fixed_rev = builder.add_joint_fixed(
            parent=-1,
            child=a_rev,
            parent_xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + upper_hz),
                q=q_id,
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, 0.0),
                q=q_id,
            ),
            label="fixed_rev_anchor",
        )
        j_revolute = builder.add_joint_revolute(
            parent=a_rev,
            child=b_rev,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, -upper_hz),
                q=q_id,
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, +cuboid_hz),
                q=q_id,
            ),
            label="revolute_a_b",
        )
        builder.add_articulation(
            [j_fixed_rev, j_revolute],
            label="revolute_articulation",
        )
        self._revolute_joint = j_revolute
        builder.joint_q[-1] = wp.pi * 0.5

        # ---------------------------------------------------------
        # PRISMATIC (slider) joint
        # ---------------------------------------------------------
        y = rows[1]
        a_pri = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + upper_hz),
                q=q_id,
            ),
        )
        b_pri = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.0),  # pyright: ignore[reportArgumentType]
            ),
            label="b_prismatic",
        )
        builder.add_shape_box(a_pri, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_pri, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        j_fixed_pri = builder.add_joint_fixed(
            parent=-1,
            child=a_pri,
            parent_xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + upper_hz),
                q=q_id,
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, 0.0),
                q=q_id,
            ),
            label="fixed_pri_anchor",
        )
        j_prismatic = builder.add_joint_prismatic(
            parent=a_pri,
            child=b_pri,
            axis=wp.vec3(0.0, 0.0, 1.0),  # slide along Z
            parent_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, -upper_hz),
                q=q_id,
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, +cuboid_hz),
                q=q_id,
            ),
            limit_lower=-0.3,
            limit_upper=0.3,
            label="prismatic_a_b",
        )
        builder.add_articulation(
            [j_fixed_pri, j_prismatic],
            label="prismatic_articulation",
        )
        self._prismatic_joint = j_prismatic

        # ---------------------------------------------------------
        # BALL (spherical) joint — sphere + cuboid
        # ---------------------------------------------------------
        y = rows[2]
        radius = 0.3
        z_offset = -1.0  # shift down so the ball hangs lower

        # Kinematic (massless) sphere as the parent anchor
        a_ball = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset),
                q=q_id,
            ),
            is_kinematic=True,
        )
        b_ball = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + z_offset),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.0),  # pyright: ignore[reportArgumentType]
            ),
            label="b_ball",
        )

        rigid_cfg = newton.ModelBuilder.ShapeConfig()
        rigid_cfg.density = 0.0
        builder.add_shape_sphere(a_ball, radius=radius, cfg=rigid_cfg)
        builder.add_shape_box(b_ball, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        # Connect parent sphere to world
        j_fixed_ball = builder.add_joint_fixed(
            parent=-1,
            child=a_ball,
            parent_xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset),
                q=q_id,
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, 0.0),
                q=q_id,
            ),
            label="fixed_ball_anchor",
        )
        j_ball = builder.add_joint_ball(
            parent=a_ball,
            child=b_ball,
            parent_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, 0.0),
                q=q_id,
            ),
            child_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, +cuboid_hz),
                q=q_id,
            ),
            label="ball_a_b",
        )
        builder.add_articulation(
            [j_fixed_ball, j_ball],
            label="ball_articulation",
        )

        # # Set initial joint orientation
        builder.joint_q[-4:] = wp.quat_rpy(0.5, 0.6, 0.7)  # ty:ignore[invalid-assignment]  # pyright: ignore[reportArgumentType]

        # Finalize
        builder.color()
        self.model = builder.finalize()
        self.state_0 = self.model.state()

        self.solver = newton.solvers.SolverUIPC(
            self.model,
            workspace="/tmp/newton_uipc/joints",
            dump_enable=True,
            dt=self.sim_dt,
            logger_level=uipc.Logger.Error,
        )
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)  # ty:ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        self.solver.initialize(self.state_0)
        self.viewer.set_model(self.model)
        self.viewer._paused = True

    def simulate(self):
        for _ in range(self.sim_substeps):
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
        self.sim_time += self.frame_dt

        # UIPC writes curr_angle / curr_distance back to state.joint_q
        # via the animator callback, so read directly — no eval_ik needed.
        joint_q_np = self.state_0.joint_q.numpy()  # pyright: ignore[reportOptionalMemberAccess]  # ty:ignore[unresolved-attribute]
        q_start = self.model.joint_q_start.numpy()  # ty:ignore[unresolved-attribute]  # pyright: ignore[reportOptionalMemberAccess]
        rev_angle = joint_q_np[q_start[self._revolute_joint]]
        pri_distance = joint_q_np[q_start[self._prismatic_joint]]
        print(f"t={self.sim_time:.3f}  revolute angle={rev_angle:.4f} rad  prismatic distance={pri_distance:.4f} m")

    def test_final(self):
        # Bodies: 0=a_rev, 1=b_rev, 2=a_pri, 3=b_pri, 4=a_ball, 5=b_ball
        # Fixed anchor links should barely move
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "anchor links are nearly stationary",
            lambda q, qd: max(abs(qd)) < 0.5,  # pyright: ignore[reportArgumentType]
            [0, 2, 4],  # a_rev, a_pri, a_ball
        )

        # Revolute child should still be above ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "revolute child above ground",
            lambda q, qd: float(q[2]) > 0.0,
            [1],  # b_rev
        )

        # Prismatic child should still be above ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "prismatic child above ground",
            lambda q, qd: float(q[2]) > 0.0,
            [3],  # b_pri
        )

        # Ball child should still be above ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "ball child above ground",
            lambda q, qd: float(q[2]) > 0.0,
            [5],  # b_ball
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
