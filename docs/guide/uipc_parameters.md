<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# UIPC Solver 参数使用汇总

本文汇总当前 Newton UIPC backend 中 `SolverUIPC`、刚体、软体和布料实际写入或影响 UIPC 的参数。内容基于以下实现文件：

- `newton/_src/solvers/uipc/solver_uipc.py`
- `newton/_src/solvers/uipc/rigid_body.py`
- `newton/_src/solvers/uipc/deformable_body.py`
- `newton/_src/solvers/uipc/cloth.py`
- `newton/_src/sim/builder.py`

本文只列出构造参数、scene 默认值、以及刚体/软体/布料构建时实际消耗的模型参数；通过运行时函数临时设置的控制 API 不单独暴露。

## SolverUIPC 全局参数

| 参数 / API | 默认值 | UIPC 目标 / 作用位置 | 说明 |
| --- | --- | --- | --- |
| `backend` | `"cuda"` | `uipc.Engine(backend_name=...)` | 选择 UIPC engine backend。 |
| `workspace` | `"/tmp/newton_uipc"` | `uipc.Engine(..., workspace=...)` | UIPC 输出目录，也用于 dump 和 profile 报告。 |
| `dt` | `1.0 / 60.0` | `scene_config["dt"]` | UIPC 固定步长 [s]。 |
| `scene_config` | `UScene.default_config()` | `uipc.Scene(scene_config)` | 可直接传入 UIPC scene 配置；构造时会覆盖 `dt`、`gravity`、部分 contact/newton 默认值。 |
| `kappa` | `100 * MPa` | `AffineBodyConstitution`、关节 builder | 刚体 AffineBody stiffness 参数 [Pa]。 |
| `default_mass_density` | `1000.0` | 刚体 / 软体 fallback 密度 | 当无法从质量和体积估计密度时使用 [kg/m^3]。 |
| `logger_level` | `ULogger.Warn` | `ULogger.set_level()` | UIPC 日志等级。 |
| `dump_enable` | `False` | solver dump 路径 | 控制是否输出 UIPC surface mesh / 调试数据。 |
| `require_profile` | `False` | UIPC timer / report | 开启后记录 step profile，可由 `save_performance_report()` 导出。 |
| `auto_sync_inertia` | `True` | 初始化后的 ABD inertia 同步 | 初始化后把 UIPC ABD 最终质量、COM、惯量同步回 Newton model。 |
| `cloth_model` | `"strain_limiting_baraff_witkin"` | `ClothBuilder` membrane constitution | 可选 `"strain_limiting_baraff_witkin"` 或 `"neo_hookean"`。 |
| `cloth_thickness` | `0.001` | UIPC shell `thickness` | 布料默认厚度 [m]。 |
| `cloth_soft_position_strength_ratio` | `100.0` | cloth `SoftPositionConstraint` | 布料软位置约束默认强度比例。 |
| `enable_soft_position_constraint` | `True` | cloth / deformable vertices | 是否给布料和软体顶点添加 dormant `SoftPositionConstraint` 属性。 |

### 默认 scene / contact 配置

| 配置项 | 当前默认值 | 说明 |
| --- | --- | --- |
| `scene_config["contact"]["enable"]` | `False` | UIPC contact 默认关闭；需要在 scene config 中打开后才启用。 |
| `scene_config["contact"]["d_hat"]` | `0.001` | IPC 接触安全层距离 [m]。 |
| `scene_config["newton"]["velocity_tol"]` | `0.001` | UIPC Newton 迭代速度容差。 |
| `scene_config["newton"]["translation_tol"]` | `0.01` | UIPC Newton 迭代平移容差。 |
| `scene_config["gravity"]` | 来自 `model.gravity` | 若 model 有 gravity，则写入 UIPC scene。 |
| 默认 contact pair `friction` | `0.5` | 内置 `env/robot/actor/ground` pair 的摩擦系数。 |
| 默认 contact pair `stiffness` | `1.0 * GPa` | 内置 contact pair 的接触 stiffness。 |
| 默认 contact pair `ccd` | 按 pair 设置 | `env-robot`、`env-actor`、`ground-robot`、`ground-actor`、`robot-actor`、`actor-actor` 默认开启 CCD；同类静态/机器人 pair 多数关闭。 |

## 刚体 / AffineBody 参数

| Newton 来源 / Solver 参数 | UIPC 目标 / 用法 | 默认 / 计算方式 | 说明 |
| --- | --- | --- | --- |
| `model.body_q` | AffineBody instance transform | 初始化时转成 `mat44` | 刚体初始世界位姿。 |
| `model.body_qd` | UIPC ABD velocity sync | 初始化 / step 同步 | 刚体速度状态同步使用。 |
| `model.body_flags` 中 `BodyFlags.KINEMATIC` | instance `is_fixed = 1` | flag 存在时设置 | Newton kinematic body 在 UIPC 中作为 fixed AffineBody instance。 |
| `model.body_mass` | `mass_density` 或 custom ABD mass matrix | `body_mass / mesh_volume`；体积不可用则 `default_mass_density` | 默认路径用密度让 UIPC 计算 ABD；custom inertia 路径直接写 mass matrix。 |
| `model.body_com` | custom ABD mass matrix | 仅 custom inertia 路径使用 | COM 使用 Newton body-local COM。 |
| `model.body_inertia` | custom ABD mass matrix | 仅 custom inertia 路径使用 | 写入前会对惯量矩阵做对称化。 |
| `kappa` | `AffineBodyConstitution.apply_to(...)` | `100 * MPa` | 刚体 AffineBody stiffness [Pa]。 |
| `default_mass_density` | fallback `mass_density` | `1000.0 kg/m^3` | body mass 缺失或 mesh volume 不可用时使用。 |
| `model.shape_body` | body -> collision shape 映射 | 只选属于当前 body 的 shape | world 静态 shape 使用 `shape_body == -1` 路径。 |
| `model.shape_type` | 生成 UIPC mesh / halfplane | 支持 mesh、convex mesh、box、sphere、capsule、cylinder、cone；plane 单独处理 | plane 不作为 AffineBody mesh；ground plane 走 `halfplane`。 |
| `model.shape_transform` | shape 局部变换 | 合成 body mesh | 用于把 shape 几何放到 body-local/world 位置。 |
| `model.shape_scale` | shape 尺寸 / scale | 合成 body mesh | primitive 和 mesh 构造时使用。 |
| `model.shape_source` | mesh / convex mesh 源 | mesh shape 使用 | primitive shape 不依赖该字段。 |
| `model.shape_flags` | 是否参与 UIPC mesh | 只使用 `ShapeFlags.COLLIDE_SHAPES` | 非碰撞 shape 不贡献 UIPC 接触几何。 |
| articulation membership | contact element `robot` | 非 free joint 的 articulation body | 机器人 link 默认归到 `robo_elem`。 |
| free joint membership | contact element `actor` | `JointType.FREE` child body | free-joint body 默认归到 `actor_elem`。 |
| 其他 body | contact element `env` | fallback | 非 articulation / free body 默认归到环境元素。 |

## 软体 / DeformableBody 参数

| Newton 来源 / Solver 参数 | UIPC 目标 / 用法 | 默认 / 计算方式 | 说明 |
| --- | --- | --- | --- |
| `model.particle_q` | UIPC tetmesh vertices | 直接拷贝选中粒子坐标 | 软体几何顶点位置 [m]。 |
| `model.tet_indices` | UIPC tetmesh tetrahedra | 按 `soft_body_ranges` 或 particle range 选取并重映射 | 软体四面体拓扑。 |
| `model.soft_body_ranges` | 分组构建 deformable object | 有 range 时逐组构建 | 避免不同软体混到一个 UIPC geometry。 |
| `model.particle_mass` | mass density 估计；fixed marker | `sum(mass) / tet_volume`；失败时 `default_mass_density` | `particle_mass <= 0.0` 的粒子会写成 UIPC vertex `is_fixed = 1`。 |
| `default_mass_density` | `StableNeoHookean.apply_to(..., mass_density=...)` fallback | `1000.0 kg/m^3` | 质量或体积不可用时使用。 |
| `model.tet_materials[:, 0]` (`k_mu`) | UIPC tet `mu` | 写入 `(4 / 3) * k_mu` | UIPC StableNeoHookean 的 `mu` 不是原样写入，而是做了变换。 |
| `model.tet_materials[:, 1]` (`k_lambda`) | UIPC tet `lambda` | 写入 `k_lambda + (5 / 6) * k_mu` | UIPC StableNeoHookean 的 `lambda` 会叠加部分 `k_mu`。 |
| `model.tet_materials[:, 2]` (`k_damp`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tet_k_damp = 0.0` | Newton 会保存该列，但当前 UIPC deformable builder 没有映射 damping 参数。 |
| `enable_soft_position_constraint` | `SoftPositionConstraint.apply_to(sc)` | `True` | 为软体顶点添加软位置约束属性；具体目标值由内部同步逻辑写入。 |
| contact element | `actor_elem` | 默认 | 当前 solver 构建 soft body 时默认归入 actor contact element。 |
| `mesh_partition(sc, 16)` | UIPC mesh partition | 固定值 `16` | 构建时用于 UIPC 内部分区。 |

## 布料 / Cloth 参数

| Newton 来源 / Solver 参数 | UIPC 目标 / 用法 | 默认 / 计算方式 | 说明 |
| --- | --- | --- | --- |
| `model.particle_q` | UIPC trimesh vertices | 直接拷贝选中 cloth 粒子坐标 | 布料几何顶点位置 [m]。 |
| `model.tri_indices` | UIPC trimesh faces | 按 `cloth_ranges` 或 legacy heuristic 选取并重映射 | 布料三角面拓扑。闭合三角网格会报错，建议闭合体使用软体或刚体。 |
| `model.cloth_ranges` | 分组构建 cloth object | 有 range 时逐组构建 | 避免多个 cloth 混成一个 UIPC geometry。 |
| `cloth_model` | membrane constitution | 默认 `StrainLimitingBaraffWitkinShell`；可选 `NeoHookeanShell` | 两种 membrane 后续都会写入 `mu/lambda` 三角属性。 |
| `cloth_thickness` | shell `thickness` | `0.001 m` | 默认布料厚度。 |
| 自定义粒子属性 `cloth_thick` | vertex `thickness` 和 `volume` 修正 | 缺失时用 `cloth_thickness`；0 值回退默认厚度；负值报错 | 属性 frequency 必须是 `PARTICLE`。 |
| `model.particle_mass` + `model.tri_areas` | membrane `mass_density` | `sum(particle_mass) / sum(tri_area) / thickness`；失败时 `100.0` | UIPC shell 使用体密度 [kg/m^3]。 |
| `model.tri_materials[:, 0]` (`tri_ke`) | UIPC triangle `mu` | 当前实现原样写入 | 虽然注释提到 Young's modulus，但实际代码把该列写到 `mu`。 |
| `model.tri_materials[:, 1]` (`tri_ka`) | UIPC triangle `lambda` | 当前实现原样写入 | 当前实现把 area stiffness / 第二列写到 `lambda`。 |
| `model.tri_materials[:, 2]` (`tri_kd`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tri_kd = 10.0` | Newton 会保存 damping 列，但当前 UIPC cloth builder 没有映射。 |
| `model.tri_materials[:, 3]` (`tri_drag`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tri_drag = 0.0` | aerodynamic drag 当前未映射到 UIPC cloth。 |
| `model.tri_materials[:, 4]` (`tri_lift`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tri_lift = 0.0` | aerodynamic lift 当前未映射到 UIPC cloth。 |
| `DiscreteShellBending.apply_to(...)` | edge `bending_stiffness` 初值 | builder 默认 `0.01` | 所有 UIPC bending edge 先应用默认 bending stiffness。 |
| `model.edge_bending_properties[:, 0]` (`edge_ke`) | UIPC edge `bending_stiffness` | `ModelBuilder.default_edge_ke = 100.0`，或 cloth/edge API 显式传入 | `_write_edge_bending_stiffness()` 用这一列覆盖 UIPC 每条 bending edge stiffness。 |
| `model.edge_bending_properties[:, 1]` (`edge_kd`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_edge_kd = 0.0` | Newton 会保存 damping，但当前 UIPC cloth builder 只使用 `edge_ke`。 |
| `enable_soft_position_constraint` | `SoftPositionConstraint.apply_to(sc, strength_ratio)` | `True` | 为布料顶点添加软位置约束属性；具体目标值由内部同步逻辑写入。 |
| `cloth_soft_position_strength_ratio` | cloth `SoftPositionConstraint` 默认强度 | `100.0` | 只在布料 builder 中作为默认 ratio 传入。 |
| contact element | `actor_elem` | 默认 | 当前 solver 构建 cloth 时默认归入 actor contact element。 |
| `mesh_partition(sc, 16)` | UIPC mesh partition | 固定值 `16` | 构建时用于 UIPC 内部分区。 |

## ModelBuilder 中与 UIPC 相关的材料默认值

| Builder 默认值 | 当前值 | 主要影响 | UIPC 当前是否使用 |
| --- | --- | --- | --- |
| `default_tri_ke` | `100.0` | `tri_materials[:, 0]` | 使用：写入 cloth triangle `mu`。 |
| `default_tri_ka` | `100.0` | `tri_materials[:, 1]` | 使用：写入 cloth triangle `lambda`。 |
| `default_tri_kd` | `10.0` | `tri_materials[:, 2]` | 未使用。 |
| `default_tri_drag` | `0.0` | `tri_materials[:, 3]` | 未使用。 |
| `default_tri_lift` | `0.0` | `tri_materials[:, 4]` | 未使用。 |
| `default_edge_ke` | `100.0` | `edge_bending_properties[:, 0]` | 使用：写入 cloth edge `bending_stiffness`。 |
| `default_edge_kd` | `0.0` | `edge_bending_properties[:, 1]` | 未使用。 |
| `default_tet_k_mu` | `1.0e3` | `tet_materials[:, 0]` | 使用：变换后写入 deformable tet `mu`。 |
| `default_tet_k_lambda` | `1.0e3` | `tet_materials[:, 1]` | 使用：变换后写入 deformable tet `lambda`。 |
| `default_tet_k_damp` | `0.0` | `tet_materials[:, 2]` | 未使用。 |
| `default_tet_density` | `1.0` | soft body particle mass authoring | 间接使用：影响 `particle_mass`，进而影响 UIPC mass density 估计。 |

## 注意事项

- `_write_edge_bending_stiffness()` 使用的是 `model.edge_bending_properties[edge_id, 0]`，也就是 `edge_ke`；`edge_kd` 当前没有写入 UIPC。
- 布料的 `tri_ke/tri_ka` 当前实现是直接写入 UIPC triangle `mu/lambda`，不是在 UIPC builder 中再转换为 Young's modulus / Poisson ratio。
- 软体的 `tet_materials` 会做 StableNeoHookean 所需的参数变换：`mu = (4/3) * k_mu`，`lambda = k_lambda + (5/6) * k_mu`。
- `SoftPositionConstraint` 只有在 `enable_soft_position_constraint=True` 时才会添加属性；目标位置、启用标记和强度比例属于运行时约束数据，不在本表展开为公开配置项。
- UIPC 全局 contact 默认关闭；即使 contact tabular 已经配置 pair，也只有 `scene_config["contact"]["enable"]` 打开后才会启用接触。
