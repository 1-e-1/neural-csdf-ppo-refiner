> Historical source documentation. See the repository root README and PUBLICATION_NOTES.md for current distribution limits.

# 工程结构说明（扁平化，2026-08-07）

> 本目录是 `D:\work_space\isaacsim_lab_zbot_6s_2` 中对应版本研究内容的**独立副本**。已取消 `h1_1111_grasp` 层级，所有内容直接位于本目录顶层；主工程保持嵌套结构不变。本副本可独立训练、评估与物理演示。

## 目录结构

| 目录 | 内容 |
|---|---|
| `src/` | 共享源码：`src/h1_1111_grasp/`（H1 双机械臂视觉抓取包）+ `src/datasets|models|planning|robot|simulation|training|utils`（条件流匹配共享依赖） |
| `configs/` | 训练/评估/机器人配置（含 `visual_cfm_v621.yaml`、`kinematic_chain.json`） |
| `scripts/` | 训练、评估、数据生成、物理演示与 USD 配置脚本 |
| `data/` | 视觉数据集（`visual_v621_full/`：6 train + 1 val + 1 test）与相机标定 |
| `outputs/` | `checkpoints/`（`visual_cfm_v621_left.pt` 等）+ `figures/`（物理演示与报告） |
| `pretrained/` | 平衡与视觉预训练 checkpoint |
| `self_collision_refiner/` | Neural C-SDF + PPO 自碰撞细化子工程（独立可运行） |
| `tests/` | 单元测试 |
| `docs/` | 论文初稿与技术文档 |
| `evaluation/`（仅 v6.2） | 版本评估报告 json |

## 运行方式

- 环境：`D:\conda_envs\env_isaaclab`（Isaac Sim / IsaacLab，不修改 `D:\Isaac_lab`）
- 训练：`D:\conda_envs\env_isaaclab\python.exe scripts\train_visual_cfm_v621.py`
- 评估：`D:\conda_envs\env_isaaclab\python.exe scripts\evaluate_visual_cfm_v621.py`
- 数据生成：`D:\conda_envs\env_isaaclab\python.exe scripts\generate_grasp_place_dataset.py`
- 单元测试：`D:\conda_envs\env_isaaclab\python.exe -m pytest tests -q`

> 注意：原 README 中出现的 `h1_1111_grasp\...` 旧路径在扁平化后已失效；`import h1_1111_grasp.*` 包导入仍然有效（包位于 `src/h1_1111_grasp/`）。

---
# 1111_copy 双辅助机械臂 H1 平衡与视觉抓取研究

本工程直接使用工作空间根目录的 `1111_copy.usd`。机械装配、关节 drive、相机 Xform 和碰撞过滤均写入主 USD，不通过外部场景覆盖临时拼装。工程不修改 `D:\Isaac_lab`、`D:\conda_envs\env_isaaclab` 或既有条件流匹配研究目录。

## 主 USD 装配

- `torso_link -> Cube`：`FixedJoint`。
- `Cube -> ZBot6S_left/foot_0`：`FixedJoint`。
- `Cube -> ZBot6S_right/foot_0`：`FixedJoint`。
- 两个 Robotiq 夹爪通过各自机械臂末端的 `GripperMountJoint` 安装；原来固定到 World 的 `root_joint` 已停用。
- H1 内部 benign collision pairs 被过滤；双臂之间、双臂与 H1 非安装部位之间的碰撞保持启用。
- 修改前备份：`..\1111_copy_before_training_20260801.usd`。
- USD 审计：`outputs\usd_audit.json`。

重新审计并写入主 USD：

```powershell
D:\conda_envs\env_isaaclab\python.exe h1_1111_grasp\scripts\configure_1111_usd.py
```

## 平衡策略

- Observation：`95-D = H1 proprioception 69-D + base xy 2-D + 双臂关节位置 12-D + 双臂关节速度 12-D`。
- Action：`19-D` H1 关节位置动作；辅助机械臂由独立轨迹生成器控制。
- Physics timestep：`0.005 s`；control decimation：`4`；控制频率：`50 Hz`。
- Reward：存活、基座高度、水平位置、姿态、速度跟踪、动作变化率、H1 力矩/加速度。
- Termination：高度过低、倾角过大、水平漂移超过 `0.30 m`、timeout。
- 训练：Isaac Lab Manager-Based RL VecEnv，512 个并行环境，RSL-RL PPO，TensorBoard 日志保存在 `outputs\balance`。

训练命令示例：

```powershell
D:\conda_envs\env_isaaclab\python.exe h1_1111_grasp\scripts\train_balance.py --stage manipulation --num_envs 512 --max_iterations 300 --checkpoint CHECKPOINT.pt --headless
```

统一评估命令：

```powershell
D:\conda_envs\env_isaaclab\python.exe h1_1111_grasp\scripts\evaluate_balance.py --checkpoint CHECKPOINT.pt --stage robust --num_envs 128 --steps 1000 --output h1_1111_grasp\outputs\evaluation\result.json --headless
```

## 视觉轨迹生成

左右机械臂均在各自 `foot_0` 坐标系中规划，共享 Visual Conditional Flow Matching 模型。

- 输入：`q_start[6] + p_goal[3] + RGB-D[4 views, 2 frames]`；不读取 H1 状态。
- 四视角：腕部、同侧 Cube、同侧胸部、头顶。
- 输出：`64x6` 关节位置序列和一个 duration。
- 推理：32 步 Heun ODE，生成 64 条候选。
- 在线筛选：视觉点云检查自碰撞、机械臂-H1、双臂、桌面和障碍物间隙。
- PhysX：只用于仿真训练标签和离线复核，不作为真实部署的在线真值输入。
- 紧急停止：视觉估计间隙低于 `0.015 m` 或出现未预期接触时停止。

数据契约见 `docs\DATA_CONTRACT.md`，核心配置见 `configs\trajectory_visual.yaml` 与 `configs\visual_cfm_transfer.yaml`。

## 验证

```powershell
D:\conda_envs\env_isaaclab\python.exe -m pytest h1_1111_grasp\tests -q
D:\conda_envs\env_isaaclab\python.exe -m py_compile h1_1111_grasp\scripts\*.py
```

真实硬件部署前必须重新标定相机内外参，并验证关节动作限幅、急停、通信延迟、视觉深度噪声、遮挡、机械碰撞和 domain randomization 覆盖范围。当前 checkpoint 只能用于 Isaac Sim/Isaac Lab 仿真研究。
