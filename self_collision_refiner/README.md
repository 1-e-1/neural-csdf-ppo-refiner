# 自碰撞感知 64→256 轨迹强化学习修正器

该子工程与 `h1_1111_grasp` 中已有 Visual CFM、H1 balance policy、数据集和 checkpoint 隔离。它只解决六自由度辅助机械臂的**自碰撞**，不读取环境障碍物、H1 状态或另一条机械臂状态，也不修改 `1111_copy.usd`。

## 1. 方法流程

1. 在周期构型空间 `[-pi, pi)^6` 中进行 scrambled Sobol sampling 与独立均匀采样。
2. 使用现有六自由度 FK 和非相邻 link capsule 几何，计算每个构型的 signed self-clearance。
3. 在碰撞边界附近自适应增加采样，训练 Neural Configuration-space Signed Distance Field（Neural C-SDF）。
4. PPO residual policy 接收原始 `64×6` 轨迹，通过 16 个控制点的内部残差逐步改变整条轨迹。
5. 使用 clamped cubic spline 输出 `256×6`；起点关节状态严格不变。若原终端关节构型不安全，则保持 TCP 目标位置不变并重新求安全终端关节解。
6. 最终使用精确 capsule oracle 对每个相邻区间自适应加密检查。

256 帧不是均匀复制。时间索引保持有序，但关节空间位移由策略调整后的 spline 决定；靠近自碰撞区域时可以改变轨迹方向和局部采样密度对应的关节变化率。

## 2. Observation / Action / Reward

### Observation

默认控制点数为 16，单个环境 observation 为 488 维：

- 当前控制点周期编码 `sin(q), cos(q)`：`16×12=192`。
- 原始控制点周期编码：`16×12=192`。
- 控制点 Neural C-SDF clearance：16。
- 14 个内部控制点的累计 residual：`14×6=84`。
- 最小 clearance、平均 clearance、碰撞帧比例和 episode 进度：4。

策略不读取 H1 机器人状态、环境点云或障碍物真值。

### Action

- Action shape：`[B,14]`，对应 14 个内部控制点的安全方向门控系数。
- 每个安全方向由 Neural C-SDF 对该控制点的 clearance 梯度给出，PPO 决定沿该方向的步长；策略不在 84 维关节空间中盲目搜索。
- 每步最大变化：0.10 rad。
- 累计 residual 限制：`[-0.45,0.45] rad`。
- 起点和 terminal projection 后的终点不是 PPO action 的一部分，因此不会被策略继续修改。

### Reward

- 势函数：最小 self-clearance、碰撞帧比例、轨迹偏移、加速度 proxy、jerk proxy。
- 势函数塑形：`gamma*Phi(s_next)-Phi(s)`，避免通过反复靠近和远离边界刷分。
- 终局奖励：所有 256 帧 Neural C-SDF clearance 均超过 1 cm。
- 动作惩罚：限制不必要的大幅关节修正。

## 3. 时间和张量契约

- 输入：`q64 [B,64,6]`。
- PPO episode：4 个决策步。
- 控制点：16，其中 14 个可修改。
- 输出：`q256 [B,256,6]`。
- 关节角：始终映射至 `[-pi,pi)`，内部 spline 使用连续展开角度。
- 输出控制时间戳：默认沿原轨迹 duration 生成 256 个时间点。
- 精确安全判据：capsule self-clearance ≥ 0.010 m。
- 不可行判据：当前起点本身低于安全 clearance 时，轨迹补帧无法修复，推理报告返回 `start_infeasible`。

## 4. Loss 与训练

Neural C-SDF 使用两个 head：

- clearance regression：边界加权 Smooth L1 loss。
- collision classification：带正样本权重的 Binary Cross Entropy。

PPO 使用 clipped policy objective、value loss、entropy regularization 和 GAE。固定随机种子为 `20260802`。

## 5. 目录

```text
self_collision_refiner/
├── config.yaml
├── README.md
├── self_collision_refiner/
│   ├── geometry.py
│   ├── csdf_data.py
│   ├── csdf_training.py
│   ├── models.py
│   ├── trajectory_env.py
│   └── ppo.py
├── scripts/
│   ├── generate_csdf_dataset.py
│   ├── train_csdf.py
│   ├── train_refiner_ppo.py
│   └── refine_trajectory.py
├── tests/
├── data/
├── checkpoints/
└── outputs/
```

## 6. 执行命令

在工程根目录执行：

```powershell
& 'D:\conda_envs\env_isaaclab\python.exe' h1_1111_grasp/self_collision_refiner/scripts/generate_csdf_dataset.py
& 'D:\conda_envs\env_isaaclab\python.exe' h1_1111_grasp/self_collision_refiner/scripts/train_csdf.py
& 'D:\conda_envs\env_isaaclab\python.exe' h1_1111_grasp/self_collision_refiner/scripts/train_refiner_ppo.py
& 'D:\conda_envs\env_isaaclab\python.exe' h1_1111_grasp/self_collision_refiner/scripts/refine_trajectory.py --index 0
```

## 7. 验证边界与风险

- capsule oracle 是现有机械臂链段的近似几何，不等于完整 mesh self-collision；最终部署仍需 PhysX 或真实碰撞模型复核。
- 自适应区间加密是 continuous collision checking 的离散近似，不提供解析 swept-volume 保证。
- 本模块不能避免机械臂与 H1、人体、障碍物或另一条机械臂的碰撞。
- 256 点提高了轨迹表示和检测分辨率，但控制器仍需执行 action clipping、速度/加速度限制和 emergency stop。
- 真实硬件验证必须重新评估通信延迟、关节跟踪误差、结构柔性及 capsule 半径误差。

本阶段不包含对比实验；输出报告仅用于功能和训练过程验证。
