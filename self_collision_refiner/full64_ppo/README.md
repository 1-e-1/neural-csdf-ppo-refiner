# 完整64点 PPO 自碰撞细化实验

该目录与原16控制点细化器隔离。模型直接使用完整 `64×6` 输入轨迹作为64个控制点，首尾固定，PPO输出62个内部点沿 Neural C-SDF 安全梯度的门控系数。该实验不使用分层或自适应控制点。

## 接口

- 输入轨迹：`[B,64,6]`
- Observation：`[B,1976]`
- Action：`[B,62]`
- 输出轨迹：`[B,256,6]`
- Episode：4个 refinement step
- 策略网络 hidden dimension：768
- 参数量：2,158,973

## 执行顺序

```powershell
& 'D:\conda_envs\env_isaaclab\python.exe' h1_1111_grasp/self_collision_refiner/full64_ppo/scripts/generate_midcollision_tasks.py
& 'D:\conda_envs\env_isaaclab\python.exe' h1_1111_grasp/self_collision_refiner/full64_ppo/scripts/train_full64_ppo.py
& 'D:\conda_envs\env_isaaclab\python.exe' h1_1111_grasp/self_collision_refiner/full64_ppo/scripts/evaluate_full64_ppo.py
```

第一版结果见 `outputs/完整64点PPO自碰撞细化实验报告.md`。

加强平滑惩罚并降低动作步长的重训练结果见 `outputs/完整64点PPO平滑重训练v2实验报告.md`。
