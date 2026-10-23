# Method scope

Primary method: self_collision_refiner/. The 16-control-point and full64_ppo variants are retained. The shared src/ kinematics and geometry are support code.

Primary source entries:

- `self_collision_refiner/full64_ppo/scripts/_bootstrap.py`
- `self_collision_refiner/full64_ppo/scripts/evaluate_full64_ppo.py`
- `self_collision_refiner/full64_ppo/scripts/generate_midcollision_tasks.py`
- `self_collision_refiner/full64_ppo/scripts/generate_random_endpoint_safe_tasks.py`
- `self_collision_refiner/full64_ppo/scripts/train_full64_ppo.py`
- `self_collision_refiner/scripts/_bootstrap.py`
- `self_collision_refiner/scripts/evaluate_integrated_pipeline.py`
- `self_collision_refiner/scripts/generate_csdf_dataset.py`
- `self_collision_refiner/scripts/refine_trajectory.py`
- `self_collision_refiner/scripts/train_csdf.py`
- `self_collision_refiner/scripts/train_refiner_ppo.py`
- `self_collision_refiner/self_collision_refiner/__init__.py`
- `self_collision_refiner/self_collision_refiner/csdf_data.py`
- `self_collision_refiner/self_collision_refiner/csdf_training.py`
- `self_collision_refiner/self_collision_refiner/geometry.py`
- `self_collision_refiner/self_collision_refiner/models.py`
- `self_collision_refiner/self_collision_refiner/ppo.py`
- `self_collision_refiner/self_collision_refiner/trajectory_env.py`
