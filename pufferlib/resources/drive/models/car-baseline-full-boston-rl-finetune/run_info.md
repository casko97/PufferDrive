# Training Run Info: car-baseline-full-boston-rl-finetune

## Run
- Description: RL finetune (no preference reward) from car-baseline-full-boston checkpoint
- W&B run: jumping-resonance-13 (4ua8yotd)
- W&B project: pufferdrive_preferences
- Launched: 2026-04-18 via launch_both_runs.sh (Run 1 of 2, sequential)
- Finished: 2026-04-19
- Uptime: ~33h
- Final checkpoint: puffer_drive_4ua8yotd.pt

## Base model
- load_model_path: pufferlib/resources/drive/models/car-baseline-full-boston/puffer_drive_my9g62z5/model_puffer_drive_004000.pt

## Deterministic settings
- CUBLAS_WORKSPACE_CONFIG=:4096:8
- PYTHONHASHSEED=42
- --train.torch-deterministic True
- seed=42

## Environment
- Git commit: 7dff1065 (full boston results)
- Python: 3.12.3
- PyTorch: 2.5.1+cu121 (CUDA 12.1)
- GPU: NVIDIA GeForce GTX 1080 Ti (driver 535.288.01)

## Training config
- total_timesteps: 2,000,000,000
- num_agents: 128
- num_workers: 6, num_envs: 6
- map_dir: datasets/nuplanCarBostonAll_training_turning (4044 maps)
- control_mode: control_sdc_only
- sdc_runtime_truck_override: true
- dynamics_model: articulated
- action_type: discrete
- preference_reward: disabled

## Final metrics (at 2.0B steps, epoch 81381)
- completion_rate: 0.953
- collision_rate: 0.068
- offroad_rate: 0.067
- lane_alignment_rate: 0.837
- episode_return: 0.902
- speed_at_goal: 12.984
- SPS: ~63.5K
