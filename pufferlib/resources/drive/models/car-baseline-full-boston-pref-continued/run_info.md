# Training Run Info: car-baseline-full-boston-pref-continued

## Run
- Description: RL finetune with preference reward from car-baseline-full-boston-truck-preference checkpoint
- W&B run: xdfgve2e
- W&B project: pufferdrive_preferences
- Launched: 2026-04-19 via launch_both_runs.sh (Run 2 of 2, sequential)
- Finished: 2026-04-20
- Uptime: ~15h 18m
- Final checkpoint: puffer_drive_xdfgve2e.pt

## Base model
- load_model_path: pufferlib/resources/drive/models/car-baseline-full-boston-truck-preference/puffer_drive_15jn4zsd.pt

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
- total_timesteps: 1,000,000,000
- num_agents: 128
- num_workers: 6, num_envs: 6
- map_dir: datasets/nuplanCarBostonAll_training_turning (4044 maps)
- control_mode: control_sdc_only
- sdc_runtime_truck_override: true
- dynamics_model: articulated
- action_type: discrete

## Preference reward config
- enabled: true
- model_dir: pufferlib/resources/drive/preferences/models/turning_all_90_10_30rounds_continue1
- checkpoint_stem: offline_truck_context
- beta: 0.01
- normalize_mode: zscore_baseline
- scale: 1.0
- clip_min: -0.5, clip_max: 0.5
- calibration_steps: 5000

## Final metrics (at 1.0B steps, epoch 40691)
- completion_rate: 0.965
- collision_rate: 0.079
- offroad_rate: 0.063
- lane_alignment_rate: 0.859
- episode_return: 0.906
- preference_reward_mean: 0.308
- speed_at_goal: 12.903
- SPS: ~30.1K
