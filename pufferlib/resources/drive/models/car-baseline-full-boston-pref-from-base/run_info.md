# Training Run Info: car-baseline-full-boston-pref-from-base

## Run
- Description: Preference reward finetuning from car-baseline-full-boston base model
- W&B project: pufferdrive_preferences
- Launched: 2026-04-22

## Base model
- load_model_path: pufferlib/resources/drive/models/car-baseline-full-boston/puffer_drive_my9g62z5/model_puffer_drive_004000.pt

## Deterministic settings
- CUBLAS_WORKSPACE_CONFIG=:4096:8
- PYTHONHASHSEED=42
- --train.torch-deterministic True
- seed=42

## Training config
- total_timesteps: 2,000,000,000
- num_agents: 256
- num_workers: 6, num_envs: 6
- batch_size: 49152, minibatch_size: 3072
- bptt_horizon: 32
- update_epochs: 2
- learning_rate: 0.001
- map_dir: datasets/nuplanCarBostonAll_training_turning (4044 maps)
- control_mode: control_sdc_only
- sdc_runtime_truck_override: true
- dynamics_model: articulated
- action_type: discrete

## Preference reward
- model_dir: pufferlib/resources/drive/preferences/models/turning_all_90_10_30rounds_continue1
- checkpoint_stem: offline_truck_context
- beta: 0.01
- normalize_mode: zscore_baseline
- calibration_steps: 5000

## Notes
- Restarted from scratch (from base model) on 2026-04-22 to add missing determinism env vars (CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED) that were absent in the initial launch.
