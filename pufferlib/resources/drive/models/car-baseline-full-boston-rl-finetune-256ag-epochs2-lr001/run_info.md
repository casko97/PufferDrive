# Training Run Info: car-baseline-full-boston-rl-finetune-256ag-epochs2-lr001

## Run
- Description: RL finetune with tuned hyperparameters (lower LR, 2 update epochs) to compensate for smaller batch size (256 agents vs 2048 in reference)
- W&B project: pufferdrive_preferences
- Launched: 2026-04-21

## Base model
- load_model_path: pufferlib/resources/drive/models/car-baseline-full-boston/puffer_drive_my9g62z5/model_puffer_drive_004000.pt

## Deterministic settings
- CUBLAS_WORKSPACE_CONFIG=:4096:8
- PYTHONHASHSEED=42
- --train.torch-deterministic True
- seed=42

## Environment
- Git commit: 7e4b3615 (Ignore large train_log.txt files)
- Python: 3.12.3
- PyTorch: 2.5.1+cu121 (CUDA 12.1)
- GPU: NVIDIA GeForce GTX 1080 Ti (driver 535.288.01)

## Key hyperparameter changes (vs car-baseline-full-boston-rl-finetune)
- num_agents: 128 → 256
- batch_size: 24576 → 49152 (256 * 6 * 32)
- minibatch_size: 1536 → 3072 (batch/16)
- update_epochs: 1 → 2
- learning_rate: 0.003 → 0.001

## Hyperparameter selection
Ran 4 experiments at 10M steps each (256 agents) comparing:
1. baseline (epochs=1, lr=0.003): score=0.983, return=0.951
2. epochs=4, lr=0.003: score=0.970, return=0.881
3. epochs=2, lr=0.001: score=0.991, return=0.980 ← selected
4. epochs=4, lr=0.001: score=0.981, return=—
Lower LR + moderate epoch count gave best results.

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
