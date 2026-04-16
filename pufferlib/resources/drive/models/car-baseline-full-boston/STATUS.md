# Car Baseline Full Boston — Training Status

## Status: ❌ Incomplete (crashed ~78.6% through)

## Run Details

| Field | Value |
|---|---|
| Run ID | `my9g62z5` |
| Wandb project | `pufferdrive` |
| Start date | ~Apr 14, 2026 |
| Last checkpoint | Apr 15, 2026 01:51 |
| Agent steps completed | 1,572,864,000 / 2,000,000,000 (~78.6%) |
| Last epoch | 4,000 |
| Dataset | `nuplanCarBostonAll_training` (37,565 maps) |
| Eval dataset | `nuplanCarBostonAll_validation` (4,173 maps) |

## Config Highlights

- **Policy**: `Drive` with `Recurrent` RNN (hidden_size=256)
- **Optimizer**: Muon (lr=0.003)
- **Action type**: discrete
- **Dynamics model**: articulated
- **Observation mode**: `sdc_only_with_trailer` (sdc_runtime_truck_override=True)
- **Batch size**: 393,216
- **Minibatch size**: 24,576
- **BPTT horizon**: 32
- **Num agents**: 2,048
- **Num envs**: 6 (Multiprocessing, 6 workers)
- **Gamma**: 0.98, GAE lambda: 0.95
- **Entropy coef**: 0.005
- **Checkpoint interval**: 1,000 epochs

## Checkpoints

- `model_puffer_drive_001000.pt`
- `model_puffer_drive_002000.pt`
- `model_puffer_drive_003000.pt`
- `model_puffer_drive_004000.pt` ← latest

## What Happened

The tmux session running this training died between epoch 4,000 and completion.
A follow-up wandb run (`8x6zmnv8`, Apr 15 04:17) was initiated but crashed
immediately — no checkpoints were produced for that run.

No final exported `.pt` file exists at `experiments/puffer_drive_my9g62z5.pt`,
confirming the run did not complete the full training loop and close cleanly.

## To Resume

Restart training from the last checkpoint:
```bash
puffer train puffer_drive --load-model-path pufferlib/resources/drive/models/car-baseline-full-boston/puffer_drive_my9g62z5/model_puffer_drive_004000.pt
```
