# Direct Paired Offline-Fit BC Training

This runbook covers the direct recurrent behavior cloning path that trains from the paired offline-fit export without first materializing a separate BC shard dataset.

## What This Trains

The first target is a recurrent 91-action Drive policy that imitates fitted car actions:

- Source format: `paired_offline_fits`
- Fit side: `car`
- Observation key: `logged_obs_default`
- Observation shape: `[T, 1120]`
- Target action shape: `[T]`
- Action IDs: integers in `[0, 90]`
- Training windows: `[batch, 32, 1120]`
- Loss: multiclass cross entropy over valid timesteps

The recurrent mask is only used to ignore padded timesteps in short final windows.

## Source Data

The source split is:

```text
/home/casko/phd-code/pufferdrive-kth/datasets/nuplanCarBostonAll_training
```

The paired fit export is:

```text
/home/casko/phd-code/pufferdrive-kth/pufferlib/resources/drive/preferences/offline_fits/nuplan_boston_all_chunk64_paired_offline_fits.pt
```

The trainer does not read old extracted BC files such as `map_000.pt`. It streams from the paired offline-fit export. That export is itself physically sharded into 653 source-fit shard files, so progress messages that mention shards refer to the source artifact, not to a materialized BC dataset.

## Config

The packaged config is:

```text
pufferlib/config/ocean/drive_bc_paired_offline_fits_car.ini
```

The important fields are:

```ini
[env]
action_type = discrete
dynamics_model = articulated
extend_classic_action_space = False
observation_mode = default

[bc_train]
source_format = "paired_offline_fits"
fit_export = "/home/casko/phd-code/pufferdrive-kth/pufferlib/resources/drive/preferences/offline_fits/nuplan_boston_all_chunk64_paired_offline_fits.pt"
source_map_dir = "/home/casko/phd-code/pufferdrive-kth/datasets/nuplanCarBostonAll_training"
fit_side = "car"
obs_key = "logged_obs_default"
seq_len = 32
sequence_stride = 10
index_log_interval = 50
```

## Smoke Tests Run

The implementation was checked with:

```bash
python3 -m py_compile pufferlib/ocean/drive/drive.py tests/test_drive_bc_paired_offline_fits.py
/home/casko/phd-code/PufferDrive/.venv/bin/python3 -m pytest -q tests/test_drive_bc_paired_offline_fits.py
```

A real 128-map CPU smoke run also completed one epoch and wrote `latest.pt`, `best.pt`, and `metrics.json`.

## GPU Run Launched

The longer GPU run was launched from the clean worktree:

```text
/tmp/PufferDrive-bc-worktree
```

Output directory:

```text
/home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-gpu-20260428-103228
```

Log file:

```text
/home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-gpu-20260428-103228/train.log
```

It used:

- Device: `cuda`
- Epochs: `10`
- Batch size: `256`
- Num workers: `2`
- Train maps: `33,809`
- Validation maps: `3,756`

The run was interrupted before GPU epochs started. At the last check, train indexing had completed:

```text
shards=653/653 maps=33809/33809 windows=293831
```

and validation indexing was in progress. Because no epoch had started, this interrupted run should not be treated as a usable trained checkpoint unless the log shows a later successful checkpoint write.

## Relaunch Command

From this branch/worktree, build the extension if needed:

```bash
/home/casko/phd-code/PufferDrive/.venv/bin/python3 setup.py build_ext --inplace
```

Then launch a detached GPU run:

```bash
tmux new-session -d -s bc_car_fit_gpu \
  "cd /tmp/PufferDrive-bc-worktree && \
   PYTHONUNBUFFERED=1 /home/casko/phd-code/PufferDrive/.venv/bin/python3 -c \"from pufferlib.ocean.drive.drive import load_drive_builder_config, train_bc_policy; args=load_drive_builder_config('/tmp/PufferDrive-bc-worktree/pufferlib/config/ocean/drive_bc_paired_offline_fits_car.ini'); args['train']['device']='cuda'; args['bc_train']['device']='cuda'; args['bc_train']['output_dir']='/home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-gpu-RELAUNCH'; args['bc_train']['epochs']=10; args['bc_train']['batch_size']=256; args['bc_train']['num_workers']=2; args['bc_train']['shard_shuffle_buffer']=4; args['bc_train']['max_maps']=-1; args['bc_train']['val_fraction']=0.1; args['bc_train']['log_interval']=25; train_bc_policy(args)\" \
   > /home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-gpu-RELAUNCH/train.log 2>&1"
```

Create the output directory before launching:

```bash
mkdir -p /home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-gpu-RELAUNCH
```

## Monitoring

Use:

```bash
tail -f /home/casko/phd-code/PufferDrive/experiments_bc/bc-car-paired-offline-fits-gpu-RELAUNCH/train.log
nvidia-smi
tmux capture-pane -pt bc_car_fit_gpu
```

During indexing, GPU usage stays low because the process is scanning source-fit shards on CPU. GPU memory and utilization should rise when epoch 1 starts.

The current indexing progress logger prints every `bc_train.index_log_interval` source-fit shards. The paired offline-fit car config defaults this to `50`, which gives regular progress updates during the long pre-training scan.
