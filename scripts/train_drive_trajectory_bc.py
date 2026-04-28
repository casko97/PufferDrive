#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pufferlib.ocean.drive.trajectory_bc import TrajectoryBCExperimentConfig, TrajectoryBCTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the trajectory Drive policy with behavior cloning on `.bin` scenarios.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pufferlib/config/ocean/drive_trajectory_bc.ini"),
        help="Path to the BC training INI config.",
    )
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument("--val-dataset-dir", type=str, default=None, help="Optional validation dataset directory override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. auto, cpu, mps, or cuda.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--max-maps", type=int, default=None, help="Optional map-count limit override.")
    parser.add_argument("--max-val-maps", type=int, default=None, help="Optional validation map-count limit override.")
    parser.add_argument("--max-train-samples-per-epoch", type=int, default=None, help="Optional training sample budget override.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional validation sample budget override.")
    parser.add_argument("--sample-stride", type=int, default=None, help="Optional stride between sampled timesteps.")
    parser.add_argument(
        "--turning-sample-stride",
        type=int,
        default=None,
        help="Optional denser stride override for turning training scenarios; disabled when < 1.",
    )
    parser.add_argument(
        "--turning-threshold-deg",
        type=float,
        default=None,
        help="Heading delta threshold for classifying turning scenarios.",
    )
    parser.add_argument(
        "--turning-manifest-path",
        type=str,
        default=None,
        help="Optional scenario_filter_manifest.json path for identifying turning maps.",
    )
    parser.add_argument(
        "--sample-start-offset",
        type=int,
        default=None,
        help="Optional number of initial timesteps to skip before creating samples.",
    )
    parser.add_argument(
        "--sample-end-offset",
        type=int,
        default=None,
        help="Optional number of final timesteps to skip when creating samples.",
    )
    full_window_group = parser.add_mutually_exclusive_group()
    full_window_group.add_argument(
        "--require-full-windows",
        dest="require_full_windows",
        action="store_true",
        default=None,
        help="Only train on samples with full valid history and prediction windows.",
    )
    full_window_group.add_argument(
        "--allow-partial-windows",
        dest="require_full_windows",
        action="store_false",
        help="Allow samples with partial history or prediction windows.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging for this run.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging even if config enables it.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional wandb project override.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional wandb group override.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional wandb run name override.")
    parser.add_argument("--wandb-tag", type=str, default=None, help="Optional wandb tag override.")
    parser.add_argument("--wandb-resume-id", type=str, default=None, help="Optional wandb run id to resume.")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint path to continue training from.")
    args = parser.parse_args()

    experiment = TrajectoryBCExperimentConfig.from_ini(args.config)
    if args.dataset_dir is not None:
        experiment.train.dataset_dir = args.dataset_dir
    if args.val_dataset_dir is not None:
        experiment.train.val_dataset_dir = args.val_dataset_dir
    if args.output_dir is not None:
        experiment.train.output_dir = args.output_dir
    if args.device is not None:
        experiment.train.device = args.device
    if args.epochs is not None:
        experiment.train.epochs = args.epochs
    if args.batch_size is not None:
        experiment.train.batch_size = args.batch_size
    if args.max_maps is not None:
        experiment.train.max_maps = args.max_maps
    if args.max_val_maps is not None:
        experiment.train.max_val_maps = args.max_val_maps
    if args.max_train_samples_per_epoch is not None:
        experiment.train.max_train_samples_per_epoch = args.max_train_samples_per_epoch
    if args.max_val_samples is not None:
        experiment.train.max_val_samples = args.max_val_samples
    if args.sample_stride is not None:
        experiment.train.sample_stride = args.sample_stride
    if args.turning_sample_stride is not None:
        experiment.train.turning_sample_stride = args.turning_sample_stride
    if args.turning_threshold_deg is not None:
        experiment.train.turning_threshold_deg = args.turning_threshold_deg
    if args.turning_manifest_path is not None:
        experiment.train.turning_manifest_path = args.turning_manifest_path
    if args.sample_start_offset is not None:
        experiment.train.sample_start_offset = args.sample_start_offset
    if args.sample_end_offset is not None:
        experiment.train.sample_end_offset = args.sample_end_offset
    if args.require_full_windows is not None:
        experiment.train.require_full_windows = args.require_full_windows
    if args.wandb:
        experiment.train.wandb = True
    if args.disable_wandb:
        experiment.train.wandb = False
    if args.wandb_project is not None:
        experiment.train.wandb_project = args.wandb_project
    if args.wandb_group is not None:
        experiment.train.wandb_group = args.wandb_group
    if args.wandb_name is not None:
        experiment.train.wandb_name = args.wandb_name
    if args.wandb_tag is not None:
        experiment.train.wandb_tag = args.wandb_tag
    if args.wandb_resume_id is not None:
        experiment.train.wandb_resume_id = args.wandb_resume_id
    if args.resume_from is not None:
        experiment.train.resume_from = args.resume_from

    trainer = TrajectoryBCTrainer(experiment)
    trainer.train()


if __name__ == "__main__":
    main()
