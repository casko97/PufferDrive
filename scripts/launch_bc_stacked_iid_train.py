#!/usr/bin/env python3
import argparse

from pufferlib.ocean.drive.drive import load_drive_builder_config, train_bc_policy


def main():
    parser = argparse.ArgumentParser(description="Launch stacked-IID BC training with simple overrides.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-shuffle-buffer", type=int, default=4)
    parser.add_argument("--max-maps", type=int, default=-1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--index-log-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    config = load_drive_builder_config(args.config)
    config["train"]["device"] = args.device
    config["train"]["learning_rate"] = args.learning_rate
    config["bc_train"]["device"] = args.device
    config["bc_train"]["output_dir"] = args.output_dir
    config["bc_train"]["epochs"] = args.epochs
    config["bc_train"]["batch_size"] = args.batch_size
    config["bc_train"]["num_workers"] = args.num_workers
    config["bc_train"]["shard_shuffle_buffer"] = args.shard_shuffle_buffer
    config["bc_train"]["max_maps"] = args.max_maps
    config["bc_train"]["val_fraction"] = args.val_fraction
    config["bc_train"]["log_interval"] = args.log_interval
    config["bc_train"]["index_log_interval"] = args.index_log_interval
    config["bc_train"]["learning_rate"] = args.learning_rate
    train_bc_policy(config)


if __name__ == "__main__":
    main()
