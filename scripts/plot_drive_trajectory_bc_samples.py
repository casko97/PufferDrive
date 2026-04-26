#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pufferlib.ocean.drive.trajectory_bc import TrajectoryBCEnvConfig
from pufferlib.ocean.drive.trajectory_bc_viz import render_trajectory_bc_dataset_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trajectory-BC observation/target sanity plots for multiple `.bin` scenarios.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/casko/phd-code/pufferdrive-kth/datasets/nuplanCarBostonAll_training"),
        help="Directory containing `.bin` scenarios.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/test-viz/trajectory_bc_samples"),
        help="Directory where plots will be written.",
    )
    parser.add_argument("--num-scenarios", type=int, default=4, help="How many scenarios to render.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into the discovered scenario list.")
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Logged timestep to visualize. Leave unset to auto-pick one with full history and future horizon.",
    )
    args = parser.parse_args()

    outputs = render_trajectory_bc_dataset_plots(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_scenarios=args.num_scenarios,
        timestep=args.timestep,
        start_index=args.start_index,
        env_config=TrajectoryBCEnvConfig(),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
