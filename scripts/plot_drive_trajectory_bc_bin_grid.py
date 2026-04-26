#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.drive.trajectory_bc import TrajectoryBCEnvConfig
from pufferlib.ocean.drive.trajectory_bc_viz import render_trajectory_bc_single_bin_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a subplot grid of all extracted BC samples from one `.bin` scenario."
    )
    parser.add_argument("--map-path", type=Path, required=True, help="Path to the `.bin` scenario.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("tests/test-viz/trajectory_bc_single_bin_grid.png"),
        help="Where to save the rendered grid plot.",
    )
    parser.add_argument(
        "--min-history-seconds",
        type=float,
        default=1.0,
        help="Minimum required history duration for a plotted sample.",
    )
    parser.add_argument(
        "--min-future-seconds",
        type=float,
        default=1.0,
        help="Minimum required future duration for a plotted sample.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=3.2,
        help="Stride between successive plotted anchor timesteps.",
    )
    parser.add_argument(
        "--road-source",
        type=str,
        choices=("map", "observation", "observation_half_length"),
        default="observation",
        help=(
            "Use decoded map geometry, the observation road segments as-is, "
            "or interpret the stored observation length as a half-segment length."
        ),
    )
    args = parser.parse_args()

    output = render_trajectory_bc_single_bin_grid(
        map_path=args.map_path,
        output_path=args.output_path,
        min_history_seconds=args.min_history_seconds,
        min_future_seconds=args.min_future_seconds,
        stride_seconds=args.stride_seconds,
        env_config=TrajectoryBCEnvConfig(),
        road_source=args.road_source,
    )
    print(output)


if __name__ == "__main__":
    main()
