from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build_truck_context_preferences import build_truck_context_preferences
from scripts.export_paired_offline_fits import (
    CAR_ROOT,
    DEFAULT_OUTPUT as DEFAULT_EXPORT_OUTPUT,
    TRUCK_ROOT,
    export_paired_fits,
)
from scripts.train_offline_truck_context_reward import DEFAULT_OUTPUT_DIR as DEFAULT_REWARD_OUTPUT_DIR, train_offline_truck_context_reward

DEFAULT_PREFERENCE_OUTPUT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")


def main():
    parser = argparse.ArgumentParser(description="Run the truck-context offline preference pipeline.")
    parser.add_argument(
        "--stage",
        choices=("fit", "preferences", "reward_train", "all"),
        default="all",
        help="Pipeline stage to execute.",
    )
    parser.add_argument("--car-root", type=Path, default=CAR_ROOT)
    parser.add_argument("--truck-root", type=Path, default=TRUCK_ROOT)
    parser.add_argument("--fit-output", type=Path, default=DEFAULT_EXPORT_OUTPUT)
    parser.add_argument("--preference-output", type=Path, default=DEFAULT_PREFERENCE_OUTPUT)
    parser.add_argument("--reward-output-dir", type=Path, default=DEFAULT_REWARD_OUTPUT_DIR)
    parser.add_argument("--window-len", type=int, default=32)
    parser.add_argument("--max-start-distance-m", type=float, default=1.0)
    parser.add_argument("--min-gap-seconds", type=float, default=2.0)
    parser.add_argument("--turning-threshold-deg", type=float, default=None)
    parser.add_argument("--observation-mode", type=str, default="default")
    parser.add_argument("--action-type", type=str, default="discrete")
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mb-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    args = parser.parse_args()

    run_fit = args.stage in ("fit", "all")
    run_preferences = args.stage in ("preferences", "all")
    run_reward_train = args.stage in ("reward_train", "all")

    fit_output = args.fit_output
    pref_output = args.preference_output

    if run_fit:
        export_paired_fits(
            car_root=args.car_root,
            truck_root=args.truck_root,
            output_path=fit_output,
            max_maps=(args.max_maps if args.max_maps > 0 else None),
        )
        print(fit_output)

    if run_preferences:
        if not fit_output.exists():
            raise FileNotFoundError(f"Fit export not found: {fit_output}")
        build_truck_context_preferences(
            export_path=fit_output,
            output_path=pref_output,
            window_len=args.window_len,
            max_start_distance_m=args.max_start_distance_m,
            min_time_diff_seconds=args.min_gap_seconds,
            turning_threshold_deg=args.turning_threshold_deg,
            observation_mode=args.observation_mode,
            action_type=args.action_type,
        )
        print(pref_output)

    if run_reward_train:
        if not pref_output.exists():
            raise FileNotFoundError(f"Preference artifact not found: {pref_output}")
        summary = train_offline_truck_context_reward(
            preference_path=pref_output,
            output_dir=args.reward_output_dir,
            ensemble_size=args.ensemble_size,
            rounds=args.rounds,
            lr=args.lr,
            mb_size=args.mb_size,
            train_batch_size=args.train_batch_size,
        )
        print(summary["output_dir"])


if __name__ == "__main__":
    main()
