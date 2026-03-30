from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scripts.build_truck_context_preferences import load_preferences_into_reward_model

DEFAULT_INPUT = Path("outputs/preferences/nuplan_boston_test10_truck_context_preferences.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/reward_model")


def _load_reward_model_class():
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from preferences.reward_model import RewardModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RewardModel dependencies are unavailable in this environment. "
            "Install the optional reward-model dependencies (for example scipy and loralib) to run offline training."
        ) from exc
    return RewardModel


def train_offline_truck_context_reward(
    preference_path: Path,
    output_dir: Path,
    ensemble_size: int = 1,
    rounds: int = 1,
    lr: float = 3e-4,
    mb_size: int = 32,
    train_batch_size: int = 32,
    activation: str = "tanh",
) -> dict:
    payload = torch.load(preference_path, map_location="cpu")
    meta = payload["metadata"]
    obs_dim = int(meta["obs_dim"])
    action_dim = int(meta["action_dim"])
    size_segment = int(meta["window_len"])
    total_windows = int(meta["total_windows"])
    if total_windows <= 0:
        raise ValueError(f"No preference windows available in {preference_path}")

    RewardModel = _load_reward_model_class()
    model = RewardModel(
        ds=obs_dim,
        da=action_dim,
        ensemble_size=ensemble_size,
        lr=lr,
        mb_size=mb_size,
        size_segment=size_segment,
        capacity=max(total_windows * 2, total_windows + 1),
        activation=activation,
    )
    model.train_batch_size = train_batch_size
    inserted = load_preferences_into_reward_model(model, payload)

    round_acc = []
    for _ in range(rounds):
        acc = model.train_reward()
        round_acc.append(np.asarray(acc, dtype=np.float32))

    final_acc = model.get_train_acc()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir), "offline_truck_context")

    summary = {
        "preference_path": str(preference_path),
        "output_dir": str(output_dir),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "size_segment": size_segment,
        "total_windows": total_windows,
        "inserted_windows": inserted,
        "ensemble_size": ensemble_size,
        "rounds": rounds,
        "round_acc": [acc.tolist() for acc in round_acc],
        "final_train_acc": float(final_acc),
    }
    summary_path = output_dir / "offline_truck_context_reward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train RewardModel from cached truck-context preferences.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mb-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--activation", type=str, default="tanh")
    args = parser.parse_args()

    summary = train_offline_truck_context_reward(
        preference_path=args.input,
        output_dir=args.output_dir,
        ensemble_size=args.ensemble_size,
        rounds=args.rounds,
        lr=args.lr,
        mb_size=args.mb_size,
        train_batch_size=args.train_batch_size,
        activation=args.activation,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
