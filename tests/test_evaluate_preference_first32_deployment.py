import json
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_preference_first32_deployment as deploy_script
from scripts.evaluate_preference_train_val_dataset import RewardBundle


class FakeRewardModel:
    def r_hat_member(self, windows, member: int):
        arr = np.asarray(windows, dtype=np.float32)
        values = arr[:, :, :1] + float(member) * 0.01
        return torch.as_tensor(values, dtype=torch.float32)


def _rollout(map_name: str, steps: int, obs_value: float, scenario_type: str = "turning"):
    return {
        "map_name": map_name,
        "scenario_type": scenario_type,
        "delta_heading_deg": 70.0 if scenario_type == "turning" else 0.0,
        "steps": steps,
        "observations": np.full((steps, 2), obs_value, dtype=np.float32),
        "actions": np.zeros((steps,), dtype=np.int64),
        "task_rewards": np.zeros((steps,), dtype=np.float32),
        "x": np.arange(steps, dtype=np.float32),
        "y": np.zeros((steps,), dtype=np.float32),
    }


def _write_payload(path: Path, model_name: str, rollouts):
    torch.save({"format": "synthetic_rollouts", "model_name": model_name, "rollouts": rollouts}, path)


def test_first32_deployment_scores_dataset_orientation_and_skips_short_rollouts(tmp_path):
    policy_car = tmp_path / "policy_car.pt"
    policy_truck = tmp_path / "policy_truck.pt"
    gt_car = tmp_path / "gt_car.pt"
    gt_truck = tmp_path / "gt_truck.pt"

    _write_payload(policy_car, "car-policy", [_rollout("map_a.bin", 2, 0.0), _rollout("map_b.bin", 1, 0.0)])
    _write_payload(policy_truck, "truck-policy", [_rollout("map_a.bin", 2, 2.0), _rollout("map_b.bin", 2, 2.0)])
    _write_payload(gt_car, "gt-car", [_rollout("map_a.bin", 2, 0.5), _rollout("map_b.bin", 2, 3.0)])
    _write_payload(gt_truck, "gt-truck", [_rollout("map_a.bin", 2, 1.5), _rollout("map_b.bin", 2, 1.0)])

    train_val_summary = tmp_path / "train_val_summary.json"
    train_val_summary.write_text(
        json.dumps(
            {
                "selected": {
                    "splits": {
                        "train": {"accuracy": 1.0, "mean_margin": 2.0},
                        "validation": {"accuracy": 0.5, "mean_margin": 1.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    reward_dir = tmp_path / "reward"
    reward_dir.mkdir()
    bundle = RewardBundle(
        obs_dim=2,
        action_dim=1,
        size_segment=2,
        ensemble_size=2,
        reward_model=FakeRewardModel(),
    )

    outputs = deploy_script.evaluate_deployment(
        policy_car=policy_car,
        policy_truck=policy_truck,
        gt_car=gt_car,
        gt_truck=gt_truck,
        reward_dir=reward_dir,
        train_val_summary_path=train_val_summary,
        output_dir=tmp_path / "out",
        checkpoint_stem="offline_truck_context",
        device="cpu",
        window_len=2,
        bundle=bundle,
    )

    for output in outputs.values():
        assert Path(output).exists()

    summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    policy_stats = summary["comparisons"]["policy_truck_vs_car"]
    gt_stats = summary["comparisons"]["gt_truck_context_vs_car_fit"]
    assert policy_stats["ok_count"] == 1
    assert policy_stats["too_short_count"] == 1
    assert policy_stats["accuracy"] == 1.0
    assert gt_stats["ok_count"] == 2

    scored = torch.load(outputs["scored"], map_location="cpu", weights_only=False)
    assert scored["format"] == deploy_script.SCORED_FORMAT_VERSION
    assert any(row["status"] == "too_short" for row in scored["rows"])
