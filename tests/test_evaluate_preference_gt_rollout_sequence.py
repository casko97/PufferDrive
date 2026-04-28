import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import evaluate_preference_gt_rollout_sequence as sequence_script
from scripts.evaluate_preference_train_val_dataset import RewardBundle


class FakeRewardModel:
    def r_hat_member(self, windows, member: int):
        arr = np.asarray(windows, dtype=np.float32)
        return torch.as_tensor(arr[:, :, :1] + float(member) * 0.01, dtype=torch.float32)


def _rollout(map_name: str, values: list[float]):
    observations = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    observations = np.concatenate([observations, np.zeros_like(observations)], axis=1)
    steps = len(values)
    return {
        "map_name": map_name,
        "source_map_name": f"source_{map_name}",
        "scenario_type": "turning",
        "delta_heading_deg": 70.0,
        "steps": steps,
        "observations": observations.astype(np.float32),
        "actions": np.zeros((steps,), dtype=np.int64),
        "x": np.arange(steps, dtype=np.float32),
        "y": np.zeros((steps,), dtype=np.float32),
    }


def _write_payload(path: Path, model_name: str, rollouts: list[dict]):
    torch.save({"format": "synthetic_rollouts", "model_name": model_name, "rollouts": rollouts}, path)


def test_gt_rollout_sequence_scores_windows_and_full_trajectory(tmp_path):
    gt_truck = tmp_path / "gt_truck.pt"
    gt_car = tmp_path / "gt_car.pt"
    _write_payload(gt_truck, "gt-truck", [_rollout("map_a.bin", [2, 2, 2, 2, 2])])
    _write_payload(gt_car, "gt-car", [_rollout("map_a.bin", [1, 1, 3, 3, 1])])

    bundle = RewardBundle(
        obs_dim=2,
        action_dim=1,
        size_segment=2,
        ensemble_size=2,
        reward_model=FakeRewardModel(),
    )
    outputs = sequence_script.evaluate_gt_rollout_sequence(
        gt_truck=gt_truck,
        gt_car=gt_car,
        reward_dir=tmp_path / "reward",
        output_dir=tmp_path / "out",
        checkpoint_stem="offline_truck_context",
        device="cpu",
        window_len=2,
        reference_scored=None,
        bundle=bundle,
    )

    for output in outputs.values():
        assert Path(output).exists()

    summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    assert summary["nonoverlap_windows"]["ok_count"] == 2
    assert summary["nonoverlap_windows"]["by_window_index"]["0"]["mean_margin"] == pytest.approx(2.0)
    assert summary["nonoverlap_windows"]["by_window_index"]["1"]["mean_margin"] == pytest.approx(-2.0)
    assert summary["full_trajectory"]["ok_count"] == 1
    assert summary["full_trajectory"]["mean_margin"] == pytest.approx(1.0)

    scored = torch.load(outputs["scored"], map_location="cpu", weights_only=False)
    assert len(scored["window_rows"]) == 2
    assert len(scored["full_rows"]) == 1
    assert len(scored["curve_rows"]) == 5
