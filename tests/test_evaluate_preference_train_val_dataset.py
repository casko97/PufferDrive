import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_preference_train_val_dataset as eval_script


class FakeRewardModel:
    def r_hat_member(self, windows, member: int):
        arr = np.asarray(windows, dtype=np.float32)
        values = arr[:, :, :1] + float(member) * 0.05
        return torch.as_tensor(values, dtype=torch.float32)


def _write_synthetic_preference_dataset(path: Path) -> None:
    preferred = np.asarray(
        [
            [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[3.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
            [[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
            [[4.0, 0.0, 1.0], [4.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    rejected = np.asarray(
        [
            [[0.5, 0.0, 1.0], [0.5, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[1.5, 0.0, 1.0], [1.5, 0.0, 1.0]],
            [[5.0, 0.0, 1.0], [5.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    payload = {
        "metadata": {
            "obs_dim": 2,
            "action_dim": 1,
            "window_len": 2,
            "total_windows": 4,
            "observation_mode": "default",
            "action_encoding": "one_hot",
        },
        "preferred_sa": preferred,
        "rejected_sa": rejected,
        "labels": np.zeros((4, 1), dtype=np.float32),
        "window_metadata": [
            {
                "map_name": "map_a.bin",
                "timestep_start": 20,
                "timestep_end": 22,
                "scenario_delta_heading_deg": 80.0,
            },
            {
                "map_name": "map_a.bin",
                "timestep_start": 0,
                "timestep_end": 2,
                "scenario_delta_heading_deg": 80.0,
            },
            {
                "map_name": "map_b.bin",
                "timestep_start": 0,
                "timestep_end": 2,
                "scenario_delta_heading_deg": -70.0,
            },
            {
                "map_name": "map_c.bin",
                "timestep_start": 0,
                "timestep_end": 2,
                "scenario_delta_heading_deg": 65.0,
            },
        ],
    }
    torch.save(payload, path)


def test_evaluate_first_windows_selects_earliest_per_map_and_writes_outputs(tmp_path):
    preference_path = tmp_path / "preferences.pt"
    _write_synthetic_preference_dataset(preference_path)

    reward_dir = tmp_path / "reward"
    reward_dir.mkdir()
    (reward_dir / "offline_truck_context_reward_summary.json").write_text(
        json.dumps(
            {
                "obs_dim": 2,
                "action_dim": 1,
                "size_segment": 2,
                "total_windows": 4,
                "ensemble_size": 2,
                "observation_mode": "default",
                "action_encoding": "one_hot",
            }
        ),
        encoding="utf-8",
    )

    eval_report = tmp_path / "eval.json"
    eval_report.write_text(
        json.dumps(
            {
                "train_indices": [0, 2],
                "validation_indices": [1, 3],
                "validation_accuracy": 0.5,
                "validation_loss": 0.7,
            }
        ),
        encoding="utf-8",
    )
    preference_config = tmp_path / "preference.ini"
    preference_config.write_text("[preference_reward]\n", encoding="utf-8")

    bundle = eval_script.RewardBundle(
        obs_dim=2,
        action_dim=1,
        size_segment=2,
        ensemble_size=2,
        reward_model=FakeRewardModel(),
    )
    outputs = eval_script.evaluate_first_windows(
        preference_path=preference_path,
        reward_dir=reward_dir,
        eval_report_path=eval_report,
        preference_config_path=preference_config,
        source_paired_fit=None,
        output_dir=tmp_path / "out",
        gallery_count=3,
        bundle=bundle,
    )

    for output in outputs.values():
        assert Path(output).exists()

    summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    assert summary["selected"]["window_count"] == 3
    assert summary["selected"]["splits"]["train"]["window_count"] == 1
    assert summary["selected"]["splits"]["validation"]["window_count"] == 2

    with Path(outputs["window_csv"]).open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert [int(row["global_index"]) for row in rows] == [1, 2, 3]
    assert rows[0]["map_name"] == "map_a.bin"
    assert rows[0]["split"] == "validation"
    assert rows[0]["timestep_start"] == "0"

    scored = torch.load(outputs["scored_windows"], map_location="cpu", weights_only=False)
    assert scored["format"] == eval_script.SCORED_FORMAT_VERSION
    assert len(scored["rows"]) == 3
