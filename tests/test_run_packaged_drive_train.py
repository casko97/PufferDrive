from pathlib import Path

from scripts.run_packaged_drive_train import _load_packaged_config, _overlay_args


def test_load_packaged_config_reads_preference_reward_section(tmp_path):
    config_path = tmp_path / "packaged.ini"
    config_path.write_text(
        "\n".join(
            [
                "[base]",
                "env_name = puffer_drive",
                "[preference_reward]",
                "enabled = true",
                'model_dir = "outputs/reward_model/example"',
                'checkpoint_stem = "offline_truck_context"',
                "beta = 0.05",
            ]
        ),
        encoding="utf-8",
    )

    data = _load_packaged_config(config_path)

    assert data["preference_reward"]["enabled"] is True
    assert data["preference_reward"]["model_dir"] == "outputs/reward_model/example"
    assert data["preference_reward"]["checkpoint_stem"] == "offline_truck_context"
    assert data["preference_reward"]["beta"] == 0.05


def test_overlay_args_merges_preference_reward_section():
    base_args = {
        "wandb": False,
        "rnn_name": None,
        "train": {"use_rnn": False},
        "preference_reward": {
            "enabled": False,
            "model_dir": "old",
            "checkpoint_stem": "offline_truck_context",
        },
    }
    packaged = {
        "preference_reward": {
            "enabled": True,
            "model_dir": str(Path("outputs/reward_model/example")),
            "checkpoint_stem": "custom_stem",
        }
    }

    args = _overlay_args(base_args, packaged)

    assert args["preference_reward"]["enabled"] is True
    assert args["preference_reward"]["model_dir"] == "outputs/reward_model/example"
    assert args["preference_reward"]["checkpoint_stem"] == "custom_stem"
