import sys
from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

from pufferlib.ocean.drive.drive import save_map_binary
from pufferlib.ocean.drive.trajectory_bc import (
    _base_obs_dim,
    _iter_sample_timesteps,
    _resolve_training_device,
    _trajectory_history_feature_dims,
    _trajectory_obs_dim,
    _TRAJECTORY_HISTORY_FEATURES,
    _TRAJECTORY_HORIZON,
    TrajectoryBCEnvConfig,
    TrajectoryBCExperimentConfig,
    TrajectoryBCTrainConfig,
    TrajectoryBCTrainer,
    iter_trajectory_bc_samples,
)
from pufferlib.ocean.drive.trajectory_bc_viz import (
    render_trajectory_bc_dataset_plots,
    render_trajectory_bc_sample_plot,
    render_trajectory_bc_single_bin_grid,
)


def _linear_traj(x0, y0, dx, dy, heading=0.0, length=91):
    return {
        "position": [{"x": float(x0 + dx * t), "y": float(y0 + dy * t), "z": 0.0} for t in range(length)],
        "velocity": [{"x": float(dx / 0.1), "y": float(dy / 0.1), "z": 0.0} for _ in range(length)],
        "heading": [float(heading) for _ in range(length)],
        "valid": [1 for _ in range(length)],
    }


def _curved_traj(x0, y0, dx, amplitude, heading_bias=0.0, length=91):
    xs = []
    ys = []
    headings = []
    velocities = []
    valid = []
    for t in range(length):
        x = float(x0 + dx * t)
        y = float(y0 + amplitude * np.sin(t / 10.0))
        xs.append(x)
        ys.append(y)
        valid.append(1)

    for t in range(length):
        prev_t = max(t - 1, 0)
        next_t = min(t + 1, length - 1)
        vx = (xs[next_t] - xs[prev_t]) / (0.1 * max(next_t - prev_t, 1))
        vy = (ys[next_t] - ys[prev_t]) / (0.1 * max(next_t - prev_t, 1))
        velocities.append({"x": float(vx), "y": float(vy), "z": 0.0})
        headings.append(float(np.arctan2(vy, vx) + heading_bias))

    return {
        "position": [{"x": x, "y": y, "z": 0.0} for x, y in zip(xs, ys)],
        "velocity": velocities,
        "heading": headings,
        "valid": valid,
    }


def _write_simple_trajectory_map(map_dir: Path):
    scenario = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}, {"track_index": 1}]},
        "objects": [
            {
                "id": 100,
                "type": "vehicle",
                "length": 4.5,
                "width": 1.9,
                "height": 1.6,
                **_curved_traj(0.0, 0.0, 0.9, 1.5),
                "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
            },
            {
                "id": 101,
                "type": "vehicle",
                "length": 4.5,
                "width": 1.9,
                "height": 1.6,
                **_curved_traj(8.0, 2.5, 0.75, 0.9, heading_bias=0.03),
                "goalPosition": {"x": 110.0, "y": 6.0, "z": 0.0},
            },
        ],
        "roads": [
            {
                "id": 200,
                "type": "lane",
                "geometry": [{"x": -10.0, "y": -4.0, "z": 0.0}, {"x": 90.0, "y": -4.0, "z": 0.0}],
                "width": 3.5,
                "length": 100.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 201,
                "type": "lane",
                "geometry": [{"x": -10.0, "y": 4.0, "z": 0.0}, {"x": 90.0, "y": 4.0, "z": 0.0}],
                "width": 3.5,
                "length": 100.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 202,
                "type": "road_edge",
                "geometry": [{"x": -10.0, "y": -8.0, "z": 0.0}, {"x": 90.0, "y": -8.0, "z": 0.0}],
                "width": 0.2,
                "length": 100.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 203,
                "type": "road_edge",
                "geometry": [{"x": -10.0, "y": 8.0, "z": 0.0}, {"x": 90.0, "y": 8.0, "z": 0.0}],
                "width": 0.2,
                "length": 100.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }
    save_map_binary(scenario, str(map_dir / "map_000.bin"), unique_map_id=9)
    return scenario


def _write_trailer_trajectory_map(map_dir: Path):
    scenario = {
        "metadata": {
            "sdc_track_index": 0,
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
            "tracks_to_predict": [{"track_index": 0}],
        },
        "objects": [
            {
                "id": 200,
                "type": "vehicle",
                "length": 6.0,
                "width": 2.5,
                "height": 3.2,
                **_linear_traj(0.0, 0.0, 1.0, 0.0),
                "goalPosition": {"x": 100.0, "y": 0.0, "z": 0.0},
            },
            {
                "id": 201,
                "type": "vehicle",
                "length": 10.0,
                "width": 2.5,
                "height": 3.2,
                **_linear_traj(-8.0, 0.0, 1.0, 0.0),
                "goalPosition": {"x": 90.0, "y": 0.0, "z": 0.0},
            },
            {
                "id": 202,
                "type": "vehicle",
                "length": 4.5,
                "width": 1.9,
                "height": 1.6,
                **_linear_traj(10.0, 0.0, 1.0, 0.0),
                "goalPosition": {"x": 110.0, "y": 0.0, "z": 0.0},
            },
        ],
        "roads": [],
    }
    save_map_binary(scenario, str(map_dir / "map_000.bin"), unique_map_id=10)
    return scenario


def test_iter_trajectory_bc_samples_produces_history_observations(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_trajectory_map(map_dir)

    samples = list(iter_trajectory_bc_samples(map_dir / "map_000.bin", TrajectoryBCEnvConfig()))
    assert samples

    observation, target = samples[0]
    assert observation.shape[0] > 160
    assert target.shape == (160,)
    assert target.reshape(32, 5)[0, 4] == pytest.approx(1.0)


def test_iter_trajectory_bc_samples_supports_extended_trailer_history(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_trailer_trajectory_map(map_dir)
    env_config = TrajectoryBCEnvConfig(
        observation_mode="trajectory_history_32_sdc_only_with_trailer",
        control_mode="control_sdc_only",
    )

    samples = list(
        iter_trajectory_bc_samples(
            map_dir / "map_000.bin",
            env_config,
            sample_stride=32,
            sample_start_offset=31,
            sample_end_offset=31,
        )
    )
    assert samples

    observation, target = samples[0]
    assert observation.shape == (_trajectory_obs_dim("classic", env_config.observation_mode),)
    assert target.shape == (160,)

    base_dim = _base_obs_dim("classic", env_config.observation_mode)
    ego_history_features, _partner_history_features = _trajectory_history_feature_dims(env_config.observation_mode)
    ego_hist_start = base_dim
    partner_hist_start = ego_hist_start + (_TRAJECTORY_HORIZON * ego_history_features)
    ego_history = observation[ego_hist_start:partner_hist_start].reshape(
        _TRAJECTORY_HORIZON,
        ego_history_features,
    )

    assert ego_history[0, 6] == pytest.approx(4.0)
    assert ego_history[0, 7] == pytest.approx(-8.0 * 0.02)
    assert ego_history[1, 7] == pytest.approx(-9.0 * 0.02)


def test_iter_trajectory_bc_samples_adds_active_partner_history_type(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_trajectory_map(map_dir)
    env_config = TrajectoryBCEnvConfig(observation_mode="trajectory_history_32_sdc_only_with_trailer")

    samples = list(
        iter_trajectory_bc_samples(
            map_dir / "map_000.bin",
            env_config,
            sample_stride=32,
            sample_start_offset=31,
            sample_end_offset=31,
        )
    )
    assert samples

    observation, _target = samples[0]
    base_dim = _base_obs_dim("classic", env_config.observation_mode)
    ego_history_features, partner_history_features = _trajectory_history_feature_dims(env_config.observation_mode)
    partner_hist_start = base_dim + (_TRAJECTORY_HORIZON * ego_history_features)
    partner_history = observation[partner_hist_start:].reshape(
        -1,
        _TRAJECTORY_HORIZON,
        partner_history_features,
    )

    assert partner_history[0, 0, 5] == pytest.approx(1.0)
    assert partner_history[0, 0, 6] == pytest.approx(1.0)


def test_iter_trajectory_bc_samples_respects_stride_and_offsets(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_trajectory_map(map_dir)

    timesteps = list(
        _iter_sample_timesteps(
            91,
            start_offset=max(4, _TRAJECTORY_HORIZON - 1),
            end_offset=max(8, _TRAJECTORY_HORIZON - 1),
            stride=3,
        )
    )
    samples = list(
        iter_trajectory_bc_samples(
            map_dir / "map_000.bin",
            TrajectoryBCEnvConfig(),
            sample_stride=3,
            sample_start_offset=4,
            sample_end_offset=8,
        )
    )

    assert timesteps
    assert len(samples) == len(timesteps)


def test_iter_trajectory_bc_samples_preserves_dense_history_with_sparse_stride(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_trajectory_map(map_dir)

    samples = list(
        iter_trajectory_bc_samples(
            map_dir / "map_000.bin",
            TrajectoryBCEnvConfig(),
            sample_stride=32,
            sample_start_offset=9,
            sample_end_offset=10,
        )
    )

    assert samples
    observation, _target = samples[0]
    ego_history = observation[
        _base_obs_dim() : _base_obs_dim() + (_TRAJECTORY_HORIZON * _TRAJECTORY_HISTORY_FEATURES)
    ].reshape(_TRAJECTORY_HORIZON, _TRAJECTORY_HISTORY_FEATURES)

    assert int((ego_history[:, 5] > 0.5).sum()) >= 10


def test_resolve_training_device_prefers_mps_when_cuda_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _FakeMPS:
        @staticmethod
        def is_built():
            return True

        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(torch.backends, "mps", _FakeMPS)

    device = _resolve_training_device("cuda")

    assert device.type == "mps"


def test_resolve_training_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _FakeMPS:
        @staticmethod
        def is_built():
            return True

        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(torch.backends, "mps", _FakeMPS)

    device = _resolve_training_device("auto")

    assert device.type == "cpu"


def test_trajectory_bc_visual_sanity_plot(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_trajectory_map(map_dir)
    output_dir = Path("tests/test-viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "trajectory_bc_visual_sanity.png"
    render_trajectory_bc_sample_plot(
        map_path=map_dir / "map_000.bin",
        output_path=plot_path,
        timestep=None,
        env_config=TrajectoryBCEnvConfig(),
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_trajectory_bc_multi_scenario_plots(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for idx in range(3):
        scenario_dir = dataset_dir / f"case_{idx}"
        scenario_dir.mkdir()
        _write_simple_trajectory_map(scenario_dir)
        (scenario_dir / "map_000.bin").rename(scenario_dir / f"map_{idx:03d}.bin")

    output_dir = tmp_path / "plots"
    outputs = render_trajectory_bc_dataset_plots(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        num_scenarios=2,
        timestep=None,
        env_config=TrajectoryBCEnvConfig(),
    )

    assert len(outputs) == 2
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0


def test_trajectory_bc_single_bin_grid_plot(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_trajectory_map(map_dir)

    output_path = tmp_path / "single_bin_grid.png"
    result = render_trajectory_bc_single_bin_grid(
        map_path=map_dir / "map_000.bin",
        output_path=output_path,
        min_history_seconds=1.0,
        min_future_seconds=1.0,
        stride_seconds=3.2,
        env_config=TrajectoryBCEnvConfig(),
        road_source="observation",
    )

    assert result.exists()
    assert result.stat().st_size > 0


def test_trajectory_bc_single_bin_grid_plot_half_length(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    _write_simple_trajectory_map(map_dir)

    output_path = tmp_path / "single_bin_grid_half_length.png"
    result = render_trajectory_bc_single_bin_grid(
        map_path=map_dir / "map_000.bin",
        output_path=output_path,
        min_history_seconds=1.0,
        min_future_seconds=1.0,
        stride_seconds=3.2,
        env_config=TrajectoryBCEnvConfig(),
        road_source="observation_half_length",
    )

    assert result.exists()
    assert result.stat().st_size > 0


def test_trajectory_bc_trainer_smoke(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_simple_trajectory_map(dataset_dir)

    experiment = TrajectoryBCExperimentConfig(
        train=TrajectoryBCTrainConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(tmp_path / "outputs"),
            device="cpu",
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            num_workers=0,
            val_fraction=0.5,
            max_maps=-1,
            save_best=True,
            log_interval=0,
            seed=7,
            max_train_samples_per_epoch=8,
            max_val_samples=4,
        ),
        env=TrajectoryBCEnvConfig(),
        policy={"input_size": 32, "hidden_size": 64},
    )

    trainer = TrajectoryBCTrainer(experiment)
    summary = trainer.train()

    assert summary["history"]
    assert (tmp_path / "outputs" / "last.pt").exists()
    assert (tmp_path / "outputs" / "metrics.json").exists()


def test_trajectory_bc_trainer_extended_trailer_history_smoke(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_trailer_trajectory_map(dataset_dir)

    experiment = TrajectoryBCExperimentConfig(
        train=TrajectoryBCTrainConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(tmp_path / "outputs_extended"),
            device="cpu",
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            num_workers=0,
            val_fraction=0.5,
            max_maps=-1,
            save_best=True,
            log_interval=0,
            seed=7,
            max_train_samples_per_epoch=8,
            max_val_samples=4,
        ),
        env=TrajectoryBCEnvConfig(
            observation_mode="trajectory_history_32_sdc_only_with_trailer",
            control_mode="control_sdc_only",
        ),
        policy={"input_size": 32, "hidden_size": 64},
    )

    trainer = TrajectoryBCTrainer(experiment)
    summary = trainer.train()

    assert summary["history"]
    assert (tmp_path / "outputs_extended" / "last.pt").exists()
    assert (tmp_path / "outputs_extended" / "metrics.json").exists()


def test_trajectory_bc_config_reads_wandb_fields(tmp_path):
    config_path = tmp_path / "trajectory_bc.ini"
    config_path.write_text(
        "\n".join(
            [
                "[policy]",
                "input_size = 32",
                "hidden_size = 64",
                "[env]",
                'observation_mode = "trajectory_history_32"',
                "[bc_train]",
                f'dataset_dir = "{tmp_path}"',
                f'output_dir = "{tmp_path / "outputs"}"',
                "wandb = true",
                'wandb_project = "demo-project"',
                'wandb_group = "demo-group"',
                'wandb_name = "demo-run"',
                'wandb_tag = "demo-tag"',
                'wandb_resume_id = "resume-123"',
            ]
        ),
        encoding="utf-8",
    )

    experiment = TrajectoryBCExperimentConfig.from_ini(config_path)

    assert experiment.train.wandb is True
    assert experiment.train.wandb_project == "demo-project"
    assert experiment.train.wandb_group == "demo-group"
    assert experiment.train.wandb_name == "demo-run"
    assert experiment.train.wandb_tag == "demo-tag"
    assert experiment.train.wandb_resume_id == "resume-123"


def test_trajectory_bc_config_reads_sample_window_fields(tmp_path):
    config_path = tmp_path / "trajectory_bc_sampling.ini"
    config_path.write_text(
        "\n".join(
            [
                "[policy]",
                "input_size = 32",
                "hidden_size = 64",
                "[env]",
                'observation_mode = "trajectory_history_32"',
                "[bc_train]",
                f'dataset_dir = "{tmp_path}"',
                f'output_dir = "{tmp_path / "outputs"}"',
                "sample_stride = 4",
                "sample_start_offset = 6",
                "sample_end_offset = 10",
            ]
        ),
        encoding="utf-8",
    )

    experiment = TrajectoryBCExperimentConfig.from_ini(config_path)

    assert experiment.train.sample_stride == 4
    assert experiment.train.sample_start_offset == 6
    assert experiment.train.sample_end_offset == 10


def test_trajectory_bc_config_reads_explicit_validation_dataset(tmp_path):
    config_path = tmp_path / "trajectory_bc_validation.ini"
    config_path.write_text(
        "\n".join(
            [
                "[policy]",
                "input_size = 32",
                "hidden_size = 64",
                "[env]",
                'observation_mode = "trajectory_history_32"',
                "[bc_train]",
                f'dataset_dir = "{tmp_path / "train"}"',
                f'val_dataset_dir = "{tmp_path / "val"}"',
                f'output_dir = "{tmp_path / "outputs"}"',
                "val_fraction = 0.0",
                "max_maps = 5",
                "max_val_maps = 3",
            ]
        ),
        encoding="utf-8",
    )

    experiment = TrajectoryBCExperimentConfig.from_ini(config_path)

    assert experiment.train.val_dataset_dir == str(tmp_path / "val")
    assert experiment.train.val_fraction == 0.0
    assert experiment.train.max_maps == 5
    assert experiment.train.max_val_maps == 3


def test_trajectory_bc_trainer_uses_explicit_validation_dataset(tmp_path):
    train_dir = tmp_path / "train_dataset"
    val_dir = tmp_path / "val_dataset"
    train_dir.mkdir()
    val_dir.mkdir()
    _write_simple_trajectory_map(train_dir)
    _write_simple_trajectory_map(val_dir)

    experiment = TrajectoryBCExperimentConfig(
        train=TrajectoryBCTrainConfig(
            dataset_dir=str(train_dir),
            val_dataset_dir=str(val_dir),
            output_dir=str(tmp_path / "outputs_explicit_val"),
            device="cpu",
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            num_workers=0,
            val_fraction=0.0,
            max_maps=-1,
            max_val_maps=-1,
            save_best=True,
            log_interval=0,
            seed=7,
            max_train_samples_per_epoch=8,
            max_val_samples=4,
        ),
        env=TrajectoryBCEnvConfig(),
        policy={"input_size": 32, "hidden_size": 64},
    )

    trainer = TrajectoryBCTrainer(experiment)
    summary = trainer.train()

    assert summary["train_map_count"] == 1
    assert summary["val_map_count"] == 1
    assert summary["val_dataset_dir"] == str(val_dir)
    assert "val" in summary["history"][0]


def test_trajectory_bc_trainer_logs_to_wandb(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_simple_trajectory_map(dataset_dir)

    events = {"init": None, "logs": [], "finished": False, "artifacts": [], "defined_metrics": []}

    class _FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path):
            self.files.append(path)

    class _FakeRun:
        def __init__(self):
            self.id = "fake-run-id"
            self.summary = {}

        def log_artifact(self, artifact):
            events["artifacts"].append((artifact.name, artifact.type, tuple(artifact.files)))

        def define_metric(self, name, **kwargs):
            events["defined_metrics"].append((name, dict(kwargs)))

    class _FakeWandb:
        class util:
            @staticmethod
            def generate_id():
                return "generated-id"

        Artifact = _FakeArtifact

        @staticmethod
        def init(**kwargs):
            events["init"] = kwargs
            return _FakeRun()

        @staticmethod
        def log(payload, step=None):
            events["logs"].append((step, dict(payload)))

        @staticmethod
        def finish():
            events["finished"] = True

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)

    experiment = TrajectoryBCExperimentConfig(
        train=TrajectoryBCTrainConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(tmp_path / "outputs_wandb"),
            device="cpu",
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            num_workers=0,
            val_fraction=0.5,
            max_maps=-1,
            save_best=True,
            log_interval=0,
            seed=7,
            max_train_samples_per_epoch=8,
            max_val_samples=4,
            wandb=True,
            wandb_project="proj",
            wandb_group="group",
            wandb_name="run-name",
            wandb_tag="tag-a",
            wandb_resume_id="resume-me",
        ),
        env=TrajectoryBCEnvConfig(),
        policy={"input_size": 32, "hidden_size": 64},
    )

    trainer = TrajectoryBCTrainer(experiment)
    summary = trainer.train()

    assert summary["wandb_run_id"] == "fake-run-id"
    assert events["init"]["project"] == "proj"
    assert events["init"]["group"] == "group"
    assert events["init"]["name"] == "run-name"
    assert events["init"]["id"] == "resume-me"
    assert ("epoch", {}) in events["defined_metrics"]
    assert ("batch_step", {}) in events["defined_metrics"]
    assert ("train/*", {"step_metric": "epoch"}) in events["defined_metrics"]
    assert ("val/*", {"step_metric": "epoch"}) in events["defined_metrics"]
    assert ("train/loss", {"step_metric": "epoch"}) in events["defined_metrics"]
    assert ("val/loss", {"step_metric": "epoch"}) in events["defined_metrics"]
    assert ("train/valid_accuracy", {"step_metric": "epoch"}) in events["defined_metrics"]
    assert ("val/valid_accuracy", {"step_metric": "epoch"}) in events["defined_metrics"]
    assert ("train_batch/*", {"step_metric": "batch_step"}) in events["defined_metrics"]
    assert ("train_batch/loss", {"step_metric": "batch_step"}) in events["defined_metrics"]
    assert ("train_batch/learning_rate", {"step_metric": "batch_step"}) in events["defined_metrics"]
    assert events["logs"]
    batch_logs = [payload for _step, payload in events["logs"] if "train_batch/loss" in payload]
    epoch_logs = [payload for _step, payload in events["logs"] if "train/loss" in payload]
    assert batch_logs
    assert epoch_logs
    assert all(step is None for step, _payload in events["logs"])
    assert events["finished"] is True
