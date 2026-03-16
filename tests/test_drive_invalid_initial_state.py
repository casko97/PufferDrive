import numpy as np
import pytest

import pufferlib.ocean.drive.drive as drive_module
from pufferlib.ocean.drive.drive import Drive


def _make_minimal_drive_instance():
    env = Drive.__new__(Drive)
    env.c_envs = object()
    env.tick = 123
    env.observations = np.zeros((0,), dtype=np.float32)
    return env


@pytest.mark.parametrize("invalid_reason", ["actor_collision", "offroad"])
def test_reset_resamples_once_when_initial_trailer_state_is_invalid(monkeypatch, invalid_reason):
    env = _make_minimal_drive_instance()
    calls = {"vec_reset": 0, "resample": 0}
    invalid_flags = iter([True, False])

    monkeypatch.setattr(
        drive_module.binding,
        "vec_reset",
        lambda c_envs, seed: calls.__setitem__("vec_reset", calls["vec_reset"] + 1),
    )
    monkeypatch.setattr(
        drive_module.binding,
        "vec_has_invalid_initial_trailer_state",
        lambda c_envs: next(invalid_flags, False),
    )
    monkeypatch.setattr(drive_module.np.random, "randint", lambda low, high: 1234)
    monkeypatch.setattr(
        env,
        "_resample_vector_envs",
        lambda seed: calls.__setitem__("resample", calls["resample"] + 1),
    )

    obs, info = Drive.reset(env, seed=7)

    assert calls["vec_reset"] == 1
    assert calls["resample"] == 1
    assert env.tick == 0
    assert obs is env.observations
    assert info == []


@pytest.mark.parametrize("invalid_reason", ["actor_collision", "offroad"])
def test_reset_raises_after_max_invalid_initial_trailer_state_resamples(monkeypatch, invalid_reason):
    env = _make_minimal_drive_instance()
    calls = {"vec_reset": 0, "resample": 0}

    monkeypatch.setattr(
        drive_module.binding,
        "vec_reset",
        lambda c_envs, seed: calls.__setitem__("vec_reset", calls["vec_reset"] + 1),
    )
    monkeypatch.setattr(drive_module.binding, "vec_has_invalid_initial_trailer_state", lambda c_envs: True)
    monkeypatch.setattr(drive_module.np.random, "randint", lambda low, high: 1234)
    monkeypatch.setattr(
        env,
        "_resample_vector_envs",
        lambda seed: calls.__setitem__("resample", calls["resample"] + 1),
    )

    with pytest.raises(RuntimeError, match="Exceeded 8 resample attempts"):
        Drive.reset(env, seed=7)

    assert calls["vec_reset"] == 1
    assert calls["resample"] == 8
