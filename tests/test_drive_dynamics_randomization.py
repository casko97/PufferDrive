from pathlib import Path

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive


MAP_DIR = str(Path(__file__).parents[1] / "resources" / "drive" / "binaries")


def make_env(*, enabled=True, action_type="continuous", episode_length=80):
    return Drive(
        num_agents=1,
        num_maps=1,
        map_dir=MAP_DIR,
        sequential_map_sampling=True,
        init_mode="create_only_controlled",
        control_mode="control_sdc_only",
        dynamics_model="jerk",
        dynamics_randomization=enabled,
        action_type=action_type,
        episode_length=episode_length,
        goal_behavior=1,
        resample_frequency=0,
        collision_behavior=0,
        offroad_behavior=0,
    )


def test_randomization_requires_jerk_dynamics():
    with pytest.raises(ValueError, match="requires dynamics_model='jerk'"):
        Drive(dynamics_model="classic", dynamics_randomization=True)


def test_disabled_randomization_uses_nominal_conditions():
    env = make_env(enabled=False)
    try:
        env.reset(seed=7)
        np.testing.assert_array_equal(env.get_dynamics_conditions(), np.ones((1, 4), dtype=np.float32))
    finally:
        env.close()


def test_mixed_uniform_ranges_resampling_and_seed_reproducibility():
    env = make_env()
    try:
        samples = []
        for seed in range(64):
            env.reset(seed=seed)
            samples.append(env.get_dynamics_conditions()[0].copy())
        samples = np.asarray(samples)

        np.testing.assert_array_less(0.8 - 1e-6, samples[:, :2])
        np.testing.assert_array_less(samples[:, :2], 1.25 + 1e-6)
        np.testing.assert_array_less((2.0 / 3.0) - 1e-6, samples[:, 2:])
        np.testing.assert_array_less(samples[:, 2:], 1.5 + 1e-6)

        # X(a) chooses the lower and upper uniform components with equal
        # probability. A modest seeded sample should exercise both branches.
        assert np.all(np.any(samples < 1.0, axis=0))
        assert np.all(np.any(samples >= 1.0, axis=0))
        assert np.unique(samples, axis=0).shape[0] > 1

        env.reset(seed=1234)
        first = env.get_dynamics_conditions().copy()
        env.reset(seed=1234)
        second = env.get_dynamics_conditions().copy()
        np.testing.assert_array_equal(first, second)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("action_type", "action"),
    [
        ("continuous", np.asarray([[1.0, 0.0]], dtype=np.float32)),
        # Joint discrete action: longitudinal index 3 (jerk +4), lateral
        # index 1 (jerk 0), encoded as 3 * 3 + 1.
        ("discrete", np.asarray([[10]], dtype=np.int32)),
    ],
)
def test_throttle_gain_applies_after_action_decoding(action_type, action):
    env = make_env(action_type=action_type)
    try:
        env.reset(seed=11)
        throttle = env.get_dynamics_conditions()[0, 0]
        env.step(action)

        # Positive longitudinal acceleration is normalized by +4 in obs[7].
        actual_acceleration = env.observations[0, 7] * 4.0
        expected_acceleration = 4.0 * throttle * env.dt
        assert actual_acceleration == pytest.approx(expected_acceleration, abs=1e-5)
    finally:
        env.close()


def test_steering_gain_applies_to_lateral_jerk():
    env = make_env(action_type="continuous")
    action = np.asarray([[0.0, 0.01]], dtype=np.float32)
    try:
        env.reset(seed=17)
        steering = env.get_dynamics_conditions()[0, 1]
        env.step(action)

        # The small command stays below the steering-rate limit, so obs[8]
        # exposes the gain-scaled lateral acceleration directly.
        actual_lateral_acceleration = env.observations[0, 8] * 4.0
        expected_lateral_acceleration = 4.0 * 0.01 * steering * env.dt
        assert actual_lateral_acceleration == pytest.approx(expected_lateral_acceleration, abs=1e-5)
    finally:
        env.close()


def test_positive_acceleration_uses_randomized_upper_clip():
    env = make_env(action_type="continuous")
    action = np.asarray([[1.0, 0.0]], dtype=np.float32)
    try:
        env.reset(seed=19)
        acceleration_scale = env.get_dynamics_conditions()[0, 2]

        for _ in range(20):
            env.step(action)

        actual_acceleration = env.observations[0, 7] * 4.0
        assert actual_acceleration == pytest.approx(2.5 * acceleration_scale, abs=1e-5)
    finally:
        env.close()


def test_forward_speed_uses_randomized_upper_clip():
    env = make_env(action_type="continuous", episode_length=200)
    action = np.asarray([[1.0, 0.0]], dtype=np.float32)
    try:
        env.reset(seed=23)
        velocity_scale = env.get_dynamics_conditions()[0, 3]

        for _ in range(150):
            env.step(action)

        actual_speed = env.observations[0, 2] * 100.0
        assert actual_speed == pytest.approx(20.0 * velocity_scale, abs=1e-4)
    finally:
        env.close()
