import math

import numpy as np
import pytest

from pufferlib.ocean.drive import binding


GOAL = 0
COLLISION = 1
BOUNDARY = 2
COMFORT = 3
LANE_ALIGN = 4
LANE_CENTER = 5
PROGRESS = 6
REVERSE = 7
TIME = 8
DENSE_TOTAL = 9


def reward_terms(*, condition=None, state=None, collision_state=0):
    condition_values = np.zeros(11, dtype=np.float32)
    state_values = np.zeros(13, dtype=np.float32)
    state_values[0] = 0.2  # dt
    state_values[8] = 100.0  # distance to goal
    if condition:
        for index, value in condition.items():
            condition_values[index] = value
    if state:
        for index, value in state.items():
            state_values[index] = value
    if not state or 11 not in state:
        state_values[11] = abs(state_values[1])  # collision speed captured before response
    return np.asarray(
        binding._test_gigaflow_reward_terms(condition_values, state_values, collision_state), dtype=np.float32
    )


def test_gigaflow_goal_reward_radius_speed_and_one_shot_gate():
    condition = {0: 10.0}
    assert reward_terms(condition=condition, state={1: 3.0, 8: 9.999})[GOAL] == 1.0
    assert reward_terms(condition=condition, state={1: 3.0, 8: 10.0})[GOAL] == 0.0
    assert reward_terms(condition=condition, state={1: 3.001, 8: 9.0})[GOAL] == 0.0
    assert reward_terms(condition=condition, state={1: 0.0, 8: 9.0, 9: 1.0})[GOAL] == 0.0
    assert reward_terms(condition=condition, state={1: 12.0, 8: 9.0, 10: 1.0})[GOAL] == 1.0


def test_gigaflow_collision_reward_has_fixed_and_speed_scaled_penalty():
    inactive = reward_terms(condition={1: 1.7}, state={1: 4.0})
    active = reward_terms(condition={1: 1.7}, state={1: 4.0}, collision_state=1)
    assert inactive[COLLISION] == 0.0
    assert active[COLLISION] == pytest.approx(-(1.7 + 0.1 * 4.0))



def test_gigaflow_collision_reward_retains_impact_speed_after_stop():
    # STOP_AGENT clears vx/vy after collision detection. The reward must use the
    # speed captured at impact rather than the now-zero motion state.
    active = reward_terms(condition={1: 1.7}, state={1: 0.0, 11: 4.0}, collision_state=1)
    assert active[COLLISION] == pytest.approx(-(1.7 + 0.1 * 4.0))


def test_gigaflow_boundary_reward_applies_only_to_offroad_contact():
    inactive = reward_terms(condition={2: 2.3})
    active = reward_terms(condition={2: 2.3}, collision_state=2)
    assert inactive[BOUNDARY] == 0.0
    assert active[BOUNDARY] == pytest.approx(-2.3)


def test_gigaflow_comfort_reward_counts_acceleration_axes_and_combined_jerk_event():
    condition = {3: 0.08}
    at_threshold = reward_terms(condition=condition, state={4: 3.0, 5: -3.0, 6: 5.0, 7: -5.0})
    three_events = reward_terms(condition=condition, state={4: 3.1, 5: -3.2, 6: 5.1, 7: 8.0})
    assert at_threshold[COMFORT] == 0.0
    assert three_events[COMFORT] == pytest.approx(-3.0 * 0.08)


def test_gigaflow_lane_alignment_reward_penalizes_wrong_way_motion():
    condition = {4: 0.01, 5: 0.4}
    aligned = reward_terms(condition=condition, state={1: 5.0, 2: 0.0})
    wrong_way = reward_terms(condition=condition, state={1: 5.0, 2: math.pi})
    aligned_expected = 0.01 * 0.2 * 0.0025
    wrong_way_expected = 0.01 * 0.2 * (-1.0 - 0.4 * 5.0 - 0.0025)
    assert aligned[LANE_ALIGN] == pytest.approx(aligned_expected)
    assert wrong_way[LANE_ALIGN] == pytest.approx(wrong_way_expected)


def test_gigaflow_lane_center_reward_uses_sampled_center_bias():
    condition = {6: 0.006, 7: 0.25}
    centered = reward_terms(condition=condition, state={2: 0.0, 3: 0.25})
    offset = reward_terms(condition=condition, state={2: 0.0, 3: 0.75})
    centered_expected = 0.006 * 0.2 * 0.05 / math.exp(-0.5)
    offset_expected = -0.006 * 0.2 * (0.5 - 0.05 * math.exp(0.0))
    assert centered[LANE_CENTER] == pytest.approx(centered_expected)
    assert offset[LANE_CENTER] == pytest.approx(offset_expected)


def test_gigaflow_progress_reward_requires_speed_and_forward_alignment():
    condition = {8: 0.0025}
    at_threshold = reward_terms(condition=condition, state={1: 2.5, 2: 0.0})
    forward = reward_terms(condition=condition, state={1: 4.0, 2: 0.0})
    wrong_way = reward_terms(condition=condition, state={1: 4.0, 2: math.pi})
    assert at_threshold[PROGRESS] == 0.0
    assert forward[PROGRESS] == pytest.approx(0.0025 * 0.2)
    assert wrong_way[PROGRESS] == 0.0


def test_no_lane_applies_boundary_penalty_and_disables_lane_rewards():
    condition = {2: 2.3, 4: 0.01, 5: 0.4, 6: 0.006, 7: 0.25, 8: 0.0025}
    terms = reward_terms(condition=condition, state={1: 4.0, 2: 0.0, 3: 0.75, 12: -1.0})
    assert terms[LANE_ALIGN] == 0.0
    assert terms[BOUNDARY] == pytest.approx(-2.3)
    assert terms[LANE_CENTER] == 0.0
    assert terms[PROGRESS] == 0.0


def test_gigaflow_reverse_reward_uses_signed_speed():
    condition = {9: 0.005}
    assert reward_terms(condition=condition, state={1: 1.0})[REVERSE] == 0.0
    assert reward_terms(condition=condition, state={1: -1.0})[REVERSE] == pytest.approx(-0.005 * 0.2)


def test_gigaflow_time_reward_applies_only_while_moving_or_accelerating():
    condition = {10: 2.5e-5}
    stopped = reward_terms(condition=condition)
    moving = reward_terms(condition=condition, state={1: 1.0})
    accelerating = reward_terms(condition=condition, state={4: 0.1})
    expected = -2.5e-5 * 0.2
    assert stopped[TIME] == 0.0
    assert moving[TIME] == pytest.approx(expected)
    assert accelerating[TIME] == pytest.approx(expected)


def test_gigaflow_dense_reward_is_sum_of_dense_components():
    condition = {3: 0.08, 4: 0.01, 5: 0.4, 6: 0.006, 7: 0.25, 8: 0.0025, 9: 0.005, 10: 2.5e-5}
    terms = reward_terms(condition=condition, state={1: -4.0, 2: math.pi, 3: 0.75, 4: 3.1})
    assert terms[DENSE_TOTAL] == pytest.approx(terms[COMFORT : TIME + 1].sum(), abs=1e-7)
