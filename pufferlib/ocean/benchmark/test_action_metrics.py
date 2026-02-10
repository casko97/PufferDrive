"""Sanity checks for action log-likelihood metrics.

This verifies that sequence likelihood (LLR) is positive for smooth action
sequences and near zero for shuffled (order-destroyed) sequences.
"""

import numpy as np

from pufferlib.ocean.benchmark import estimators


def _generate_markov_sequences(
    rng: np.random.Generator,
    num_agents: int,
    num_rollouts: int,
    num_steps: int,
    num_bins: int,
    stay_prob: float,
) -> np.ndarray:
    """Generate categorical sequences with temporal dependence."""
    seqs = np.empty((num_agents, num_rollouts, num_steps), dtype=np.int32)
    for a in range(num_agents):
        for r in range(num_rollouts):
            s = np.empty(num_steps, dtype=np.int32)
            s[0] = rng.integers(num_bins)
            for t in range(1, num_steps):
                if rng.random() < stay_prob:
                    s[t] = s[t - 1]
                else:
                    s[t] = rng.integers(num_bins)
            seqs[a, r] = s
    return seqs


def _compute_llr(gt: np.ndarray, sim: np.ndarray, num_bins: int, smoothing: float):
    marg = estimators.log_likelihood_estimate_categorical_timeseries(
        log_values=gt,
        sim_values=sim,
        num_bins=num_bins,
        additive_smoothing=smoothing,
    )
    seq = estimators.log_likelihood_estimate_conditional_categorical_timeseries(
        log_values=gt,
        sim_values=sim,
        num_bins=num_bins,
        additive_smoothing=smoothing,
    )
    llr = seq - marg
    return np.nanmean(marg), np.nanmean(seq), np.nanmean(llr)


def test_action_llr_signs():
    rng = np.random.default_rng(0)
    num_agents = 2
    num_rollouts = 200
    num_steps = 60
    num_bins = 5
    stay_prob = 0.9
    smoothing = 0.1

    sim = _generate_markov_sequences(rng, num_agents, num_rollouts, num_steps, num_bins, stay_prob)

    gt = _generate_markov_sequences(rng, num_agents, 1, num_steps, num_bins, stay_prob)[:, 0, :]
    gt_shuffled = gt.copy()
    for a in range(num_agents):
        rng.shuffle(gt_shuffled[a])

    marg_s, seq_s, llr_s = _compute_llr(gt, sim, num_bins, smoothing)
    marg_sh, seq_sh, llr_sh = _compute_llr(gt_shuffled, sim, num_bins, smoothing)

    # Marginal should be insensitive to shuffling
    assert abs(marg_s - marg_sh) < 1e-6

    # Sequence likelihood should reward smooth sequences
    assert seq_s > seq_sh + 0.05

    # LLR should be positive for smooth sequences and reduced when order is destroyed
    assert llr_s > 0.05
    assert llr_s - llr_sh > 0.5


if __name__ == "__main__":
    test_action_llr_signs()
    print("Action metric sanity checks passed.")
