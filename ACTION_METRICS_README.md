# Action Log-Likelihood Metrics (Implementation Notes)

This document summarizes the action log-likelihood metrics implemented in this repo, the design choices, limitations, and how to extend them later (state conditioning, joint coupling, etc.).

**What was implemented**
- New discrete action metrics based on rollouts:
- `likelihood_action_long_marginal`
- `likelihood_action_long_sequence`
- `likelihood_action_long_ratio`
- `likelihood_action_lat_marginal`
- `likelihood_action_lat_sequence`
- `likelihood_action_lat_ratio`

These are computed separately for longitudinal and lateral actions (factorized), using inferred ground-truth actions from kinematics. State conditioning is not included yet.

**Where it lives**
- Metrics core logic: `pufferlib/ocean/benchmark/evaluator.py`
- Categorical estimators: `pufferlib/ocean/benchmark/estimators.py`
- Action inference and helpers: `pufferlib/ocean/benchmark/metrics.py`
- Sanity test: `pufferlib/ocean/benchmark/test_action_metrics.py`

**Action space context**
- Current eval config: `pufferlib/config/ocean/drive.ini` sets `action_type = discrete` and `dynamics_model = classic`.
- Classic discrete action space is joint 7 x 13 (accel x steering).
- Jerk discrete action space is joint 4 x 3 (not supported yet for inference).
- Action decoding in the env uses: `action_val = long_idx * num_lat + lat_idx`.

**Ground-truth inputs used**
We do not have direct GT actions, so they are inferred from GT trajectories:
- GT trajectories come from `Drive.get_ground_truth_trajectories()` in `pufferlib/ocean/drive/drive.py`.
- Fields used: `x`, `y`, `heading`, `valid`, plus `agent_length` from `agent_state`.
- Kinematic features computed in `pufferlib/ocean/benchmark/metrics.py`:
- `linear_speed` and `linear_accel` from central differences of (x, y).
- `angular_speed` from wrapped central difference of heading.
- Validity masks derived from `valid` with central logical-and (see `compute_kinematic_validity`).

**GT action inference**
- Longitudinal action index:
- Take `ref_linear_accel`.
- Choose nearest value in `ACCELERATION_VALUES` (classic) or `JERK_LONG_VALUES` (jerk).
- Lateral action index (classic only):
- Use `ref_angular_speed`, `ref_linear_speed`, and `agent_length`.
- Compute predicted yaw rates for each steering value using the same formula as the env:
- `beta = tanh(0.5 * tan(steer))`
- `yaw_rate = (speed * cos(beta) * tan(steer)) / length`
- Pick the steering index whose yaw rate is closest to `ref_angular_speed`.
- If `linear_speed` is near zero, default to steering closest to zero.

**Rollout action collection**
- During evaluation rollouts, actions are captured in `collect_simulated_trajectories()`.
- For discrete classic:
- If policy output is joint index, decode to `(long_idx, lat_idx)`.
- If policy output is already 2D, use it directly.
- These are stored as `action_long` and `action_lat` in the simulated trajectories dict.

**Metrics theory (implemented version)**
For each dimension (long, lat) and each agent:
- Marginal log-likelihood:
- `L_marg = mean_t log p(a_t)`
- Sequence log-likelihood:
- `L_seq = mean_{t>=1} log p(a_t | a_{t-1})`
- Likelihood ratio:
- `L_LLR = mean_{t>=1} [log p(a_t | a_{t-1}) - log p(a_t)]`

Interpretation:
- `L_marg` checks if actions are globally plausible under rollout distribution.
- `L_seq` checks if action transitions look like rollout transitions.
- `L_LLR` measures how much temporal ordering adds beyond the marginal.

These metrics are computed separately for long and lat.

**Smoothing**
- Additive smoothing is applied to categorical histograms to avoid `log(0)`.
- Config key: `eval.action_loglik_smoothing` (default 0.1) in the runtime config dict.
- Tradeoff:
- Higher smoothing: more stable, less sensitive to rare actions.
- Lower smoothing: sharper, but more brittle with sparse data.

**Validity and masking**
- Marginal metrics use validity masks from kinematics:
- Longitudinal uses acceleration validity.
- Lateral uses speed validity.
- Sequence metrics require both current and previous timestep to be valid.
- The first timestep is set to invalid for sequence metrics.

**Outputs**
- New fields are included in per-agent and scene-level results.
- They are not added to the WOSAC meta-score in `wosac.ini`.

**Test added**
File: `pufferlib/ocean/benchmark/test_action_metrics.py`
- Creates smooth Markov sequences for rollouts.
- Compares GT vs shuffled GT:
- Marginal likelihood is unchanged by shuffling.
- Sequence likelihood and LLR decrease when order is destroyed.

Run:
```bash
pufferdrive.venv/bin/python -m pufferlib.ocean.benchmark.test_action_metrics
```

Note: Matplotlib may warn about cache location. It is harmless.

**Known limitations**
- No state conditioning. Metrics are global and can confound state-driven action changes.
- Jerk dynamics not supported for lateral inference (see TODO in `metrics.py`).
- GT actions are inferred from kinematics, not logged.
- Coupling between long and lat is not modeled (factorized only).

**Future extensions**
1. State-conditioned metrics
- Goal: `p(a_t | s_t)` and `p(a_t | s_t, a_{t-1})` with backoff.
- Needed: state abstraction (bucketed features) since state is continuous and high dimensional.
- Suggested buckets: speed bin, heading bin, distance-to-road-edge bin, TTC bin, lane alignment, etc.
- With state buckets, build:
- `p0(a)` global
- `p_marg(a | s)`
- `p_seq(a | s, a_prev)`
- Use interpolation: `p_seq~ = λ2 p_seq + λ1 p_marg + λ0 p0` with λ based on counts.

2. Joint long/lat coupling
- Build joint histograms `p(a_long, a_lat)` and `p(a_long, a_lat | a_prev_long, a_prev_lat)`.
- Use mixture/backoff to combine joint and factorized models.
- Add a coupling diagnostic:
- `log p_joint - log p_long - log p_lat`.

3. Jerk dynamics support
- Define a mapping from jerk actions to angular speed for lateral inference.
- Update `discretize_lateral_actions()` to support `dynamics_model = jerk`.

**Implementation notes for state conditioning**
- Since obs space is continuous, choose a bucketization strategy and document it.
- Keep bucket counts small to avoid sparsity.
- Consider sharing buckets across long and lat for simplicity.
- Always include a global backoff distribution to handle rare states.

