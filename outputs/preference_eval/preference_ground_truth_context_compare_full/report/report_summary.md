# Preference Reward Rollout Comparison

- Favored model overall: `ground-truth-car-fit`
- Normalization mode: `offline_sample`
- Calibration steps used: `1800` / `5000`

## Overall
- `ground-truth-car-fit`: mean_cumulative_pref_shaped=-2.9733, mean_pref_std=0.2977, smoothness=0.1133
- `ground-truth-truck-context-replay`: mean_cumulative_pref_shaped=-5.6586, mean_pref_std=0.2994, smoothness=0.1169

## Turning vs Straight
- `turning`:
  - `ground-truth-car-fit`: mean_cumulative_pref_shaped=6.6168, rollout_count=7
  - `ground-truth-truck-context-replay`: mean_cumulative_pref_shaped=4.1891, rollout_count=7
- `straight`:
  - `ground-truth-car-fit`: mean_cumulative_pref_shaped=-25.3504, rollout_count=3
  - `ground-truth-truck-context-replay`: mean_cumulative_pref_shaped=-28.6366, rollout_count=3

## Interpretation
- `mean_cumulative_pref_shaped` estimates which rollout style the reward model prefers on the sampled scenes.
- `mean_pref_std` summarizes ensemble disagreement; higher values indicate greater uncertainty.
- `smoothness` is the mean absolute step-to-step change in shaped reward; higher values indicate noisier reward traces.
