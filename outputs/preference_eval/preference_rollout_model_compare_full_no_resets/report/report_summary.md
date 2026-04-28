# Preference Reward Rollout Comparison

- Favored model overall: `car-baseline-full-boston`
- Normalization mode: `offline_sample`
- Calibration steps used: `954` / `5000`

## Overall
- `car-baseline-full-boston`: mean_cumulative_pref_shaped=5.4799, mean_pref_std=0.2109, smoothness=0.2184
- `truck-baseline-full-boston`: mean_cumulative_pref_shaped=3.5478, mean_pref_std=0.2407, smoothness=0.3354

## Turning vs Straight
- `turning`:
  - `car-baseline-full-boston`: mean_cumulative_pref_shaped=6.3720, rollout_count=7
  - `truck-baseline-full-boston`: mean_cumulative_pref_shaped=3.9588, rollout_count=7
- `straight`:
  - `car-baseline-full-boston`: mean_cumulative_pref_shaped=3.3982, rollout_count=3
  - `truck-baseline-full-boston`: mean_cumulative_pref_shaped=2.5889, rollout_count=3

## Interpretation
- `mean_cumulative_pref_shaped` estimates which rollout style the reward model prefers on the sampled scenes.
- `mean_pref_std` summarizes ensemble disagreement; higher values indicate greater uncertainty.
- `smoothness` is the mean absolute step-to-step change in shaped reward; higher values indicate noisier reward traces.
