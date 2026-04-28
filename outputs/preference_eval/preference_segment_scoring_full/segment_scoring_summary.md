# Segment Scoring Summary

32-step matched sliding-window comparisons using the reward model as a true segment scorer.

## gt_car_vs_truck_context
- map_count: `10`
- mean_window_margin: `1.3386`
- mean_left_win_fraction: `0.6322`
- mean_pair_probability_left_beats_right: `0.6134`
- mean_top_k_margin: `1.9222`

## policy_car_vs_truck
- map_count: `6`
- mean_window_margin: `3.4109`
- mean_left_win_fraction: `0.7487`
- mean_pair_probability_left_beats_right: `0.6936`
- mean_top_k_margin: `3.3717`

| Comparison | Map | Type | Windows | Mean margin | Median margin | Left win frac | Pair prob | Top-k margin |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gt_car_vs_truck_context | map_11089.bin | straight | 59 | 1.409 | 2.190 | 0.746 | 0.712 | 1.178 |
| gt_car_vs_truck_context | map_11239.bin | turning | 59 | -3.610 | -3.858 | 0.085 | 0.118 | -3.305 |
| policy_car_vs_truck | map_13634.bin | turning | 22 | 2.070 | 3.544 | 0.636 | 0.533 | -2.420 |
| gt_car_vs_truck_context | map_13634.bin | turning | 59 | 2.724 | -0.127 | 0.492 | 0.666 | 9.664 |
| policy_car_vs_truck | map_13872.bin | turning | 20 | 1.581 | 1.583 | 0.850 | 0.637 | 2.540 |
| gt_car_vs_truck_context | map_13872.bin | turning | 59 | 0.759 | 0.866 | 0.627 | 0.589 | -0.063 |
| gt_car_vs_truck_context | map_14822.bin | straight | 59 | 3.270 | 3.834 | 1.000 | 0.891 | 3.862 |
| policy_car_vs_truck | map_15185.bin | turning | 22 | 11.871 | 13.489 | 1.000 | 0.988 | 11.990 |
| gt_car_vs_truck_context | map_15185.bin | turning | 59 | 5.328 | 4.688 | 1.000 | 0.902 | 2.311 |
| policy_car_vs_truck | map_20434.bin | turning | 10 | -0.915 | -1.202 | 0.300 | 0.371 | -0.611 |
| gt_car_vs_truck_context | map_20434.bin | turning | 59 | -1.143 | -1.091 | 0.407 | 0.359 | -2.253 |
| gt_car_vs_truck_context | map_21823.bin | turning | 59 | 0.930 | 0.820 | 1.000 | 0.694 | 1.867 |
| policy_car_vs_truck | map_23497.bin | turning | 17 | 2.026 | 3.459 | 0.706 | 0.686 | 4.898 |
| gt_car_vs_truck_context | map_23497.bin | turning | 59 | 3.999 | 6.022 | 0.780 | 0.771 | 5.706 |
| policy_car_vs_truck | map_4685.bin | straight | 3 | 3.833 | 3.720 | 1.000 | 0.947 | 3.833 |
| gt_car_vs_truck_context | map_4685.bin | straight | 59 | -0.279 | -0.134 | 0.186 | 0.433 | 0.255 |
