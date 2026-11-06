# Domain Randomization

PufferDrive supports optional per-agent domain randomization based on the Gigaflow.

## Dynamics

### Configuration

Add these values under `[env]` in `pufferlib/config/ocean/drive.ini`:

```ini
dynamics_model = jerk
dynamics_randomization = True
```

`dynamics_randomization` toggles the feature. When enabled, it always uses
the full Gigaflow ranges below.

### Sampling and application

At every episode reset, each controlled agent independently samples:

```text
C_throttle, C_steer ~ X(1.25)
C_acc, C_vel        ~ X(1.5)
X(a) = 0.5 U(1/a, 1) + 0.5 U(1, a)
```

The condition remains fixed during the episode, including agent respawns. Longitudinal and lateral jerk commands are multiplied by `C_throttle` and `C_steer` after discrete or continuous action decoding. The resulting limits are:

```text
longitudinal acceleration: [-5, 2.5 * C_acc] m/s²
signed speed:              [-2, 20 * C_vel] m/s
```

This feature requires `dynamics_model = jerk`; the classic model uses different action semantics. It does not change action or observation dimensions. Sampled values are available through `env.get_dynamics_conditions()` in the order throttle, steering, acceleration, velocity.

## Reward

### Configuration

```ini
reward_model = "gigaflow"
reward_randomization = True
```

`reward_model = "pufferdrive"` is the backward-compatible default. The Gigaflow model may also be used with `reward_randomization = False`, which selects the conservative Table A4 evaluation condition.

### Sampling and conditioning

At every episode reset, each controlled agent independently samples the supported Table A2 ranges for goal radius, collision, boundary, comfort, lane alignment, velocity alignment, lane centering, center bias, and reverse weights. The velocity and timestep coefficients are fixed as in the paper. The stop-line term is omitted because PufferDrive does not expose traffic-light phase state.
The 11 supported parameters are normalized to `[-1, 1]` and appended to that agent's ego observation in Table A2 order. Other agents never expose their conditions. Use `env.get_reward_conditions()` to retrieve the raw values.

Conditions remain fixed through agent respawns and are resampled only at episode reset. Enabling the Gigaflow model adds 11 ego features, and the Drive policy automatically sizes its ego encoder accordingly.
Fixed mode uses `[10, 3, 3, 0.05, 0.025, 1, 0.0038, 0, 0.0025, 0.005, 0.000025]`.

### Reward terms

The implementation follows Table A2 for goal completion, speed-scaled collisions, road-boundary contact, acceleration/jerk comfort, lane direction and velocity alignment, biased lane centering, forward motion, reverse motion, and moving-time cost. Final goal collection in Gigaflow mode requires speed at or below 3 m/s.

Every active reward is derived from state that PufferDrive already stores or computes:

| Reward term | PufferDrive source |
| --- | --- |
| Goal | Agent position/speed plus the binary goal position and configured global goal behavior |
| Collision | Existing oriented-box collision detection plus speed captured before collision response |
| Boundary | Vehicle-box intersection with binary `ROAD_EDGE` segments, or no compatible `ROAD_LANE` match |
| Comfort | Runtime longitudinal/lateral acceleration and jerk from the selected dynamics model |
| Lane alignment and centering | Heading error and signed offset from a selected binary `ROAD_LANE` segment |
| Forward progress | Agent speed and the selected `ROAD_LANE` heading |
| Reverse | Velocity projected onto the agent heading |
| Time | Runtime velocity and acceleration state |

Lane alignment, centering, and progress are zero when no real lane segment is selected. In GIGAFLOW mode this state receives the sampled boundary penalty, without invoking PufferDrive's physical STOP/REMOVE off-road response. No reward is inferred from unavailable polygon, traffic-light, or stop-line state.

The paper's stop-line term is intentionally absent until PufferDrive exposes traffic-light phase and violation state.

### Lane/Frenet adaptation

PufferDrive binaries expose directed lane centerlines rather than GIGAFLOW-style lane polygons. The lane-dependent reward state is therefore defined directly on this representation:

1. Search nearby `ROAD_LANE` segments and compute point-to-segment distance.
2. Add a 3 m surcharge when the absolute heading difference exceeds 30 degrees.
3. Select the segment with the smallest adjusted distance, rejecting the selection when it exceeds 4 m.
4. Compute heading error and signed lateral offset from the selected directed segment.

This makes the behavior explicit but introduces discontinuities when nearby centerlines exchange rank and at the 4 m cutoff. The checked-in `map_000.bin` regression covers aligned motion, a heading-dependent lane switch, retained opposite-heading motion, and both sides of the cutoff. Generate its labeled visualization with:

```bash
./.venv/bin/python3 scripts/plot_drive_lane_frenet_cases.py
```

The output is `artifacts/lane_frenet_cases_map_000.svg`.

Classic dynamics derives longitudinal/lateral acceleration and jerk for the comfort term; jerk dynamics uses its native state.

## Embodiment

Todo

*"Embodiment is randomized separately per agent: length U(0.8, 7)m, width U(0.8, 3)m (capped at length), and wheelbase is 60% of length. That spans something pedestrian-like through large vehicles. Vehicles are placed at random valid map locations/orientations and then filtered to form a collision-free initial scene; each receives a randomly sampled route and 0–3 intermediate waypoints"*

