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

Todo

*"The reward is not one fixed scalarization. At the start of an episode, each agent samples its own reward parameters; those parameters are included in that agent’s observation/conditioning vector.
This makes the population heterogeneous: some drivers are strongly risk-averse and smooth; some are impatient, weakly lane-aligned, or willing to reverse. Since an agent cannot observe other agents’ sampled preferences, it must drive defensively around latent behavioural variation. GIGAFLOW reward table (A2)"*

## Embodiment

Todo

*"Embodiment is randomized separately per agent: length U(0.8, 7)m, width U(0.8, 3)m (capped at length), and wheelbase is 60% of length. That spans something pedestrian-like through large vehicles. Vehicles are placed at random valid map locations/orientations and then filtered to form a collision-free initial scene; each receives a randomly sampled route and 0–3 intermediate waypoints"*

