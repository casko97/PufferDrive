# Tractor–Trailer Ego Vehicle

PufferDrive can represent the scenario self-driving car (SDC) as a tractor–trailer combination. The tractor and trailer remain separate oriented rectangular `Entity` objects for rendering, observations, collision detection, and off-road checks. The policy controls only the tractor; the trailer pose is derived from tractor motion and the rigid hitch constraint.

## Build and Run

### Environment Setup

The examples below use the project environment used during implementation:

```bash
uv venv pufferdrive-november --python 3.12
source pufferdrive-november/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force
```

The Drive map directory must contain `map_000.bin` and any other requested maps.

### Generate .mp4

Make sure to have your trailer-configured `pufferlib/config/ocean/drive.ini` file in place.

````
mkdir -p tractor_trailer_rollouts

for run in $(seq 1 5); do
    marker=$(mktemp)
    puffer eval puffer_drive \
        --load-model-path pufferlib/resources/drive/pufferdrive_weights.pt \
        --train.seed "$run"

    video=$(find . -maxdepth 1 -name '*.mp4' -newer "$marker" -print -quit)
    mv "$video" "tractor_trailer_rollouts/rollout_${run}.mp4"
    rm -f "$marker"
done
```

## Configuring the Simulation

Set `ego_vehicle_type="tractor_trailer"` in the `[env]` section of the ini config:

```ini
[env]
ego_vehicle_type = "tractor_trailer"
max_articulation_angle = 1.22173048
control_mode = "control_sdc_only"
action_type = "continuous"
dynamics_model = "classic"
collision_behavior = 1
offroad_behavior = 1
```

Omit `max_articulation_angle` to leave articulation unlimited. `collision_behavior` and `offroad_behavior` use the existing Drive values:

| Value | Behavior |
| ---: | --- |
| 0 | Record the event and continue. |
| 1 | Stop the complete rig. |
| 2 | Remove the complete rig. |

`ego_vehicle_type` accepts:

| Value | Behavior |
| --- | --- |
| `"car"` | Backward-compatible default. No trailer is created. |
| `"tractor_trailer"` | Convert a valid SDC into the tractor and synthesize its trailer. |

`max_articulation_angle` is measured in radians and must be in `(0, pi]`. Use `None`, or omit it, to disable the clamp. The articulation angle is

$$
\alpha = \operatorname{wrap}(\psi_t - \psi).
$$

## Trailer Model

### Vehicle geometry

Define

$$
e(\theta) = \begin{bmatrix}\cos\theta \\ \sin\theta\end{bmatrix}.
$$

The configured distances are:

- `L`: tractor rear axle to tractor front axle.
- `a`: tractor rear axle to hitch, measured backward.
- `b`: trailer axle to hitch, measured forward.
- `c_r`: tractor rear axle to entity center, positive forward.
- `c_t`: trailer axle to entity center, positive forward.

Both hitch distances are axle-referenced. The common hitch position must satisfy

$$
p_h = p_r - a e(\psi) = p_t + b e(\psi_t).
$$

The 0.9 m tractor rear-bumper-to-hitch distance is derived from the 1.5 m rear-axle-to-rear-bumper distance minus `a = 0.6 m`.

Geometry is currently resolved from the hard-coded `DEFAULT_NON_KINEMATIC_VEHICLE_PARAMS` vector. Its field order is part of the implementation contract:

| Index | Field | Default |
| ---: | --- | ---: |
| 0 | Tractor nominal length | 6.3 m |
| 1 | Trailer nominal length | 12.5 m |
| 2 | Tractor width | 2.55 m |
| 3 | Trailer width | 2.55 m |
| 4 | Tractor height | 3.5 m |
| 5 | Trailer height | 3.5 m |
| 6 | Tractor rear axle to hitch, `a` | 0.6 m |
| 7 | Trailer front bumper to hitch | 2.2 m |
| 8 | Tractor rear axle to rear bumper | 1.5 m |
| 9 | Tractor rear axle to front axle, `L` | 3.6 m |
| 10 | Tractor front axle to front bumper | 1.2 m |
| 11 | Trailer axle to rear bumper | 2.0 m |
| 12 | Trailer axle to front bumper | 10.5 m |

Derived dynamics values are

$$
\begin{aligned}
L &= 3.6\ \text{m}, & a &= 0.6\ \text{m}, & b &= 10.5 - 2.2 = 8.3\ \text{m}, \\
c_r &= \frac{(3.6 + 1.2) - 1.5}{2} = 1.65\ \text{m}, &
c_t &= \frac{10.5 - 2.0}{2} = 4.25\ \text{m}.
\end{aligned}
$$

The bumper-derived entity lengths are

$$
\ell_r = 1.5 + 3.6 + 1.2 = 6.3\ \text{m}, \qquad
\ell_t = 2.0 + 10.5 = 12.5\ \text{m}.
$$

The nominal entries are aligned with these bumper-derived lengths.

For axle-to-center offset `c`,

$$
p_{center} = p_{axle} + c e(\psi), \qquad
p_{axle} = p_{center} - c e(\psi).
$$

The oriented rectangle dimensions remain attached to the entity center. Overall length and center pose alone are not used to infer axle or hitch locations.

### Simulation stepping

**Tractor bicycle model:**

The tractor uses its rear axle as the reference point. With curvature

$$
\kappa = \frac{\tan\delta}{L},
$$

its rates are

$$
\dot{x}_r = v\cos\psi, \qquad
\dot{y}_r = v\sin\psi, \qquad
\dot{\psi} = v\kappa.
$$

Existing Drive controls are preserved:

- `classic`: acceleration updates signed speed and steering directly commands the steering angle.
- `jerk`: existing longitudinal/lateral jerk integration is used; steering is rate-limited to 0.6 rad/s and clipped to `[-0.55, 0.55]` radians before articulated integration.
- Continuous and discrete action scaling is unchanged.
- Signed speed supports both forward and reverse motion.

**Hitch and trailer model:**

The tractor-derived hitch position and velocity are

$$
p_h = p_r - a e(\psi),
$$

$$
\dot p_h =
\begin{bmatrix}
\dot x_r + a\dot\psi\sin\psi \\
\dot y_r - a\dot\psi\cos\psi
\end{bmatrix}.
$$

The trailer axle remains `b` behind the hitch:

$$
p_t = p_h - b e(\psi_t).
$$

With no lateral slip at the trailer axle,

$$
\dot\psi_t = \frac{(p_h-p_t)^\perp\cdot\dot p_h}{b^2}, \qquad
\dot p_t = e(\psi_t)\left(\dot p_h\cdot e(\psi_t)\right),
$$

where `(u_x, u_y)^perp = (-u_y, u_x)`.

**RK2 integration and hitch projection:**

The axle states are integrated together with midpoint RK2. Speed and steering are linearly interpolated within the simulator step. The substep count is

$$
N = \max\left(
\left\lceil\frac{|\Delta t|}{0.02\ \mathrm{s}}\right\rceil,
\left\lceil\frac{\max(|v_0|, |v_1|)|\Delta t|}{0.25\ \mathrm{m}}\right\rceil
\right).
$$

The trailer axle is reprojected from the tractor-derived hitch after the RK2 midpoint and after every completed substep. This prevents numerical drift and enforces

$$
p_r - a e(\psi) = p_t + b e(\psi_t)
$$

to floating-point precision.

If configured, the articulation limit clamps `psi_t - psi` after every substep and then reprojects the trailer axle. This is particularly useful during reverse motion.

**Entity pose and velocity write-back:**

After stepping, axle states are converted back to entity-center poses. Center velocities include the rotational contribution from each axle-to-center offset, so observations, rendering, and collision checks receive physically consistent motion. Both entity `vz` values are zero for the planar articulated model.

## Simulator Adaptions

### Entity and policy model

The tractor and trailer are separate `Entity` rectangles. Each stores a center pose `(x, y, z)`, heading, velocity, and dimensions `(length, width, height)`.

The articulated dynamics state instead uses axle reference points:

- Tractor rear-axle pose `(x_r, y_r, z_r, psi)`.
- Trailer axle pose `(x_t, y_t, z_t, psi_t)`.
- Signed longitudinal tractor speed `v`.
- Tractor steering angle `delta`.

The trailer is a static, non-policy actor linked to the tractor. Therefore one rig consumes two actor slots, but only one active-agent slot and one action. `MAX_AGENTS` still limits the combined number of tractor, trailer, and other actors. Self observations and action semantics remain tractor-based.

### Initialization, reset, and respawn

When tractor–trailer mode is enabled, map loading:

1. Locates the SDC through `sdc_track_index`.
2. Validates that it is a vehicle with a valid sample at `init_steps`.
3. Uses the SDC entity as the tractor and applies the articulated dimensions and wheelbase.
4. Synthesizes a separate trailer entity and inserts it before the road entities.
5. For every valid trajectory sample, sets `psi_t = psi` and derives the trailer position so both hitch calculations coincide.
6. Registers the tractor as controlled and the trailer as linked but non-policy-controlled.

With the default geometry and aligned headings, the trailer center starts 6.3 m behind the tractor center.

Reset restores both entity poses and velocities from their trajectories, then reconstructs axle state from the entity centers. Respawning the tractor also resets the trailer pose, velocity, collision flags, stopped/removed flags, and dynamics accumulators.

If the SDC is missing, invalid, not a vehicle, or invalid at `init_steps`, articulated initialization is skipped. The environment remains usable and its articulated-state validity mask is false.

### Collision and off-road behavior

Both oriented rectangles participate independently in Drive checks:

- Each rectangle undergoes vehicle collision and off-road checks.
- A third party can contact only the trailer and still affect the complete rig.
- The intentional tractor/trailer rectangle overlap is ignored only for that linked pair.
- Every other entity can collide normally with either rectangle.

Results are aggregated after both rectangles are checked:

1. A vehicle collision on either rectangle becomes a vehicle collision for both.
2. Otherwise, off-road on either rectangle becomes off-road for both.
3. `STOP` stops both entities and zeros articulated speed and entity velocities.
4. `REMOVE` removes both entities. For vehicle collisions, the external collider follows existing Drive removal behavior.

Vehicle collision has priority over off-road if the two rectangles report different events in the same step.

### Observations and rendering

- The owning tractor policy does not observe its attached trailer as a partner.
- Other controlled agents observe tractor and trailer as separate partner rectangles when in range.
- The owner receives one self observation and supplies one action for the complete rig.
- The trailer is rendered as a separate oriented rectangle.
- It inherits tractor selection and collision colors so the complete rig is highlighted consistently.

### Exporting articulated state

`Drive.get_articulated_ego_state()` returns one row per vectorized simulator environment:

```python
state = env.get_articulated_ego_state()

valid = state["valid"]
tractor_center = state["tractor_center_pose"]
trailer_center = state["trailer_center_pose"]
tractor_axle = state["tractor_rear_axle_pose"]
trailer_axle = state["trailer_axle_pose"]
hitch = state["hitch_position"]
articulation = state["articulation_angle"]
```

| Key | Shape | Contents |
| --- | --- | --- |
| `valid` | `(num_envs,)` | Whether the environment has an articulated SDC. |
| `tractor_center_pose` | `(num_envs, 4)` | Global `[x, y, z, heading]`. |
| `trailer_center_pose` | `(num_envs, 4)` | Global `[x, y, z, heading]`. |
| `tractor_rear_axle_pose` | `(num_envs, 3)` | Global `[x, y, heading]`. |
| `trailer_axle_pose` | `(num_envs, 3)` | Global `[x, y, heading]`. |
| `hitch_position` | `(num_envs, 3)` | Global `[x, y, z]`, derived from the tractor. |
| `tractor_velocity` | `(num_envs, 3)` | Center `[vx, vy, vz]`. |
| `trailer_velocity` | `(num_envs, 3)` | Center `[vx, vy, vz]`. |
| `speed` | `(num_envs,)` | Signed tractor rear-axle speed. |
| `steering_angle` | `(num_envs,)` | Tractor steering angle in radians. |
| `articulation_angle` | `(num_envs,)` | Wrapped `psi_t - psi` in radians. |
| `tractor_dimensions` | `(num_envs, 3)` | `[length, width, height]`. |
| `trailer_dimensions` | `(num_envs, 3)` | `[length, width, height]`. |

Pose coordinates are restored to the global/world frame, matching `get_global_agent_state()`. Invalid rows are zero-filled and have `valid=False`.

## Trailer Specific Tests and Visualizations

| Area | Test |
| --- | --- |
| Geometry and conversions | `tests/test_tractor_trailer_geometry.py` |
| Geometry plot | `tests/test_tractor_trailer_geometry_plot.py` |
| Entity lifecycle | `tests/test_tractor_trailer_lifecycle.py` |
| Coupled dynamics | `tests/test_tractor_trailer_dynamics.py` |
| Native dynamics rollouts | `tests/test_tractor_trailer_dynamics_rollout_plots.py` |
| Collisions, observations, and rendering | `tests/test_tractor_trailer_interactions.py` |
| Interaction plot | `tests/test_tractor_trailer_interaction_plot.py` |
| State export and reset | `tests/test_tractor_trailer_state_export.py` |
| Actual simulator rollouts | `tests/test_tractor_trailer_simulator_rollout_plots.py` |

Run all tractor–trailer tests with:

```bash
source pufferdrive-november/bin/activate
python -m pytest -q tests/test_tractor_trailer_*.py
```

Generate actual `Drive.step()` rollout panels and GIFs with:

```bash
python tests/drive/plot_tractor_trailer_simulator_rollouts.py \
    --output-dir /tmp/tractor_trailer_simulator_rollouts
```

The output contains straight-forward, constant-steering, and signed-reverse scenarios. Frames annotate both entity boxes, axle paths, the shared hitch, signed speed, steering, articulation, and hitch residual.

## Current limitations

- Only the SDC can be converted into a tractor–trailer rig.
- There is at most one articulated rig per simulator environment.
- The 13 geometry values are hard-coded and are not yet Python or INI options.
- Synthesized initialization starts aligned (`psi_t = psi`); arbitrary initial articulation is not a public option.
- The planar model does not include suspension, roll, pitch, tire slip, or hitch compliance.
