# Continuous Action Space Implementation

## Summary

Successfully implemented support for both discrete and continuous action spaces in PufferDrive's model loading from `.bin` files. The implementation is parameterized by the `action_type` setting in `drive.ini`.

## Changes Made

### 1. **drivenet.h** - Core Network Changes
- Added `action_type` and `action_dim` fields to `DriveNet` struct
- Modified `init_drivenet()` signature to accept `action_type` parameter
- Updated action space initialization logic:
  - **Continuous**: `action_size = 2` (acceleration/jerk_long + steering/jerk_lat)
  - **Discrete**: `action_size = 91` (classic) or `12` (jerk)
- Modified `forward()` function:
  - Changed signature to accept `void *actions` instead of `int *actions`
  - Added branching logic based on `action_type`:
    - **Discrete**: Uses `softmax_multidiscrete()` to sample discrete actions
    - **Continuous**: Uses `tanh()` to bound actions to [-1, 1]
- Only initializes `multidiscrete` layer for discrete actions

### 2. **visualize.c** - Visualizer Updates
- Updated call to `init_drivenet()` to pass `env.action_type` from config

### 3. **env_config.h** - Config Parser (Already Supported)
- Config parser already had support for `action_type` parameter
- Parses both string ("discrete"/"continuous") and numeric (0/1) values

### 4. **drive.ini** - Configuration File
- `action_type` parameter controls the action space:
  - `action_type = discrete` → Discrete actions (0)
  - `action_type = continuous` → Continuous actions (1)

## Test Results

### Test 1: Discrete Action Space
```bash
Model: pufferlib/resources/drive/puffer_drive_weights_disc.bin
Config: action_type = discrete
Weights: 614,364 floats
Result: ✅ SUCCESS
Output: DriveNet initialized: action_type=0 (discrete), action_dim=1, dynamics_model=0 (classic)
        Wrote 182 frames in 5.24 seconds (34.75 FPS)
```

### Test 2: Continuous Action Space
```bash
Model: pufferlib/resources/drive/puffer_drive_weights_cont.bin
Config: action_type = continuous
Weights: 592,005 floats
Result: ✅ SUCCESS
Output: DriveNet initialized: action_type=1 (continuous), action_dim=2, dynamics_model=0 (classic)
        Wrote 182 frames in 5.13 seconds (35.49 FPS)
```

## Usage

### Training with Continuous Actions
1. Set in `drive.ini`:
   ```ini
   action_type = continuous
   ```
2. Train your policy (it will output continuous actions)
3. Export to `.bin` file

### Training with Discrete Actions
1. Set in `drive.ini`:
   ```ini
   action_type = discrete
   ```
2. Train your policy (it will output discrete action indices)
3. Export to `.bin` file

### Running Visualizer
```bash
# For continuous actions
bash scripts/build_ocean.sh visualize fast
./visualize --policy-name path/to/continuous_model.bin --map-name path/to/map.bin

# For discrete actions (change drive.ini first)
bash scripts/build_ocean.sh visualize fast
./visualize --policy-name path/to/discrete_model.bin --map-name path/to/map.bin
```

## Implementation Details

### Continuous Action Output
- Actor network outputs 2 values per agent
- Values are passed through `tanh()` to bound to [-1, 1]
- Actions are stored as `float[num_agents * 2]`
- Environment scales these to actual acceleration/steering ranges

### Discrete Action Output
- Actor network outputs logits (91 for classic, 12 for jerk)
- Logits are passed through `softmax_multidiscrete()` for sampling
- Actions are stored as `int[num_agents]`
- Environment decodes indices to acceleration/steering values

### Weight File Compatibility
- **Discrete model**: Requires actor layer with 91 outputs (classic) or 12 (jerk)
- **Continuous model**: Requires actor layer with 2 outputs
- Models trained with one action type cannot be used with the other
- Weight count difference: ~22,359 weights (discrete has larger actor layer)

## Architecture Compatibility

The implementation maintains backward compatibility:
- Existing discrete models continue to work
- Config file controls behavior at runtime
- No changes needed to Python training code (already supported both)
- C inference now matches Python training capabilities

## Files Modified

1. `pufferlib/ocean/drive/drivenet.h` - Network architecture
2. `pufferlib/ocean/drive/visualize.c` - Visualizer integration
3. `pufferlib/config/ocean/drive.ini` - Configuration (action_type setting)

## Notes

- The `action_type` in config must match the model's training configuration
- Mismatched action types will cause weight loading errors or incorrect behavior
- Both action types work with both dynamics models (classic/jerk)
- Continuous actions use deterministic policy (mean only, no sampling)
