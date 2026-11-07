#!/usr/bin/env python3
"""Render the fixed real-map embodiment-conflict resolution regression."""
from pathlib import Path
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive.embodiment_resolution_diagnostics import plot_embodiment_resolution


def main():
    env = Drive(num_agents=1, num_maps=1, map_dir="pufferlib/resources/drive/binaries",
                sequential_map_sampling=True, init_mode="create_only_controlled",
                control_mode="control_vehicles", dynamics_model="jerk",
                embodiment_randomization=True, action_type="continuous",
                episode_length=91, init_steps=70, resample_frequency=0)
    try:
        env.reset(seed=3)
        output_path = Path("artifacts/embodiment_resolution_map_000.svg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = plot_embodiment_resolution(
            env.get_global_agent_state(), env.get_embodiment_resolution(), output_path,
            map_file=Path("pufferlib/resources/drive/binaries/map_000.bin"),
        )
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
