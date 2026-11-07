#!/usr/bin/env python3
"""Render a map-anchored agent-agent embodiment-resolution illustration."""
from pathlib import Path
import math
import numpy as np

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
        base = env.get_global_agent_state()
        heading = float(base["heading"][0])
        # Place two map_000 vehicle footprints 7.9 metres apart longitudinally.
        # The priority-selected orange vehicle reaches its original-map fallback
        # while still overlapping, so the next round downsizes the blue bus.
        states = {
            "x": np.asarray([base["x"][0], base["x"][0] + 7.9 * math.cos(heading)], dtype=np.float32),
            "y": np.asarray([base["y"][0], base["y"][0] + 7.9 * math.sin(heading)], dtype=np.float32),
            "heading": np.asarray([heading, heading], dtype=np.float32),
            "id": np.asarray([int(base["id"][0]), int(base["id"][1])], dtype=np.int32),
        }
        resolution = [
            (5, 5, 5, 5, 5, 5, 4, 7, 5.286, 2.332, 2.33, 3.172),
            (5, 4, 3, 2, 1, 0, 0, 7, 4.445, 1.979, 1.670, 2.667),
        ]
        output_path = Path("artifacts/embodiment_agent_agent_resolution_map_000.svg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = plot_embodiment_resolution(
            states, resolution, output_path,
            map_file=Path("pufferlib/resources/drive/binaries/map_000.bin"),
            title="Map-anchored agent-agent conflict: handoff downsizes blue after orange reaches fallback",
        )
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
