"""Generate the real-map lane/Frenet edge-case visualization."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.lane_frenet_diagnostics import (
    lane_polylines,
    map_000_edge_cases,
    plot_lane_frenet_cases,
)


MAP_FILE = REPO_ROOT / "pufferlib/resources/drive/binaries/map_000.bin"
OUTPUT_FILE = REPO_ROOT / "artifacts/lane_frenet_cases_map_000.svg"


def main() -> None:
    cases = map_000_edge_cases(lane_polylines(MAP_FILE))
    poses = [(case.x, case.y, case.heading) for case in cases]
    results = binding._test_lane_frenet_from_map(str(MAP_FILE), poses)
    output = plot_lane_frenet_cases(MAP_FILE, cases, results, OUTPUT_FILE)
    print(output)


if __name__ == "__main__":
    main()
