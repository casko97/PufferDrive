import argparse
import os
import struct
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


EXTENSION_MAGIC = 0x54524C52  # "TRLR"


@dataclass
class ObjectRecord:
    scenario_id: int
    entity_type: int
    entity_id: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    heading: np.ndarray
    valid: np.ndarray
    width: float
    length: float
    source_track_id_hash: int = 0
    is_trailer: int = 0
    parent_track_index: int = -1


@dataclass
class RoadRecord:
    scenario_id: int
    entity_type: int
    entity_id: int
    x: np.ndarray
    y: np.ndarray


def _read_i32(file_obj):
    return struct.unpack("<i", file_obj.read(4))[0]


def _read_f32_array(file_obj, n):
    return np.array(struct.unpack(f"<{n}f", file_obj.read(4 * n)), dtype=np.float32)


def _read_i32_array(file_obj, n):
    return np.array(struct.unpack(f"<{n}i", file_obj.read(4 * n)), dtype=np.int32)


def _object_corners(x, y, heading, length, width):
    hl = 0.5 * length
    hw = 0.5 * width
    corners_local = np.array(
        [
            [hl, hw],
            [hl, -hw],
            [-hl, -hw],
            [-hl, hw],
        ],
        dtype=np.float32,
    )
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = corners_local @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def parse_map_binary(binary_path):
    with open(binary_path, "rb") as file_obj:
        sdc_track_index = _read_i32(file_obj)
        num_tracks_to_predict = _read_i32(file_obj)
        tracks_to_predict = [_read_i32(file_obj) for _ in range(num_tracks_to_predict)]

        num_objects = _read_i32(file_obj)
        num_roads = _read_i32(file_obj)

        objects = []
        for _ in range(num_objects):
            scenario_id = _read_i32(file_obj)
            entity_type = _read_i32(file_obj)
            entity_id = _read_i32(file_obj)
            trajectory_length = _read_i32(file_obj)

            x = _read_f32_array(file_obj, trajectory_length)
            y = _read_f32_array(file_obj, trajectory_length)
            z = _read_f32_array(file_obj, trajectory_length)

            # Velocity arrays exist in the binary, but are not needed for plotting.
            _ = _read_f32_array(file_obj, trajectory_length)
            _ = _read_f32_array(file_obj, trajectory_length)
            _ = _read_f32_array(file_obj, trajectory_length)

            heading = _read_f32_array(file_obj, trajectory_length)
            valid = _read_i32_array(file_obj, trajectory_length)

            width = struct.unpack("<f", file_obj.read(4))[0]
            length = struct.unpack("<f", file_obj.read(4))[0]

            # height + goal xyz + mark_as_expert
            file_obj.seek((4 * 4) + 4, os.SEEK_CUR)

            objects.append(
                ObjectRecord(
                    scenario_id=scenario_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    x=x,
                    y=y,
                    z=z,
                    heading=heading,
                    valid=valid,
                    width=width,
                    length=length,
                )
            )

        roads = []
        for _ in range(num_roads):
            scenario_id = _read_i32(file_obj)
            entity_type = _read_i32(file_obj)
            entity_id = _read_i32(file_obj)
            array_size = _read_i32(file_obj)

            x = _read_f32_array(file_obj, array_size)
            y = _read_f32_array(file_obj, array_size)

            # z + scalar payload (width, length, height, goal xyz, mark_as_expert)
            file_obj.seek((4 * array_size) + (6 * 4) + 4, os.SEEK_CUR)

            roads.append(
                RoadRecord(
                    scenario_id=scenario_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    x=x,
                    y=y,
                )
            )

        extension = {
            "present": False,
            "version": None,
            "has_ego_trailer": 0,
            "ego_trailer_track_index": -1,
        }

        remaining = file_obj.read()
        if len(remaining) >= 20:
            cursor = 0
            magic = struct.unpack_from("<i", remaining, cursor)[0]
            cursor += 4
            if magic == EXTENSION_MAGIC:
                extension["present"] = True
                extension["version"] = struct.unpack_from("<i", remaining, cursor)[0]
                cursor += 4
                extension["has_ego_trailer"] = struct.unpack_from("<i", remaining, cursor)[0]
                cursor += 4
                extension["ego_trailer_track_index"] = struct.unpack_from("<i", remaining, cursor)[0]
                cursor += 4
                object_meta_count = struct.unpack_from("<i", remaining, cursor)[0]
                cursor += 4

                for idx in range(min(object_meta_count, len(objects))):
                    source_hash = struct.unpack_from("<Q", remaining, cursor)[0]
                    cursor += 8
                    is_trailer = struct.unpack_from("<i", remaining, cursor)[0]
                    cursor += 4
                    parent_track_index = struct.unpack_from("<i", remaining, cursor)[0]
                    cursor += 4
                    objects[idx].source_track_id_hash = source_hash
                    objects[idx].is_trailer = is_trailer
                    objects[idx].parent_track_index = parent_track_index

    return {
        "sdc_track_index": sdc_track_index,
        "tracks_to_predict": tracks_to_predict,
        "objects": objects,
        "roads": roads,
        "extension": extension,
    }


def _draw_roads(ax, roads):
    for road in roads:
        if road.entity_type == 6:
            ax.plot(road.x, road.y, color="black", linewidth=1.5, alpha=0.9, zorder=1)
        else:
            ax.plot(road.x, road.y, color="0.75", linewidth=1.0, alpha=0.6, zorder=0)


def _draw_object(ax, obj, t, color, label, zorder):
    if t >= len(obj.valid) or obj.valid[t] <= 0:
        return
    corners = _object_corners(obj.x[t], obj.y[t], obj.heading[t], obj.length, obj.width)
    poly = Polygon(corners, closed=True, facecolor=color, edgecolor=color, alpha=0.35, linewidth=1.5, zorder=zorder)
    ax.add_patch(poly)
    ax.plot(obj.x[t], obj.y[t], marker="o", markersize=3, color=color, zorder=zorder + 1)
    if label:
        ax.text(obj.x[t], obj.y[t], label, color=color, fontsize=8, zorder=zorder + 1)


def plot_trailer_scene(binary_path, output_prefix, frames, plot_all_objects=True):
    data = parse_map_binary(binary_path)
    objects = data["objects"]
    roads = data["roads"]
    sdc_idx = data["sdc_track_index"]
    trailer_idx = data["extension"]["ego_trailer_track_index"] if data["extension"]["has_ego_trailer"] else -1

    max_t = max(len(obj.x) for obj in objects) - 1 if objects else 0
    frames = [min(max(0, int(t)), max_t) for t in frames]

    use_progress = len(frames) > 1
    frame_iter = frames
    progress_mode = None
    if use_progress:
        try:
            from tqdm import tqdm

            frame_iter = tqdm(frames, desc="Rendering frames", unit="frame")
            progress_mode = "tqdm"
        except Exception:
            frame_iter = frames
            progress_mode = "simple"

    total_frames = len(frames)
    for idx, t in enumerate(frame_iter, start=1):
        fig, ax = plt.subplots(figsize=(11, 9))
        _draw_roads(ax, roads)

        if plot_all_objects:
            for idx, obj in enumerate(objects):
                if idx in (sdc_idx, trailer_idx):
                    continue
                _draw_object(ax, obj, t, color="0.55", label=None, zorder=2)

        if 0 <= sdc_idx < len(objects):
            _draw_object(ax, objects[sdc_idx], t, color="tab:blue", label="tractor", zorder=4)

        if 0 <= trailer_idx < len(objects):
            _draw_object(ax, objects[trailer_idx], t, color="tab:orange", label="trailer", zorder=5)

        if 0 <= sdc_idx < len(objects) and 0 <= trailer_idx < len(objects):
            tractor = objects[sdc_idx]
            trailer = objects[trailer_idx]
            if t < len(tractor.valid) and t < len(trailer.valid) and tractor.valid[t] > 0 and trailer.valid[t] > 0:
                ax.plot(
                    [tractor.x[t], trailer.x[t]],
                    [tractor.y[t], trailer.y[t]],
                    linestyle="--",
                    color="tab:orange",
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=6,
                )

        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            f"{os.path.basename(binary_path)} | t={t} | has_ego_trailer={data['extension']['has_ego_trailer']}"
        )

        handles = [
            plt.Line2D([0], [0], color="black", lw=1.5, label="road_edge (type=6)"),
            plt.Line2D([0], [0], color="tab:blue", lw=2, label="tractor"),
            plt.Line2D([0], [0], color="tab:orange", lw=2, label="trailer"),
        ]
        ax.legend(handles=handles, loc="best")

        if len(frames) == 1:
            out_path = output_prefix
        else:
            root, ext = os.path.splitext(output_prefix)
            ext = ext or ".png"
            out_path = f"{root}_t{t:03d}{ext}"

        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        if not use_progress:
            print(f"Wrote {out_path}")
        elif progress_mode == "simple":
            print(f"\rRendering frames: {idx}/{total_frames}", end="", flush=True)

    if use_progress and progress_mode == "simple":
        print("")


def main():
    parser = argparse.ArgumentParser(description="Plot trailer/tractor positions and road edges from a map .bin")
    parser.add_argument("--bin", required=True, help="Path to map binary (e.g. map_000.bin)")
    parser.add_argument("--output", required=True, help="Output image path or prefix")
    parser.add_argument(
        "--frames",
        default="0",
        help="Comma-separated timesteps to visualize, e.g. 0,1,5,10",
    )
    parser.add_argument(
        "--no-other-objects",
        action="store_true",
        help="Only draw tractor/trailer and roads",
    )
    args = parser.parse_args()

    frame_list = [int(v.strip()) for v in args.frames.split(",") if v.strip()]
    if not frame_list:
        raise ValueError("--frames must contain at least one integer")

    plot_trailer_scene(
        binary_path=args.bin,
        output_prefix=args.output,
        frames=frame_list,
        plot_all_objects=(not args.no_other_objects),
    )


if __name__ == "__main__":
    main()
