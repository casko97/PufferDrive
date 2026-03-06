import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import shutil
import struct

import pufferlib.ocean.drive.drive as drive_module
from pufferlib.ocean.drive.drive import (
    DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN,
    Drive,
    _load_non_kinematic_vehicle_params_from_bin,
)
from pufferlib.ocean.drive.trailer_viz import parse_map_binary

TESTS_DIR = Path(__file__).resolve().parent
BOSTON_REFERENCE_BIN = (
    TESTS_DIR
    / "artifacts"
    / "drive"
    / "traversing_traffic_light_intersection__97be27351e915863__97be27351e915863.bin"
)
TRAINING_CAR_REFERENCE_BIN = TESTS_DIR / "artifacts" / "drive" / "training_map_024_turning_car.bin"
BOSTON_CAR_HEADER_BIN = (
    TESTS_DIR
    / "artifacts"
    / "drive"
    / "traversing_traffic_light_intersection__97be27351e915863__97be27351e915863__car_header.bin"
)


def _constant_trajectory(x, y, z=0.0, steps=91):
    return [{"x": x, "y": y, "z": z} for _ in range(steps)]


def _constant_scalar(value, steps=91):
    return [value for _ in range(steps)]


def _load_boston_reference_map_and_trajectories(tmp_path):
    """Step 1 helper: load Boston .bin for env use and cache GT tractor/trailer trajectories."""
    if not BOSTON_REFERENCE_BIN.exists():
        pytest.skip(f"Boston reference bin not found: {BOSTON_REFERENCE_BIN}")

    # Prepare a Drive-compatible map directory expected by the C env loader.
    map_dir = tmp_path / "boston_reference_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BOSTON_REFERENCE_BIN, map_dir / "map_000.bin")

    parsed = parse_map_binary(str(BOSTON_REFERENCE_BIN))
    sdc_idx = int(parsed["sdc_track_index"])
    trailer_idx = int(parsed["extension"]["ego_trailer_track_index"])
    assert int(parsed["extension"]["has_ego_trailer"]) == 1
    assert 0 <= sdc_idx < len(parsed["objects"])
    assert 0 <= trailer_idx < len(parsed["objects"])

    tractor = parsed["objects"][sdc_idx]
    trailer = parsed["objects"][trailer_idx]

    # Ground-truth references saved for future behavior/regression checks.
    gt_refs = {
        "tractor": {
            "x": np.asarray(tractor.x, dtype=np.float32),
            "y": np.asarray(tractor.y, dtype=np.float32),
            "heading": np.asarray(tractor.heading, dtype=np.float32),
            "valid": np.asarray(tractor.valid, dtype=np.int32),
            "length": float(tractor.length),
            "width": float(tractor.width),
        },
        "trailer": {
            "x": np.asarray(trailer.x, dtype=np.float32),
            "y": np.asarray(trailer.y, dtype=np.float32),
            "heading": np.asarray(trailer.heading, dtype=np.float32),
            "valid": np.asarray(trailer.valid, dtype=np.int32),
            "length": float(trailer.length),
            "width": float(trailer.width),
        },
    }
    return map_dir, gt_refs


def _load_sdc_trajectory_from_base_bin(binary_path):
    """Read SDC trajectory fields from base map binary layout (works for legacy bins)."""
    with open(binary_path, "rb") as f:
        sdc_track_index = struct.unpack("<i", f.read(4))[0]
        num_tracks_to_predict = struct.unpack("<i", f.read(4))[0]
        f.seek(4 * num_tracks_to_predict, 1)
        num_objects = struct.unpack("<i", f.read(4))[0]
        _ = struct.unpack("<i", f.read(4))[0]  # num_roads

        tractor = None
        for obj_idx in range(num_objects):
            _ = struct.unpack("<i", f.read(4))[0]  # scenario_id
            _ = struct.unpack("<i", f.read(4))[0]  # type
            _ = struct.unpack("<i", f.read(4))[0]  # id
            trajectory_length = struct.unpack("<i", f.read(4))[0]

            x = np.array(struct.unpack(f"<{trajectory_length}f", f.read(4 * trajectory_length)), dtype=np.float32)
            y = np.array(struct.unpack(f"<{trajectory_length}f", f.read(4 * trajectory_length)), dtype=np.float32)

            # z + vx/vy/vz
            f.seek(4 * trajectory_length * 4, 1)
            heading = np.array(struct.unpack(f"<{trajectory_length}f", f.read(4 * trajectory_length)), dtype=np.float32)
            valid = np.array(struct.unpack(f"<{trajectory_length}i", f.read(4 * trajectory_length)), dtype=np.int32)

            width = float(struct.unpack("<f", f.read(4))[0])
            length = float(struct.unpack("<f", f.read(4))[0])
            # height + goal xyz + mark_as_expert
            f.seek((4 * 4) + 4, 1)

            if obj_idx == sdc_track_index:
                tractor = {
                    "x": x,
                    "y": y,
                    "heading": heading,
                    "valid": valid,
                    "length": length,
                    "width": width,
                }

    if tractor is None:
        raise ValueError(f"SDC track index {sdc_track_index} not found in {binary_path}")
    return tractor


def _load_roads_from_base_bin(binary_path):
    """Read road polylines from base map binary layout (works for legacy bins)."""
    roads = []
    with open(binary_path, "rb") as f:
        _ = struct.unpack("<i", f.read(4))[0]  # sdc_track_index
        num_tracks_to_predict = struct.unpack("<i", f.read(4))[0]
        f.seek(4 * num_tracks_to_predict, 1)
        num_objects = struct.unpack("<i", f.read(4))[0]
        num_roads = struct.unpack("<i", f.read(4))[0]

        for _ in range(num_objects):
            _ = struct.unpack("<i", f.read(4))[0]  # scenario_id
            _ = struct.unpack("<i", f.read(4))[0]  # type
            _ = struct.unpack("<i", f.read(4))[0]  # id
            trajectory_length = struct.unpack("<i", f.read(4))[0]
            # x,y,z,vx,vy,vz,heading,valid
            f.seek(4 * trajectory_length * 8, 1)
            # width, length, height, goal xyz, mark_as_expert
            f.seek((6 * 4) + 4, 1)

        for _ in range(num_roads):
            _ = struct.unpack("<i", f.read(4))[0]  # scenario_id
            entity_type = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]  # id
            array_size = struct.unpack("<i", f.read(4))[0]
            x = np.array(struct.unpack(f"<{array_size}f", f.read(4 * array_size)), dtype=np.float32)
            y = np.array(struct.unpack(f"<{array_size}f", f.read(4 * array_size)), dtype=np.float32)
            # z + width, length, height, goal xyz, mark_as_expert
            f.seek((1 * 4 * array_size) + (6 * 4) + 4, 1)
            roads.append({"entity_type": int(entity_type), "x": x, "y": y})
    return roads


def _load_training_car_reference_map_and_trajectory(tmp_path):
    """Load a turning car map artifact and parse the SDC reference trajectory."""
    if not TRAINING_CAR_REFERENCE_BIN.exists():
        pytest.skip(f"Training car reference bin not found: {TRAINING_CAR_REFERENCE_BIN}")

    map_dir = tmp_path / "training_car_reference_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAINING_CAR_REFERENCE_BIN, map_dir / "map_000.bin")
    tractor_gt = _load_sdc_trajectory_from_base_bin(TRAINING_CAR_REFERENCE_BIN)
    roads = _load_roads_from_base_bin(TRAINING_CAR_REFERENCE_BIN)
    return map_dir, tractor_gt, roads


def _load_boston_car_header_map_and_trajectories(tmp_path):
    """Load patched Boston car-header map and return GT tractor/trailer trajectories from object data."""
    if not BOSTON_CAR_HEADER_BIN.exists():
        pytest.skip(f"Boston car-header bin not found: {BOSTON_CAR_HEADER_BIN}")

    map_dir = tmp_path / "boston_car_header_map"
    map_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BOSTON_CAR_HEADER_BIN, map_dir / "map_000.bin")

    parsed = parse_map_binary(str(BOSTON_CAR_HEADER_BIN))
    sdc_idx = int(parsed["sdc_track_index"])
    assert 0 <= sdc_idx < len(parsed["objects"])
    tractor = parsed["objects"][sdc_idx]

    trailer_candidates = [
        obj for obj in parsed["objects"] if int(getattr(obj, "is_trailer", 0)) == 1 and int(obj.parent_track_index) == sdc_idx
    ]
    assert len(trailer_candidates) >= 1
    trailer = trailer_candidates[0]

    gt_refs = {
        "tractor": {
            "x": np.asarray(tractor.x, dtype=np.float32),
            "y": np.asarray(tractor.y, dtype=np.float32),
            "heading": np.asarray(tractor.heading, dtype=np.float32),
            "valid": np.asarray(tractor.valid, dtype=np.int32),
            "length": float(tractor.length),
            "width": float(tractor.width),
        },
        "trailer": {
            "x": np.asarray(trailer.x, dtype=np.float32),
            "y": np.asarray(trailer.y, dtype=np.float32),
            "heading": np.asarray(trailer.heading, dtype=np.float32),
            "valid": np.asarray(trailer.valid, dtype=np.int32),
            "length": float(trailer.length),
            "width": float(trailer.width),
        },
    }
    return map_dir, gt_refs


def _extract_initial_state_from_ground_truth(gt_refs):
    """Step 2 helper: build initial tractor/trailer state from GT timestep 0."""
    return {
        "tractor": {
            "x": float(gt_refs["tractor"]["x"][0]),
            "y": float(gt_refs["tractor"]["y"][0]),
            "heading": float(gt_refs["tractor"]["heading"][0]),
            "valid": int(gt_refs["tractor"]["valid"][0]),
        },
        "trailer": {
            "x": float(gt_refs["trailer"]["x"][0]),
            "y": float(gt_refs["trailer"]["y"][0]),
            "heading": float(gt_refs["trailer"]["heading"][0]),
            "valid": int(gt_refs["trailer"]["valid"][0]),
        },
    }


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _pure_pursuit_action_from_gt(
    x,
    y,
    heading,
    gt_x,
    gt_y,
    lookahead_distance=6.0,
    lookahead_distance_start=None,
    steer_gain=12.0,
    accel_idx=4,
):
    """Compute classic discrete action from GT path using a pure-pursuit target."""
    path = np.column_stack((gt_x, gt_y))
    pos = np.array([x, y], dtype=np.float32)
    dists = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))

    # Optionally use shorter lookahead near the beginning and ramp it up later.
    if lookahead_distance_start is not None and len(path) > 1:
        progress = nearest / float(len(path) - 1)
        lookahead_distance = float(
            lookahead_distance_start + (lookahead_distance - lookahead_distance_start) * progress
        )

    target = path[-1]
    for i in range(nearest, len(path)):
        if np.linalg.norm(path[i] - pos) >= lookahead_distance:
            target = path[i]
            break

    dx = float(target[0] - x)
    dy = float(target[1] - y)
    alpha = _wrap_angle(np.arctan2(dy, dx) - heading)
    # Pure-pursuit curvature for a point target at distance Ld.
    Ld = max(lookahead_distance, 1e-3)
    curvature = (2.0 * np.sin(alpha)) / Ld

    # Map curvature to discrete steering bins centered at idx 6.
    steer_idx = int(np.clip(np.round(6.0 + steer_gain * curvature), 0, 12))
    return _classic_joint_action(accel_idx=accel_idx, steer_idx=steer_idx)


def _rollout_pure_pursuit_and_collect_trajectories(
    env,
    gt_x,
    gt_y,
    num_steps,
    lookahead_distance=6.0,
    lookahead_distance_start=2.5,
    steer_gain=12.0,
    accel_idx=4,
):
    """Step env with pure-pursuit and collect tractor/trailer trajectories from env APIs."""
    env.reset(seed=0)
    max_steps = int(num_steps)
    if env.episode_length is not None:
        # Stop before horizon rollover to avoid plotting wrapped trajectories.
        max_steps = min(max_steps, max(1, int(env.episode_length) - 1))

    history = {
        "tractor_x": [],
        "tractor_y": [],
        "tractor_heading": [],
        "tractor_length": [],
        "tractor_width": [],
        "trailer_x": [],
        "trailer_y": [],
        "trailer_heading": [],
        "trailer_length": [],
        "trailer_width": [],
        "track_error": [],
        "stopped_on_reset": np.array([0], dtype=np.int32),
    }
    initial_xy = None

    for _ in range(max_steps):
        tractor = env.get_global_agent_state()
        trailer = env.get_sdc_trailer_state()
        assert int(trailer["has_trailer"][0]) == 1

        x = float(tractor["x"][0])
        y = float(tractor["y"][0])
        heading = float(tractor["heading"][0])
        tx = float(trailer["x"][0])
        ty = float(trailer["y"][0])
        theading = float(trailer["heading"][0])

        history["tractor_x"].append(x)
        history["tractor_y"].append(y)
        history["tractor_heading"].append(heading)
        history["tractor_length"].append(float(tractor["length"][0]))
        history["tractor_width"].append(float(tractor["width"][0]))
        history["trailer_x"].append(tx)
        history["trailer_y"].append(ty)
        history["trailer_heading"].append(theading)
        history["trailer_length"].append(float(trailer["length"][0]))
        history["trailer_width"].append(float(trailer["width"][0]))

        err = np.sqrt(np.min((gt_x - x) ** 2 + (gt_y - y) ** 2))
        history["track_error"].append(float(err))
        if initial_xy is None:
            initial_xy = np.array([x, y], dtype=np.float32)

        action = _pure_pursuit_action_from_gt(
            x=x,
            y=y,
            heading=heading,
            gt_x=gt_x,
            gt_y=gt_y,
            lookahead_distance=lookahead_distance,
            lookahead_distance_start=lookahead_distance_start,
            steer_gain=steer_gain,
            accel_idx=accel_idx,
        )
        obs, rewards, terminals, truncations, _ = env.step(np.full_like(env.actions, action))
        assert np.isfinite(obs).all()
        assert np.isfinite(rewards).all()

        # If env signals end-of-episode/reset, stop simulation cleanly.
        if bool(terminals[0]) or bool(truncations[0]):
            history["stopped_on_reset"][0] = 1
            break

        # Fallback reset detection: sudden teleport back near initial state.
        nx = float(env.get_global_agent_state()["x"][0])
        ny = float(env.get_global_agent_state()["y"][0])
        step_jump = float(np.hypot(nx - x, ny - y))
        if initial_xy is not None:
            near_start = float(np.hypot(nx - initial_xy[0], ny - initial_xy[1])) < 1.0
            if step_jump > 5.0 and near_start:
                history["stopped_on_reset"][0] = 1
                break

    for key in history:
        if key == "stopped_on_reset":
            continue
        history[key] = np.asarray(history[key], dtype=np.float32)
    return history


def _make_boston_rollout_animation(
    output_path,
    gt_tractor_xy,
    gt_trailer_xy,
    pred_tractor_xy,
    pred_trailer_xy,
    gt_tractor_heading,
    gt_trailer_heading,
    pred_tractor_heading,
    pred_trailer_heading,
    pred_tractor_length,
    pred_tractor_width,
    pred_trailer_length,
    pred_trailer_width,
    tractor_ade_curve,
    trailer_ade_curve,
    map_binary_path=None,
    map_roads=None,
):
    """Create GIF animation: XY traces + heading curves + articulation + ADE curves."""
    n_exec = min(
        len(pred_tractor_xy),
        len(pred_trailer_xy),
        len(pred_tractor_heading),
        len(pred_trailer_heading),
        len(tractor_ade_curve),
        len(trailer_ade_curve),
        len(pred_tractor_length),
        len(pred_tractor_width),
        len(pred_trailer_length),
        len(pred_trailer_width),
    )
    if n_exec <= 1:
        raise ValueError("Not enough frames to animate")

    # Keep GT full-length; only executed series are truncated to available rollout horizon.
    pred_tractor_xy = pred_tractor_xy[:n_exec]
    pred_trailer_xy = pred_trailer_xy[:n_exec]
    pred_tractor_heading = pred_tractor_heading[:n_exec]
    pred_trailer_heading = pred_trailer_heading[:n_exec]
    pred_tractor_length = pred_tractor_length[:n_exec]
    pred_tractor_width = pred_tractor_width[:n_exec]
    pred_trailer_length = pred_trailer_length[:n_exec]
    pred_trailer_width = pred_trailer_width[:n_exec]
    tractor_ade_curve = tractor_ade_curve[:n_exec]
    trailer_ade_curve = trailer_ade_curve[:n_exec]
    gt_tractor_heading = _wrap_angle(np.asarray(gt_tractor_heading, dtype=np.float32))
    gt_trailer_heading = _wrap_angle(np.asarray(gt_trailer_heading, dtype=np.float32))
    pred_tractor_heading = _wrap_angle(np.asarray(pred_tractor_heading, dtype=np.float32))
    pred_trailer_heading = _wrap_angle(np.asarray(pred_trailer_heading, dtype=np.float32))

    step_axis_gt = np.arange(max(len(gt_tractor_heading), len(gt_trailer_heading)), dtype=np.int32)
    step_axis_exec = np.arange(n_exec, dtype=np.int32)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1, 1])
    ax_xy = fig.add_subplot(gs[:, 0])   # large simulation/map panel
    ax_h = fig.add_subplot(gs[0, 1])    # heading curves
    ax_art = fig.add_subplot(gs[1, 1])  # articulation angle
    ax_ade = fig.add_subplot(gs[2, 1])  # ADE curves

    # Static map backdrop (if provided) + GT traces.
    if map_roads is not None:
        for road in map_roads:
            if int(road["entity_type"]) == 6:
                ax_xy.plot(road["x"], road["y"], color="black", linewidth=1.2, alpha=0.85, zorder=0)
            else:
                ax_xy.plot(road["x"], road["y"], color="0.7", linewidth=0.8, alpha=0.55, zorder=0)
    elif map_binary_path is not None and Path(map_binary_path).exists():
        parsed = parse_map_binary(str(map_binary_path))
        for road in parsed["roads"]:
            if road.entity_type == 6:
                ax_xy.plot(road.x, road.y, color="black", linewidth=1.2, alpha=0.85, zorder=0)
            else:
                ax_xy.plot(road.x, road.y, color="0.7", linewidth=0.8, alpha=0.55, zorder=0)

    ax_xy.plot(gt_tractor_xy[:, 0], gt_tractor_xy[:, 1], "--", color="tab:blue", alpha=0.5, label="GT tractor")
    ax_xy.plot(gt_trailer_xy[:, 0], gt_trailer_xy[:, 1], "--", color="tab:orange", alpha=0.5, label="GT trailer")
    pred_tr_line, = ax_xy.plot([], [], "-", color="tab:blue", lw=2, label="Executed tractor")
    pred_tl_line, = ax_xy.plot([], [], "-", color="tab:orange", lw=2, label="Executed trailer")
    pred_tr_dot, = ax_xy.plot([], [], "o", color="tab:blue", ms=4)
    pred_tl_dot, = ax_xy.plot([], [], "o", color="tab:orange", ms=4)
    pred_tr_box = Polygon(np.zeros((4, 2), dtype=np.float32), closed=True, facecolor="none", edgecolor="tab:blue", lw=2.0)
    pred_tl_box = Polygon(
        np.zeros((4, 2), dtype=np.float32), closed=True, facecolor="none", edgecolor="tab:orange", lw=2.0
    )
    ax_xy.add_patch(pred_tr_box)
    ax_xy.add_patch(pred_tl_box)
    ax_xy.set_title("Trajectory Traces (GT vs Executed)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(alpha=0.25)
    ax_xy.legend(loc="best")

    ax_h.plot(np.arange(len(gt_tractor_heading)), gt_tractor_heading, "--", color="tab:blue", alpha=0.5, label="GT tractor")
    ax_h.plot(np.arange(len(gt_trailer_heading)), gt_trailer_heading, "--", color="tab:orange", alpha=0.5, label="GT trailer")
    pred_tr_h_line, = ax_h.plot([], [], "-", color="tab:blue", lw=2, label="Exec tractor")
    pred_tl_h_line, = ax_h.plot([], [], "-", color="tab:orange", lw=2, label="Exec trailer")
    ax_h.set_title("Headings")
    ax_h.set_xlabel("Step")
    ax_h.set_ylabel("Heading [rad]")
    ax_h.grid(alpha=0.25)
    ax_h.legend(loc="best")

    gt_articulation = _wrap_angle(gt_tractor_heading - gt_trailer_heading)
    pred_articulation = _wrap_angle(pred_tractor_heading - pred_trailer_heading)
    gt_art_line, = ax_art.plot(
        np.arange(len(gt_articulation)),
        gt_articulation,
        "--",
        color="tab:green",
        alpha=0.5,
        label="GT articulation",
    )
    pred_art_line, = ax_art.plot([], [], "-", color="tab:green", lw=2, label="Exec articulation")
    ax_art.set_title("Trailer Articulation (Tractor - Trailer)")
    ax_art.set_xlabel("Step")
    ax_art.set_ylabel("Angle [rad]")
    ax_art.grid(alpha=0.25)
    ax_art.legend(loc="best")

    tr_ade_line, = ax_ade.plot([], [], color="tab:blue", lw=2, label="Tractor ADE(t)")
    tl_ade_line, = ax_ade.plot([], [], color="tab:orange", lw=2, label="Trailer ADE(t)")
    ax_ade.set_title("Cumulative ADE")
    ax_ade.set_xlabel("Step")
    ax_ade.set_ylabel("ADE [m]")
    ax_ade.grid(alpha=0.25)
    ax_ade.legend(loc="best")

    # Stable axes
    all_xy = np.vstack([gt_tractor_xy, gt_trailer_xy, pred_tractor_xy, pred_trailer_xy])
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    pad = 3.0
    x_lo = float(x_min - pad)
    x_hi = float(x_max + pad)
    y_lo = float(y_min - pad)
    y_hi = float(y_max + pad)

    # Keep equal unit scaling but bias framing to be wider horizontally.
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    target_ratio = 1.8  # desired x_span / y_span for a landscape simulation panel
    if y_span > 1e-6 and x_span / y_span < target_ratio:
        extra = 0.5 * (target_ratio * y_span - x_span)
        x_lo -= extra
        x_hi += extra

    ax_xy.set_xlim(x_lo, x_hi)
    ax_xy.set_ylim(y_lo, y_hi)

    h_min = min(
        np.min(gt_tractor_heading),
        np.min(gt_trailer_heading),
        np.min(pred_tractor_heading),
        np.min(pred_trailer_heading),
    )
    h_max = max(
        np.max(gt_tractor_heading),
        np.max(gt_trailer_heading),
        np.max(pred_tractor_heading),
        np.max(pred_trailer_heading),
    )
    h_pad = 0.1 * max(1e-3, h_max - h_min)
    ax_h.set_xlim(0, max(len(step_axis_gt) - 1, len(step_axis_exec) - 1))
    ax_h.set_ylim(h_min - h_pad, h_max + h_pad)

    art_min = min(np.min(gt_articulation), np.min(pred_articulation))
    art_max = max(np.max(gt_articulation), np.max(pred_articulation))
    art_pad = 0.1 * max(1e-3, art_max - art_min)
    ax_art.set_xlim(0, max(len(step_axis_gt) - 1, len(step_axis_exec) - 1))
    ax_art.set_ylim(art_min - art_pad, art_max + art_pad)

    ade_max = max(float(np.max(tractor_ade_curve)), float(np.max(trailer_ade_curve)), 1e-3)
    ax_ade.set_xlim(0, max(1, len(step_axis_exec) - 1))
    ax_ade.set_ylim(0, ade_max * 1.1)

    def update(frame_idx):
        def _box_corners(x, y, heading, length, width):
            hl = 0.5 * length
            hw = 0.5 * width
            local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
            c = np.cos(heading)
            s = np.sin(heading)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            pts = local @ rot.T
            pts[:, 0] += x
            pts[:, 1] += y
            return pts

        end = frame_idx + 1
        pred_tr_line.set_data(pred_tractor_xy[:end, 0], pred_tractor_xy[:end, 1])
        pred_tl_line.set_data(pred_trailer_xy[:end, 0], pred_trailer_xy[:end, 1])
        pred_tr_dot.set_data([pred_tractor_xy[frame_idx, 0]], [pred_tractor_xy[frame_idx, 1]])
        pred_tl_dot.set_data([pred_trailer_xy[frame_idx, 0]], [pred_trailer_xy[frame_idx, 1]])
        pred_tr_box.set_xy(
            _box_corners(
                x=float(pred_tractor_xy[frame_idx, 0]),
                y=float(pred_tractor_xy[frame_idx, 1]),
                heading=float(pred_tractor_heading[frame_idx]),
                length=float(pred_tractor_length[frame_idx]),
                width=float(pred_tractor_width[frame_idx]),
            )
        )
        pred_tl_box.set_xy(
            _box_corners(
                x=float(pred_trailer_xy[frame_idx, 0]),
                y=float(pred_trailer_xy[frame_idx, 1]),
                heading=float(pred_trailer_heading[frame_idx]),
                length=float(pred_trailer_length[frame_idx]),
                width=float(pred_trailer_width[frame_idx]),
            )
        )

        pred_tr_h_line.set_data(step_axis_exec[:end], pred_tractor_heading[:end])
        pred_tl_h_line.set_data(step_axis_exec[:end], pred_trailer_heading[:end])
        pred_art_line.set_data(step_axis_exec[:end], pred_articulation[:end])

        tr_ade_line.set_data(step_axis_exec[:end], tractor_ade_curve[:end])
        tl_ade_line.set_data(step_axis_exec[:end], trailer_ade_curve[:end])

        fig.suptitle(f"Boston Pure-Pursuit Rollout | Step {frame_idx}/{n_exec - 1}", fontsize=11)
        return (
            pred_tr_line,
            pred_tl_line,
            pred_tr_dot,
            pred_tl_dot,
            pred_tr_h_line,
            pred_tl_h_line,
            pred_art_line,
            tr_ade_line,
            tl_ade_line,
            pred_tr_box,
            pred_tl_box,
        )

    anim = FuncAnimation(fig, update, frames=n_exec, interval=120, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def test_generated_bin_loads_and_steps_in_drive(generated_conversion_bin, monkeypatch):
    map_dir = generated_conversion_bin.parent

    # Some builds of the C binding require this kwarg explicitly.
    original_shared = drive_module.binding.shared

    def shared_with_default(*args, **kwargs):
        kwargs.setdefault("sequential_map_sampling", 0)
        return original_shared(*args, **kwargs)

    monkeypatch.setattr(drive_module.binding, "shared", shared_with_default)

    try:
        env = Drive(
            num_agents=2,
            num_maps=1,
            map_dir=str(map_dir),
            resample_frequency=0,
            episode_length=10,
            control_mode="control_vehicles",
            init_mode="create_all_valid",
        )
    except Exception as exc:
        pytest.fail(f"Failed to initialize Drive with generated binary: {exc}")

    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert obs.shape[0] == env.num_agents

        actions = np.zeros_like(env.actions)
        obs, rewards, terminals, truncations, info = env.step(actions)
        assert obs.shape[0] == env.num_agents
        assert rewards.shape[0] == env.num_agents
        assert terminals.shape[0] == env.num_agents
        assert truncations.shape[0] == env.num_agents
    finally:
        env.close()


def test_boston_tractor_trailer_setup(tmp_path, monkeypatch):
    map_dir, gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    initial_state = _extract_initial_state_from_ground_truth(gt_refs)
    _shared_patch(monkeypatch)

    # Confirm the copied reference map can initialize the simulation environment.
    env = _make_env(map_dir=map_dir, episode_length=20)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] == env.num_agents
        tractor_state = env.get_global_agent_state()
        trailer_state = env.get_sdc_trailer_state()
    finally:
        env.close()

    # Step 1 validation: GT references are populated and shape-consistent.
    for key in ("tractor", "trailer"):
        traj = gt_refs[key]
        n = traj["x"].shape[0]
        assert n > 0
        assert traj["y"].shape[0] == n
        assert traj["heading"].shape[0] == n
        assert traj["valid"].shape[0] == n
        assert np.any(traj["valid"] > 0)
        assert traj["length"] > 0.0
        assert traj["width"] > 0.0

    # Step 2 validation: env initializes at GT timestep 0 for tractor and trailer.
    assert initial_state["tractor"]["valid"] > 0
    assert initial_state["trailer"]["valid"] > 0
    assert int(trailer_state["has_trailer"][0]) == 1
    assert np.isclose(float(tractor_state["x"][0]), initial_state["tractor"]["x"], atol=1e-4)
    assert np.isclose(float(tractor_state["y"][0]), initial_state["tractor"]["y"], atol=1e-4)
    assert np.isclose(float(tractor_state["heading"][0]), initial_state["tractor"]["heading"], atol=1e-4)
    assert np.isclose(float(trailer_state["x"][0]), initial_state["trailer"]["x"], atol=1e-4)
    assert np.isclose(float(trailer_state["y"][0]), initial_state["trailer"]["y"], atol=1e-4)
    assert np.isclose(float(trailer_state["heading"][0]), initial_state["trailer"]["heading"], atol=1e-4)

    # Step 3: run pure-pursuit on the tractor against GT tractor trajectory.
    valid = gt_refs["tractor"]["valid"] > 0
    gt_x = gt_refs["tractor"]["x"][valid]
    gt_y = gt_refs["tractor"]["y"][valid]
    assert gt_x.shape[0] >= 10

    rollout_steps = 70  # configurable number of rollout steps for controller-driven simulation
    episode_length = rollout_steps + 5
    env = _make_env(map_dir=map_dir, episode_length=episode_length)
    try:
        history = _rollout_pure_pursuit_and_collect_trajectories(
            env=env,
            gt_x=gt_x,
            gt_y=gt_y,
            num_steps=rollout_steps,
        )
    finally:
        env.close()

    # Step 3 output checks: trajectories were gathered for both tractor and trailer.
    assert history["tractor_x"].shape[0] <= rollout_steps
    assert history["trailer_x"].shape[0] <= rollout_steps
    assert history["tractor_x"].shape[0] > 0
    assert history["trailer_x"].shape[0] > 0
    assert history["tractor_y"].shape == history["tractor_x"].shape
    assert history["trailer_y"].shape == history["trailer_x"].shape
    assert history["tractor_heading"].shape == history["tractor_x"].shape
    assert history["trailer_heading"].shape == history["trailer_x"].shape

    # Step 4: compute ADE (Average Displacement Error) against GT for tractor and trailer.
    tractor_valid = gt_refs["tractor"]["valid"] > 0
    trailer_valid = gt_refs["trailer"]["valid"] > 0
    gt_tractor_xy = np.column_stack((gt_refs["tractor"]["x"][tractor_valid], gt_refs["tractor"]["y"][tractor_valid]))
    gt_trailer_xy = np.column_stack((gt_refs["trailer"]["x"][trailer_valid], gt_refs["trailer"]["y"][trailer_valid]))

    n_eval = min(
        rollout_steps,
        gt_tractor_xy.shape[0],
        gt_trailer_xy.shape[0],
        history["tractor_x"].shape[0],
        history["trailer_x"].shape[0],
    )
    assert n_eval > 0

    pred_tractor_xy = np.column_stack((history["tractor_x"][:n_eval], history["tractor_y"][:n_eval]))
    pred_trailer_xy = np.column_stack((history["trailer_x"][:n_eval], history["trailer_y"][:n_eval]))

    tractor_ade = float(np.mean(np.linalg.norm(pred_tractor_xy - gt_tractor_xy[:n_eval], axis=1)))
    trailer_ade = float(np.mean(np.linalg.norm(pred_trailer_xy - gt_trailer_xy[:n_eval], axis=1)))

    assert np.isfinite(tractor_ade)
    assert np.isfinite(trailer_ade)
    # Tighter bounds for the current pure-pursuit baseline on the Boston reference map.
    assert tractor_ade < 10.0
    assert trailer_ade < 15.0

    tractor_step_err = np.linalg.norm(pred_tractor_xy - gt_tractor_xy[:n_eval], axis=1).astype(np.float32)
    trailer_step_err = np.linalg.norm(pred_trailer_xy - gt_trailer_xy[:n_eval], axis=1).astype(np.float32)
    tractor_ade_curve = np.cumsum(tractor_step_err) / (np.arange(n_eval, dtype=np.float32) + 1.0)
    trailer_ade_curve = np.cumsum(trailer_step_err) / (np.arange(n_eval, dtype=np.float32) + 1.0)

    # Plot/animation artifact for manual inspection.
    gif_path = tmp_path / "boston_pure_pursuit_rollout.gif"
    _make_boston_rollout_animation(
        output_path=gif_path,
        gt_tractor_xy=gt_tractor_xy[:n_eval],
        gt_trailer_xy=gt_trailer_xy[:n_eval],
        pred_tractor_xy=pred_tractor_xy,
        pred_trailer_xy=pred_trailer_xy,
        gt_tractor_heading=gt_refs["tractor"]["heading"][tractor_valid][:n_eval],
        gt_trailer_heading=gt_refs["trailer"]["heading"][trailer_valid][:n_eval],
        pred_tractor_heading=history["tractor_heading"][:n_eval],
        pred_trailer_heading=history["trailer_heading"][:n_eval],
        pred_tractor_length=history["tractor_length"][:n_eval],
        pred_tractor_width=history["tractor_width"][:n_eval],
        pred_trailer_length=history["trailer_length"][:n_eval],
        pred_trailer_width=history["trailer_width"][:n_eval],
        tractor_ade_curve=tractor_ade_curve,
        trailer_ade_curve=trailer_ade_curve,
        map_binary_path=map_dir / "map_000.bin",
    )
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0

    # Also store a stable copy under outputs/ for easy inspection outside pytest tmp dirs.
    output_dir = Path("outputs/test_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_gif_path = output_dir / "boston_pure_pursuit_rollout.gif"
    shutil.copy2(gif_path, stable_gif_path)
    assert stable_gif_path.exists()
    assert stable_gif_path.stat().st_size > 0

    track_err = history["track_error"]
    assert np.isfinite(track_err).all()
    # Track-error bounds tightened to catch controller regressions.
    assert float(np.mean(track_err)) < 2.0
    assert float(np.max(track_err)) < 5.0


def test_training_car_map_supports_runtime_truck_override(tmp_path, monkeypatch):
    map_dir, gt_refs = _load_boston_car_header_map_and_trajectories(tmp_path)
    _shared_patch(monkeypatch)

    # Step 1: baseline car map should initialize and report no SDC trailer.
    env = _make_env(map_dir=map_dir, episode_length=20)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] == env.num_agents
        base_tractor = env.get_global_agent_state()
        base_trailer = env.get_sdc_trailer_state()
        assert int(base_trailer["has_trailer"][0]) == 0
        base_tractor_length = float(base_tractor["length"][0])
        base_tractor_width = float(base_tractor["width"][0])
    finally:
        env.close()

    # Step 2: enabling runtime truck override should inject truck/trailer params at reset.
    env = _make_env(map_dir=map_dir, episode_length=180, sdc_runtime_truck_override=True)
    try:
        expected_params = _load_non_kinematic_vehicle_params_from_bin(
            str(Path(DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN).resolve())
        )
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] == env.num_agents
        tractor = env.get_global_agent_state()
        trailer = env.get_sdc_trailer_state()
        assert int(trailer["has_trailer"][0]) == 1
        assert float(tractor["length"][0]) > 0.0
        assert float(tractor["width"][0]) > 0.0
        assert float(trailer["length"][0]) > 0.0
        assert float(trailer["width"][0]) > 0.0
        # Runtime geometry should match reference non-kinematic truck params.
        assert np.isclose(float(tractor["length"][0]), float(expected_params[0]), atol=1e-6)
        assert np.isclose(float(tractor["width"][0]), float(expected_params[2]), atol=1e-6)
        assert np.isclose(float(trailer["length"][0]), float(expected_params[1]), atol=1e-6)
        assert np.isclose(float(trailer["width"][0]), float(expected_params[3]), atol=1e-6)

        actions = np.zeros_like(env.actions)
        obs, rewards, terminals, truncations, _ = env.step(actions)
        assert np.isfinite(obs).all()
        assert np.isfinite(rewards).all()
        assert terminals.shape[0] == env.num_agents
        assert truncations.shape[0] == env.num_agents

        # Track the map's reference tractor trajectory with pure-pursuit.
        tractor_valid = gt_refs["tractor"]["valid"] > 0
        trailer_valid = gt_refs["trailer"]["valid"] > 0
        gt_x = gt_refs["tractor"]["x"][tractor_valid]
        gt_y = gt_refs["tractor"]["y"][tractor_valid]
        gt_heading = gt_refs["tractor"]["heading"][tractor_valid]
        gt_trailer_x = gt_refs["trailer"]["x"][trailer_valid]
        gt_trailer_y = gt_refs["trailer"]["y"][trailer_valid]
        gt_trailer_heading = gt_refs["trailer"]["heading"][trailer_valid]
        assert gt_x.shape[0] >= 10

        rollout_steps = 140
        history = _rollout_pure_pursuit_and_collect_trajectories(
            env=env,
            gt_x=gt_x,
            gt_y=gt_y,
            num_steps=rollout_steps,
        )
        n_eval = min(
            history["tractor_x"].shape[0],
            history["trailer_x"].shape[0],
            gt_x.shape[0],
            gt_y.shape[0],
            gt_heading.shape[0],
            gt_trailer_x.shape[0],
            gt_trailer_y.shape[0],
            gt_trailer_heading.shape[0],
        )
        assert n_eval > 1

        gt_tractor_xy = np.column_stack((gt_x[:n_eval], gt_y[:n_eval]))
        gt_trailer_xy = np.column_stack((gt_trailer_x[:n_eval], gt_trailer_y[:n_eval]))
        pred_tractor_xy = np.column_stack((history["tractor_x"][:n_eval], history["tractor_y"][:n_eval]))
        pred_trailer_xy = np.column_stack((history["trailer_x"][:n_eval], history["trailer_y"][:n_eval]))
        tractor_step_err = np.linalg.norm(pred_tractor_xy - gt_tractor_xy, axis=1).astype(np.float32)
        trailer_step_err = np.linalg.norm(pred_trailer_xy - gt_trailer_xy, axis=1).astype(np.float32)
        tractor_ade_curve = np.cumsum(tractor_step_err) / (np.arange(n_eval, dtype=np.float32) + 1.0)
        trailer_ade_curve = np.cumsum(trailer_step_err) / (np.arange(n_eval, dtype=np.float32) + 1.0)
        assert np.isfinite(tractor_ade_curve).all()
        assert np.isfinite(trailer_ade_curve).all()

        gif_path = tmp_path / "training_car_runtime_truck_override_rollout.gif"
        _make_boston_rollout_animation(
            output_path=gif_path,
            gt_tractor_xy=gt_tractor_xy,
            gt_trailer_xy=gt_trailer_xy,
            pred_tractor_xy=pred_tractor_xy,
            pred_trailer_xy=pred_trailer_xy,
            gt_tractor_heading=gt_heading[:n_eval],
            gt_trailer_heading=gt_trailer_heading[:n_eval],
            pred_tractor_heading=history["tractor_heading"][:n_eval],
            pred_trailer_heading=history["trailer_heading"][:n_eval],
            pred_tractor_length=history["tractor_length"][:n_eval],
            pred_tractor_width=history["tractor_width"][:n_eval],
            pred_trailer_length=history["trailer_length"][:n_eval],
            pred_trailer_width=history["trailer_width"][:n_eval],
            tractor_ade_curve=tractor_ade_curve,
            trailer_ade_curve=trailer_ade_curve,
            map_binary_path=map_dir / "map_000.bin",
        )
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0

        output_dir = Path("outputs/test_visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        stable_gif_path = output_dir / "training_car_runtime_truck_override_rollout.gif"
        shutil.copy2(gif_path, stable_gif_path)
        assert stable_gif_path.exists()
        assert stable_gif_path.stat().st_size > 0
    finally:
        env.close()


def _make_map(tmp_path, scenario, unique_map_id=0):
    map_dir = tmp_path / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    map_path = map_dir / "map_000.bin"
    drive_module.save_map_binary(scenario, str(map_path), unique_map_id=unique_map_id)
    return map_dir


def _shared_patch(monkeypatch):
    # Some builds of the C binding require this kwarg explicitly.
    original_shared = drive_module.binding.shared

    def shared_with_default(*args, **kwargs):
        kwargs.setdefault("sequential_map_sampling", 0)
        return original_shared(*args, **kwargs)

    monkeypatch.setattr(drive_module.binding, "shared", shared_with_default)


def _run_steps(map_dir, reward_vehicle_collision, reward_offroad_collision, num_steps=1):
    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        resample_frequency=0,
        episode_length=10,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        reward_vehicle_collision=reward_vehicle_collision,
        reward_offroad_collision=reward_offroad_collision,
    )
    try:
        env.reset(seed=0)
        reward_total = 0.0
        collision_flag = 0.0
        for _ in range(num_steps):
            # Neutral classic-discrete action: accel idx=3, steer idx=6 => 3*13+6 = 45.
            actions = np.full_like(env.actions, 45)
            obs, rewards, _, _, _ = env.step(actions)
            reward_total += float(rewards[0])
            collision_flag = max(collision_flag, float(obs[0][5]))
        return reward_total, collision_flag
    finally:
        env.close()


def test_trailer_pose_follow_triggers_collision_after_step(tmp_path, monkeypatch):
    # A/B test: with trailer enabled, coupled trailer motion creates collision;
    # without trailer metadata, ego should not collide in this setup.
    scenario = {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}],
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
        },
        "objects": [
            {
                "id": 0,
                "source_track_id": "ego",
                "type": "vehicle",
                "position": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "velocity": [{"x": 10.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 1.0,
                "length": 4.8,
                "height": 1.7,
                "goalPosition": {"x": 40.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 1,
                "source_track_id": "ego_trailer",
                "type": "vehicle",
                "position": [{"x": -3.32, "y": 0.0, "z": 0.0}],
                "velocity": [{"x": 10.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 2.0,
                "length": 4.0,
                "height": 2.0,
                "goalPosition": {"x": -3.32, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 2,
                "source_track_id": "static_obstacle",
                "type": "vehicle",
                "position": [{"x": 0.3, "y": 0.9, "z": 0.0}],
                "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 1.0,
                "length": 1.0,
                "height": 1.0,
                "goalPosition": {"x": 0.3, "y": 0.9, "z": 0.0},
                "mark_as_expert": 1,
            },
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [{"x": -20.0, "y": 0.0, "z": 0.0}, {"x": 60.0, "y": 0.0, "z": 0.0}],
                "width": 3.5,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 101,
                "type": "road_edge",
                "geometry": [{"x": -20.0, "y": 8.0, "z": 0.0}, {"x": 60.0, "y": 8.0, "z": 0.0}],
                "width": 0.2,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }
    map_dir = _make_map(tmp_path / "with_trailer", scenario)
    scenario_no_trailer = dict(scenario)
    scenario_no_trailer["metadata"] = dict(scenario["metadata"])
    scenario_no_trailer["metadata"]["has_ego_trailer"] = False
    scenario_no_trailer["metadata"]["ego_trailer_track_index"] = -1
    map_dir_no_trailer = _make_map(tmp_path / "without_trailer", scenario_no_trailer)
    _shared_patch(monkeypatch)

    reward_with, collision_flag_with = _run_steps(
        map_dir=map_dir,
        reward_vehicle_collision=-1.0,
        reward_offroad_collision=0.0,
        num_steps=6,
    )
    reward_without, collision_flag_without = _run_steps(
        map_dir=map_dir_no_trailer,
        reward_vehicle_collision=-1.0,
        reward_offroad_collision=0.0,
        num_steps=6,
    )

    # Use a small margin: reward scaling can differ across builds, but
    # trailer-enabled case should still be measurably worse than baseline.
    assert reward_with < reward_without - 0.1
    assert collision_flag_with > collision_flag_without


def test_trailer_only_offroad_penalty(tmp_path, monkeypatch):
    # A/B test: road edge intersects trailer footprint only; offroad penalty should
    # appear when trailer metadata is enabled.
    scenario = {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}],
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
        },
        "objects": [
            {
                "id": 0,
                "source_track_id": "ego",
                "type": "vehicle",
                "position": _constant_trajectory(0.0, 0.0),
                "velocity": _constant_trajectory(0.0, 0.0),
                "heading": _constant_scalar(0.0),
                "valid": _constant_scalar(1),
                "width": 1.0,
                "length": 4.8,
                "height": 1.7,
                "goalPosition": {"x": 20.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 1,
                "source_track_id": "ego_trailer",
                "type": "vehicle",
                "position": _constant_trajectory(-3.32, 0.0),
                "velocity": _constant_trajectory(0.0, 0.0),
                "heading": _constant_scalar(0.0),
                "valid": _constant_scalar(1),
                "width": 2.0,
                "length": 4.0,
                "height": 2.0,
                "goalPosition": {"x": -3.32, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [{"x": -20.0, "y": 0.0, "z": 0.0}, {"x": 60.0, "y": 0.0, "z": 0.0}],
                "width": 3.5,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 101,
                "type": "road_edge",
                "geometry": [{"x": -10.0, "y": 0.6, "z": 0.0}, {"x": 2.0, "y": 0.6, "z": 0.0}],
                "width": 0.2,
                "length": 12.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }
    map_dir = _make_map(tmp_path / "with_trailer", scenario)
    scenario_no_trailer = dict(scenario)
    scenario_no_trailer["metadata"] = dict(scenario["metadata"])
    scenario_no_trailer["objects"] = [dict(obj) for obj in scenario["objects"]]
    scenario_no_trailer["metadata"]["has_ego_trailer"] = False
    scenario_no_trailer["metadata"]["ego_trailer_track_index"] = -1
    # Keep the A/B comparison focused on trailer-coupled offroad behavior:
    # without trailer metadata, this second box would otherwise be treated as a
    # regular nearby vehicle and can trigger plain vehicle-collision penalties,
    # masking the trailer-offroad signal this test is meant to isolate.
    scenario_no_trailer["objects"][1]["valid"] = _constant_scalar(0)
    map_dir_no_trailer = _make_map(tmp_path / "without_trailer", scenario_no_trailer)
    _shared_patch(monkeypatch)

    reward_with, collision_flag_with = _run_steps(
        map_dir=map_dir,
        reward_vehicle_collision=0.0,
        reward_offroad_collision=-1.0,
        num_steps=1,
    )
    reward_without, collision_flag_without = _run_steps(
        map_dir=map_dir_no_trailer,
        reward_vehicle_collision=0.0,
        reward_offroad_collision=-1.0,
        num_steps=1,
    )

    # Use a small margin: reward scaling can differ across builds, but
    # trailer-enabled case should still be measurably worse than baseline.
    assert reward_with < reward_without - 0.1
    assert collision_flag_with > collision_flag_without


def test_offroad_penalty_baseline_for_tractor(tmp_path, monkeypatch):
    # Baseline: ensure offroad pipeline is active when road edge intersects the tractor.
    scenario = {
        "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 0}]},
        "objects": [
            {
                "id": 0,
                "type": "vehicle",
                "position": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 1.2,
                "length": 4.8,
                "height": 1.7,
                "goalPosition": {"x": 20.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            }
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [{"x": -20.0, "y": 0.0, "z": 0.0}, {"x": 60.0, "y": 0.0, "z": 0.0}],
                "width": 3.5,
                "length": 80.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 101,
                "type": "road_edge",
                "geometry": [{"x": -5.0, "y": 0.3, "z": 0.0}, {"x": 5.0, "y": 0.3, "z": 0.0}],
                "width": 0.2,
                "length": 10.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }
    map_dir = _make_map(tmp_path / "tractor_offroad", scenario)
    _shared_patch(monkeypatch)

    reward, collision_flag = _run_steps(
        map_dir=map_dir,
        reward_vehicle_collision=0.0,
        reward_offroad_collision=-1.0,
        num_steps=1,
    )
    assert reward <= -0.5
    assert collision_flag > 0.5


def _classic_joint_action(accel_idx, steer_idx):
    """Classic discrete action encoding: joint index = accel_idx * 13 + steer_idx."""
    return accel_idx * 13 + steer_idx


def _make_env(map_dir, episode_length, **kwargs):
    """Create a Drive env for SDC-only curved-rollout tests."""
    return Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        resample_frequency=0,
        episode_length=episode_length,
        control_mode="control_sdc_only",
        init_mode="create_all_valid",
        reward_vehicle_collision=-1.0,
        reward_offroad_collision=-1.0,
        **kwargs,
    )


def test_get_global_agent_types_exposes_sim_entity_type(tmp_path):
    map_dir, tractor_gt, _ = _load_training_car_reference_map_and_trajectory(tmp_path)
    env = _make_env(map_dir, episode_length=len(tractor_gt["x"]))
    try:
        env.reset(seed=0)
        types = env.get_global_agent_types()

        assert isinstance(types, np.ndarray)
        assert types.dtype == np.int32
        assert types.shape == (env.num_agents,)
        assert int(types[0]) == 1  # VEHICLE

        state_with_types = env.get_global_agent_state(include_types=True)
        assert "type" in state_with_types
        np.testing.assert_array_equal(state_with_types["type"], types)
    finally:
        env.close()


def test_sdc_only_with_trailer_observation_has_trailer_features(tmp_path):
    map_dir, gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    episode_length = len(gt_refs["tractor"]["x"])
    env = _make_env(map_dir, episode_length=episode_length, observation_mode="sdc_only_with_trailer")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.num_agents, env.num_obs)
        assert env.ego_features == env._base_ego_features + 5
        assert env.partner_features == env._base_partner_features + 1

        base_ego = env._base_ego_features
        trailer_start = base_ego
        ego_type_idx = base_ego + 4

        # trailer length and width should be populated when trailer exists
        assert float(obs[0, trailer_start + 2]) > 0.0
        assert float(obs[0, trailer_start + 3]) > 0.0
        assert int(obs[0, ego_type_idx]) == 1  # truck-with-trailer
    finally:
        env.close()


def test_sdc_only_with_trailer_observation_zero_pads_without_trailer(tmp_path):
    map_dir, tractor_gt, _ = _load_training_car_reference_map_and_trajectory(tmp_path)
    env = _make_env(
        map_dir,
        episode_length=len(tractor_gt["x"]),
        observation_mode="sdc_only_with_trailer",
        force_truck_params_from_ref_bin=None,
    )
    try:
        obs, _ = env.reset(seed=0)
        base_ego = env._base_ego_features
        trailer_start = base_ego
        ego_type_idx = base_ego + 4

        np.testing.assert_allclose(obs[0, trailer_start : trailer_start + 4], np.zeros(4, dtype=np.float32))
        assert int(obs[0, ego_type_idx]) == 0  # car
    finally:
        env.close()


def test_sdc_only_with_trailer_policy_forward(tmp_path):
    import torch
    from pufferlib.ocean.torch import Drive as DrivePolicy

    map_dir, gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    env = _make_env(map_dir, episode_length=len(gt_refs["tractor"]["x"]), observation_mode="sdc_only_with_trailer")
    try:
        obs, _ = env.reset(seed=0)
        obs_t = torch.as_tensor(obs)
        policy = DrivePolicy(env, input_size=64, hidden_size=64)
        logits, value = policy(obs_t)
        assert value.shape == (env.num_agents, 1)
        assert isinstance(logits, tuple)
        assert len(logits) >= 1
    finally:
        env.close()


def test_default_observation_mode_policy_forward(tmp_path):
    import torch
    from pufferlib.ocean.torch import Drive as DrivePolicy

    map_dir, gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    env = _make_env(map_dir, episode_length=len(gt_refs["tractor"]["x"]), observation_mode="default")
    try:
        obs, _ = env.reset(seed=0)
        obs_t = torch.as_tensor(obs)
        policy = DrivePolicy(env, input_size=64, hidden_size=64)
        logits, value = policy(obs_t)
        assert value.shape == (env.num_agents, 1)
        assert isinstance(logits, tuple)
        assert len(logits) >= 1
    finally:
        env.close()


def test_sdc_only_with_trailer_exact_trailer_xy_transform(tmp_path):
    map_dir, gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    env = _make_env(
        map_dir,
        episode_length=len(gt_refs["tractor"]["x"]),
        observation_mode="sdc_only_with_trailer",
        init_steps=0,
    )
    try:
        obs, _ = env.reset(seed=0)
        base_ego = env._base_ego_features
        trailer_start = base_ego

        tractor_x = float(gt_refs["tractor"]["x"][0])
        tractor_y = float(gt_refs["tractor"]["y"][0])
        tractor_heading = float(gt_refs["tractor"]["heading"][0])
        trailer_x = float(gt_refs["trailer"]["x"][0])
        trailer_y = float(gt_refs["trailer"]["y"][0])

        dx = trailer_x - tractor_x
        dy = trailer_y - tractor_y
        cos_h = np.cos(tractor_heading)
        sin_h = np.sin(tractor_heading)

        expected_x_local = (dx * cos_h + dy * sin_h) * 0.02
        expected_y_local = (-dx * sin_h + dy * cos_h) * 0.02

        np.testing.assert_allclose(float(obs[0, trailer_start]), expected_x_local, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(float(obs[0, trailer_start + 1]), expected_y_local, rtol=0.0, atol=1e-5)
    finally:
        env.close()


def test_sdc_only_with_trailer_dims_and_sdc_type(tmp_path):
    map_dir, gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    env = _make_env(
        map_dir,
        episode_length=len(gt_refs["tractor"]["x"]),
        observation_mode="sdc_only_with_trailer",
        init_steps=0,
    )
    try:
        obs, _ = env.reset(seed=0)
        base_ego = env._base_ego_features
        trailer_start = base_ego
        ego_type_idx = base_ego + 4

        expected_len = float(gt_refs["trailer"]["length"]) / 30.0
        expected_width = float(gt_refs["trailer"]["width"]) / 15.0

        np.testing.assert_allclose(float(obs[0, trailer_start + 2]), expected_len, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(float(obs[0, trailer_start + 3]), expected_width, rtol=0.0, atol=1e-6)
        assert int(obs[0, ego_type_idx]) == 1  # truck_with_trailer
    finally:
        env.close()


def test_sdc_only_with_trailer_partner_type_slots_exact_indices(tmp_path):
    map_dir, gt_refs = _load_boston_reference_map_and_trajectories(tmp_path)
    env = _make_env(
        map_dir,
        episode_length=len(gt_refs["tractor"]["x"]),
        observation_mode="sdc_only_with_trailer",
        init_steps=0,
    )
    try:
        obs, _ = env.reset(seed=0)
        partner_start = env.ego_features
        slot_size = env.partner_features
        type_offset = env._base_partner_features

        partner_types_raw = env.get_partner_types()[0]
        expected = np.asarray([env._map_policy_type(int(t), is_ego=False, has_trailer=False) for t in partner_types_raw])
        observed = np.asarray(
            [int(obs[0, partner_start + p * slot_size + type_offset]) for p in range(env.max_partner_objects)],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(observed, expected)

        # Boston fixture has multiple observed partners; enforce that this is not a trivial single-slot check.
        raw_partner_count = int((partner_types_raw != 0).sum())
        assert raw_partner_count >= 2
    finally:
        env.close()


def test_sdc_only_with_trailer_control_vehicles_zero_pads_trailer_slots(tmp_path):
    map_dir, tractor_gt, _ = _load_training_car_reference_map_and_trajectory(tmp_path)
    env = Drive(
        num_agents=1,
        num_maps=1,
        map_dir=str(map_dir),
        resample_frequency=0,
        episode_length=len(tractor_gt["x"]),
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        observation_mode="sdc_only_with_trailer",
    )
    try:
        obs, _ = env.reset(seed=0)
        base_ego = env._base_ego_features
        trailer_start = base_ego
        ego_type_idx = base_ego + 4

        np.testing.assert_allclose(obs[0, trailer_start : trailer_start + 4], np.zeros(4, dtype=np.float32))
        assert int(obs[0, ego_type_idx]) == 0  # car
    finally:
        env.close()


def test_default_observation_mode_keeps_base_shape(tmp_path):
    map_dir, tractor_gt, _ = _load_training_car_reference_map_and_trajectory(tmp_path)
    env = _make_env(map_dir, episode_length=len(tractor_gt["x"]), observation_mode="default")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.num_agents, env.num_obs)
        assert env.num_obs == env._sim_num_obs
        assert env.ego_features == env._base_ego_features
        assert env.partner_features == env._base_partner_features
    finally:
        env.close()
