import struct

from pufferlib.ocean.drive.drive import save_map_binary


def _read_int(f):
    return struct.unpack("i", f.read(4))[0]


def _read_float(f):
    return struct.unpack("f", f.read(4))[0]


def _skip_object_payload(f, trajectory_length):
    # x, y, z
    f.seek(trajectory_length * 3 * 4, 1)
    # vx, vy, vz
    f.seek(trajectory_length * 3 * 4, 1)
    # heading
    f.seek(trajectory_length * 4, 1)
    # valid
    f.seek(trajectory_length * 4, 1)
    # width, length, height, goal x/y/z, mark_as_expert
    f.seek((6 * 4) + 4, 1)


def _skip_road_payload(f, size):
    # x, y, z arrays
    f.seek(size * 3 * 4, 1)
    # width, length, height, goal x/y/z, mark_as_expert
    f.seek((6 * 4) + 4, 1)


def test_drive_json_to_bin_writes_core_and_trailer_extension(generated_conversion_bin):
    trajectory_length = 91

    with open(generated_conversion_bin, "rb") as f:
        # Header
        assert _read_int(f) == 0  # sdc_track_index
        assert _read_int(f) == 2  # num_tracks_to_predict
        assert _read_int(f) == 0
        assert _read_int(f) == 1
        assert _read_int(f) == 2  # num_objects
        assert _read_int(f) == 2  # num_roads

        # Object 0 base fields
        assert _read_int(f) == 42  # scenario_id / unique_map_id
        assert _read_int(f) == 1  # type=vehicle
        assert _read_int(f) == 10  # id
        assert _read_int(f) == trajectory_length
        _skip_object_payload(f, trajectory_length)

        # Object 1 base fields
        assert _read_int(f) == 42
        assert _read_int(f) == 1
        assert _read_int(f) == 11
        assert _read_int(f) == trajectory_length
        _skip_object_payload(f, trajectory_length)

        # Road base fields
        assert _read_int(f) == 42
        assert _read_int(f) == 4  # lane remapped to ROAD_LANE bucket
        assert _read_int(f) == 100
        road_size = _read_int(f)
        assert road_size == 2
        _skip_road_payload(f, road_size)

        assert _read_int(f) == 42
        assert _read_int(f) == 6  # road_edge remapped to ROAD_EDGE enum bucket
        assert _read_int(f) == 101
        road_size = _read_int(f)
        assert road_size == 2
        _skip_road_payload(f, road_size)

        # Extension block
        assert _read_int(f) == 0x54524C52  # "TRLR"
        assert _read_int(f) == 2  # extension version
        assert _read_int(f) == 1  # has_ego_trailer
        assert _read_int(f) == 1  # ego_trailer_track_index
        assert _read_int(f) == 2  # object meta count

        # Object extension metadata
        src0 = struct.unpack("Q", f.read(8))[0]
        is_trailer0 = _read_int(f)
        parent0 = _read_int(f)

        src1 = struct.unpack("Q", f.read(8))[0]
        is_trailer1 = _read_int(f)
        parent1 = _read_int(f)

        assert src0 != 0
        assert src1 != 0
        assert src0 != src1
        assert is_trailer0 == 0
        assert parent0 == -1
        assert is_trailer1 == 1
        assert parent1 == 0

        # Non-kinematic vehicle params extension payload
        assert _read_int(f) == 13
        params = [_read_float(f) for _ in range(13)]
        assert abs(params[0] - 5.3) < 1e-5  # tractor_length
        assert abs(params[1] - 13.7) < 1e-5  # trailer_length
        assert abs(params[6] - 0.6) < 1e-5  # tractor2hitch
        assert abs(params[7] - 2.2) < 1e-5  # trailer2hitch

        # Ensure we consumed full file.
        assert f.read() == b""


def test_drive_json_to_bin_handles_large_and_string_ids(tmp_path):
    scenario = {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}],
            "has_ego_trailer": False,
            "ego_trailer_track_index": -1,
            "non_kinematic_vehicle_params": {
                "tractor_length": 5.3,
                "trailer_length": 13.7,
                "width": 2.55,
                "trailer_width": 2.55,
                "vehicle_height": 3.5,
                "trailer_height": 3.5,
                "tractor2hitch": 0.6,
                "trailer2hitch": 2.2,
                "tractor_d_rear_axle2rear_bumper": 1.5,
                "tractor_d_rear_axle2front_axle": 3.6,
                "tractor_d_front_axle2front_bumper": 1.2,
                "trailer_d_rear_axel2_rear_bumper": 2.0,
                "trailer_d_real_axel2_front_bumper": 10.5,
            },
        },
        "objects": [
            {
                "id": 9223372036854775807,
                "source_track_id": "ego",
                "type": "vehicle",
                "position": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 2.0,
                "length": 5.0,
                "height": 1.7,
                "goalPosition": {"x": 10.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            }
        ],
        "roads": [
            {
                "id": "road_edge_very_large_or_string_id",
                "type": "road_edge",
                "geometry": [{"x": -5.0, "y": 2.0, "z": 0.0}, {"x": 5.0, "y": 2.0, "z": 0.0}],
                "width": 0.2,
                "length": 10.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            }
        ],
    }

    out1 = tmp_path / "map_large_id_1.bin"
    out2 = tmp_path / "map_large_id_2.bin"
    save_map_binary(scenario, str(out1), unique_map_id=7)
    save_map_binary(scenario, str(out2), unique_map_id=7)
    assert out1.exists()
    assert out2.exists()

    # Conversion should be deterministic for hashed/string ids.
    assert out1.read_bytes() == out2.read_bytes()
