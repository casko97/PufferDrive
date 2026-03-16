from pufferlib.ocean.drive.drive import save_map_binary


def build_conversion_scenario():
    return {
        "metadata": {
            "sdc_track_index": 0,
            "tracks_to_predict": [{"track_index": 0}, {"track_index": 1}],
            "has_ego_trailer": True,
            "ego_trailer_track_index": 1,
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
                "id": 10,
                "source_track_id": "ego",
                "type": "vehicle",
                "position": [{"x": 1.0, "y": 2.0, "z": 0.0}],
                "velocity": [{"x": 0.1, "y": 0.2, "z": 0.0}],
                "heading": [0.0],
                "valid": [1],
                "width": 2.0,
                "length": 5.0,
                "height": 1.7,
                "goalPosition": {"x": 20.0, "y": 30.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 11,
                "source_track_id": "ego_trailer",
                "type": "vehicle",
                "position": [{"x": -1.0, "y": -2.0, "z": 0.0}],
                "velocity": [{"x": 0.0, "y": 0.0, "z": 0.0}],
                "heading": [0.1],
                "valid": [1],
                "width": 2.5,
                "length": 13.7,
                "height": 3.0,
                "goalPosition": {"x": 10.0, "y": 12.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
        "roads": [
            {
                "id": 100,
                "type": "lane",
                "geometry": [
                    {"x": -10.0, "y": 0.0, "z": 0.0},
                    {"x": 50.0, "y": 0.0, "z": 0.0},
                ],
                "width": 3.5,
                "length": 60.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
            {
                "id": 101,
                "type": "road_edge",
                "geometry": [
                    {"x": -10.0, "y": 6.0, "z": 0.0},
                    {"x": 50.0, "y": 6.0, "z": 0.0},
                ],
                "width": 0.2,
                "length": 60.0,
                "height": 0.0,
                "goalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "mark_as_expert": 0,
            },
        ],
    }


def write_conversion_bin(output_file, unique_map_id=42):
    save_map_binary(build_conversion_scenario(), str(output_file), unique_map_id=unique_map_id)
