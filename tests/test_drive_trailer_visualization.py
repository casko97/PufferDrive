from pufferlib.ocean.drive.trailer_viz import parse_map_binary, plot_trailer_scene


def test_parse_and_plot_trailer_scene(generated_conversion_bin, tmp_path):
    parsed = parse_map_binary(str(generated_conversion_bin))

    assert parsed["sdc_track_index"] == 0
    assert parsed["extension"]["version"] == 2
    assert parsed["extension"]["has_ego_trailer"] == 1
    assert parsed["extension"]["ego_trailer_track_index"] == 1
    assert "tractor2hitch" in parsed["extension"]["non_kinematic_vehicle_params"]
    assert len(parsed["objects"]) == 2
    assert parsed["objects"][1].is_trailer == 1

    output_path = tmp_path / "trailer_scene.png"
    plot_trailer_scene(
        binary_path=str(generated_conversion_bin),
        output_prefix=str(output_path),
        frames=[0],
        plot_all_objects=True,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
