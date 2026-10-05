"""Tests for satellite_pass_predictor.globe.

Synthetic ground tracks stand in for propagated ones. Assertions inspect pydeck's Layer/View
objects directly, which keep their constructor arguments as plain attributes.
"""

import numpy as np
import pytest
from skyfield.timelib import Timescale

from satellite_pass_predictor.colors import hex_to_rgb
from satellite_pass_predictor.config import (
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    KOUROU_MARKER_COLOR,
    SATELLITE_COLORS,
    THEME_BACKGROUND_COLOR,
)
from satellite_pass_predictor.globe import (
    _HOVER_POINT_STRIDE,
    _split_path_at_antimeridian,
    build_globe_deck,
)
from satellite_pass_predictor.propagation import GroundTrackDict


def _fake_track(ts: Timescale, latitudes: list[float], longitudes: list[float]) -> GroundTrackDict:
    """A minimal GroundTrackDict; the time field must be a real Skyfield Time."""
    n = len(latitudes)
    assert len(longitudes) == n
    return {
        "latitude_deg": np.array(latitudes, dtype=float),
        "longitude_deg": np.array(longitudes, dtype=float),
        "altitude_km": np.linspace(400.0, 420.0, n),
        "time": ts.utc(2026, 1, 1, 0, np.arange(n), 0),
    }


def _layers_by_type(deck, type_name: str) -> list:
    return [layer for layer in deck.layers if layer.type == type_name]


def test_one_path_and_one_scatter_layer_per_satellite_plus_kourou(ts: Timescale) -> None:
    """Per satellite: a PathLayer and a hover ScatterplotLayer. Plus Kourou and one basemap."""
    ground_tracks = {
        "ISS (ZARYA)": _fake_track(ts, [10.0, 20.0, 30.0, 40.0], [0.0, 10.0, 20.0, 30.0]),
        "SWISSCUBE": _fake_track(ts, [-10.0, -20.0, -30.0, -40.0], [50.0, 60.0, 70.0, 80.0]),
    }

    deck = build_globe_deck(ground_tracks)

    path_layers = _layers_by_type(deck, "PathLayer")
    scatter_layers = _layers_by_type(deck, "ScatterplotLayer")
    geojson_layers = _layers_by_type(deck, "GeoJsonLayer")
    assert len(path_layers) == len(ground_tracks)
    assert len(scatter_layers) == len(ground_tracks) + 1  # +1 for Kourou
    assert len(geojson_layers) == 1
    assert len(deck.layers) == len(path_layers) + len(scatter_layers) + len(geojson_layers)


def test_kourou_layer_is_at_kourous_coordinates_and_visually_distinct(
    ts: Timescale,
) -> None:
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, [10.0, 20.0], [0.0, 10.0])}

    deck = build_globe_deck(ground_tracks)

    scatter_layers = _layers_by_type(deck, "ScatterplotLayer")
    kourou_layer = next(layer for layer in scatter_layers if layer.data[0]["name"] == "Kourou")
    satellite_layer = next(
        layer for layer in scatter_layers if layer.data[0]["name"] == "ISS (ZARYA)"
    )

    assert kourou_layer.get_position == [KOUROU_LONGITUDE_DEG, KOUROU_LATITUDE_DEG]
    assert kourou_layer.get_fill_color == hex_to_rgb(KOUROU_MARKER_COLOR)
    assert kourou_layer.get_fill_color != satellite_layer.data[0]["color"]
    assert kourou_layer.stroked is True


def test_satellite_track_color_comes_from_the_shared_color_mapping(ts: Timescale) -> None:
    """Colors are looked up by satellite name, so selection order doesn't change them."""
    ground_tracks = {
        "MICROSCOPE": _fake_track(ts, [1.0, 2.0], [3.0, 4.0]),
        "ISS (ZARYA)": _fake_track(ts, [5.0, 6.0], [7.0, 8.0]),
    }

    deck = build_globe_deck(ground_tracks)

    path_layers = _layers_by_type(deck, "PathLayer")
    scatter_layers = _layers_by_type(deck, "ScatterplotLayer")
    for name in ground_tracks:
        expected_rgb = hex_to_rgb(SATELLITE_COLORS[name])
        path_layer = next(layer for layer in path_layers if layer.data[0]["color"] == expected_rgb)
        assert path_layer.data[0]["color"] == expected_rgb
        scatter_layer = next(
            layer for layer in scatter_layers if layer.data and layer.data[0].get("name") == name
        )
        assert scatter_layer.data[0]["color"] == expected_rgb


def test_hover_points_are_unaffected_by_antimeridian_splitting(ts: Timescale) -> None:
    """Hover points come from the real samples only, never the synthetic boundary points."""
    longitudes = [170.0, 179.0, -179.0, -170.0]  # crosses the antimeridian once
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, [10.0, 11.0, 12.0, 13.0], longitudes)}

    deck = build_globe_deck(ground_tracks)

    scatter_layer = next(
        layer
        for layer in _layers_by_type(deck, "ScatterplotLayer")
        if layer.data and layer.data[0].get("name") == "ISS (ZARYA)"
    )
    assert len(scatter_layer.data) == len(range(0, len(longitudes), _HOVER_POINT_STRIDE))
    assert [point["lon"] for point in scatter_layer.data] == longitudes[::_HOVER_POINT_STRIDE]


def test_hover_points_contain_name_time_latlon_and_altitude(ts: Timescale) -> None:
    n = _HOVER_POINT_STRIDE * 2  # enough points that some survive thinning
    lats = [12.34] * n
    lons = [56.78] * n
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, lats, lons)}

    deck = build_globe_deck(ground_tracks)

    scatter_layers = _layers_by_type(deck, "ScatterplotLayer")
    satellite_layer = next(
        layer for layer in scatter_layers if layer.data[0]["name"] == "ISS (ZARYA)"
    )
    point = satellite_layer.data[0]

    assert point["name"] == "ISS (ZARYA)"
    assert "2026-01-01" in point["time"] and "UTC" in point["time"]
    assert point["lat_str"] == "12.34"
    assert point["lon_str"] == "56.78"
    assert point["alt_str"] == "400.0"  # first value of _fake_track's altitude


def test_hover_point_layer_is_thinned_but_path_layer_stays_full_resolution(
    ts: Timescale,
) -> None:
    n = 20
    lats = list(np.linspace(0.0, 10.0, n))
    lons = list(np.linspace(0.0, 10.0, n))
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, lats, lons)}

    deck = build_globe_deck(ground_tracks)

    path_layer = _layers_by_type(deck, "PathLayer")[0]
    scatter_layer = next(
        layer
        for layer in _layers_by_type(deck, "ScatterplotLayer")
        if layer.data[0]["name"] == "ISS (ZARYA)"
    )

    assert len(path_layer.data[0]["path"]) == n
    assert len(scatter_layer.data) == len(range(0, n, _HOVER_POINT_STRIDE))


def test_canvas_clear_color_matches_the_app_theme_background(ts: Timescale) -> None:
    deck = build_globe_deck({"ISS (ZARYA)": _fake_track(ts, [10.0], [20.0])})

    expected = [c / 255.0 for c in hex_to_rgb(THEME_BACKGROUND_COLOR)] + [1.0]
    assert deck.parameters == {"clearColor": expected}


def test_uses_globe_view_centered_near_kourou(ts: Timescale) -> None:
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, [10.0], [20.0])}

    deck = build_globe_deck(ground_tracks)

    assert len(deck.views) == 1
    assert deck.views[0].type == "_GlobeView"
    assert deck.initial_view_state.latitude == KOUROU_LATITUDE_DEG
    assert deck.initial_view_state.longitude == KOUROU_LONGITUDE_DEG


def test_antimeridian_crossing_splits_the_path_into_separate_data_entries(ts: Timescale) -> None:
    """A crossing yields two sub-paths, each ending exactly on the meridian it was cut at."""
    longitudes = [170.0, 179.0, -179.0, -170.0]  # crosses the antimeridian once
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, [10.0, 11.0, 12.0, 13.0], longitudes)}

    deck = build_globe_deck(ground_tracks)

    path_layer = _layers_by_type(deck, "PathLayer")[0]
    assert len(path_layer.data) == 2

    first_path = path_layer.data[0]["path"]
    second_path = path_layer.data[1]["path"]
    assert [lon for lon, _lat in first_path[:2]] == [170.0, 179.0]
    assert [lon for lon, _lat in second_path[-2:]] == [-179.0, -170.0]
    assert first_path[-1][0] == 180.0
    assert second_path[0][0] == -180.0
    # Both boundary points share one interpolated latitude, so the halves meet without a gap.
    assert first_path[-1][1] == pytest.approx(second_path[0][1])


def test_no_path_segment_jumps_more_than_180_degrees_in_longitude(ts: Timescale) -> None:
    """With several crossings, no sub-path ever jumps across the map the long way round."""
    longitudes = [170.0, 179.0, -179.0, -170.0, 170.0, 179.0, -175.0]
    latitudes = [float(i) for i in range(len(longitudes))]
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, latitudes, longitudes)}

    deck = build_globe_deck(ground_tracks)

    path_layer = _layers_by_type(deck, "PathLayer")[0]
    for sub_path_entry in path_layer.data:
        lons = [point[0] for point in sub_path_entry["path"]]
        for lon_a, lon_b in zip(lons, lons[1:], strict=False):
            assert abs(lon_b - lon_a) <= 180.0


def test_split_path_at_antimeridian_leaves_a_non_crossing_track_unchanged(ts: Timescale) -> None:
    longitudes = np.array([10.0, 20.0, 30.0, 40.0])
    latitudes = np.array([1.0, 2.0, 3.0, 4.0])

    paths = _split_path_at_antimeridian(longitudes, latitudes)

    assert len(paths) == 1
    assert paths[0] == [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]
