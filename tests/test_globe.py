"""
Tests for satellite_pass_predictor.globe.

Presentation code, like reporting.py -- similar scope: check the
properties that actually matter if this broke (right layers get built,
Kourou's marker present and visually distinct, hover data has the
expected fields, the globe view is actually configured), not an
exhaustive pixel-level check of the rendered globe.

PURE LOGIC / no real satellite needed for most of this: build_globe_deck()
only reads whatever GroundTrackDict-shaped data it's given, so synthetic
lat/lon/alt arrays stand in for real propagated ones. The one exception is
the "time" field, which must be a real Skyfield Time (for .utc_strftime()),
so the `ts` fixture from conftest.py builds one.

Introspects pydeck's own Layer/View/ViewState objects directly (their
constructor kwargs are stored as plain attributes, and `.data` is kept
as the original Python list/dict -- not yet JSON-encoded), rather than
round-tripping through deck.to_json(), which would require decoding
pydeck's "@@=field" function-reference syntax for no benefit here.
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
    """A minimal, valid-shaped GroundTrackDict -- the actual propagation
    is tested in test_propagation.py, this just needs representative
    data to render."""
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
    """Each satellite contributes a PathLayer (the visible track) and a
    ScatterplotLayer (thinned, for hover) -- plus one more
    ScatterplotLayer for Kourou, plus one GeoJsonLayer for the
    land/coastline basemap (shared, not per-satellite)."""
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
    assert len(geojson_layers) == 1  # the basemap, shared across all satellites
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
    # Exactly config.KOUROU_MARKER_COLOR (white), not an arbitrary color --
    # and distinct from every satellite layer in color: Kourou is the
    # ground station, not a satellite, and should read as visually
    # different at a glance. (pydeck's ScatterplotLayer only draws
    # circles, so "symbol" distinctness here means color + size + stroke
    # outline -- kourou_layer.stroked below -- rather than a literal star
    # shape; see build_globe_deck()'s docstring for that trade-off.)
    assert kourou_layer.get_fill_color == hex_to_rgb(KOUROU_MARKER_COLOR)
    assert kourou_layer.get_fill_color != satellite_layer.data[0]["color"]
    assert kourou_layer.stroked is True


def test_satellite_track_color_comes_from_the_shared_color_mapping(ts: Timescale) -> None:
    """
    Every satellite's PathLayer/ScatterplotLayer color should be exactly
    config.SATELLITE_COLORS[name] -- the same mapping the passes table,
    sky plot, and matplotlib figure all read -- looked up by name, not
    by position: st.multiselect() (app.py) can return satellites in
    whatever order the user selected them in, and a positional color
    would then reassign colors as the selection order changes.
    """
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
    """
    The hover ScatterplotLayer should reflect only the real, original
    samples -- exactly _HOVER_POINT_STRIDE-thinned, same as a track that
    never crosses the antimeridian -- not the extra synthetic boundary
    points _split_path_at_antimeridian() adds to the PathLayer's own
    data. Those synthetic points have no real timestamp/altitude to
    report in a hover tooltip, so they must never leak into this layer.
    """
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
    # enough points that at least one survives the _HOVER_POINT_STRIDE thinning
    n = _HOVER_POINT_STRIDE * 2
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
    assert point["alt_str"] == "400.0"  # from _fake_track's linspace(400, 420, ...)


def test_hover_point_layer_is_thinned_but_path_layer_stays_full_resolution(
    ts: Timescale,
) -> None:
    """The visible line should use every sample (for a smooth-looking
    track); the pickable hover layer should be thinned to
    _HOVER_POINT_STRIDE (see build_globe_deck()'s docstring for why:
    hovering one specific point among hundreds of closely-spaced ones
    was verified to be impractical at full density)."""
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
    """
    The WebGL canvas's clear color (the "void" around the sphere, since
    map_provider=None means there's no basemap tile layer to fill it) is
    set to the app's own THEME_BACKGROUND_COLOR, normalized to deck.gl's
    0-1 parameter convention -- rather than left at deck.gl's own
    default, which would show as a plain rectangle around the globe that
    doesn't match the surrounding Streamlit page.
    """
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
    """
    CONFIRMED bug (observed directly in the running app, rotated to the
    Pacific/New Zealand view): a single PathLayer "path" spanning a
    +-180 degree crossing drew a spurious straight line the long way
    round (through 0 degrees) connecting the two broken ends, instead of
    the short hop across the seam -- see build_globe_deck()'s docstring
    for the full story of why an earlier version of this code assumed
    (incorrectly) that GlobeView needed no such split at all.

    The fix: one PathLayer, but multiple "path" entries in its `data`
    list, split at each crossing -- each sub-path capped with an
    interpolated point exactly on the meridian it was cut at, so the two
    halves still meet with no visible gap.
    """
    longitudes = [170.0, 179.0, -179.0, -170.0]  # crosses the antimeridian once
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, [10.0, 11.0, 12.0, 13.0], longitudes)}

    deck = build_globe_deck(ground_tracks)

    path_layer = _layers_by_type(deck, "PathLayer")[0]
    assert len(path_layer.data) == 2  # split into exactly two sub-paths by the one crossing

    first_path = path_layer.data[0]["path"]
    second_path = path_layer.data[1]["path"]
    # The original, real samples are preserved in order, split across
    # the two sub-paths at the crossing (between 179.0 and -179.0)...
    assert [lon for lon, _lat in first_path[:2]] == [170.0, 179.0]
    assert [lon for lon, _lat in second_path[-2:]] == [-179.0, -170.0]
    # ...and each sub-path ends/starts with an interpolated point placed
    # exactly on the meridian it was cut at -- +180 for the segment
    # moving east into the crossing, -180 for the one continuing from it
    # -- not at the original +-179 samples themselves.
    assert first_path[-1][0] == 180.0
    assert second_path[0][0] == -180.0
    # Both boundary points share the same (interpolated) latitude, so
    # the two halves visually meet with no gap at the meridian.
    assert first_path[-1][1] == pytest.approx(second_path[0][1])


def test_no_path_segment_jumps_more_than_180_degrees_in_longitude(ts: Timescale) -> None:
    """
    Direct regression test for the confirmed bug: across every PathLayer
    sub-path this module produces, no two consecutive points should ever
    be more than 180 degrees apart in longitude -- that's exactly the
    "long way round" wrong-direction line the antimeridian fix exists to
    eliminate. Covers a track with multiple crossings (plausible over a
    multi-orbit globe window), not just a single one.
    """
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
