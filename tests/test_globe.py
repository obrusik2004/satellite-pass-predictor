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
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import (
    KOUROU_ELEVATION_M,
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
)
from satellite_pass_predictor.globe import _HOVER_POINT_STRIDE, build_globe_deck
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
    satellite_layer = next(layer for layer in scatter_layers if layer.data[0]["name"] == "ISS (ZARYA)")

    assert kourou_layer.get_position == [KOUROU_LONGITUDE_DEG, KOUROU_LATITUDE_DEG]
    # Distinct from every satellite layer in color: Kourou is the ground
    # station, not a satellite, and should read as visually different at
    # a glance. (pydeck's ScatterplotLayer only draws circles, so
    # "symbol" distinctness here means color + size + stroke outline --
    # kourou_layer.stroked below -- rather than a literal star shape;
    # see build_globe_deck()'s docstring for that trade-off.)
    assert kourou_layer.get_fill_color != satellite_layer.data[0]["color"]
    assert kourou_layer.stroked is True


def test_hover_points_contain_name_time_latlon_and_altitude(ts: Timescale) -> None:
    # enough points that at least one survives the _HOVER_POINT_STRIDE thinning
    n = _HOVER_POINT_STRIDE * 2
    lats = [12.34] * n
    lons = [56.78] * n
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, lats, lons)}

    deck = build_globe_deck(ground_tracks)

    scatter_layers = _layers_by_type(deck, "ScatterplotLayer")
    satellite_layer = next(layer for layer in scatter_layers if layer.data[0]["name"] == "ISS (ZARYA)")
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
        layer for layer in _layers_by_type(deck, "ScatterplotLayer")
        if layer.data[0]["name"] == "ISS (ZARYA)"
    )

    assert len(path_layer.data[0]["path"]) == n
    assert len(scatter_layer.data) == len(range(0, n, _HOVER_POINT_STRIDE))


def test_uses_globe_view_centered_near_kourou(ts: Timescale) -> None:
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, [10.0], [20.0])}

    deck = build_globe_deck(ground_tracks)

    assert len(deck.views) == 1
    assert deck.views[0].type == "_GlobeView"
    assert deck.initial_view_state.latitude == KOUROU_LATITUDE_DEG
    assert deck.initial_view_state.longitude == KOUROU_LONGITUDE_DEG


def test_antimeridian_crossing_data_passes_through_unsplit(ts: Timescale) -> None:
    """
    Unlike visualization._split_at_antimeridian(), build_globe_deck()
    should NOT insert any break at a +-180 degree longitude jump: on a
    true sphere there's no seam to break at (verified directly -- see
    build_globe_deck()'s docstring -- that GlobeView interpolates along
    the short 3D chord between points, not a 2D lon/lat line, so it
    never draws the "wrong way around"). A longitude sequence that
    crosses the antimeridian should come out in the path exactly as
    given, with no points inserted or removed.
    """
    longitudes = [170.0, 179.0, -179.0, -170.0]  # crosses the antimeridian
    ground_tracks = {"ISS (ZARYA)": _fake_track(ts, [10.0, 11.0, 12.0, 13.0], longitudes)}

    deck = build_globe_deck(ground_tracks)

    path_layer = _layers_by_type(deck, "PathLayer")[0]
    path = path_layer.data[0]["path"]
    assert [lon for lon, _lat in path] == longitudes
    assert len(path) == len(longitudes)
