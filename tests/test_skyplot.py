"""
Tests for satellite_pass_predictor.skyplot.

Same fixture-based approach as test_visibility.py's
test_compute_altaz_and_compute_passes_* group: uses the real
`iss_satellite` fixture (frozen TLE, no live fetch) and the real Kourou
observer, so a genuine PassDict with genuine start/end times feeds
build_sky_plot_figure(). Since compute_altaz() is deterministic given a
fixed satellite/observer/time, the figure's data can be checked against
an independently-recomputed expected array rather than a hand-picked
constant -- exact equality, not just "some plausible shape".
"""

import numpy as np
import pytest
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import KOUROU
from satellite_pass_predictor.skyplot import SKY_PLOT_STEP_SECONDS, build_sky_plot_figure
from satellite_pass_predictor.time_utils import build_time_grid
from satellite_pass_predictor.visibility import PassDict, compute_altaz, compute_passes


def _a_real_pass_with_some_duration(ts: Timescale, iss_satellite) -> PassDict:
    """
    A genuine detected pass, long enough to have more than one sample --
    tests that care about a smooth arc (not a single-point edge case)
    use this rather than risking a single-sample low_confidence graze.
    """
    passes = compute_passes(
        iss_satellite, KOUROU, ts,
        start_time=ts.utc(2026, 9, 21, 0, 0, 0), duration_hours=24 * 7, step_minutes=1,
    )
    return next(p for p in passes if p["duration_minutes"] > 1.0)


def _line_trace(fig):
    return next(tr for tr in fig.data if tr.mode == "lines")


def _marker_traces(fig):
    return [tr for tr in fig.data if tr.mode == "markers"]


def test_radius_is_90_minus_elevation_and_zenith_maps_to_plot_center(
    ts: Timescale, iss_satellite
) -> None:
    """
    The core convention this whole chart depends on: radius = 90 -
    elevation, so the highest-elevation sample (closest to zenith) gets
    the *smallest* radius (closest to the plot's center) and the
    lowest-elevation sample gets the largest radius (closest to the
    outer edge) -- inverted from a naive "radius = elevation" mapping.
    """
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts)

    # Recomputed independently, over the same window/step, rather than
    # trusting build_sky_plot_figure()'s own internals -- same geometry,
    # same deterministic function, so this must match exactly.
    duration_hours = (pass_["end_time"].tt - pass_["start_time"].tt) * 24.0
    t = build_time_grid(
        ts, start_time=pass_["start_time"],
        duration_hours=duration_hours, step_minutes=SKY_PLOT_STEP_SECONDS / 60.0,
    )
    altaz = compute_altaz(iss_satellite, KOUROU, t)
    expected_radius = 90.0 - altaz["elevation_deg"]

    line = _line_trace(fig)
    assert np.allclose(line.r, expected_radius)
    assert np.allclose(line.theta, altaz["azimuth_deg"])
    # The point nearest the plot's center (smallest radius) is the one
    # with the highest elevation, not the one with the lowest -- this is
    # the actual "zenith at center" claim, checked directly rather than
    # inferred from the formula alone.
    highest_elevation_idx = int(np.argmax(altaz["elevation_deg"]))
    assert int(np.argmin(line.r)) == highest_elevation_idx


def test_radial_axis_spans_horizon_to_zenith_with_inverted_labels(
    ts: Timescale, iss_satellite
) -> None:
    """The radial axis covers the full [0, 90] radius range, labeled
    with the elevation each radius represents -- 90 deg (zenith) at
    radius 0 (the center), 0 deg (horizon) at radius 90 (the edge)."""
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts)

    radial = fig.layout.polar.radialaxis
    assert tuple(radial.range) == (0, 90)
    assert list(radial.tickvals) == [0, 30, 60, 90]
    assert list(radial.ticktext) == ["90°", "60°", "30°", "0°"]


def test_angular_axis_uses_compass_convention_not_math_convention(
    ts: Timescale, iss_satellite
) -> None:
    """
    0 deg azimuth (North) should sit at the top of the chart with angle
    increasing clockwise through E/S/W -- not Plotly's polar-chart
    default (0 on the right, increasing counterclockwise).
    """
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts)

    angular = fig.layout.polar.angularaxis
    assert angular.rotation == 90
    assert angular.direction == "clockwise"
    assert list(angular.tickvals) == [0, 90, 180, 270]
    assert list(angular.ticktext) == ["N", "E", "S", "W"]


def test_rise_and_set_markers_present_distinct_and_at_pass_endpoints(
    ts: Timescale, iss_satellite
) -> None:
    """Two marker traces, visually distinct from each other, positioned
    at the pass's own recorded start/end azimuth -- so the direction of
    travel across the sky is clear even without reading the line."""
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts)

    markers = _marker_traces(fig)
    assert len(markers) == 2
    names = {tr.name for tr in markers}
    assert names == {"Rise", "Set"}

    rise = next(tr for tr in markers if tr.name == "Rise")
    set_ = next(tr for tr in markers if tr.name == "Set")
    assert rise.marker.color != set_.marker.color
    assert rise.marker.symbol != set_.marker.symbol
    assert rise.theta[0] == pytest.approx(pass_["start_azimuth_deg"])
    assert set_.theta[0] == pytest.approx(pass_["end_azimuth_deg"])


def test_recomputes_at_finer_resolution_than_the_tables_one_minute_grid(
    ts: Timescale, iss_satellite
) -> None:
    """
    The pass table (app.py) detects passes on a step_minutes=1 grid --
    too coarse for a smooth-looking arc over a pass that may only last a
    few minutes. build_sky_plot_figure() should recompute at its own,
    finer SKY_PLOT_STEP_SECONDS resolution, not just reuse the table's
    two (start, end) samples or a per-minute grid.
    """
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)
    coarse_sample_count = int(pass_["duration_minutes"]) + 1  # what a 1-min grid would give

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts)

    assert len(_line_trace(fig).r) > coarse_sample_count
