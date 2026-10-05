"""Tests for satellite_pass_predictor.skyplot.

Uses the frozen ISS TLE and the real Kourou observer. Figure data is compared against an
independently recomputed look-angle array.
"""

import numpy as np
import pytest
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import KOUROU, SATELLITE_COLORS
from satellite_pass_predictor.skyplot import SKY_PLOT_STEP_SECONDS, build_sky_plot_figure
from satellite_pass_predictor.time_utils import build_time_grid
from satellite_pass_predictor.visibility import PassDict, compute_altaz, compute_passes

_ISS_NAME = "ISS (ZARYA)"


def _a_real_pass_with_some_duration(ts: Timescale, iss_satellite) -> PassDict:
    """A detected pass with more than one sample, so the arc isn't a single-point graze."""
    passes = compute_passes(
        iss_satellite,
        KOUROU,
        ts,
        start_time=ts.utc(2026, 9, 21, 0, 0, 0),
        duration_hours=24 * 7,
        step_minutes=1,
    )
    return next(p for p in passes if p["duration_minutes"] > 1.0)


def _line_trace(fig):
    return next(tr for tr in fig.data if tr.mode == "lines")


def _marker_traces(fig):
    return [tr for tr in fig.data if tr.mode == "markers"]


def test_track_color_comes_from_the_shared_satellite_color_mapping(
    ts: Timescale, iss_satellite
) -> None:
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    assert _line_trace(fig).line.color == SATELLITE_COLORS[_ISS_NAME]


def test_radius_is_90_minus_elevation_and_zenith_maps_to_plot_center(
    ts: Timescale, iss_satellite
) -> None:
    """Radius is 90 - elevation, so the highest point is nearest the center."""
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    duration_hours = (pass_["end_time"].tt - pass_["start_time"].tt) * 24.0
    t = build_time_grid(
        ts,
        start_time=pass_["start_time"],
        duration_hours=duration_hours,
        step_minutes=SKY_PLOT_STEP_SECONDS / 60.0,
    )
    altaz = compute_altaz(iss_satellite, KOUROU, t)
    expected_radius = 90.0 - altaz["elevation_deg"]

    line = _line_trace(fig)
    assert np.allclose(line.r, expected_radius)
    assert np.allclose(line.theta, altaz["azimuth_deg"])
    highest_elevation_idx = int(np.argmax(altaz["elevation_deg"]))
    assert int(np.argmin(line.r)) == highest_elevation_idx


def test_radial_axis_spans_horizon_to_zenith_with_inverted_labels(
    ts: Timescale, iss_satellite
) -> None:
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    radial = fig.layout.polar.radialaxis
    assert tuple(radial.range) == (0, 90)
    assert list(radial.tickvals) == [0, 30, 60, 90]
    assert list(radial.ticktext) == ["90°", "60°", "30°", "0°"]


def test_angular_axis_uses_compass_convention_not_math_convention(
    ts: Timescale, iss_satellite
) -> None:
    """North at the top, increasing clockwise."""
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    angular = fig.layout.polar.angularaxis
    assert angular.rotation == 90
    assert angular.direction == "clockwise"
    assert list(angular.tickvals) == [0, 90, 180, 270]
    assert list(angular.ticktext) == ["N", "E", "S", "W"]


def test_rise_and_set_markers_present_distinct_and_at_pass_endpoints(
    ts: Timescale, iss_satellite
) -> None:
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    markers = _marker_traces(fig)
    assert len(markers) == 2
    names = {tr.name for tr in markers}
    assert names == {"AOS (rise)", "LOS (set)"}

    rise = next(tr for tr in markers if tr.name == "AOS (rise)")
    set_ = next(tr for tr in markers if tr.name == "LOS (set)")
    assert rise.marker.symbol != set_.marker.symbol
    assert rise.theta[0] == pytest.approx(pass_["start_azimuth_deg"])
    assert set_.theta[0] == pytest.approx(pass_["end_azimuth_deg"])


def test_recomputes_at_finer_resolution_than_the_tables_one_minute_grid(
    ts: Timescale, iss_satellite
) -> None:
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)
    coarse_sample_count = int(pass_["duration_minutes"]) + 1  # what a 1-min grid would give

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    assert len(_line_trace(fig).r) > coarse_sample_count
