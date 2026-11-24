"""Tests for satellite_pass_predictor.skyplot.

Uses the frozen ISS TLE and the real Kourou observer. Figure data is compared against look
angles recomputed at the pass's own AOS and LOS.
"""

import numpy as np
import pytest
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import KOUROU, MIN_PASS_ELEVATION_DEG, SATELLITE_COLORS
from satellite_pass_predictor.skyplot import SKY_PLOT_STEP_SECONDS, build_sky_plot_figure
from satellite_pass_predictor.visibility import PassDict, compute_altaz, compute_passes

_ISS_NAME = "ISS (ZARYA)"


def _a_real_pass_with_some_duration(ts: Timescale, iss_satellite) -> PassDict:
    """A complete pass longer than a minute, so the arc isn't a single-point graze."""
    passes = compute_passes(
        iss_satellite,
        KOUROU,
        ts,
        start_time=ts.utc(2026, 9, 21, 0, 0, 0),
        duration_hours=24 * 7,
    )
    return next(
        p
        for p in passes
        if p["duration_minutes"] > 1.0 and not (p["start_truncated"] or p["end_truncated"])
    )


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

    line = _line_trace(fig)
    elevation = np.asarray(line.customdata)
    assert np.allclose(line.r, 90.0 - elevation)
    assert int(np.argmin(line.r)) == int(np.argmax(elevation))
    assert elevation.max() == pytest.approx(pass_["max_elevation_deg"], abs=0.1)


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


def test_track_runs_from_exact_aos_to_exact_los_on_the_mask(ts: Timescale, iss_satellite) -> None:
    """The first and last samples are AOS and LOS themselves, so both markers sit on the mask."""
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    line = _line_trace(fig)
    edges = compute_altaz(
        iss_satellite, KOUROU, ts.tt_jd([pass_["start_time"].tt, pass_["end_time"].tt])
    )
    assert np.array_equal([line.theta[0], line.theta[-1]], edges["azimuth_deg"])
    assert np.array_equal([line.customdata[0], line.customdata[-1]], edges["elevation_deg"])
    for marker in _marker_traces(fig):
        assert marker.customdata[0] >= MIN_PASS_ELEVATION_DEG
        assert marker.customdata[0] == pytest.approx(MIN_PASS_ELEVATION_DEG, abs=1e-3)


def test_samples_are_no_further_apart_than_the_step(ts: Timescale, iss_satellite) -> None:
    pass_ = _a_real_pass_with_some_duration(ts, iss_satellite)

    fig = build_sky_plot_figure(iss_satellite, pass_, KOUROU, ts, _ISS_NAME)

    sample_count = len(_line_trace(fig).r)
    assert (sample_count - 1) * SKY_PLOT_STEP_SECONDS >= pass_["duration_minutes"] * 60
