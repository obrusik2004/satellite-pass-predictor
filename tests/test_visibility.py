"""Tests for satellite_pass_predictor.visibility.

The find_passes tests are pure logic on synthetic arrays; the compute_* tests use the frozen
ISS TLE and the real Kourou observer, and check invariants rather than exact pass times.
"""

import numpy as np
import pytest
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import KOUROU, MIN_PASS_ELEVATION_DEG
from satellite_pass_predictor.time_utils import build_time_grid
from satellite_pass_predictor.visibility import compute_altaz, compute_passes, find_passes

# find_passes(): the threshold-crossing logic, on a synthetic time grid.


def _synthetic_time_grid(ts: Timescale, n_samples: int):
    """A 1-minute-spaced time grid of `n_samples` for indexing and subtraction."""
    return build_time_grid(
        ts,
        start_time=ts.utc(2026, 1, 1, 0, 0, 0),
        duration_hours=(n_samples - 1) / 60,
        step_minutes=1,
    )


def test_find_passes_core_state_machine_on_a_clean_pass(ts: Timescale) -> None:
    """A pass fully inside the window has no flags and lands on the expected indices."""
    elevation = np.array([2, 5, 12, 18, 22, 15, 8, 3], dtype=float)
    azimuth = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
    t = _synthetic_time_grid(ts, len(elevation))

    passes = find_passes(t, elevation, azimuth, min_elevation_deg=10.0)

    assert len(passes) == 1
    p = passes[0]
    assert p["start_time"].tt == t[2].tt
    assert p["start_azimuth_deg"] == 90.0
    assert not p["start_truncated"]
    assert p["end_time"].tt == t[5].tt
    assert p["end_azimuth_deg"] == 225.0
    assert not p["end_truncated"]
    assert p["max_elevation_deg"] == 22.0
    assert p["max_elevation_time"].tt == t[4].tt
    assert not p["max_elevation_truncated"]
    assert p["duration_minutes"] == pytest.approx(3.0)  # index 5 - index 2 = 3 minutes
    assert not p["low_confidence"]  # 4 samples >= MIN_PASS_SAMPLES_FOR_CONFIDENCE


def test_find_passes_returns_empty_list_when_never_above_threshold(ts: Timescale) -> None:
    elevation = np.array([1, 2, 3, 4, 5], dtype=float)
    azimuth = np.zeros_like(elevation)
    t = _synthetic_time_grid(ts, len(elevation))

    assert find_passes(t, elevation, azimuth, min_elevation_deg=10.0) == []


def test_find_passes_pass_already_above_threshold_at_window_start(ts: Timescale) -> None:
    """Above the mask at sample 0: start_truncated, with the first sample as the start."""
    elevation = np.array([15.0, 12.0, 5.0, 2.0], dtype=float)
    azimuth = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    t = _synthetic_time_grid(ts, len(elevation))

    passes = find_passes(t, elevation, azimuth, min_elevation_deg=10.0)

    assert len(passes) == 1
    p = passes[0]
    assert p["start_truncated"]
    assert p["start_time"].tt == t[0].tt
    assert p["start_azimuth_deg"] == 10.0
    assert not p["end_truncated"]
    assert p["end_time"].tt == t[1].tt


def test_find_passes_pass_still_above_threshold_at_window_end(ts: Timescale) -> None:
    """Still rising at the last sample: end_truncated and max_elevation_truncated."""
    elevation = np.array([2.0, 5.0, 8.0, 11.0, 16.0, 20.0, 25.0, 30.0], dtype=float)
    azimuth = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
    t = _synthetic_time_grid(ts, len(elevation))

    passes = find_passes(t, elevation, azimuth, min_elevation_deg=10.0)

    assert len(passes) == 1
    p = passes[0]
    assert not p["start_truncated"]
    assert p["start_time"].tt == t[3].tt  # first sample >= 10.0
    assert p["end_truncated"]
    assert p["end_time"].tt == t[7].tt  # last sample in the window
    assert p["max_elevation_deg"] == 30.0
    assert p["max_elevation_time"].tt == t[7].tt
    assert p["max_elevation_truncated"]
    assert not p["low_confidence"]  # 5 samples >= 3


def test_find_passes_boolean_fields_are_genuine_python_bool_not_numpy_bool(
    ts: Timescale,
) -> None:
    """The flags must be real bools: numpy.bool_ fails `is True` checks and isn't JSON-serializable.

    Uses the still-rising case, the only one that reaches the numpy comparison.
    """
    elevation = np.array([2.0, 5.0, 8.0, 11.0, 16.0, 20.0, 25.0, 30.0], dtype=float)
    azimuth = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
    t = _synthetic_time_grid(ts, len(elevation))

    p = find_passes(t, elevation, azimuth, min_elevation_deg=10.0)[0]

    for field in ("start_truncated", "end_truncated", "max_elevation_truncated", "low_confidence"):
        assert type(p[field]) is bool, f"{field} is {type(p[field])!r}, not bool"


def test_find_passes_short_marginal_crossing_flagged_low_confidence(ts: Timescale) -> None:
    """A single sample above the mask is kept, but flagged low_confidence."""
    elevation = np.array([5.0, 12.0, 4.0], dtype=float)
    azimuth = np.array([100.0, 110.0, 120.0], dtype=float)
    t = _synthetic_time_grid(ts, len(elevation))

    passes = find_passes(t, elevation, azimuth, min_elevation_deg=10.0)

    assert len(passes) == 1
    p = passes[0]
    assert not p["start_truncated"]
    assert not p["end_truncated"]
    assert p["start_time"].tt == t[1].tt
    assert p["end_time"].tt == t[1].tt
    assert p["duration_minutes"] == pytest.approx(0.0)
    assert p["low_confidence"]  # 1 sample < MIN_PASS_SAMPLES_FOR_CONFIDENCE (3)


# compute_altaz() / compute_passes() against the frozen ISS TLE.


def test_compute_altaz_returns_physically_valid_ranges(ts: Timescale, iss_satellite) -> None:
    """Catches swapped angles, radians-vs-degrees and sign errors."""
    t = build_time_grid(
        ts, start_time=ts.utc(2026, 9, 21, 0, 0, 0), duration_hours=24, step_minutes=5
    )
    altaz = compute_altaz(iss_satellite, KOUROU, t)

    assert np.all(altaz["elevation_deg"] >= -90.0) and np.all(altaz["elevation_deg"] <= 90.0)
    assert np.all(altaz["azimuth_deg"] >= 0.0) and np.all(altaz["azimuth_deg"] < 360.0)
    assert np.all(altaz["distance_km"] > 0.0)


def test_compute_passes_detects_at_least_one_pass_in_a_week(ts: Timescale, iss_satellite) -> None:
    """The ISS crosses Kourou's sky on most orbits, so a week always contains a pass."""
    t0 = ts.utc(2026, 9, 21, 0, 0, 0)
    passes = compute_passes(
        iss_satellite, KOUROU, ts, start_time=t0, duration_hours=24 * 7, step_minutes=1
    )

    assert len(passes) > 0
    for p in passes:
        assert p["start_time"].tt <= p["end_time"].tt
        assert p["max_elevation_deg"] >= MIN_PASS_ELEVATION_DEG
        assert 0.0 <= p["start_azimuth_deg"] < 360.0
        assert 0.0 <= p["end_azimuth_deg"] < 360.0
