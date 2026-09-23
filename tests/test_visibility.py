"""
Tests for satellite_pass_predictor.visibility.

Split into two groups, named accordingly:

- test_find_passes_* : PURE LOGIC. Plain synthetic numpy elevation/
  azimuth arrays plus a synthetic Skyfield time grid built straight from
  ts.utc() + an offset -- no TLE, no EarthSatellite, no orbital mechanics
  at all. These are exact, hand-computed-expected-value tests of the
  threshold-crossing state machine in isolation.

- test_compute_altaz_and_compute_passes_* : FIXTURE-BASED. Use the real
  `iss_satellite` fixture (frozen TLE, no live fetch -- see conftest.py)
  and the real Kourou observer, exercising compute_altaz()/
  compute_passes() as actual integration tests of the full geometry.
  Since the exact pass times/count depend on the real (frozen, so still
  deterministic) orbital geometry, these check structural invariants
  rather than hardcoding exact expected values -- keeps them robust to
  e.g. a future Skyfield/SGP4 version producing a slightly different
  (still physically valid) numeric result.
"""

import numpy as np
import pytest
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import KOUROU, MIN_PASS_ELEVATION_DEG
from satellite_pass_predictor.time_utils import build_time_grid
from satellite_pass_predictor.visibility import compute_altaz, compute_passes, find_passes

# ---------------------------------------------------------------------------
# Pure logic: find_passes()'s state machine, no satellite/Skyfield object
# involved beyond a synthetic time grid.
# ---------------------------------------------------------------------------


def _synthetic_time_grid(ts: Timescale, n_samples: int):
    """1-minute-spaced synthetic time grid, `n_samples` long, for tests
    that only care about find_passes()'s index/threshold logic and need
    *some* real Skyfield Time to index into and subtract."""
    return build_time_grid(
        ts, start_time=ts.utc(2026, 1, 1, 0, 0, 0),
        duration_hours=(n_samples - 1) / 60, step_minutes=1,
    )


def test_find_passes_core_state_machine_on_a_clean_pass(ts: Timescale) -> None:
    """
    The "textbook" case: elevation rises above threshold, peaks, and
    falls back below, comfortably inside the sampled window. All three
    truncation/confidence flags should be False, and start/end/max
    should land on exactly the indices the raw numbers dictate.
    """
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
    """No samples above threshold at all -> no passes, not an error."""
    elevation = np.array([1, 2, 3, 4, 5], dtype=float)
    azimuth = np.zeros_like(elevation)
    t = _synthetic_time_grid(ts, len(elevation))

    assert find_passes(t, elevation, azimuth, min_elevation_deg=10.0) == []


def test_find_passes_pass_already_above_threshold_at_window_start(ts: Timescale) -> None:
    """
    Edge case: elevation is already >= threshold at sample 0. We can't
    know when it actually rose above threshold (that happened before our
    window started), so start_truncated should be True and start_time
    should be sample 0 -- "first time we can see it", not a guessed rise
    time.
    """
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
    """
    Edge case: elevation is still >= threshold -- and still *rising* --
    at the very last sample. end_truncated should be True (we don't know
    when/if it eventually dropped below threshold), and because the
    reported max is the last sample and it was still climbing,
    max_elevation_truncated should also be True: the real peak lies
    beyond the window and may be higher than what we observed.
    """
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
    assert p["max_elevation_truncated"]  # still rising at cutoff
    assert not p["low_confidence"]  # 5 samples >= 3


def test_find_passes_boolean_fields_are_genuine_python_bool_not_numpy_bool(
    ts: Timescale,
) -> None:
    """
    Regression test: max_elevation_truncated is computed via a chain of
    `and`s ending in a numpy array comparison (elev_segment[-1] >
    elev_segment[-2]), which -- without an explicit bool(...) wrap --
    produces a numpy.bool_ at runtime rather than the Python bool
    PassDict declares. numpy.bool_ is truthy-compatible (so it silently
    "worked" everywhere the codebase only ever checked truthiness), but
    it fails `is True`/`is False` identity checks and isn't accepted by
    json.dumps() (unlike numpy.float64, which genuinely subclasses
    float and *is* JSON-safe -- this is specifically a bool problem,
    since CPython doesn't allow subclassing bool at all).

    Uses the same "still rising at cutoff" scenario as
    test_find_passes_pass_still_above_threshold_at_window_end() above,
    since that's the only shape of input that actually reaches the
    numpy-comparison operand -- a check on any other scenario wouldn't
    have caught this.
    """
    elevation = np.array([2.0, 5.0, 8.0, 11.0, 16.0, 20.0, 25.0, 30.0], dtype=float)
    azimuth = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
    t = _synthetic_time_grid(ts, len(elevation))

    p = find_passes(t, elevation, azimuth, min_elevation_deg=10.0)[0]

    for field in ("start_truncated", "end_truncated", "max_elevation_truncated", "low_confidence"):
        assert type(p[field]) is bool, f"{field} is {type(p[field])!r}, not bool"


def test_find_passes_short_marginal_crossing_flagged_low_confidence(ts: Timescale) -> None:
    """
    Edge case: only a single sample grazes above threshold. This is not
    treated as sampling noise and dropped -- SGP4's output is a smooth,
    deterministic curve, so a brief graze is a real (if marginal) pass --
    but with only one sample we can't resolve *when* within that minute
    the threshold was actually crossed, so it's flagged low_confidence
    rather than reported as if fully resolved.
    """
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


# ---------------------------------------------------------------------------
# Fixture-based: compute_altaz() / compute_passes() against the real,
# frozen ISS TLE and the real Kourou observer.
# ---------------------------------------------------------------------------


def test_compute_altaz_returns_physically_valid_ranges(ts: Timescale, iss_satellite) -> None:
    """
    Whatever the actual numbers are, elevation/azimuth/distance must fall
    within their physically valid ranges -- this would catch e.g. a
    swapped alt/az, a radians-vs-degrees bug, or a sign error.
    """
    t = build_time_grid(ts, start_time=ts.utc(2026, 9, 21, 0, 0, 0), duration_hours=24, step_minutes=5)
    altaz = compute_altaz(iss_satellite, KOUROU, t)

    assert np.all(altaz["elevation_deg"] >= -90.0) and np.all(altaz["elevation_deg"] <= 90.0)
    assert np.all(altaz["azimuth_deg"] >= 0.0) and np.all(altaz["azimuth_deg"] < 360.0)
    assert np.all(altaz["distance_km"] > 0.0)


def test_compute_passes_detects_at_least_one_pass_in_a_week(ts: Timescale, iss_satellite) -> None:
    """
    ISS's ~51.6 deg inclination ground track sweeps well past Kourou's
    ~5.2 deg latitude on essentially every orbit (~90 min), so over a
    full week it is a near-certainty that at least one pass clears the
    10 deg threshold. This is a light integration smoke test, not a
    precise prediction check -- see test_find_passes_* above for exact
    value assertions on the underlying logic.
    """
    t0 = ts.utc(2026, 9, 21, 0, 0, 0)
    passes = compute_passes(iss_satellite, KOUROU, ts, start_time=t0, duration_hours=24 * 7, step_minutes=1)

    assert len(passes) > 0
    for p in passes:
        assert p["start_time"].tt <= p["end_time"].tt
        assert p["max_elevation_deg"] >= MIN_PASS_ELEVATION_DEG
        assert 0.0 <= p["start_azimuth_deg"] < 360.0
        assert 0.0 <= p["end_azimuth_deg"] < 360.0
