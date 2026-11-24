"""Tests for satellite_pass_predictor.visibility.

Uses the frozen ISS TLE and the real Kourou observer over a fixed week. Event times are checked
against an independent brute-force reference: elevation sampled every second, with crossings
linearly interpolated between samples.
"""

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import KOUROU, MIN_PASS_ELEVATION_DEG
from satellite_pass_predictor.time_utils import build_time_grid
from satellite_pass_predictor.visibility import PassDict, compute_altaz, compute_passes

MASK = MIN_PASS_ELEVATION_DEG
SECOND = 1.0 / 86400.0  # in days
WEEK_HOURS = 24 * 7


@pytest.fixture(scope="module")
def week_passes(ts: Timescale, iss_satellite: EarthSatellite) -> list[PassDict]:
    return compute_passes(
        iss_satellite, KOUROU, ts, start_time=ts.utc(2026, 9, 21), duration_hours=WEEK_HOURS
    )


def _elevation(sat: EarthSatellite, ts: Timescale, jd: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(compute_altaz(sat, KOUROU, ts.tt_jd(jd))["elevation_deg"], dtype=np.float64)


def _reference_crossings(
    sat: EarthSatellite, ts: Timescale, center_jd: float, half_width_s: int = 90
) -> tuple[list[float], list[float]]:
    """Upward and downward mask crossings (TT JD) from 1 s samples within `half_width_s`."""
    jd = center_jd + np.arange(-half_width_s, half_width_s + 1) * SECOND
    above = _elevation(sat, ts, jd) - MASK
    up: list[float] = []
    down: list[float] = []
    for k in np.flatnonzero(np.diff(np.sign(above))):
        crossing = jd[k] - above[k] / (above[k + 1] - above[k]) * (jd[k + 1] - jd[k])
        (up if above[k + 1] > above[k] else down).append(float(crossing))
    return up, down


def _complete_pass(passes: list[PassDict], min_rise_to_peak_s: float = 120) -> PassDict:
    """A pass with no truncation and enough time between rise and peak to cut inside it."""
    return next(
        p
        for p in passes
        if not (p["start_truncated"] or p["end_truncated"])
        and (p["max_elevation_time"].tt - p["start_time"].tt) / SECOND > min_rise_to_peak_s
    )


def test_compute_altaz_returns_physically_valid_ranges(ts: Timescale, iss_satellite) -> None:
    """Catches swapped angles, radians-vs-degrees and sign errors."""
    t = build_time_grid(
        ts, start_time=ts.utc(2026, 9, 21, 0, 0, 0), duration_hours=24, step_minutes=5
    )
    altaz = compute_altaz(iss_satellite, KOUROU, t)

    assert np.all(altaz["elevation_deg"] >= -90.0) and np.all(altaz["elevation_deg"] <= 90.0)
    assert np.all(altaz["azimuth_deg"] >= 0.0) and np.all(altaz["azimuth_deg"] < 360.0)
    assert np.all(altaz["distance_km"] > 0.0)


def test_aos_and_los_match_brute_force_crossings(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    assert len(week_passes) > 10
    for p in week_passes:
        up, _ = _reference_crossings(iss_satellite, ts, p["start_time"].tt)
        _, down = _reference_crossings(iss_satellite, ts, p["end_time"].tt)
        assert len(up) == 1 and len(down) == 1
        assert abs(p["start_time"].tt - up[0]) / SECOND < 0.01
        assert abs(p["end_time"].tt - down[0]) / SECOND < 0.01
        assert p["duration_minutes"] * 60 == pytest.approx((down[0] - up[0]) / SECOND, abs=0.02)


def test_aos_and_los_sit_on_the_mask(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    """AOS/LOS are the first/last 1 ms tick at or above the mask."""
    for p in week_passes:
        edges = np.array([p["start_time"].tt, p["end_time"].tt])
        assert np.all(_elevation(iss_satellite, ts, edges) - MASK >= 0.0)
        assert np.all(_elevation(iss_satellite, ts, edges) - MASK < 1e-3)
        assert np.all(
            _elevation(iss_satellite, ts, edges + np.array([-1, 1]) * 1e-3 * SECOND) < MASK
        )


def test_max_elevation_matches_brute_force_peak(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    """The peak is at least every 1 s sample, and agrees with a 10 ms scan around it."""
    for p in week_passes:
        peak_jd = p["max_elevation_time"].tt
        coarse = _elevation(iss_satellite, ts, peak_jd + np.arange(-90, 91) * SECOND)
        fine = _elevation(iss_satellite, ts, peak_jd + np.arange(-200, 201) * 0.01 * SECOND)
        assert p["max_elevation_deg"] >= coarse.max() - 1e-9
        assert p["max_elevation_deg"] == pytest.approx(fine.max(), abs=1e-4)
        assert p["start_time"].tt < peak_jd < p["end_time"].tt


def test_no_pass_is_missed_or_spurious(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    """Every run of 30 s samples above the mask lies in exactly one pass, and vice versa."""
    t0 = ts.utc(2026, 9, 21)
    jd = t0.tt + np.arange(int(WEEK_HOURS * 3600 / 30) + 1) * 30 * SECOND
    above = np.concatenate(([False], _elevation(iss_satellite, ts, jd) >= MASK, [False]))
    edges = np.flatnonzero(np.diff(above.astype(int)))
    runs = list(zip(jd[edges[0::2]], jd[edges[1::2] - 1], strict=True))

    assert len(runs) == len(week_passes)
    for (run_start, run_end), p in zip(runs, week_passes, strict=True):
        assert p["start_time"].tt <= run_start and run_end <= p["end_time"].tt


def test_window_starting_after_the_peak_truncates_start_and_max(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    full = _complete_pass(week_passes)
    t0 = ts.tt_jd(full["max_elevation_time"].tt + 60 * SECOND)

    p = compute_passes(iss_satellite, KOUROU, ts, start_time=t0, duration_hours=1)[0]

    assert p["start_truncated"] and p["start_time"].tt == t0.tt
    assert p["max_elevation_truncated"] and p["max_elevation_time"].tt == t0.tt
    assert p["max_elevation_deg"] == pytest.approx(_elevation(iss_satellite, ts, t0.tt))
    assert p["max_elevation_deg"] < full["max_elevation_deg"]
    assert not p["end_truncated"] and p["end_time"].tt == full["end_time"].tt


def test_window_starting_before_the_peak_keeps_the_true_peak(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    full = _complete_pass(week_passes)
    t0 = ts.tt_jd(full["start_time"].tt + 30 * SECOND)

    p = compute_passes(iss_satellite, KOUROU, ts, start_time=t0, duration_hours=1)[0]

    assert p["start_truncated"] and p["start_time"].tt == t0.tt
    assert not p["max_elevation_truncated"]
    assert p["max_elevation_time"].tt == full["max_elevation_time"].tt
    assert p["max_elevation_deg"] == full["max_elevation_deg"]


def test_window_ending_before_the_peak_truncates_end_and_max(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    full = _complete_pass(week_passes)
    t0 = ts.tt_jd(full["start_time"].tt - 600 * SECOND)
    t1_jd = full["max_elevation_time"].tt - 60 * SECOND

    p = compute_passes(
        iss_satellite, KOUROU, ts, start_time=t0, duration_hours=(t1_jd - t0.tt) * 24
    )[-1]

    assert not p["start_truncated"] and p["start_time"].tt == full["start_time"].tt
    assert p["end_truncated"] and p["end_time"].tt == pytest.approx(t1_jd, abs=1e-6 * SECOND)
    assert p["max_elevation_truncated"] and p["max_elevation_time"].tt == p["end_time"].tt
    assert p["max_elevation_deg"] < full["max_elevation_deg"]


def test_window_ending_after_the_peak_keeps_the_true_peak(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    full = _complete_pass(week_passes)
    t0 = ts.tt_jd(full["start_time"].tt - 600 * SECOND)
    t1_jd = full["max_elevation_time"].tt + 60 * SECOND

    p = compute_passes(
        iss_satellite, KOUROU, ts, start_time=t0, duration_hours=(t1_jd - t0.tt) * 24
    )[-1]

    assert p["end_truncated"]
    assert not p["max_elevation_truncated"]
    assert p["max_elevation_time"].tt == full["max_elevation_time"].tt


def test_window_inside_a_pass_is_one_pass_truncated_at_both_ends(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    """No rise, culmination or set in the window: the pass comes from the elevation at t0."""
    full = _complete_pass(week_passes)
    t0 = ts.tt_jd(full["start_time"].tt + 20 * SECOND)

    passes = compute_passes(iss_satellite, KOUROU, ts, start_time=t0, duration_hours=1 / 60)

    assert len(passes) == 1
    p = passes[0]
    assert p["start_truncated"] and p["end_truncated"]
    # Still rising, so the highest in-window point is the window end.
    assert p["max_elevation_truncated"] and p["max_elevation_time"].tt == p["end_time"].tt


def test_several_culminations_report_the_highest(
    ts: Timescale, iss_satellite, week_passes: list[PassDict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second, lower culmination in the same pass doesn't displace the true peak."""
    full = _complete_pass(week_passes)
    t0 = ts.tt_jd(full["start_time"].tt - 600 * SECOND)
    real_find_events = iss_satellite.find_events

    def find_events_with_extra_culmination(*args, **kwargs):
        t, events = real_find_events(*args, **kwargs)
        extra_jd = full["start_time"].tt + 60 * SECOND
        jd = np.concatenate((t.tt, [extra_jd]))
        events = np.concatenate((events, np.array([1], dtype=events.dtype)))
        order = np.argsort(jd)
        return ts.tt_jd(jd[order]), events[order]

    monkeypatch.setattr(iss_satellite, "find_events", find_events_with_extra_culmination)
    p = compute_passes(iss_satellite, KOUROU, ts, start_time=t0, duration_hours=1)[0]

    assert p["max_elevation_time"].tt == full["max_elevation_time"].tt
    assert p["max_elevation_deg"] == full["max_elevation_deg"]
    assert not p["max_elevation_truncated"]


def test_a_day_has_several_ordered_disjoint_passes(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    passes = compute_passes(
        iss_satellite, KOUROU, ts, start_time=ts.utc(2026, 9, 22), duration_hours=24
    )

    assert len(passes) >= 2
    for earlier, later in zip(passes, passes[1:], strict=False):
        assert earlier["end_time"].tt < later["start_time"].tt
    for p in passes:
        assert p["start_time"].tt <= p["max_elevation_time"].tt <= p["end_time"].tt
        assert p["max_elevation_deg"] >= MASK
        assert 0.0 <= p["start_azimuth_deg"] < 360.0
        assert 0.0 <= p["end_azimuth_deg"] < 360.0


def test_a_window_between_passes_has_none(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    before, after = week_passes[1], week_passes[2]
    t0 = ts.tt_jd(before["end_time"].tt + 300 * SECOND)
    t1_jd = after["start_time"].tt - 300 * SECOND

    passes = compute_passes(
        iss_satellite, KOUROU, ts, start_time=t0, duration_hours=(t1_jd - t0.tt) * 24
    )

    assert passes == []


def test_pass_times_are_identical_across_reruns_and_window_starts(
    ts: Timescale, iss_satellite, week_passes: list[PassDict]
) -> None:
    """Event times sit on absolute ticks, so moving the window start doesn't move them at all."""

    def complete_pass_times(passes: list[PassDict]) -> dict[float, tuple[float, float, float]]:
        return {
            p["start_time"].tt: (
                p["end_time"].tt,
                p["max_elevation_time"].tt,
                p["max_elevation_deg"],
            )
            for p in passes
            if not (p["start_truncated"] or p["end_truncated"])
        }

    reference = complete_pass_times(week_passes)
    t0 = ts.utc(2026, 9, 21)
    for shift_s in (0.0, 0.3, 17 * 60 + 23.4, 5 * 3600 + 0.123):
        shifted = complete_pass_times(
            compute_passes(
                iss_satellite,
                KOUROU,
                ts,
                start_time=ts.tt_jd(t0.tt + shift_s * SECOND),
                duration_hours=WEEK_HOURS,
            )
        )
        common = reference.keys() & shifted.keys()
        assert len(common) >= len(reference) - 2
        assert all(shifted[start] == reference[start] for start in common)


def test_pass_flags_are_python_bools(week_passes: list[PassDict]) -> None:
    """numpy.bool_ fails `is True` checks and isn't JSON-serializable."""
    for p in week_passes:
        for field in ("start_truncated", "end_truncated", "max_elevation_truncated"):
            assert type(p[field]) is bool, f"{field} is {type(p[field])!r}, not bool"
