"""Tests for satellite_pass_predictor.propagation, using the frozen ISS TLE (no network)."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from skyfield.timelib import Timescale

from satellite_pass_predictor.propagation import (
    compute_ground_track,
    get_subpoint,
)

# Close to the fixture TLE's epoch (2026-09-20 ~18:54 UTC), where SGP4 is most accurate.
FIXED_TIME_ARGS = (2026, 9, 21, 0, 0, 0)


def test_get_subpoint_returns_plausible_iss_altitude(ts: Timescale, iss_satellite) -> None:
    """A wildly wrong altitude would indicate a units or frame bug."""
    t = ts.utc(*FIXED_TIME_ARGS)
    subpoint = get_subpoint(iss_satellite, t)
    assert 300 < subpoint["altitude_km"] < 450


def test_ground_track_latitude_stays_within_inclination_envelope(
    ts: Timescale, iss_satellite
) -> None:
    """Ground-track latitude can't exceed the orbital inclination.

    get_subpoint() returns geodetic latitude while inclination bounds geocentric latitude, which
    differs by up to about 0.19 degrees, so a small tolerance is allowed.
    """
    inclination_deg = np.degrees(iss_satellite.model.inclo)
    geodetic_vs_geocentric_tolerance_deg = 0.5

    t0 = ts.utc(*FIXED_TIME_ARGS)
    track = compute_ground_track(
        iss_satellite, ts, start_time=t0, duration_hours=24, step_minutes=1
    )

    max_abs_latitude = np.max(np.abs(track["latitude_deg"]))
    assert max_abs_latitude <= inclination_deg + geodetic_vs_geocentric_tolerance_deg


def test_ground_track_shape_matches_requested_grid(ts: Timescale, iss_satellite) -> None:
    """A 24h/1min grid has 24*60 + 1 = 1441 samples, and every array agrees."""
    t0 = ts.utc(*FIXED_TIME_ARGS)
    track = compute_ground_track(
        iss_satellite, ts, start_time=t0, duration_hours=24, step_minutes=1
    )

    assert len(track["time"]) == 1441
    assert cast(NDArray[np.float64], track["latitude_deg"]).shape == (1441,)
    assert cast(NDArray[np.float64], track["longitude_deg"]).shape == (1441,)
    assert cast(NDArray[np.float64], track["altitude_km"]).shape == (1441,)


def test_ground_track_longitude_stays_in_valid_range(ts: Timescale, iss_satellite) -> None:
    """Longitude is in (-180, 180], which the antimeridian handling relies on."""
    t0 = ts.utc(*FIXED_TIME_ARGS)
    track = compute_ground_track(
        iss_satellite, ts, start_time=t0, duration_hours=24, step_minutes=1
    )

    assert np.all(track["longitude_deg"] > -180.0)
    assert np.all(track["longitude_deg"] <= 180.0)
