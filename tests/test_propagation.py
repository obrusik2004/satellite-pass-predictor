"""
Tests for satellite_pass_predictor.propagation.

FIXTURE-BASED: every test here uses the frozen `iss_satellite` fixture
(a real EarthSatellite parsed from tests/fixtures/iss_tle.txt), so these
exercise real SGP4 propagation end to end -- but never touch Celestrak,
since the TLE is a committed file, not a live fetch.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from skyfield.timelib import Timescale

from satellite_pass_predictor.propagation import (
    compute_ground_track,
    get_subpoint,
)

# A fixed instant near the fixture TLE's epoch (2026-09-20 ~18:54 UTC).
# SGP4 accuracy degrades over days-to-weeks as the real satellite drifts
# from what the TLE predicted, so evaluating close to the epoch keeps
# this a test of the *code*, not a referendum on how stale the fixture
# has become.
FIXED_TIME_ARGS = (2026, 9, 21, 0, 0, 0)


def test_get_subpoint_returns_plausible_iss_altitude(ts: Timescale, iss_satellite) -> None:
    """
    Sanity check on get_subpoint(): ISS orbits at roughly 400-430 km.
    A wildly different altitude (e.g. off by a factor of Earth's radius,
    or negative) would indicate a units or frame bug, not normal orbital
    variation -- ISS's altitude doesn't wander outside a fairly narrow
    band between reboosts.
    """
    t = ts.utc(*FIXED_TIME_ARGS)
    subpoint = get_subpoint(iss_satellite, t)
    assert 300 < subpoint["altitude_km"] < 450


def test_ground_track_latitude_stays_within_inclination_envelope(
    ts: Timescale, iss_satellite
) -> None:
    """
    Physical constraint: a satellite's ground-track latitude can never
    exceed its orbital inclination (~51.6 deg for ISS) -- the orbital
    plane's tilt relative to the equator is exactly what bounds how far
    north/south the sub-satellite point can reach, regardless of Earth's
    rotation underneath it.

    One real subtlety, discovered while writing this test rather than
    assumed: get_subpoint() reports *geodetic* latitude (WGS84-ellipsoid,
    the GPS-comparable convention -- see get_subpoint()'s docstring for
    why), while orbital inclination bounds *geocentric* latitude. Because
    Earth is an oblate ellipsoid, geodetic latitude runs slightly higher
    than geocentric latitude away from the equator and poles -- up to
    about 0.19 degrees at WGS84's flattening, peaking around 45 degrees
    latitude. So the correct assertion is inclination + a small ellipsoid
    tolerance, not a bare inclination bound (which this test failed
    against by about 0.16 degrees before that tolerance was added).
    """
    inclination_deg = np.degrees(iss_satellite.model.inclo)
    geodetic_vs_geocentric_tolerance_deg = 0.5  # comfortably covers the ~0.19 deg max

    t0 = ts.utc(*FIXED_TIME_ARGS)
    track = compute_ground_track(iss_satellite, ts, start_time=t0, duration_hours=24, step_minutes=1)

    max_abs_latitude = np.max(np.abs(track["latitude_deg"]))
    assert max_abs_latitude <= inclination_deg + geodetic_vs_geocentric_tolerance_deg


def test_ground_track_shape_matches_requested_grid(ts: Timescale, iss_satellite) -> None:
    """
    A 24h/1min grid should produce 24*60 + 1 = 1441 samples (both
    endpoints inclusive), and every returned array/the time grid itself
    should agree on that length.
    """
    t0 = ts.utc(*FIXED_TIME_ARGS)
    track = compute_ground_track(iss_satellite, ts, start_time=t0, duration_hours=24, step_minutes=1)

    # latitude/longitude/altitude are typed float | NDArray (see
    # propagation.FloatOrArray) since get_subpoint() can be called with a
    # scalar time too -- but compute_ground_track() always builds a
    # vectorized grid, so these are always arrays in practice, same as
    # the cast()s in the application code itself (propagation.py,
    # visibility.py, visualization.py).
    assert len(track["time"]) == 1441
    assert cast(NDArray[np.float64], track["latitude_deg"]).shape == (1441,)
    assert cast(NDArray[np.float64], track["longitude_deg"]).shape == (1441,)
    assert cast(NDArray[np.float64], track["altitude_km"]).shape == (1441,)


def test_ground_track_longitude_stays_in_valid_range(ts: Timescale, iss_satellite) -> None:
    """
    Longitude should always come back in Skyfield's standard (-180, 180]
    convention. This is really a guard on _split_at_antimeridian()'s
    assumption in visualization.py: that function only knows how to
    handle a +180/-180 wraparound, not e.g. a 0-360 convention.
    """
    t0 = ts.utc(*FIXED_TIME_ARGS)
    track = compute_ground_track(iss_satellite, ts, start_time=t0, duration_hours=24, step_minutes=1)

    assert np.all(track["longitude_deg"] > -180.0)
    assert np.all(track["longitude_deg"] <= 180.0)
