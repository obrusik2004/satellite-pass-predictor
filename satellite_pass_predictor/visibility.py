"""
Visibility: topocentric elevation/azimuth from a ground observer, and
detecting visibility passes (contiguous stretches above an elevation
threshold) from that.
"""

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time, Timescale
from skyfield.toposlib import GeographicPosition

from .config import MIN_PASS_ELEVATION_DEG, MIN_PASS_SAMPLES_FOR_CONFIDENCE
from .time_utils import build_time_grid

# See propagation.py's FloatOrArray for why this isn't just `float`:
# compute_altaz() is called both with a single time (a scalar result)
# and a vectorized time grid (an array result).
FloatOrArray = float | NDArray[np.float64]


class AltAzDict(TypedDict):
    """Topocentric look angles: compute_altaz()'s fixed, known keys."""
    elevation_deg: FloatOrArray
    azimuth_deg: FloatOrArray
    distance_km: FloatOrArray


class PassDict(TypedDict):
    """One detected visibility pass: find_passes()'s fixed, known keys."""
    start_time: Time
    start_azimuth_deg: float
    start_truncated: bool
    end_time: Time
    end_azimuth_deg: float
    end_truncated: bool
    max_elevation_deg: float
    max_elevation_time: Time
    max_elevation_truncated: bool
    duration_minutes: float
    low_confidence: bool


def compute_altaz(
    sat: EarthSatellite, observer: GeographicPosition, t: Time
) -> AltAzDict:
    """
    Compute a satellite's elevation, azimuth, and range as seen from
    `observer` at time(s) `t`.

    get_subpoint() answers "where on Earth is the satellite" (geocentric).
    This answers the different question a ground station actually cares
    about -- "where do I have to point to see it" -- which needs a
    topocentric (observer-relative) position instead. `sat - observer`
    builds that relative geometry; `.at(t)` evaluates it; `.altaz()`
    converts it into altitude (elevation above the horizon), azimuth
    (compass bearing), and range, correctly accounting for the observer's
    position on the WGS84 ellipsoid and Earth's rotation at time `t`.

    Returns a dict of elevation_deg, azimuth_deg, distance_km (arrays if
    `t` is a vectorized time).
    """
    topocentric = (sat - observer).at(t)
    alt, az, distance = topocentric.altaz()
    return {
        "elevation_deg": alt.degrees,
        "azimuth_deg": az.degrees,
        "distance_km": distance.km,
    }


def find_passes(
    t: Time,
    elevation_deg: NDArray[np.float64],
    azimuth_deg: NDArray[np.float64],
    min_elevation_deg: float = MIN_PASS_ELEVATION_DEG,
) -> list[PassDict]:
    """
    Detect visibility passes: contiguous stretches of the sampled time
    grid where elevation stays at or above `min_elevation_deg`.

    Edge cases, handled deliberately rather than left implicit:

    - Pass already in progress at the start of the window (elevation is
      already above threshold at sample 0): we can't know when it
      actually rose above the threshold, since that happened before our
      window starts. Rather than guessing, the pass is reported with
      start_truncated=True and start_time/start_azimuth taken from
      sample 0 -- i.e. "first time we can see it", clearly marked as not
      the true rise time.
    - Pass still in progress at the end of the window (elevation is still
      above threshold at the last sample): symmetric handling,
      end_truncated=True, end_time/end_azimuth taken from the last
      sample. Additionally, if elevation is still *rising* at that last
      sample, the true peak lies beyond our window and hasn't been seen
      yet -- flagged as max_elevation_truncated=True so the reported
      max_elevation_deg isn't mistaken for the actual peak.
    - Two threshold crossings only 1-2 samples apart (a pass that barely
      grazes the threshold): this is not assumed to be sampling noise and
      silently dropped -- SGP4's output is a smooth, deterministic curve,
      not a noisy signal, so a genuine brief graze above 10 degrees is a
      real (if marginal) pass. But with only a couple of samples, we
      literally cannot resolve *when* within that ~1-minute step the
      threshold was actually crossed, so the reported start/end/max for
      such a pass are only accurate to about one sampling step. These are
      kept in the results but flagged low_confidence=True rather than
      dropped, with a suggestion to re-run with a finer step_minutes over
      just that time range to confirm.

    Returns a list of dicts, one per detected pass, each with:
      start_time, start_azimuth_deg, start_truncated
      end_time, end_azimuth_deg, end_truncated
      max_elevation_deg, max_elevation_time, max_elevation_truncated
      duration_minutes, low_confidence
    """
    elevation_deg = np.asarray(elevation_deg, dtype=float)
    azimuth_deg = np.asarray(azimuth_deg, dtype=float)
    above = elevation_deg >= min_elevation_deg
    n = len(elevation_deg)

    passes: list[PassDict] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue

        start_idx = i
        start_truncated = start_idx == 0

        j = start_idx
        while j < n and above[j]:
            j += 1
        end_idx = j - 1
        end_truncated = end_idx == n - 1

        elev_segment = elevation_deg[start_idx:end_idx + 1]
        n_samples = end_idx - start_idx + 1
        max_local_idx = start_idx + int(np.argmax(elev_segment))

        # If we were cut off at the end of the window while elevation was
        # still climbing, the true maximum hasn't been observed yet.
        max_elevation_truncated = (
            end_truncated
            and max_local_idx == end_idx
            and n_samples > 1
            and elev_segment[-1] > elev_segment[-2]
        )

        passes.append({
            "start_time": t[start_idx],
            "start_azimuth_deg": azimuth_deg[start_idx],
            "start_truncated": start_truncated,
            "end_time": t[end_idx],
            "end_azimuth_deg": azimuth_deg[end_idx],
            "end_truncated": end_truncated,
            "max_elevation_deg": elevation_deg[max_local_idx],
            "max_elevation_time": t[max_local_idx],
            "max_elevation_truncated": max_elevation_truncated,
            "duration_minutes": (t[end_idx] - t[start_idx]) * 1440.0,
            "low_confidence": n_samples < MIN_PASS_SAMPLES_FOR_CONFIDENCE,
        })
        i = j

    return passes


def compute_passes(
    sat: EarthSatellite,
    observer: GeographicPosition,
    ts: Timescale,
    start_time: Time | None = None,
    duration_hours: float = 24,
    step_minutes: float = 1,
    min_elevation_deg: float = MIN_PASS_ELEVATION_DEG,
) -> list[PassDict]:
    """
    Convenience wrapper: build the time grid, compute alt/az across it,
    and run find_passes() over the result for a single satellite.
    """
    t = build_time_grid(ts, start_time, duration_hours, step_minutes)
    altaz = compute_altaz(sat, observer, t)
    # compute_altaz()'s fields are typed float | NDArray because a scalar
    # `t` would produce scalars -- but `t` here always comes from
    # build_time_grid(), which always returns a vectorized time, so these
    # are always arrays in practice. cast() tells mypy that without
    # changing anything at runtime (it's a no-op).
    elevation_deg = cast(NDArray[np.float64], altaz["elevation_deg"])
    azimuth_deg = cast(NDArray[np.float64], altaz["azimuth_deg"])
    return find_passes(
        t, elevation_deg, azimuth_deg,
        min_elevation_deg=min_elevation_deg,
    )
