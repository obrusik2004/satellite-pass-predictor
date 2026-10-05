"""Visibility: look angles from a ground observer, and passes above an elevation mask."""

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time, Timescale
from skyfield.toposlib import GeographicPosition

from .config import MIN_PASS_ELEVATION_DEG, MIN_PASS_SAMPLES_FOR_CONFIDENCE
from .propagation import FloatOrArray
from .time_utils import build_time_grid


class AltAzDict(TypedDict):
    """Topocentric look angles in degrees, and slant range in km."""

    elevation_deg: FloatOrArray
    azimuth_deg: FloatOrArray
    distance_km: FloatOrArray


class PassDict(TypedDict):
    """One detected pass. The `*_truncated` flags mark values cut off by the window edge."""

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


def compute_altaz(sat: EarthSatellite, observer: GeographicPosition, t: Time) -> AltAzDict:
    """Return elevation, azimuth and range of `sat` as seen from `observer` at time(s) `t`."""
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
    """Find contiguous runs of samples at or above `min_elevation_deg`.

    Start, end and peak are the first, last and highest sample of a run, so they are only
    accurate to the sampling step. Window edges and short runs are flagged rather than dropped:

    - Above the mask at the first/last sample: `start_truncated` / `end_truncated`; the true
      rise/set time is outside the window.
    - Still rising at the last sample: `max_elevation_truncated`; the true peak is later.
    - Fewer than MIN_PASS_SAMPLES_FOR_CONFIDENCE samples: `low_confidence`; the pass is real
      but its times are unresolved within one step.

    `duration_minutes` is the time from the first to the last sample of the run.
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

        elev_segment = elevation_deg[start_idx : end_idx + 1]
        n_samples = end_idx - start_idx + 1
        max_local_idx = start_idx + int(np.argmax(elev_segment))

        # bool() so a numpy.bool_ doesn't end up in PassDict (it isn't JSON-serializable).
        max_elevation_truncated = bool(
            end_truncated
            and max_local_idx == end_idx
            and n_samples > 1
            and elev_segment[-1] > elev_segment[-2]
        )

        passes.append(
            {
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
            }
        )
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
    """Sample `sat`'s look angles over a time grid and return its passes (see find_passes)."""
    t = build_time_grid(ts, start_time, duration_hours, step_minutes)
    altaz = compute_altaz(sat, observer, t)
    # The grid is always an array, so the scalar half of the union never applies here.
    elevation_deg = cast(NDArray[np.float64], altaz["elevation_deg"])
    azimuth_deg = cast(NDArray[np.float64], altaz["azimuth_deg"])
    return find_passes(
        t,
        elevation_deg,
        azimuth_deg,
        min_elevation_deg=min_elevation_deg,
    )
