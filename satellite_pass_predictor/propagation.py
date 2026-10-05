"""Orbit propagation: a satellite's geodetic subpoint (lat/lon/altitude) at one time or a grid."""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from skyfield.api import wgs84
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time, Timescale

from .time_utils import build_time_grid

# Scalar for a single time, array for a time grid.
FloatOrArray = float | NDArray[np.float64]


class SubpointDict(TypedDict):
    """Geodetic subpoint: latitude/longitude in degrees, altitude in km."""

    latitude_deg: FloatOrArray
    longitude_deg: FloatOrArray
    altitude_km: FloatOrArray


class GroundTrackDict(SubpointDict):
    """A SubpointDict plus the time grid it was sampled at."""

    time: Time


def get_subpoint(sat: EarthSatellite, t: Time) -> SubpointDict:
    """Return the point on the WGS84 ellipsoid directly below `sat` at time(s) `t`."""
    geocentric = sat.at(t)
    subpoint = wgs84.subpoint(geocentric)
    return {
        "latitude_deg": subpoint.latitude.degrees,
        "longitude_deg": subpoint.longitude.degrees,
        "altitude_km": subpoint.elevation.km,
    }


def compute_ground_track(
    sat: EarthSatellite,
    ts: Timescale,
    start_time: Time | None = None,
    duration_hours: float = 24,
    step_minutes: float = 1,
) -> GroundTrackDict:
    """Propagate `sat`'s subpoint over an evenly spaced grid (see build_time_grid)."""
    t = build_time_grid(ts, start_time, duration_hours, step_minutes)
    subpoint = get_subpoint(sat, t)
    return GroundTrackDict(
        latitude_deg=subpoint["latitude_deg"],
        longitude_deg=subpoint["longitude_deg"],
        altitude_km=subpoint["altitude_km"],
        time=t,
    )
