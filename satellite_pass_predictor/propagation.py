"""
Orbit propagation: turning a satellite's TLE into positions -- geocentric
subpoints (lat/lon/altitude) at a single time or across a time grid.
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from skyfield.api import wgs84
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time, Timescale

from .time_utils import build_time_grid

# Skyfield's Angle/Distance .degrees / .km properties return a plain
# (numpy) float64 when `t` is a single time, or an ndarray when `t` is a
# vectorized time -- the same code path serves both callers (a scalar
# "current position" query vs. a 1441-sample ground-track grid), and the
# return type genuinely depends on the shape of the input. float64 is a
# float subclass, so `float` alone would technically cover the scalar
# case, but spelling out the array alternative makes that dependency
# visible at the type level instead of hiding it.
FloatOrArray = float | NDArray[np.float64]

# TypedDict, not a dataclass, for these result types (here and in
# visibility.py's AltAzDict/PassDict): the whole codebase already passes
# these around as plain dicts accessed by string key (pos['latitude_deg'],
# p["start_time"], etc.) in main.py, visibility.py and visualization.py.
# A dataclass would mean attribute access (pos.latitude_deg) instead,
# which means changing every one of those call sites -- real code churn,
# not just typing. TypedDict adds precise, checked types for a fixed,
# known set of keys with zero runtime change (it erases to a plain dict
# at runtime) and zero call-site changes.


class SubpointDict(TypedDict):
    """Geodetic subpoint: get_subpoint()'s fixed, known keys."""
    latitude_deg: FloatOrArray
    longitude_deg: FloatOrArray
    altitude_km: FloatOrArray


class GroundTrackDict(SubpointDict):
    """A SubpointDict plus the time grid it was sampled at."""
    time: Time


def get_subpoint(sat: EarthSatellite, t: Time) -> SubpointDict:
    """
    Compute a satellite's geodetic subpoint (the point directly below it
    on Earth's surface) at a single Skyfield time `t`.

    This is deliberately a pure "one satellite, one time -> one position"
    function rather than something that loops over a time range itself.
    The ground-track step needs to call this once per satellite for every
    minute of a 24h window, and the pass-prediction step needs to call it
    at whatever times its search happens to land on -- so the time grid
    belongs to the caller, not to this function.

    `sat.at(t)` returns the satellite's position as seen from Earth's
    center (geocentric) in the GCRS inertial frame. `wgs84.subpoint()`
    rotates that into Earth-fixed coordinates and maps it onto the WGS84
    ellipsoid, giving latitude/longitude/altitude -- the same ellipsoid
    model GPS uses, which is why this is directly comparable to what a
    tracking site or a GPS receiver would report.

    Returns a dict with latitude_deg, longitude_deg, altitude_km.
    """
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
    """
    Propagate one satellite's subpoint over an evenly-spaced time grid.

    Skyfield/SGP4 vectorize cleanly over time arrays, so this is one
    batched computation (e.g. 1441 samples at once for a 24h/1min grid)
    rather than 1441 separate Python-level calls -- exactly what
    get_subpoint()'s "one time, caller's choice of grid" design from the
    propagation step was meant to make easy.

    Returns a dict with 'time' (the Skyfield Time array) plus
    'latitude_deg', 'longitude_deg', 'altitude_km' (numpy arrays, one
    entry per sample).
    """
    t = build_time_grid(ts, start_time, duration_hours, step_minutes)
    subpoint = get_subpoint(sat, t)
    # Built as a new dict (rather than mutating `subpoint` in place with
    # subpoint["time"] = t, as before typing was added) because
    # SubpointDict has no "time" key -- a TypedDict rejects assigning a
    # key outside its declared set. Same resulting dict either way.
    return GroundTrackDict(
        latitude_deg=subpoint["latitude_deg"],
        longitude_deg=subpoint["longitude_deg"],
        altitude_km=subpoint["altitude_km"],
        time=t,
    )
