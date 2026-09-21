"""
Orbit propagation: turning a satellite's TLE into positions -- geocentric
subpoints (lat/lon/altitude) at a single time or across a time grid.
"""

from skyfield.api import wgs84

from .time_utils import build_time_grid


def get_subpoint(sat, t):
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


def compute_ground_track(sat, ts, start_time=None, duration_hours=24,
                          step_minutes=1):
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
    track = get_subpoint(sat, t)
    track["time"] = t
    return track
