"""Visibility: look angles from a ground observer, and passes above an elevation mask."""

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time, Timescale
from skyfield.toposlib import GeographicPosition

from .config import MIN_PASS_ELEVATION_DEG
from .propagation import FloatOrArray

# Event codes returned by EarthSatellite.find_events().
_RISE, _CULMINATE, _SET = 0, 1, 2

# Events are refined onto a fixed grid of absolute TT ticks (1 ms), so a pass's times depend
# only on the orbit, not on where the search window happens to start.
_TICKS_PER_DAY = 86_400_000

# find_events() places each event within 0.5 s of the true one, so +-1 s brackets it.
_BRACKET_TICKS = 1000


class AltAzDict(TypedDict):
    """Topocentric look angles in degrees, and slant range in km."""

    elevation_deg: FloatOrArray
    azimuth_deg: FloatOrArray
    distance_km: FloatOrArray


class PassDict(TypedDict):
    """One pass. The `*_truncated` flags mark values cut off by the window edge."""

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


def compute_altaz(sat: EarthSatellite, observer: GeographicPosition, t: Time) -> AltAzDict:
    """Return elevation, azimuth and range of `sat` as seen from `observer` at time(s) `t`."""
    topocentric = (sat - observer).at(t)
    alt, az, distance = topocentric.altaz()
    return {
        "elevation_deg": alt.degrees,
        "azimuth_deg": az.degrees,
        "distance_km": distance.km,
    }


def _as_array(values: FloatOrArray) -> NDArray[np.float64]:
    """Narrow a look-angle value computed for a Time array (never a scalar) for the type checker."""
    return cast(NDArray[np.float64], values)


def _refine_event_times(
    sat: EarthSatellite,
    observer: GeographicPosition,
    ts: Timescale,
    event_jd: NDArray[np.float64],
    events: NDArray[np.uint8],
    min_elevation_deg: float,
) -> NDArray[np.float64]:
    """Refine find_events() times (TT Julian dates) to the 1 ms tick grid by bisection.

    Returns the first tick at or above the mask for a rise, the last such tick for a set, and
    the highest tick for a culmination. The ticks are absolute, so the result is identical
    whatever search window found the event.
    """
    n = len(event_jd)
    if n == 0:
        return event_jd
    is_rise = events == _RISE
    is_set = events == _SET
    is_culmination = events == _CULMINATE

    # Tick counts (~2e14) are integer-valued floats, exact well below 2**53.
    ticks = event_jd * _TICKS_PER_DAY
    lo = np.floor(ticks) - _BRACKET_TICKS
    # A rise/set sample from find_events() is already past the crossing; a peak may be either side.
    hi = np.ceil(ticks) + np.where(is_culmination, _BRACKET_TICKS, 0)
    # Keep each rise/set bracket after the previous one, in case a pass lasts under a second.
    transitions = np.flatnonzero(~is_culmination)
    lo[transitions[1:]] = np.maximum(lo[transitions[1:]], np.ceil(ticks[transitions[:-1]]))

    # Invariant: the event has not happened at `lo` and has at `hi`.
    while np.any(hi - lo > 1):
        mid = np.floor((lo + hi) / 2)
        t = ts.tt_jd(np.concatenate((mid, mid + 1)) / _TICKS_PER_DAY)
        elevation = _as_array(compute_altaz(sat, observer, t)["elevation_deg"])
        el_mid, el_next = elevation[:n], elevation[n:]
        happened = np.where(
            is_rise,
            el_mid >= min_elevation_deg,
            np.where(is_set, el_mid < min_elevation_deg, el_next < el_mid),
        )
        hi = np.where(happened, mid, hi)
        lo = np.where(happened, lo, mid)

    refined: NDArray[np.float64] = np.where(is_set, hi - 1, hi) / _TICKS_PER_DAY
    return refined


def compute_passes(
    sat: EarthSatellite,
    observer: GeographicPosition,
    ts: Timescale,
    start_time: Time | None = None,
    duration_hours: float = 24,
    min_elevation_deg: float = MIN_PASS_ELEVATION_DEG,
) -> list[PassDict]:
    """Return `sat`'s passes above `min_elevation_deg` within `duration_hours` of `start_time`.

    Rise, culmination and set come from Skyfield's find_events(), refined to 1 ms. A pass cut
    by the window edge is clamped to it: `start_truncated` / `end_truncated` mark a clamped
    start/end, and `max_elevation_truncated` marks a highest in-window point on a clamped edge,
    meaning the true peak lies outside the window. Several culminations: the highest wins.
    `start_time` defaults to now.
    """
    t0 = ts.now() if start_time is None else start_time
    t1 = t0 + duration_hours / 24.0
    raw_t, events = sat.find_events(observer, t0, t1, altitude_degrees=min_elevation_deg)
    event_jd = _refine_event_times(sat, observer, ts, raw_t.tt, events, min_elevation_deg)

    transitions = events[events != _CULMINATE]
    if len(transitions):
        in_pass = bool(transitions[0] == _SET)
    else:
        in_pass = bool(compute_altaz(sat, observer, t0)["elevation_deg"] >= min_elevation_deg)

    # Group events into passes: (rise index, set index, culmination indices), None = window edge.
    spans: list[tuple[int | None, int | None, list[int]]] = []
    rise_idx: int | None = None
    culmination_idxs: list[int] = []
    for i, event in enumerate(events):
        if event == _RISE:
            in_pass, rise_idx, culmination_idxs = True, i, []
        elif event == _CULMINATE:
            if in_pass:
                culmination_idxs.append(i)
        elif in_pass:
            spans.append((rise_idx, i, culmination_idxs))
            in_pass, rise_idx, culmination_idxs = False, None, []
    if in_pass:
        spans.append((rise_idx, None, culmination_idxs))

    passes: list[PassDict] = []
    for rise, set_, culminations in spans:
        start_jd = t0.tt if rise is None else event_jd[rise]
        end_jd = t1.tt if set_ is None else event_jd[set_]
        # Peak candidates: both ends (which win only when the pass is cut before or after its
        # peak) and every culmination, clamped into the pass.
        candidate_jd = np.concatenate(
            ([start_jd, end_jd], np.clip(event_jd[culminations], start_jd, end_jd))
        )
        t = ts.tt_jd(candidate_jd)
        altaz = compute_altaz(sat, observer, t)
        elevation = _as_array(altaz["elevation_deg"])
        azimuth = _as_array(altaz["azimuth_deg"])
        peak = int(np.argmax(elevation))
        peak_jd = candidate_jd[peak]

        passes.append(
            {
                "start_time": t[0],
                "start_azimuth_deg": float(azimuth[0]),
                "start_truncated": rise is None,
                "end_time": t[1],
                "end_azimuth_deg": float(azimuth[1]),
                "end_truncated": set_ is None,
                "max_elevation_deg": float(elevation[peak]),
                "max_elevation_time": t[peak],
                "max_elevation_truncated": bool(
                    (rise is None and peak_jd == start_jd) or (set_ is None and peak_jd == end_jd)
                ),
                "duration_minutes": float((end_jd - start_jd) * 1440.0),
            }
        )
    return passes
