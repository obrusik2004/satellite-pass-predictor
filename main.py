"""
Satellite Pass Predictor
=========================
Loads current TLE (Two-Line Element) data for a small set of satellites
from Celestrak. This is the data-ingestion step that later stages
(orbit propagation, ground tracks, visibility windows over Kourou) build on.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import load, wgs84

# Satellites tracked by this tool, identified by NORAD Catalog Number
# (the stable numeric ID Celestrak, Space-Track etc. all key on -- names
# are not a reliable lookup key since they change over a mission's life).
#
# ISS (ZARYA)  - ~400 km orbit, ~90 min period. Extremely well tracked,
#                which makes it the easiest case to sanity-check against
#                Heavens-Above / N2YO before trusting anything else here.
# SWISSCUBE    - 1U CubeSat, Switzerland's first satellite (launched 2009).
# BEESAT-1     - 1U CubeSat, TU Berlin (launched on the same 2009 rideshare
#                as SwissCube). Two satellites launched together into
#                similar orbits, ~17 years ago -- a natural pair for later
#                comparing how much their orbits have diverged under drag.
# MICROSCOPE   - CNES microsatellite (~300 kg, Myriade-class), launched from
#                Kourou on Soyuz VS14 in April 2016. Flew twin accelerometers
#                to test Einstein's Weak Equivalence Principle to ~1e-15
#                precision -- the most precise test of it ever flown -- and
#                ties directly into this project's Kourou/ESA theme.
#                (NB: EyeSat, NORAD 44877, also launched from Kourou via
#                CNES and would have been a nice small-CubeSat comparison,
#                but it decayed 2023-11-19 and Celestrak has no current
#                elements for it -- confirmed via CATNR lookup before
#                ruling it out.)
SATELLITES = {
    "ISS (ZARYA)": 25544,
    "SWISSCUBE": 35932,
    "BEESAT-1": 35933,
    "MICROSCOPE": 41457,
}

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?CATNR={norad_id}&FORMAT=TLE"
)

# How long a cached TLE is trusted before we bother re-downloading it.
# TLEs are only accurate for a matter of days (drag perturbations aren't
# modeled by SGP4), so we don't want to cache forever -- but we also
# don't want to hit Celestrak's servers on every single run.
MAX_TLE_AGE_DAYS = 1.0

TLE_CACHE_DIR = "data"
OUTPUT_DIR = "output"

# Guiana Space Centre (Centre Spatial Guyanais), Kourou, French Guiana --
# the ESA/CNES/Arianespace launch site this whole project is themed
# around. Coordinates per the Guiana Space Centre's Wikipedia infobox
# (5°10'08"N 52°41'25"W -> 5.169, -52.6903).
#
# Worth being upfront about: the CSG complex isn't a single point, it's
# tens of km of coastline hosting several separate launch pads (Ariane 6,
# Vega, Soyuz), and different official sources cite slightly different
# reference coordinates for "Kourou" as a result (ESA's own spaceport
# page cites 5°3'N for the general area, ~13 km south of the Wikipedia
# point). At the range of a satellite hundreds of km up, moving the
# observer by a few tens of km changes computed elevation by a small
# fraction of a degree -- negligible next to our 10-degree pass
# threshold. So one specific, citable reference point is used for
# reproducibility, not because sub-km precision matters for this
# calculation.
KOUROU_LATITUDE_DEG = 5.169
KOUROU_LONGITUDE_DEG = -52.6903
KOUROU_ELEVATION_M = 0  # coastal, effectively sea level; irrelevant here

KOUROU = wgs84.latlon(
    KOUROU_LATITUDE_DEG, KOUROU_LONGITUDE_DEG, elevation_m=KOUROU_ELEVATION_M
)

# A satellite is considered "visible" for pass-prediction purposes once
# it's at least this many degrees above the horizon -- low elevations are
# usually unusable anyway (obstructions, atmospheric extinction), and
# 10 degrees is the conventional default for both amateur satellite
# tracking and this kind of pass table.
MIN_PASS_ELEVATION_DEG = 10.0

# A detected pass spanning fewer samples than this has its start/end/peak
# resolved only to within one sampling step, since we don't know what
# happened *between* samples -- see find_passes() for how this is used.
MIN_PASS_SAMPLES_FOR_CONFIDENCE = 3


def load_satellites(satellites=SATELLITES):
    """
    Fetch current TLE data for each satellite, from a local cache when it's
    fresh enough or from Celestrak otherwise.

    Returns a dict mapping display name -> skyfield EarthSatellite.
    """
    os.makedirs(TLE_CACHE_DIR, exist_ok=True)

    result = {}
    for name, norad_id in satellites.items():
        url = CELESTRAK_URL.format(norad_id=norad_id)
        filename = os.path.join(TLE_CACHE_DIR, f"tle_{norad_id}.txt")

        stale = (
            not load.exists(filename)
            or load.days_old(filename) > MAX_TLE_AGE_DAYS
        )
        try:
            entries = load.tle_file(url, filename=filename, reload=stale)
        except OSError as e:
            # Celestrak being briefly unreachable/rate-limited shouldn't
            # crash the whole pipeline if we already have a usable (if a
            # bit stale) copy on disk -- fall back to it, but say so.
            if not load.exists(filename):
                raise
            age = load.days_old(filename)
            print(
                f"Warning: could not refresh TLE for {name} ({e}); "
                f"using cached copy from {age:.1f} day(s) ago instead."
            )
            entries = load.tle_file(filename)

        # gp.php with a single CATALOG_NUMBER always returns exactly one
        # satellite, so entries[0] is safe here.
        result[name] = entries[0]
    return result


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


def _build_time_grid(ts, start_time=None, duration_hours=24, step_minutes=1):
    """
    Build a single vectorized Skyfield time spanning `duration_hours`
    starting at `start_time` (default: now), sampled every `step_minutes`.

    Shared by the ground-track and pass-prediction steps: both need "many
    evenly-spaced instants over the next N hours" and both hand the
    result straight to a function built to vectorize over a time array
    (get_subpoint(), compute_altaz()) rather than looping per-sample in
    Python.
    """
    if start_time is None:
        start_time = ts.now()

    n_steps = int(duration_hours * 60 / step_minutes) + 1
    minutes = np.arange(n_steps) * step_minutes
    return start_time + minutes / 1440.0  # Skyfield Time + fractional days


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
    t = _build_time_grid(ts, start_time, duration_hours, step_minutes)
    track = get_subpoint(sat, t)
    track["time"] = t
    return track


def _split_at_antimeridian(longitudes, latitudes):
    """
    Insert NaN breaks wherever consecutive longitude samples jump across
    the +180/-180 antimeridian (International Date Line).

    A ground track crossing that line jumps from e.g. +179.9 deg to
    -179.9 deg between two adjacent samples that are actually right next
    to each other on the map. Plotted naively, that reads as one sample
    spanning almost 360 degrees -- a wrong horizontal streak all the way
    across the plot. matplotlib skips over NaN values in a line plot, so
    inserting one at each such jump breaks the line into separate
    segments there instead, without needing to change how the underlying
    lat/lon data was computed.
    """
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)

    jumps = np.where(np.abs(np.diff(longitudes)) > 180.0)[0]
    if len(jumps) == 0:
        return longitudes, latitudes

    insert_at = jumps + 1
    longitudes = np.insert(longitudes, insert_at, np.nan)
    latitudes = np.insert(latitudes, insert_at, np.nan)
    return longitudes, latitudes


def plot_ground_tracks(satellites, ts, start_time=None, duration_hours=24,
                        step_minutes=1, output_path=None):
    """
    Plot every satellite's ground track over the next `duration_hours` on
    a plain lat/lon grid and save it as a PNG.

    Deliberately no coastlines/continent outlines here (see README) --
    just a 30-degree lat/lon grid, axis labels, and a legend. All
    satellites share the same start_time so the tracks are directly
    comparable on one plot.
    """
    if start_time is None:
        start_time = ts.now()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "ground_tracks.png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, sat in satellites.items():
        track = compute_ground_track(
            sat, ts, start_time=start_time,
            duration_hours=duration_hours, step_minutes=step_minutes,
        )
        lon, lat = _split_at_antimeridian(
            track["longitude_deg"], track["latitude_deg"]
        )
        ax.plot(lon, lat, linewidth=1, label=name)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(
        "Ground tracks -- next {}h from {}".format(
            duration_hours, start_time.utc_strftime("%Y-%m-%d %H:%M UTC")
        )
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def compute_altaz(sat, observer, t):
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


def find_passes(t, elevation_deg, azimuth_deg,
                 min_elevation_deg=MIN_PASS_ELEVATION_DEG):
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

    passes = []
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


def compute_passes(sat, observer, ts, start_time=None, duration_hours=24,
                    step_minutes=1, min_elevation_deg=MIN_PASS_ELEVATION_DEG):
    """
    Convenience wrapper: build the time grid, compute alt/az across it,
    and run find_passes() over the result for a single satellite.
    """
    t = _build_time_grid(ts, start_time, duration_hours, step_minutes)
    altaz = compute_altaz(sat, observer, t)
    return find_passes(
        t, altaz["elevation_deg"], altaz["azimuth_deg"],
        min_elevation_deg=min_elevation_deg,
    )


def print_passes_table(passes_by_satellite):
    """
    Print one row per detected pass across all satellites, sorted by
    start time, as a plain fixed-width text table.
    """
    rows = []
    for name, passes in passes_by_satellite.items():
        for p in passes:
            notes = []
            if p["start_truncated"]:
                notes.append("IN PROGRESS AT START")
            if p["end_truncated"]:
                notes.append("CUT OFF AT END")
            if p["max_elevation_truncated"]:
                notes.append("MAX MAY BE HIGHER (still rising at cutoff)")
            if p["low_confidence"]:
                notes.append("LOW CONFIDENCE (rerun with finer step)")

            rows.append({
                "satellite": name,
                "start": p["start_time"].utc_strftime("%Y-%m-%d %H:%M:%S"),
                "start_az": f"{p['start_azimuth_deg']:.1f}",
                "max_elev": f"{p['max_elevation_deg']:.1f}",
                "max_elev_time": p["max_elevation_time"].utc_strftime("%H:%M:%S"),
                "end": p["end_time"].utc_strftime("%Y-%m-%d %H:%M:%S"),
                "end_az": f"{p['end_azimuth_deg']:.1f}",
                "duration_min": f"{p['duration_minutes']:.1f}",
                "notes": ", ".join(notes),
                "_sort_key": p["start_time"],
            })

    if not rows:
        print("No passes above threshold in this window.")
        return

    rows.sort(key=lambda r: r["_sort_key"])

    columns = [
        ("satellite", "Satellite", 12),
        ("start", "Start (UTC)", 19),
        ("start_az", "Start Az", 8),
        ("max_elev", "Max El", 6),
        ("max_elev_time", "Max El Time", 11),
        ("end", "End (UTC)", 19),
        ("end_az", "End Az", 7),
        ("duration_min", "Dur (min)", 9),
        ("notes", "Notes", 40),
    ]

    header = "  ".join(f"{title:<{width}}" for _, title, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{row[key]:<{width}}" for key, _, width in columns))


def main():
    satellites = load_satellites()

    print(f"Loaded {len(satellites)} satellite(s):\n")
    for name, sat in satellites.items():
        print(f"{name}  (NORAD {sat.model.satnum})")
        print(f"  TLE epoch: {sat.epoch.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

    ts = load.timescale()
    t = ts.now()
    print(f"Current positions at {t.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}:\n")
    for name, sat in satellites.items():
        pos = get_subpoint(sat, t)
        print(f"{name}")
        print(f"  latitude:  {pos['latitude_deg']:+.4f} deg")
        print(f"  longitude: {pos['longitude_deg']:+.4f} deg")
        print(f"  altitude:  {pos['altitude_km']:.1f} km")
        print()

    output_path = plot_ground_tracks(satellites, ts, start_time=t)
    print(f"Ground track plot saved to {output_path}")

    print(
        f"\nVisibility passes over Kourou "
        f"({KOUROU_LATITUDE_DEG:.4f}, {KOUROU_LONGITUDE_DEG:.4f}) "
        f"in the next 24h, elevation >= {MIN_PASS_ELEVATION_DEG:.0f} deg:\n"
    )
    passes_by_satellite = {
        name: compute_passes(sat, KOUROU, ts, start_time=t)
        for name, sat in satellites.items()
    }
    print_passes_table(passes_by_satellite)


if __name__ == "__main__":
    main()
