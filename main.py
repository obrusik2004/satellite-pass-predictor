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


def compute_ground_track(sat, ts, start_time=None, duration_hours=24,
                          step_minutes=1):
    """
    Propagate one satellite's subpoint over an evenly-spaced time grid.

    Builds a single vectorized Skyfield time spanning `duration_hours`
    starting at `start_time` (default: now), sampled every `step_minutes`,
    and calls get_subpoint() once across the whole array. Skyfield/SGP4
    vectorize cleanly over time arrays, so this is one batched computation
    (e.g. 1441 samples at once for a 24h/1min grid) rather than 1441
    separate Python-level calls -- exactly what get_subpoint()'s "one
    time, caller's choice of grid" design from the propagation step was
    meant to make easy.

    Returns a dict with 'time' (the Skyfield Time array) plus
    'latitude_deg', 'longitude_deg', 'altitude_km' (numpy arrays, one
    entry per sample).
    """
    if start_time is None:
        start_time = ts.now()

    n_steps = int(duration_hours * 60 / step_minutes) + 1
    minutes = np.arange(n_steps) * step_minutes
    t = start_time + minutes / 1440.0  # Skyfield Time + fractional days

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


if __name__ == "__main__":
    main()
