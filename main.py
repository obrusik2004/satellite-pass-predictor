"""
Satellite Pass Predictor
=========================
Loads current TLE (Two-Line Element) data for a small set of satellites
from Celestrak. This is the data-ingestion step that later stages
(orbit propagation, ground tracks, visibility windows over Kourou) build on.
"""

import os

from skyfield.api import load

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


def main():
    satellites = load_satellites()

    print(f"Loaded {len(satellites)} satellite(s):\n")
    for name, sat in satellites.items():
        print(f"{name}  (NORAD {sat.model.satnum})")
        print(f"  TLE epoch: {sat.epoch.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()


if __name__ == "__main__":
    main()
