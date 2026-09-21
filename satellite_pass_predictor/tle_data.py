"""
TLE ingestion: fetching (and locally caching) current orbital element data
for the tracked satellites from Celestrak.
"""

import os

from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite

from .config import CELESTRAK_URL, MAX_TLE_AGE_DAYS, SATELLITES, TLE_CACHE_DIR


def load_satellites(
    satellites: dict[str, int] = SATELLITES,
) -> dict[str, EarthSatellite]:
    """
    Fetch current TLE data for each satellite, from a local cache when it's
    fresh enough or from Celestrak otherwise.

    Returns a dict mapping display name -> skyfield EarthSatellite.
    """
    os.makedirs(TLE_CACHE_DIR, exist_ok=True)

    result: dict[str, EarthSatellite] = {}
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
