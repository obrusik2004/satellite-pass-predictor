"""
TLE ingestion: fetching (and locally caching) current orbital element data
for the tracked satellites from Celestrak.
"""

import os
import time
from typing import cast

from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite

from .config import (
    CELESTRAK_URL,
    MAX_TLE_AGE_DAYS,
    SATELLITES,
    TLE_CACHE_DIR,
    TLE_FETCH_MAX_RETRIES,
    TLE_FETCH_RETRY_BASE_DELAY_SECONDS,
)


def _fetch_tle_with_retries(
    url: str, filename: str, reload: bool
) -> list[EarthSatellite]:
    """
    load.tle_file()'s actual network fetch (the download itself, not
    the reading-a-local-file fallback load_satellites() uses when this
    is exhausted), retried with exponential backoff on OSError -- see
    config.TLE_FETCH_MAX_RETRIES's comment for why this exists at all
    (Streamlit Community Cloud's outbound networking is intermittently
    flaky) rather than being an arbitrary safety net.

    Only OSError is retried: that's specifically the network-failure
    exception Skyfield's downloader raises (a dropped connection,
    timeout, DNS failure, HTTP error, ...). Any other exception is a
    different problem this retry loop has no business papering over,
    and propagates immediately, same as if this function didn't exist.

    After TLE_FETCH_MAX_RETRIES retries, the final attempt is made with
    no surrounding try/except, so its OSError (if it still fails)
    propagates to load_satellites() completely unchanged -- that
    function's own handling (fall back to a cached copy if one exists,
    otherwise re-raise with an actionable message) still runs exactly
    as it did before this retry loop existed. This function only
    absorbs a *transient* blip; a persistent failure ends up in exactly
    the same place it always did.
    """
    for attempt in range(TLE_FETCH_MAX_RETRIES):
        try:
            # load.tle_file() is skyfield's own untyped call (see
            # mypy.ini's skyfield.* override) -- cast() tells mypy this
            # returns the list[EarthSatellite] it actually does, the
            # same pattern used throughout this package wherever a
            # skyfield call's result needs a precise type.
            return cast(list[EarthSatellite], load.tle_file(url, filename=filename, reload=reload))
        except OSError:
            time.sleep(TLE_FETCH_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
    return cast(list[EarthSatellite], load.tle_file(url, filename=filename, reload=reload))


def load_satellites(
    satellites: dict[str, int] = SATELLITES,
) -> dict[str, EarthSatellite]:
    """
    Fetch current TLE data for each satellite, from a local cache when it's
    fresh enough or from Celestrak otherwise.

    Raises OSError (with a message naming the satellite and pointing at
    the likely cause) if a fetch fails and there's no cached copy to fall
    back to -- e.g. the very first run on a machine with no internet
    connection. Raises ValueError if Celestrak (or a cached file) returns
    a response with no usable TLE in it at all -- e.g. an unrecognized or
    no-longer-tracked NORAD ID; Celestrak has been observed to signal
    this both as an HTTP error and as an HTTP 200 with an empty/"No GP
    data found" body, and only the former is caught by the OSError
    handling below, so this is checked separately.

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
            entries = _fetch_tle_with_retries(url, filename, reload=stale)
        except OSError as e:
            # Celestrak being briefly unreachable/rate-limited shouldn't
            # crash the whole pipeline if we already have a usable (if a
            # bit stale) copy on disk -- fall back to it, but say so.
            if not load.exists(filename):
                # No cache to fall back to either -- e.g. the very first
                # run on a machine with no internet connection. Skyfield's
                # own OSError here is accurate but low-level (a raw
                # urllib/HTTP message with no mention of which satellite
                # or why it matters); re-raise with context instead of
                # letting that bubble straight up to the user.
                raise OSError(
                    f"Could not fetch a TLE for {name} (NORAD {norad_id}) "
                    f"from Celestrak, and no cached copy exists at "
                    f"{filename!r} to fall back to. Check your internet "
                    f"connection and try again. Original error: {e}"
                ) from e
            age = load.days_old(filename)
            print(
                f"Warning: could not refresh TLE for {name} ({e}); "
                f"using cached copy from {age:.1f} day(s) ago instead."
            )
            entries = load.tle_file(filename)

        if not entries:
            # Skyfield's TLE parser doesn't raise on empty/malformed
            # input, it just returns an empty list -- so without this
            # check, an unrecognized/decayed NORAD ID that Celestrak
            # answers with "200 OK, empty body" (rather than an HTTP
            # error, which the except block above already handles) would
            # otherwise surface as a bare "IndexError: list index out of
            # range" on entries[0] below, with no hint of the real cause.
            raise ValueError(
                f"Celestrak returned no usable TLE data for {name} "
                f"(NORAD {norad_id}). It may no longer be tracked (e.g. "
                f"decayed) or the NORAD ID in config.py may be wrong -- "
                f"check {url} directly."
            )

        # gp.php with a single CATALOG_NUMBER returns exactly one
        # satellite whenever it returns anything at all (see the
        # empty-response check above for when it doesn't).
        result[name] = entries[0]
    return result
