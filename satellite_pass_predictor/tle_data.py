"""TLE ingestion: fetching, caching and staleness checks for the tracked satellites' elements."""

import os
import time
from datetime import UTC, datetime, timedelta
from typing import Literal, TypedDict, cast

import requests
from skyfield.api import load
from skyfield.iokit import parse_tle_file
from skyfield.sgp4lib import EarthSatellite

from .config import (
    CELESTRAK_URL,
    MAX_TLE_AGE_DAYS,
    MIRROR_REFRESH_WARNING_HOURS,
    SATELLITES,
    TLE_CACHE_DIR,
    TLE_EPOCH_WARNING_DAYS,
    TLE_FETCH_MAX_RETRIES,
    TLE_FETCH_RETRY_BASE_DELAY_SECONDS,
    TLE_FETCH_TIMEOUT_SECONDS,
    TLE_MIRROR_CACHE_AGE_HOURS,
    TLE_MIRROR_METADATA_URL,
    TLE_MIRROR_URL,
)

# "celestrak" fetches from Celestrak directly; "mirror" reads the tle-data branch
# (see docs/DECISIONS.md).
TLESource = Literal["celestrak", "mirror"]


def _download_and_cache_tle(url: str, filename: str) -> list[EarthSatellite]:
    """Download `url` to `filename` and parse it.

    Uses requests rather than Skyfield's downloader so each call has its own timeout
    (socket.setdefaulttimeout is process-global and Streamlit sessions are threads).
    Any requests failure is raised as OSError. A response with no valid TLE parses to an
    empty list rather than raising.
    """
    try:
        response = requests.get(url, timeout=TLE_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise OSError(f"cannot download {url} because {e}") from e

    with open(filename, "wb") as f:
        f.write(response.content)

    return list(parse_tle_file(response.content.splitlines()))


def _fetch_tle_with_retries(url: str, filename: str, reload: bool) -> list[EarthSatellite]:
    """Fetch with exponential backoff on OSError, or just read the cached file if not `reload`.

    The final attempt is unguarded, so a persistent failure propagates unchanged to the caller.
    """
    if not reload:
        return cast(list[EarthSatellite], load.tle_file(url, filename=filename, reload=False))

    for attempt in range(TLE_FETCH_MAX_RETRIES):
        try:
            return _download_and_cache_tle(url, filename)
        except OSError:
            time.sleep(TLE_FETCH_RETRY_BASE_DELAY_SECONDS * (2**attempt))
    return _download_and_cache_tle(url, filename)


def load_satellites(
    satellites: dict[str, int] | None = None,
    source: TLESource = "celestrak",
) -> dict[str, EarthSatellite]:
    """Load the current TLE of each satellite (default: SATELLITES), keyed by display name.

    A cached copy is used while it is younger than the source's max age (MAX_TLE_AGE_DAYS for
    Celestrak, TLE_MIRROR_CACHE_AGE_HOURS for the mirror); otherwise `source` is queried.

    Raises:
        OSError: the fetch failed and there is no cached copy to fall back to. With a stale
            cache, a warning is printed and the cached copy is used instead.
        ValueError: the response contained no usable TLE (e.g. a decayed or wrong NORAD ID).
            Celestrak signals this with an empty HTTP 200 as well as with an error status.
    """
    tracked = SATELLITES if satellites is None else satellites
    os.makedirs(TLE_CACHE_DIR, exist_ok=True)

    if source == "celestrak":
        url_template = CELESTRAK_URL
        max_age_days = MAX_TLE_AGE_DAYS
    else:
        url_template = TLE_MIRROR_URL
        max_age_days = TLE_MIRROR_CACHE_AGE_HOURS / 24.0

    result: dict[str, EarthSatellite] = {}
    for name, norad_id in tracked.items():
        url = url_template.format(norad_id=norad_id)
        filename = os.path.join(TLE_CACHE_DIR, f"tle_{norad_id}.txt")

        stale = not load.exists(filename) or load.days_old(filename) > max_age_days
        try:
            entries = _fetch_tle_with_retries(url, filename, reload=stale)
        except OSError as e:
            if not load.exists(filename):
                # Nothing to fall back to: re-raise with the satellite and a likely cause.
                if source == "mirror":
                    raise OSError(
                        f"Could not fetch a TLE for {name} (NORAD {norad_id}) "
                        f"from the GitHub mirror ({url}), and no cached copy "
                        f"exists at {filename!r} to fall back to. This "
                        f"usually means the tle-data branch doesn't exist "
                        f"yet or the mirror is unreachable -- run the "
                        f"'Refresh TLE mirror' GitHub Actions workflow "
                        f"(.github/workflows/refresh-tles.yml) and try "
                        f"again. Original error: {e}"
                    ) from e
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
            raise ValueError(
                f"Celestrak returned no usable TLE data for {name} "
                f"(NORAD {norad_id}). It may no longer be tracked (e.g. "
                f"decayed) or the NORAD ID in config.py may be wrong -- "
                f"check {url} directly."
            )

        # A single-CATNR query returns at most one satellite.
        result[name] = entries[0]
    return result


class TLESatelliteMetadata(TypedDict):
    """One satellite's entry in the mirror's metadata.json (written by fetch_tles.py)."""

    name: str
    fetched_at: str  # ISO 8601 UTC, last successful fetch of this satellite
    tle_epoch: str  # ISO 8601 UTC
    source_url: str


class TLEMirrorMetadata(TypedDict):
    """The mirror's metadata.json."""

    # ISO 8601 UTC; when the refresh workflow last ran, even if some fetches failed.
    generated_at: str
    # Keyed by NORAD ID as a string (JSON object keys are always strings).
    satellites: dict[str, TLESatelliteMetadata]


def fetch_mirror_metadata() -> TLEMirrorMetadata | None:
    """Fetch the mirror's metadata.json, or None on any network, JSON or shape problem.

    Only used for provenance display, so it is best-effort and not retried.
    """
    try:
        response = requests.get(TLE_MIRROR_METADATA_URL, timeout=TLE_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("generated_at"), str)
        or not isinstance(data.get("satellites"), dict)
    ):
        return None

    return cast(TLEMirrorMetadata, data)


def is_older_than(
    reference_time: datetime, max_age: timedelta, now: datetime | None = None
) -> bool:
    """Whether the timezone-aware `reference_time` is more than `max_age` before `now`.

    `now` defaults to the current UTC time and is injectable for deterministic tests.
    """
    if now is None:
        now = datetime.now(UTC)
    return (now - reference_time) > max_age


def parse_iso_utc(timestamp: str) -> datetime | None:
    """Parse a metadata.json timestamp like "2026-09-28T09:00:03Z"; None if malformed."""
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


class StalenessWarnings(TypedDict):
    """Which staleness banners the app should show."""

    mirror_stale: bool
    stale_satellite_names: list[str]


def compute_staleness_warnings(
    metadata: TLEMirrorMetadata | None,
    satellite_epochs: dict[str, datetime],
    now: datetime | None = None,
) -> StalenessWarnings:
    """Decide which staleness warnings apply.

    The mirror is stale when its last refresh is older than MIRROR_REFRESH_WARNING_HOURS; unknown
    metadata (None) is not reported as stale. A satellite is stale when its TLE epoch is older
    than TLE_EPOCH_WARNING_DAYS; `satellite_epochs` should come from the loaded satellites.
    """
    mirror_stale = False
    if metadata is not None:
        generated_at = parse_iso_utc(metadata["generated_at"])
        if generated_at is not None:
            mirror_stale = is_older_than(
                generated_at, timedelta(hours=MIRROR_REFRESH_WARNING_HOURS), now=now
            )

    stale_satellite_names = [
        name
        for name, epoch in satellite_epochs.items()
        if is_older_than(epoch, timedelta(days=TLE_EPOCH_WARNING_DAYS), now=now)
    ]

    return {"mirror_stale": mirror_stale, "stale_satellite_names": stale_satellite_names}
