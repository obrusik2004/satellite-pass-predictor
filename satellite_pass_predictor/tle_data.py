"""TLE ingestion: fetching, validating, caching and loading the tracked satellites' elements."""

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import requests
from platformdirs import user_cache_dir
from skyfield.api import load
from skyfield.iokit import parse_tle_file
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Timescale

from .config import (
    CACHE_DIR_ENV_VAR,
    CELESTRAK_URL,
    GITHUB_REPO_URL,
    MAX_TLE_AGE_DAYS,
    MIRROR_REFRESH_WARNING_HOURS,
    SATELLITES,
    TIME_WINDOW_MAX_HOURS,
    TLE_EPOCH_MAX_DAYS,
    TLE_EPOCH_WARNING_DAYS,
    TLE_FETCH_MAX_RETRIES,
    TLE_FETCH_RETRY_BASE_DELAY_SECONDS,
    TLE_FETCH_TIMEOUT_SECONDS,
    TLE_MIRROR_CACHE_AGE_HOURS,
    TLE_MIRROR_METADATA_URL,
    TLE_MIRROR_URL,
)

logger = logging.getLogger(__name__)

# "celestrak" fetches from Celestrak directly; "mirror" reads the tle-data branch
# (see docs/DECISIONS.md).
TLESource = Literal["celestrak", "mirror"]

_SOURCE_LABELS: dict[TLESource, str] = {"celestrak": "Celestrak", "mirror": "GitHub mirror"}
_MIRROR_HINT = "Run the 'Refresh TLE mirror' workflow (.github/workflows/refresh-tles.yml)."

USER_AGENT = f"satellite-pass-predictor/0.9 (+{GITHUB_REPO_URL})"


class TLELoadError(Exception):
    """A satellite's TLE could not be loaded or is unusable; the message says why."""


@dataclass(frozen=True)
class SatelliteFailure:
    """Why one satellite could not be loaded."""

    name: str
    norad_id: int
    reason: str


@dataclass(frozen=True)
class TLELoadResult:
    """The satellites that loaded, plus per-satellite failures and non-fatal notices."""

    satellites: dict[str, EarthSatellite]
    failures: dict[str, SatelliteFailure]
    # Non-fatal problems worth showing a user, e.g. a stale cache used after a failed refresh.
    warnings: dict[str, str]
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def default_cache_dir() -> Path:
    """The TLE cache directory: $SATPASS_CACHE_DIR if set, else the per-user cache location."""
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    return (
        Path(override)
        if override
        else Path(user_cache_dir("satellite-pass-predictor", appauthor=False))
    )


def _is_transient(error: requests.RequestException) -> bool:
    """Whether retrying could help: connection problems, timeouts, HTTP 5xx and 429."""
    if isinstance(error, requests.HTTPError):
        status = error.response.status_code if error.response is not None else 0
        return status == 429 or status >= 500
    return isinstance(error, requests.ConnectionError | requests.Timeout)


def _describe(error: requests.RequestException | None) -> str:
    """A short description of a failed request (the HTTP status instead of the full URL)."""
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code} {error.response.reason or ''}".strip()
    return str(error)


def fetch_with_retries(url: str) -> bytes:
    """GET `url`, retrying transient failures with exponential backoff.

    Permanent failures (e.g. HTTP 404 or 403) are not retried. Raises OSError if the fetch fails.
    """
    last_error: requests.RequestException | None = None
    for attempt in range(TLE_FETCH_MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=TLE_FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            last_error = e
            if not _is_transient(e) or attempt == TLE_FETCH_MAX_RETRIES:
                break
            time.sleep(TLE_FETCH_RETRY_BASE_DELAY_SECONDS * (2**attempt))
    raise OSError(_describe(last_error)) from last_error


def _parse_single(raw: bytes, expected_norad_id: int) -> EarthSatellite:
    """Parse `raw` as exactly one TLE for `expected_norad_id`; raises ValueError otherwise."""
    entries = list(parse_tle_file(raw.splitlines()))
    if len(entries) != 1:
        raise ValueError(
            f"expected exactly 1 TLE, got {len(entries)} -- the response may "
            f'be an HTML error page, an empty/"No GP data found" body, or '
            f"contain more entries than expected"
        )
    sat = entries[0]
    actual_norad_id = sat.model.satnum
    if actual_norad_id != expected_norad_id:
        raise ValueError(
            f"NORAD ID mismatch: requested {expected_norad_id}, response is for {actual_norad_id}"
        )
    return sat


def validate_tle(raw: bytes, expected_norad_id: int) -> tuple[bytes, datetime]:
    """Confirm `raw` is exactly one valid TLE for `expected_norad_id`; returns (raw, epoch).

    Raises ValueError naming the problem otherwise.
    """
    return raw, _parse_single(raw, expected_norad_id).epoch.utc_datetime()


def epoch_age_days(sat: EarthSatellite, now: datetime | None = None) -> float:
    """Age of `sat`'s TLE epoch in days at `now` (default: the current time)."""
    now = now or datetime.now(UTC)
    return float((now - sat.epoch.utc_datetime()).total_seconds() / 86400)


def check_tle_usable(
    sat: EarthSatellite, now: datetime | None = None, ts: Timescale | None = None
) -> None:
    """Raise TLELoadError if `sat`'s TLE is too old or SGP4 cannot propagate it.

    SGP4 is sampled across the longest window the app offers. Skyfield reports propagation
    errors (e.g. a decayed orbit) in `position.message` rather than raising.
    """
    now = now or datetime.now(UTC)
    age = epoch_age_days(sat, now)
    if age > TLE_EPOCH_MAX_DAYS:
        raise TLELoadError(
            f"TLE epoch is {age:.0f} days old (limit {TLE_EPOCH_MAX_DAYS:.0f}); "
            "predictions would be unreliable"
        )

    ts = ts or load.timescale()
    times = ts.from_datetime(now) + np.array([0.0, 0.5, 1.0]) * TIME_WINDOW_MAX_HOURS / 24.0
    position = sat.at(times)
    messages = np.atleast_1d(np.asarray(getattr(position, "message", None), dtype=object))
    errors = sorted({str(m) for m in messages if m})
    if errors or np.isnan(position.position.km).any():
        raise TLELoadError(f"SGP4 cannot propagate this TLE: {'; '.join(errors) or 'NaN position'}")


def _cache_age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400


def _read_cached(path: Path, norad_id: int) -> EarthSatellite | None:
    """The satellite in cache file `path`, or None if it is missing or not a valid TLE."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        return _parse_single(raw, norad_id)
    except ValueError as e:
        logger.warning("Ignoring invalid cached TLE %s: %s", path, e)
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    """Write `data` to `path` so a reader never sees a partial file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _prepare_cache_dir(cache_dir: Path) -> Path:
    """Create `cache_dir`, falling back to a temporary directory if that is impossible."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    except OSError as e:
        fallback = Path(tempfile.gettempdir()) / "satellite-pass-predictor"
        logger.warning("Cannot create cache directory %s (%s); using %s", cache_dir, e, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _no_cache_message(source: TLESource, error: OSError | ValueError) -> str:
    """The failure reason when a fetch failed and there is no cached copy to fall back to."""
    label = _SOURCE_LABELS[source]
    if isinstance(error, OSError):
        problem = f"{label} unreachable: {error}"
        hint = _MIRROR_HINT if source == "mirror" else "Check your internet connection."
    else:
        problem = f"{label} returned an invalid TLE: {error}"
        hint = (
            _MIRROR_HINT
            if source == "mirror"
            else "The satellite may no longer be tracked, or its NORAD ID in config.py is wrong."
        )
    return f"{problem}; no cached copy exists to fall back to. {hint}"


def _load_one(
    norad_id: int,
    source: TLESource,
    cache_dir: Path,
    now: datetime,
    ts: Timescale,
) -> tuple[EarthSatellite, str | None]:
    """Load one satellite from the cache or `source`; returns it with an optional notice.

    The notice says that a stale cached copy was used because the refresh failed.
    """
    if source == "celestrak":
        url = CELESTRAK_URL.format(norad_id=norad_id)
        max_age_days = MAX_TLE_AGE_DAYS
    else:
        url = TLE_MIRROR_URL.format(norad_id=norad_id)
        max_age_days = TLE_MIRROR_CACHE_AGE_HOURS / 24.0

    path = cache_dir / f"tle_{norad_id}.txt"
    cached = _read_cached(path, norad_id)
    notice = None
    if cached is not None and _cache_age_days(path) <= max_age_days:
        sat = cached
    else:
        try:
            raw = fetch_with_retries(url)
            sat = _parse_single(raw, norad_id)
        except (OSError, ValueError) as e:
            if cached is None:
                raise TLELoadError(_no_cache_message(source, e)) from e
            notice = (
                f"{_SOURCE_LABELS[source]} refresh failed ({e}); "
                f"using the cached copy from {_cache_age_days(path):.1f} day(s) ago."
            )
            logger.warning("NORAD %d: %s", norad_id, notice)
            sat = cached
        else:
            try:
                _write_atomic(path, raw)
            except OSError as e:
                logger.warning("Could not cache %s: %s", path, e)

    check_tle_usable(sat, now, ts)
    age = epoch_age_days(sat, now)
    if age > TLE_EPOCH_WARNING_DAYS:
        logger.warning(
            "NORAD %d: TLE epoch is %.1f days old (warning threshold %.0f); "
            "predictions may be less accurate.",
            norad_id,
            age,
            TLE_EPOCH_WARNING_DAYS,
        )
    return sat, notice


def load_satellites(
    satellites: dict[str, int] | None = None,
    source: TLESource = "celestrak",
    now: datetime | None = None,
    cache_dir: Path | None = None,
) -> TLELoadResult:
    """Load the current TLE of each satellite (default: SATELLITES), keyed by display name.

    A valid cached copy is used while it is younger than the source's max age (MAX_TLE_AGE_DAYS
    for Celestrak, TLE_MIRROR_CACHE_AGE_HOURS for the mirror); otherwise `source` is queried.
    A response is validated before it replaces the cache, and a failed refresh falls back to the
    cached copy. A satellite that cannot be loaded (no usable data, a TLE older than
    TLE_EPOCH_MAX_DAYS, or an SGP4 propagation error) is reported in `failures` and the others
    still load. `now` and `cache_dir` default to the current time and default_cache_dir().
    """
    tracked = SATELLITES if satellites is None else satellites
    now = now or datetime.now(UTC)
    cache_dir = _prepare_cache_dir(cache_dir or default_cache_dir())
    ts = load.timescale()

    loaded: dict[str, EarthSatellite] = {}
    failures: dict[str, SatelliteFailure] = {}
    notices: dict[str, str] = {}
    for name, norad_id in tracked.items():
        try:
            sat, notice = _load_one(norad_id, source, cache_dir, now, ts)
        except TLELoadError as e:
            reason = str(e)
        except Exception as e:
            logger.exception("Unexpected error loading %s (NORAD %d)", name, norad_id)
            reason = f"unexpected error: {type(e).__name__}: {e}"
        else:
            loaded[name] = sat
            if notice:
                notices[name] = notice
            continue
        logger.warning("Could not load %s (NORAD %d): %s", name, norad_id, reason)
        failures[name] = SatelliteFailure(name=name, norad_id=norad_id, reason=reason)
    return TLELoadResult(satellites=loaded, failures=failures, warnings=notices)


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
