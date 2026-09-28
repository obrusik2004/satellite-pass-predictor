"""
TLE ingestion: fetching (and locally caching) current orbital element data
for the tracked satellites from Celestrak.
"""

import os
import time
from datetime import datetime, timedelta, timezone
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

# "celestrak": Celestrak's own gp.php endpoint directly -- what main.py/
# the CLI uses by default, and works fine outside Streamlit Cloud.
# "mirror": the GitHub-Actions-refreshed tle-data branch mirror -- what
# the Streamlit app uses, since Streamlit Cloud cannot reach Celestrak
# at all (see TLE_MIRROR_URL's comment in config.py).
TLESource = Literal["celestrak", "mirror"]


def _download_and_cache_tle(url: str, filename: str) -> list[EarthSatellite]:
    """
    Fetch `url`'s TLE text via requests.get(timeout=...) -- not
    Skyfield's own download()/tle_file(), which fetches through
    urlopen() with no timeout of its own (confirmed directly against
    skyfield/iokit.py's source). A first version of this retry/timeout
    fix used socket.setdefaulttimeout() to bound that urlopen() call,
    which turned out to be unsafe: it mutates a process-global default,
    but Streamlit runs each user session's script in its own thread,
    all sharing that one process. Two sessions fetching concurrently
    could have one session's `finally` restore the timeout to unbounded
    while the other session's fetch was still relying on it being
    bounded -- silently reintroducing the exact unbounded-hang problem
    this exists to fix, intermittently and only under real concurrent
    load, which a single-threaded test can't catch. requests.get()'s
    own `timeout` argument is a genuinely per-call setting with no
    global/shared state at all, so it doesn't have this problem.

    Writes the raw response bytes to `filename` -- the same on-disk
    cache Skyfield's own downloader would have produced -- then parses
    them with parse_tle_file(), the same parser Skyfield's tle_file()
    uses internally. This only changes *how* the bytes get here, not
    how they're written or interpreted once they do.

    requests.RequestException (a timeout, a dropped connection, an HTTP
    error status via raise_for_status(), ...) is translated to OSError
    so everything downstream -- this module's retry loop, then
    load_satellites()'s fallback-to-cache/re-raise-with-context -- sees
    the same exception type it always has, regardless of which library
    actually made the request.
    """
    try:
        response = requests.get(url, timeout=TLE_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise OSError(f"cannot download {url} because {e}") from e

    with open(filename, "wb") as f:
        f.write(response.content)

    # parse_tle_file() is skyfield's own untyped call (see mypy.ini's
    # skyfield.* override), so list(...) of it is already exactly
    # list[EarthSatellite] as far as mypy can tell (EarthSatellite
    # itself resolves to Any under that same override) -- no cast()
    # needed here, unlike load.tle_file() below, whose own return type
    # isn't inferred as a list at all without one.
    #
    # Doesn't raise on a response with no valid TLE pairs in it (e.g.
    # Celestrak's "200 OK, empty body" case) -- it just yields nothing,
    # same as Skyfield's own tle_file() would -- so that case reaches
    # load_satellites()'s existing empty-entries check unchanged.
    return list(parse_tle_file(response.content.splitlines()))


def _fetch_tle_with_retries(
    url: str, filename: str, reload: bool
) -> list[EarthSatellite]:
    """
    The actual network fetch (via _download_and_cache_tle(), not the
    reading-a-local-file fallback load_satellites() uses when this is
    exhausted), retried with exponential backoff on OSError -- see
    config.TLE_FETCH_MAX_RETRIES's comment for why this exists at all
    (Streamlit Community Cloud's outbound networking is intermittently
    flaky) rather than being an arbitrary safety net.

    reload=False skips all of this and goes straight to Skyfield's own
    tle_file(): load_satellites() only ever passes reload=False when
    the cache file already exists, and Skyfield's own reload=False
    behavior in that case is to just open the existing local file --
    no network call happens at all, so there's nothing here that needs
    retrying or a bounded timeout, and no reason not to reuse Skyfield's
    already-correct local-file-reading logic rather than reimplementing
    it.

    Only OSError is retried: that's specifically the network-failure
    exception _download_and_cache_tle() raises (a dropped connection,
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
    if not reload:
        return cast(list[EarthSatellite], load.tle_file(url, filename=filename, reload=False))

    for attempt in range(TLE_FETCH_MAX_RETRIES):
        try:
            return _download_and_cache_tle(url, filename)
        except OSError:
            time.sleep(TLE_FETCH_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
    return _download_and_cache_tle(url, filename)


def load_satellites(
    satellites: dict[str, int] = SATELLITES,
    source: TLESource = "celestrak",
) -> dict[str, EarthSatellite]:
    """
    Fetch current TLE data for each satellite, from a local cache when it's
    fresh enough or from `source` otherwise.

    `source` picks the URL template (CELESTRAK_URL or TLE_MIRROR_URL) and
    the on-disk cache's max age (MAX_TLE_AGE_DAYS or
    TLE_MIRROR_CACHE_AGE_HOURS -- see TLESource and those constants'
    comments in config.py for why the two need different cache ages).
    Both sources share every other part of this function -- retry/
    timeout behavior, the on-disk cache, and the fallback/error handling
    below -- unchanged.

    Raises OSError (with a message naming the satellite and pointing at
    the likely cause -- worded differently per source, since "check your
    internet connection" isn't the right advice for a missing/unreachable
    mirror branch) if a fetch fails and there's no cached copy to fall
    back to -- e.g. the very first run on a machine with no internet
    connection, or the very first run before the mirror has ever been
    published. Raises ValueError if the source (or a cached file) returns
    a response with no usable TLE in it at all -- e.g. an unrecognized or
    no-longer-tracked NORAD ID; Celestrak has been observed to signal
    this both as an HTTP error and as an HTTP 200 with an empty/"No GP
    data found" body, and only the former is caught by the OSError
    handling below, so this is checked separately.

    Returns a dict mapping display name -> skyfield EarthSatellite.
    """
    os.makedirs(TLE_CACHE_DIR, exist_ok=True)

    if source == "celestrak":
        url_template = CELESTRAK_URL
        max_age_days = MAX_TLE_AGE_DAYS
    else:
        url_template = TLE_MIRROR_URL
        max_age_days = TLE_MIRROR_CACHE_AGE_HOURS / 24.0

    result: dict[str, EarthSatellite] = {}
    for name, norad_id in satellites.items():
        url = url_template.format(norad_id=norad_id)
        filename = os.path.join(TLE_CACHE_DIR, f"tle_{norad_id}.txt")

        stale = (
            not load.exists(filename)
            or load.days_old(filename) > max_age_days
        )
        try:
            entries = _fetch_tle_with_retries(url, filename, reload=stale)
        except OSError as e:
            # The source being briefly unreachable/rate-limited shouldn't
            # crash the whole pipeline if we already have a usable (if a
            # bit stale) copy on disk -- fall back to it, but say so.
            if not load.exists(filename):
                # No cache to fall back to either -- e.g. the very first
                # run on a machine with no internet connection (celestrak
                # source), or the very first run before the mirror has
                # ever been published (mirror source, most likely a 404
                # for a branch that doesn't exist yet). Either way,
                # Skyfield's/requests' own OSError here is accurate but
                # low-level (a raw urllib/HTTP message with no mention of
                # which satellite or why it matters); re-raise with
                # context instead of letting that bubble straight up to
                # the user.
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


class TLESatelliteMetadata(TypedDict):
    """One satellite's entry in the mirror's metadata.json (see
    scripts/fetch_tles.py, which writes this file)."""
    name: str
    fetched_at: str  # ISO 8601 UTC -- last time this satellite's fetch actually succeeded
    tle_epoch: str  # ISO 8601 UTC -- that TLE's own epoch
    source_url: str


class TLEMirrorMetadata(TypedDict):
    """The mirror's metadata.json, as fetch_mirror_metadata() returns it."""
    generated_at: str  # ISO 8601 UTC -- when the refresh workflow last *ran*, regardless of whether every satellite's fetch succeeded that run
    satellites: dict[str, TLESatelliteMetadata]  # keyed by NORAD ID as a string (JSON object keys are always strings)


def fetch_mirror_metadata() -> TLEMirrorMetadata | None:
    """
    Best-effort fetch of the mirror's metadata.json -- purely for
    app.py's data-provenance display ("last refreshed ...", staleness
    warnings), never required for the TLEs themselves to load or for
    load_satellites() to succeed.

    Returns None on any problem at all: a network failure, a missing
    tle-data branch (same "doesn't exist yet" case load_satellites()
    handles for the TLE files themselves), malformed JSON, or JSON that
    doesn't have the shape this expects -- rather than raising. A user
    should still be able to see satellite passes even if this one
    auxiliary file is unavailable or unreadable; the caller (app.py) is
    expected to just skip the provenance display in that case, not treat
    it as a load_satellites()-style failure.

    Deliberately no retry here, unlike the TLE fetch itself: this is
    non-critical, best-effort information, and retrying would only add
    latency for something that's allowed to simply fail.
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


def is_older_than(reference_time: datetime, max_age: timedelta, now: datetime | None = None) -> bool:
    """
    Whether `reference_time` (a timezone-aware UTC datetime) is more
    than `max_age` older than `now` (defaults to the real current time
    if not given, so callers with a fixed clock -- and tests, for
    deterministic boundary-exact assertions -- can pass one explicitly).

    A pure function of its inputs -- no Streamlit, no network, no
    Skyfield -- so app.py's two staleness checks (the mirror refresh's
    own age, via metadata.json's generated_at; each satellite's TLE
    epoch age, via EarthSatellite.epoch) share one tested implementation
    instead of two inline datetime computations. Parsing stays the
    caller's concern rather than this function's: metadata.json's
    generated_at is an ISO string (parse with
    datetime.fromisoformat(ts.replace("Z", "+00:00"))), while an
    EarthSatellite's epoch is a Skyfield Time (convert with
    .epoch.utc_datetime()) -- different enough sources that baking one
    parsing convention into this function would make it awkward for the
    other caller, for no benefit to the actual comparison it does.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    return (now - reference_time) > max_age


def parse_iso_utc(timestamp: str) -> datetime | None:
    """
    Parse an ISO 8601 UTC timestamp as metadata.json stores them (e.g.
    "2026-09-28T09:00:03Z") into a timezone-aware datetime. Returns None
    for anything that doesn't parse, rather than raising -- used both by
    compute_staleness_warnings() below and directly by app.py wherever
    it displays one of metadata.json's timestamps, since a malformed
    value there shouldn't crash the app any more than a missing
    metadata.json should (see fetch_mirror_metadata()).
    """
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


class StalenessWarnings(TypedDict):
    """What app.py's st.warning banners should say, decided once here
    rather than inline in the Streamlit script -- see
    compute_staleness_warnings()."""
    mirror_stale: bool
    stale_satellite_names: list[str]


def compute_staleness_warnings(
    metadata: TLEMirrorMetadata | None,
    satellite_epochs: dict[str, datetime],
    now: datetime | None = None,
) -> StalenessWarnings:
    """
    Pure decision of which staleness warnings app.py should show, given
    the mirror's metadata (or None if fetch_mirror_metadata() couldn't
    get it) and each currently-displayed satellite's real TLE epoch.

    `satellite_epochs` should come from each loaded EarthSatellite's own
    `.epoch.utc_datetime()`, not metadata.json's own copy of the same
    value -- the satellite that was actually loaded and is actually
    being used for propagation is the more direct source of truth for
    its own epoch than a JSON file recorded at fetch time.

    mirror_stale is False (not True/unknown) when `metadata` is None:
    "we don't know how fresh the mirror is" is a different claim than
    "it's confirmed stale", and app.py already shows a separate
    provenance-unavailable message for the None case rather than
    conflating it with this warning.
    """
    mirror_stale = False
    if metadata is not None:
        generated_at = parse_iso_utc(metadata["generated_at"])
        if generated_at is not None:
            mirror_stale = is_older_than(
                generated_at, timedelta(hours=MIRROR_REFRESH_WARNING_HOURS), now=now
            )

    stale_satellite_names = [
        name for name, epoch in satellite_epochs.items()
        if is_older_than(epoch, timedelta(days=TLE_EPOCH_WARNING_DAYS), now=now)
    ]

    return {"mirror_stale": mirror_stale, "stale_satellite_names": stale_satellite_names}
