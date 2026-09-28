"""
Tests for satellite_pass_predictor.tle_data.

load_satellites() is mostly a thin wrapper around Skyfield's own Loader
(load.tle_file/load.exists/load.days_old) for the no-network-needed path,
plus a direct requests.get() call (not Skyfield's own download(), see
_download_and_cache_tle()'s docstring for why) for the actual network
fetch -- exercising *that* for real would mean either hitting the network
in a test suite (flaky, slow, exactly what we're avoiding) or asserting
on real file mtimes (fragile, timing-dependent). Not worth it: Skyfield's
own caching mechanics and requests' own HTTP handling are each library's
responsibility to test, not ours.

What *is* worth testing is the logic this module adds on top of both:
the staleness decision (does a fresh cache skip the network entirely,
and does a stale/missing one trigger a fetch?), the OSError fallback-to-
cache behavior (does a failed fetch degrade to the cached copy when one
exists, and re-raise with an actionable message when it doesn't?), the
empty-response check (does a response with no usable TLE in it -- e.g.
an unrecognized/decayed NORAD ID -- raise a clear ValueError instead of
a bare IndexError?), the retry-with-backoff around the actual fetch
(does it retry the right number of times and eventually succeed, and
does it still give up and raise after exhausting retries rather than
looping forever?), and the per-call timeout passed to requests.get()
(is it actually passed, and does a requests exception get translated to
the same OSError type the rest of this module already handles?). All
exercised here with Skyfield's Loader methods and requests.get() mocked
out, and time.sleep mocked out where retries are exercised -- no
network, no real TLE files beyond the one fixture used as realistic
response bytes, no timing dependence, no test that actually waits out a
real retry delay or a real timeout.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from satellite_pass_predictor import tle_data

FAKE_SATELLITES = {"TESTSAT": 99999}

# Real TLE bytes (the same fixture conftest.py's iss_satellite fixture
# uses) for tests that exercise the real requests.get() -> write-to-
# cache -> parse_tle_file() pipeline -- using genuine, valid TLE text
# here (rather than a mocked-out parse step) is what actually proves
# that pipeline is wired together correctly end to end.
_FAKE_TLE_BYTES = (
    b"ISS (ZARYA)\n"
    b"1 25544U 98067A   26263.78762384  .00007766  00000+0  14793-3 0  9996\n"
    b"2 25544  51.6308 186.9475 0004819 163.3027 196.8121 15.49200337586572\n"
)


def _fake_response(content: bytes = _FAKE_TLE_BYTES) -> MagicMock:
    """A requests.Response stand-in: raise_for_status() is a no-op (as
    it is for any real 2xx response) and .content is the given bytes."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.content = content
    return response


def test_uses_cached_copy_without_reload_when_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache file that exists and is younger than MAX_TLE_AGE_DAYS
    should be used as-is (reload=False) via Skyfield's own tle_file(),
    not re-fetched -- no network call (requests.get) should happen."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 0.1)
    mock_tle_file = MagicMock(return_value=["sentinel-satellite"])
    monkeypatch.setattr(tle_data.load, "tle_file", mock_tle_file)
    mock_get = MagicMock(side_effect=AssertionError("should not fetch when cache is fresh"))
    monkeypatch.setattr(tle_data.requests, "get", mock_get)

    result = tle_data.load_satellites(FAKE_SATELLITES)

    assert result == {"TESTSAT": "sentinel-satellite"}
    _, kwargs = mock_tle_file.call_args
    assert kwargs["reload"] is False
    mock_get.assert_not_called()


def test_reloads_when_cache_is_older_than_max_age(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache file older than MAX_TLE_AGE_DAYS should trigger a
    network fetch (via requests.get(), not Skyfield's tle_file()), even
    though a cached copy exists."""
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 5.0)
    mock_get = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(tle_data.requests, "get", mock_get)

    result = tle_data.load_satellites(FAKE_SATELLITES)

    assert result["TESTSAT"].name == "ISS (ZARYA)"
    args, kwargs = mock_get.call_args
    assert args[0] == tle_data.CELESTRAK_URL.format(norad_id=99999)
    assert kwargs["timeout"] == tle_data.TLE_FETCH_TIMEOUT_SECONDS


def test_reloads_when_no_cache_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No cache file at all should also trigger a network fetch,
    regardless of whatever days_old() would say (it shouldn't even be
    consulted -- `not exists` short-circuits the `or`)."""
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)

    def _unexpected_days_old(filename: str) -> float:
        raise AssertionError("days_old() should not be called when the cache doesn't exist")

    monkeypatch.setattr(tle_data.load, "days_old", _unexpected_days_old)
    mock_get = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(tle_data.requests, "get", mock_get)

    result = tle_data.load_satellites(FAKE_SATELLITES)

    assert result["TESTSAT"].name == "ISS (ZARYA)"
    mock_get.assert_called_once()


def test_fetch_writes_the_response_to_the_cache_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful fetch should write the raw response bytes to the
    local cache file -- the same on-disk artifact Skyfield's own
    downloader would have produced -- so a later run can use it as the
    cache."""
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=_fake_response()))

    tle_data.load_satellites(FAKE_SATELLITES)

    cached_file = tmp_path / "tle_99999.txt"
    assert cached_file.read_bytes() == _FAKE_TLE_BYTES


def test_falls_back_to_cache_when_fetch_fails_but_cache_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A network hiccup shouldn't crash the whole pipeline if a usable
    (if stale) cached copy already exists -- it should fall back to
    that cached copy (read via Skyfield's own tle_file(), unrelated to
    the requests-based fetch) and print a warning explaining why,
    rather than silently pretending nothing happened.

    The mocked fetch fails on every call, so this also exercises (as a
    side effect) _fetch_tle_with_retries() exhausting all of its retries
    before load_satellites()'s own fallback-to-cache logic ever sees the
    OSError -- time.sleep is mocked out so that doesn't actually slow
    this test down by several seconds.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 3.0)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests, "get",
        MagicMock(side_effect=requests.ConnectionError("503 Service Unavailable")),
    )
    # The fallback read (load.tle_file(filename), no reload/url) is
    # Skyfield's own local-file path, untouched by this change.
    monkeypatch.setattr(tle_data.load, "tle_file", lambda *a, **k: ["sentinel-satellite"])

    result = tle_data.load_satellites(FAKE_SATELLITES)

    assert result == {"TESTSAT": "sentinel-satellite"}
    warning = capsys.readouterr().out
    assert "TESTSAT" in warning
    assert "3.0 day" in warning


def test_reraises_with_actionable_message_when_fetch_fails_and_no_cache_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    No cached fallback available (e.g. the very first run ever, with no
    internet) -- the failure should propagate as an OSError, but wrapped
    with a message naming the satellite/NORAD ID and pointing at the
    likely cause, not just a raw requests exception passed straight
    through.

    Also exercises _fetch_tle_with_retries() exhausting its retries
    first (time.sleep mocked out, same reasoning as the test above).
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests, "get",
        MagicMock(side_effect=requests.ConnectionError("<urlopen error [Errno 11001] getaddrinfo failed>")),
    )

    with pytest.raises(OSError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message
    assert "internet connection" in message


def test_retries_a_flaky_fetch_and_succeeds_without_exhausting_all_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    A transient failure (fewer than TLE_FETCH_MAX_RETRIES blips) should
    be absorbed by _fetch_tle_with_retries() -- the fetch should retry
    exactly as many times as it takes to succeed, not more, and
    load_satellites() should return the eventually-successful result as
    if nothing had gone wrong (no fallback-to-cache, no warning).
    """
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    sleep_calls: list[float] = []
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    call_count = 0

    def _get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # fails twice, succeeds on the 3rd attempt
            raise requests.Timeout("Connection timed out")
        return _fake_response()

    monkeypatch.setattr(tle_data.requests, "get", _get)

    result = tle_data.load_satellites(FAKE_SATELLITES)

    assert result["TESTSAT"].name == "ISS (ZARYA)"
    assert call_count == 3
    # Exponential backoff: 1s after the 1st failure, 2s after the 2nd --
    # no 3rd sleep, since the 3rd attempt succeeded.
    assert sleep_calls == [1.0, 2.0]


def test_gives_up_after_max_retries_rather_than_retrying_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A persistent failure (every attempt fails) should still raise the
    same informative OSError load_satellites() already raised before
    retries existed -- retries absorb a transient blip, they don't turn
    a genuine, ongoing outage into an infinite/silent retry loop.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)

    call_count = 0

    def _get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise requests.Timeout("Connection timed out")

    monkeypatch.setattr(tle_data.requests, "get", _get)

    with pytest.raises(OSError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    # TLE_FETCH_MAX_RETRIES retries plus the initial attempt -- not one
    # call more (no infinite loop) and not one fewer (retries actually
    # happened).
    assert call_count == tle_data.TLE_FETCH_MAX_RETRIES + 1
    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message
    assert "internet connection" in message


def test_fetch_passes_the_configured_timeout_to_requests_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    requests.get()'s own `timeout` argument is what bounds this fetch
    now (see _download_and_cache_tle()'s docstring for why this
    replaced an earlier socket.setdefaulttimeout()-based approach: that
    mutated a process-global default, unsafe under Streamlit's real
    per-session-thread concurrency, where one session's cleanup could
    silently unbound another session's still-in-flight fetch).
    requests.get()'s timeout is a plain per-call argument with no
    shared state at all, so this just confirms it's actually passed,
    not some other mechanism silently doing nothing.
    """
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    mock_get = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(tle_data.requests, "get", mock_get)

    tle_data.load_satellites(FAKE_SATELLITES)

    _, kwargs = mock_get.call_args
    assert kwargs["timeout"] == tle_data.TLE_FETCH_TIMEOUT_SECONDS


def test_requests_exception_is_translated_to_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A requests exception (timeout, dropped connection, HTTP error
    status, ...) must surface as OSError, not a raw requests exception
    -- that's what the retry loop's `except OSError` and
    load_satellites()'s own fallback/re-raise handling both key on.
    Checked with two different requests exception types, since
    requests.RequestException is a base class covering several.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)

    for exc in [requests.Timeout("timed out"), requests.ConnectionError("refused")]:
        monkeypatch.setattr(tle_data.requests, "get", MagicMock(side_effect=exc))

        with pytest.raises(OSError):
            tle_data.load_satellites(FAKE_SATELLITES)


def test_http_error_status_is_checked_and_not_handed_to_the_tle_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A non-2xx response (e.g. Celestrak returning a 404/500 error page)
    must not be treated as successful content and handed to
    parse_tle_file() as if it were real TLE data -- _download_and_cache_tle()
    calls response.raise_for_status() specifically to catch this, before
    ever looking at the response body.

    Uses a real requests.Response (not a mock standing in for it) with
    an actual error status code, so this proves raise_for_status()
    itself is what's being relied on -- not just that some exception,
    however it might arise, happens to get translated to OSError.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)

    error_response = requests.Response()
    error_response.status_code = 404
    error_response._content = b"<html>Not Found</html>"  # would fail to parse as TLE anyway
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=error_response))

    with pytest.raises(OSError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    assert "TESTSAT" in str(exc_info.value)


def test_raises_value_error_when_response_has_no_usable_tle_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Celestrak (or a cached file) can return a 200 OK with an empty or
    "No GP data found" body for an unrecognized/decayed NORAD ID --
    Skyfield's TLE parser doesn't raise on that, it just returns an
    empty list. Without a check for this, entries[0] would raise a bare
    IndexError with no indication of the real cause -- this should
    instead be a clear ValueError naming the satellite and NORAD ID.
    Exercised here via the fresh-cache/no-network path (Skyfield's own
    tle_file()); the requests-based fetch path shares the same
    parse-then-check logic (parse_tle_file() -> load_satellites()'s
    `if not entries` check), not a separate implementation of it.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 0.1)
    monkeypatch.setattr(tle_data.load, "tle_file", lambda *a, **k: [])

    with pytest.raises(ValueError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message


def test_raises_value_error_when_fetched_response_has_no_usable_tle_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Same check as above, but for a freshly-*fetched* (not cached)
    empty/malformed response -- proves the requests-based fetch path's
    real parse_tle_file() call (not a mock standing in for it) also
    reaches load_satellites()'s empty-entries check correctly, end to
    end.
    """
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=_fake_response(b"")))

    with pytest.raises(ValueError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message


# ---------------------------------------------------------------------------
# source="mirror": same machinery as source="celestrak" (default), but a
# different URL template and a much shorter on-disk cache age -- see
# TLE_MIRROR_URL/TLE_MIRROR_CACHE_AGE_HOURS's comments in config.py.
# ---------------------------------------------------------------------------


def test_default_source_is_celestrak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No `source` argument at all -- the existing call signature every
    caller before this feature used -- should still hit Celestrak, not
    the mirror."""
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    mock_get = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(tle_data.requests, "get", mock_get)

    tle_data.load_satellites(FAKE_SATELLITES)

    args, _ = mock_get.call_args
    assert args[0] == tle_data.CELESTRAK_URL.format(norad_id=99999)


def test_mirror_source_fetches_from_the_mirror_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    mock_get = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(tle_data.requests, "get", mock_get)

    result = tle_data.load_satellites(FAKE_SATELLITES, source="mirror")

    assert result["TESTSAT"].name == "ISS (ZARYA)"
    args, _ = mock_get.call_args
    assert args[0] == tle_data.TLE_MIRROR_URL.format(norad_id=99999)


def test_mirror_source_uses_its_own_much_shorter_cache_age(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    A cache age exactly between the mirror's short threshold and
    Celestrak's day-long one should be treated as fresh for "celestrak"
    but stale for "mirror" -- proving the two sources genuinely use
    different max ages, not just different URLs.
    """
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    between_the_two_thresholds = (
        tle_data.TLE_MIRROR_CACHE_AGE_HOURS / 24.0 + tle_data.MAX_TLE_AGE_DAYS
    ) / 2
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: between_the_two_thresholds)
    mock_tle_file = MagicMock(return_value=["sentinel-satellite"])
    monkeypatch.setattr(tle_data.load, "tle_file", mock_tle_file)
    mock_get = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(tle_data.requests, "get", mock_get)

    tle_data.load_satellites(FAKE_SATELLITES, source="celestrak")
    mock_get.assert_not_called()  # fresh enough for Celestrak's own 1-day threshold

    tle_data.load_satellites(FAKE_SATELLITES, source="mirror")
    mock_get.assert_called_once()  # but stale for the mirror's own ~1h threshold


def test_mirror_source_error_message_points_at_the_refresh_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An unreachable mirror with no local cache to fall back to (e.g. the
    tle-data branch doesn't exist yet, or has never been reachable from
    this machine) should tell the user to run the refresh workflow --
    "check your internet connection" (the celestrak-source message)
    would be actively misleading here.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests, "get",
        MagicMock(side_effect=requests.ConnectionError("404 Not Found")),
    )

    with pytest.raises(OSError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES, source="mirror")

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "refresh-tles.yml" in message or "Refresh TLE mirror" in message
    assert "check your internet connection" not in message.lower()


# ---------------------------------------------------------------------------
# is_older_than(): pure staleness logic, no network/Skyfield/Streamlit --
# shared by app.py's mirror-refresh-age and TLE-epoch-age warnings.
# ---------------------------------------------------------------------------


def test_is_older_than_false_when_well_within_max_age() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    reference_time = now - tle_data.timedelta(hours=1)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is False


def test_is_older_than_true_when_well_past_max_age() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    reference_time = now - tle_data.timedelta(hours=48)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is True


def test_is_older_than_boundary_exactly_at_max_age_is_not_yet_stale() -> None:
    """Exactly at the threshold should not (yet) count as stale -- the
    comparison is a strict `>`, not `>=`."""
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    reference_time = now - tle_data.timedelta(hours=24)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is False


def test_is_older_than_boundary_one_second_past_max_age_is_stale() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    reference_time = now - tle_data.timedelta(hours=24, seconds=1)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is True


def test_is_older_than_uses_the_real_current_time_when_now_not_given() -> None:
    """Without an explicit `now`, this should compare against the actual
    wall clock -- checked with a reference_time far enough in the past
    that the result is unambiguous regardless of when the test runs."""
    long_ago = tle_data.datetime(2000, 1, 1, tzinfo=tle_data.timezone.utc)

    assert tle_data.is_older_than(long_ago, tle_data.timedelta(days=1)) is True


# ---------------------------------------------------------------------------
# fetch_mirror_metadata(): best-effort, never raises -- app.py's
# provenance display must degrade gracefully, not break, if this is
# unavailable or malformed.
# ---------------------------------------------------------------------------


def test_fetch_mirror_metadata_returns_parsed_data_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "generated_at": "2026-09-28T09:00:00Z",
        "satellites": {
            "99999": {
                "name": "TESTSAT", "fetched_at": "2026-09-28T09:00:03Z",
                "tle_epoch": "2026-09-27T18:00:00Z", "source_url": "https://example/",
            },
        },
    }
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=response))

    metadata = tle_data.fetch_mirror_metadata()

    assert metadata == payload


def test_fetch_mirror_metadata_returns_none_on_request_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tle_data.requests, "get",
        MagicMock(side_effect=requests.ConnectionError("refused")),
    )

    assert tle_data.fetch_mirror_metadata() is None


def test_fetch_mirror_metadata_returns_none_on_http_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tle-data branch is a 404 from raw.githubusercontent.com
    -- raise_for_status() must actually be checked here too, not just
    for the TLE fetch itself."""
    error_response = requests.Response()
    error_response.status_code = 404
    error_response._content = b"404: Not Found"
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=error_response))

    assert tle_data.fetch_mirror_metadata() is None


def test_fetch_mirror_metadata_returns_none_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(side_effect=ValueError("not valid JSON"))
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=response))

    assert tle_data.fetch_mirror_metadata() is None


def test_fetch_mirror_metadata_returns_none_on_unexpected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid JSON, but missing the keys app.py's provenance display
    actually needs -- treated the same as "unavailable", not a crash."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"unexpected": "shape"})
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=response))

    assert tle_data.fetch_mirror_metadata() is None


# ---------------------------------------------------------------------------
# parse_iso_utc(): a small, forgiving ISO-timestamp parser used both by
# compute_staleness_warnings() below and directly by app.py's display code.
# ---------------------------------------------------------------------------


def test_parse_iso_utc_parses_a_real_metadata_timestamp() -> None:
    parsed = tle_data.parse_iso_utc("2026-09-28T09:00:03Z")

    assert parsed == tle_data.datetime(2026, 9, 28, 9, 0, 3, tzinfo=tle_data.timezone.utc)


def test_parse_iso_utc_returns_none_for_garbage() -> None:
    assert tle_data.parse_iso_utc("not a timestamp") is None


# ---------------------------------------------------------------------------
# compute_staleness_warnings(): the actual decision app.py's st.warning
# banners are based on -- pure, no Streamlit/network involved.
# ---------------------------------------------------------------------------


def _metadata_generated_at(timestamp: str) -> tle_data.TLEMirrorMetadata:
    return {"generated_at": timestamp, "satellites": {}}


def test_no_warnings_when_mirror_and_all_epochs_are_fresh() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    metadata = _metadata_generated_at("2026-09-28T11:00:00Z")  # 1h old
    epochs = {"ISS (ZARYA)": now - tle_data.timedelta(days=1)}  # 1 day old

    result = tle_data.compute_staleness_warnings(metadata, epochs, now=now)

    assert result == {"mirror_stale": False, "stale_satellite_names": []}


def test_mirror_stale_when_generated_at_older_than_warning_threshold() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    metadata = _metadata_generated_at("2026-09-25T12:00:00Z")  # 3 days old

    result = tle_data.compute_staleness_warnings(metadata, {}, now=now)

    assert result["mirror_stale"] is True


def test_mirror_stale_is_false_not_true_when_metadata_is_none() -> None:
    """"Provenance unavailable" and "confirmed stale" are different
    claims -- app.py shows a separate message for the None case rather
    than this warning firing on missing data."""
    result = tle_data.compute_staleness_warnings(None, {}, now=tle_data.datetime.now(tle_data.timezone.utc))

    assert result["mirror_stale"] is False


def test_satellite_flagged_stale_when_its_epoch_is_old_but_others_are_not() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    epochs = {
        "ISS (ZARYA)": now - tle_data.timedelta(days=1),
        "OLDSAT": now - tle_data.timedelta(days=10),
    }

    result = tle_data.compute_staleness_warnings(_metadata_generated_at("2026-09-28T11:00:00Z"), epochs, now=now)

    assert result["stale_satellite_names"] == ["OLDSAT"]


def test_compute_staleness_warnings_boundary_exactly_at_threshold_is_not_stale() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.timezone.utc)
    exactly_at_threshold = now - tle_data.timedelta(hours=tle_data.MIRROR_REFRESH_WARNING_HOURS)

    result = tle_data.compute_staleness_warnings(
        _metadata_generated_at(exactly_at_threshold.strftime("%Y-%m-%dT%H:%M:%SZ")), {}, now=now,
    )

    assert result["mirror_stale"] is False
