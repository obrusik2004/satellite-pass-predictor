"""
Tests for satellite_pass_predictor.tle_data.

load_satellites() is mostly a thin wrapper around Skyfield's own
Loader (load.tle_file/load.exists/load.days_old) plus a network call to
Celestrak -- exercising *that* for real would mean either hitting the
network in a test suite (flaky, slow, exactly what we're avoiding) or
asserting on real file mtimes (fragile, timing-dependent). Not worth it:
Skyfield's own caching mechanics are Skyfield's responsibility to test,
not ours.

What *is* worth testing is the logic this module adds on top of
Skyfield: the staleness decision (does it ask for reload=True at the
right times?), the OSError fallback-to-cache behavior (does a failed
fetch degrade to the cached copy when one exists, and re-raise with an
actionable message when it doesn't?), and the empty-response check (does
a response with no usable TLE in it -- e.g. an unrecognized/decayed
NORAD ID -- raise a clear ValueError instead of a bare IndexError?).
All exercised here with Skyfield's Loader methods mocked out -- no
network, no real TLE files, no timing dependence.
"""

from unittest.mock import MagicMock

import pytest

from satellite_pass_predictor import tle_data

FAKE_SATELLITES = {"TESTSAT": 99999}


def test_uses_cached_copy_without_reload_when_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache file that exists and is younger than MAX_TLE_AGE_DAYS
    should be used as-is (reload=False), not re-fetched."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 0.1)
    mock_tle_file = MagicMock(return_value=["sentinel-satellite"])
    monkeypatch.setattr(tle_data.load, "tle_file", mock_tle_file)

    result = tle_data.load_satellites(FAKE_SATELLITES)

    assert result == {"TESTSAT": "sentinel-satellite"}
    _, kwargs = mock_tle_file.call_args
    assert kwargs["reload"] is False


def test_reloads_when_cache_is_older_than_max_age(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache file older than MAX_TLE_AGE_DAYS should trigger a
    refetch (reload=True), even though a cached copy exists."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 5.0)
    mock_tle_file = MagicMock(return_value=["sentinel-satellite"])
    monkeypatch.setattr(tle_data.load, "tle_file", mock_tle_file)

    tle_data.load_satellites(FAKE_SATELLITES)

    _, kwargs = mock_tle_file.call_args
    assert kwargs["reload"] is True


def test_reloads_when_no_cache_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cache file at all should also trigger reload=True, regardless
    of whatever days_old() would say (it shouldn't even be consulted --
    `not exists` short-circuits the `or`)."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)

    def _unexpected_days_old(filename: str) -> float:
        raise AssertionError("days_old() should not be called when the cache doesn't exist")

    monkeypatch.setattr(tle_data.load, "days_old", _unexpected_days_old)
    mock_tle_file = MagicMock(return_value=["sentinel-satellite"])
    monkeypatch.setattr(tle_data.load, "tle_file", mock_tle_file)

    tle_data.load_satellites(FAKE_SATELLITES)

    _, kwargs = mock_tle_file.call_args
    assert kwargs["reload"] is True


def test_falls_back_to_cache_when_fetch_fails_but_cache_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A network hiccup shouldn't crash the whole pipeline if a usable
    (if stale) cached copy already exists -- it should fall back to
    that cached copy and print a warning explaining why, rather than
    silently pretending nothing happened.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 3.0)

    def _tle_file(*args, **kwargs):
        # First call (the network fetch, called with url=... reload=...)
        # fails; the fallback call (just the cached filename, positional,
        # no reload kwarg) succeeds.
        if "reload" in kwargs:
            raise OSError("503 Service Unavailable")
        return ["sentinel-satellite"]

    monkeypatch.setattr(tle_data.load, "tle_file", _tle_file)

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
    likely cause, not just Skyfield's low-level "cannot download ..."
    text passed straight through.
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)

    def _tle_file(*args, **kwargs):
        raise OSError("<urlopen error [Errno 11001] getaddrinfo failed>")

    monkeypatch.setattr(tle_data.load, "tle_file", _tle_file)

    with pytest.raises(OSError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message
    assert "internet connection" in message


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
    """
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 0.1)
    monkeypatch.setattr(tle_data.load, "tle_file", lambda *a, **k: [])

    with pytest.raises(ValueError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message
