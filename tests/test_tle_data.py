"""Tests for satellite_pass_predictor.tle_data.

Skyfield's Loader methods, requests.get and time.sleep are mocked, so nothing touches the
network and no test waits out a real retry delay or timeout.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from satellite_pass_predictor import tle_data

FAKE_SATELLITES = {"TESTSAT": 99999}

# Real TLE text, so the fetch -> cache file -> parse pipeline is exercised end to end.
_FAKE_TLE_BYTES = (
    b"ISS (ZARYA)\n"
    b"1 25544U 98067A   26263.78762384  .00007766  00000+0  14793-3 0  9996\n"
    b"2 25544  51.6308 186.9475 0004819 163.3027 196.8121 15.49200337586572\n"
)


def _fake_response(content: bytes = _FAKE_TLE_BYTES) -> MagicMock:
    """A successful requests.Response stand-in with the given body."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.content = content
    return response


def test_uses_cached_copy_without_reload_when_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh cache is read as-is, with no network call."""
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
    """A cache older than MAX_TLE_AGE_DAYS triggers a fetch."""
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
    """A missing cache triggers a fetch without consulting the cache age."""
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
    """The raw response bytes are written to the cache file."""
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=_fake_response()))

    tle_data.load_satellites(FAKE_SATELLITES)

    cached_file = tmp_path / "tle_99999.txt"
    assert cached_file.read_bytes() == _FAKE_TLE_BYTES


def test_falls_back_to_cache_when_fetch_fails_but_cache_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed fetch (after all retries) falls back to the stale cache and prints a warning."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: True)
    monkeypatch.setattr(tle_data.load, "days_old", lambda filename: 3.0)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests,
        "get",
        MagicMock(side_effect=requests.ConnectionError("503 Service Unavailable")),
    )
    monkeypatch.setattr(tle_data.load, "tle_file", lambda *a, **k: ["sentinel-satellite"])

    result = tle_data.load_satellites(FAKE_SATELLITES)

    assert result == {"TESTSAT": "sentinel-satellite"}
    warning = capsys.readouterr().out
    assert "TESTSAT" in warning
    assert "3.0 day" in warning


def test_reraises_with_actionable_message_when_fetch_fails_and_no_cache_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no cache to fall back to, the OSError names the satellite and the likely cause."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests,
        "get",
        MagicMock(
            side_effect=requests.ConnectionError("<urlopen error [Errno 11001] getaddrinfo failed>")
        ),
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
    """A transient failure is retried with exponential backoff until it succeeds."""
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
    assert sleep_calls == [1.0, 2.0]


def test_gives_up_after_max_retries_rather_than_retrying_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent failure makes exactly one initial attempt plus the configured retries."""
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

    assert call_count == tle_data.TLE_FETCH_MAX_RETRIES + 1
    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message
    assert "internet connection" in message


def test_fetch_passes_the_configured_timeout_to_requests_get(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each fetch passes TLE_FETCH_TIMEOUT_SECONDS to requests.get."""
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
    """requests exceptions surface as OSError, which the retry and fallback logic key on."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)

    for exc in [requests.Timeout("timed out"), requests.ConnectionError("refused")]:
        monkeypatch.setattr(tle_data.requests, "get", MagicMock(side_effect=exc))

        with pytest.raises(OSError):
            tle_data.load_satellites(FAKE_SATELLITES)


def test_http_error_status_is_checked_and_not_handed_to_the_tle_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real 404 response raises OSError via raise_for_status()."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)

    error_response = requests.Response()
    error_response.status_code = 404
    error_response._content = b"<html>Not Found</html>"
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=error_response))

    with pytest.raises(OSError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    assert "TESTSAT" in str(exc_info.value)


def test_raises_value_error_when_response_has_no_usable_tle_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty parse result (e.g. a decayed NORAD ID) raises ValueError, not IndexError."""
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
    """Same as above for a freshly fetched empty response."""
    monkeypatch.setattr(tle_data, "TLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=_fake_response(b"")))

    with pytest.raises(ValueError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES)

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "99999" in message


# source="mirror": same machinery as Celestrak, with a different URL and a shorter cache age.


def test_default_source_is_celestrak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An age between the two thresholds is fresh for Celestrak but stale for the mirror."""
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
    mock_get.assert_not_called()

    tle_data.load_satellites(FAKE_SATELLITES, source="mirror")
    mock_get.assert_called_once()


def test_mirror_source_error_message_points_at_the_refresh_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable mirror with no cache points at the refresh workflow, not the network."""
    monkeypatch.setattr(tle_data.load, "exists", lambda filename: False)
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests,
        "get",
        MagicMock(side_effect=requests.ConnectionError("404 Not Found")),
    )

    with pytest.raises(OSError) as exc_info:
        tle_data.load_satellites(FAKE_SATELLITES, source="mirror")

    message = str(exc_info.value)
    assert "TESTSAT" in message
    assert "refresh-tles.yml" in message or "Refresh TLE mirror" in message
    assert "check your internet connection" not in message.lower()


# is_older_than(): pure staleness logic.


def test_is_older_than_false_when_well_within_max_age() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    reference_time = now - tle_data.timedelta(hours=1)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is False


def test_is_older_than_true_when_well_past_max_age() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    reference_time = now - tle_data.timedelta(hours=48)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is True


def test_is_older_than_boundary_exactly_at_max_age_is_not_yet_stale() -> None:
    """The comparison is strict: exactly max_age old is not stale."""
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    reference_time = now - tle_data.timedelta(hours=24)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is False


def test_is_older_than_boundary_one_second_past_max_age_is_stale() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    reference_time = now - tle_data.timedelta(hours=24, seconds=1)

    assert tle_data.is_older_than(reference_time, tle_data.timedelta(hours=24), now=now) is True


def test_is_older_than_uses_the_real_current_time_when_now_not_given() -> None:
    long_ago = tle_data.datetime(2000, 1, 1, tzinfo=tle_data.UTC)

    assert tle_data.is_older_than(long_ago, tle_data.timedelta(days=1)) is True


# fetch_mirror_metadata(): best-effort, returns None instead of raising.


def test_fetch_mirror_metadata_returns_parsed_data_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "generated_at": "2026-09-28T09:00:00Z",
        "satellites": {
            "99999": {
                "name": "TESTSAT",
                "fetched_at": "2026-09-28T09:00:03Z",
                "tle_epoch": "2026-09-27T18:00:00Z",
                "source_url": "https://example/",
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
        tle_data.requests,
        "get",
        MagicMock(side_effect=requests.ConnectionError("refused")),
    )

    assert tle_data.fetch_mirror_metadata() is None


def test_fetch_mirror_metadata_returns_none_on_http_error_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tle-data branch is a 404, which must be treated as unavailable."""
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
    """Valid JSON without the expected keys counts as unavailable."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"unexpected": "shape"})
    monkeypatch.setattr(tle_data.requests, "get", MagicMock(return_value=response))

    assert tle_data.fetch_mirror_metadata() is None


# parse_iso_utc()


def test_parse_iso_utc_parses_a_real_metadata_timestamp() -> None:
    parsed = tle_data.parse_iso_utc("2026-09-28T09:00:03Z")

    assert parsed == tle_data.datetime(2026, 9, 28, 9, 0, 3, tzinfo=tle_data.UTC)


def test_parse_iso_utc_returns_none_for_garbage() -> None:
    assert tle_data.parse_iso_utc("not a timestamp") is None


# compute_staleness_warnings()


def _metadata_generated_at(timestamp: str) -> tle_data.TLEMirrorMetadata:
    return {"generated_at": timestamp, "satellites": {}}


def test_no_warnings_when_mirror_and_all_epochs_are_fresh() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    metadata = _metadata_generated_at("2026-09-28T11:00:00Z")  # 1h old
    epochs = {"ISS (ZARYA)": now - tle_data.timedelta(days=1)}  # 1 day old

    result = tle_data.compute_staleness_warnings(metadata, epochs, now=now)

    assert result == {"mirror_stale": False, "stale_satellite_names": []}


def test_mirror_stale_when_generated_at_older_than_warning_threshold() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    metadata = _metadata_generated_at("2026-09-25T12:00:00Z")  # 3 days old

    result = tle_data.compute_staleness_warnings(metadata, {}, now=now)

    assert result["mirror_stale"] is True


def test_mirror_stale_is_false_not_true_when_metadata_is_none() -> None:
    """Unknown provenance is not reported as stale."""
    result = tle_data.compute_staleness_warnings(None, {}, now=tle_data.datetime.now(tle_data.UTC))

    assert result["mirror_stale"] is False


def test_satellite_flagged_stale_when_its_epoch_is_old_but_others_are_not() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    epochs = {
        "ISS (ZARYA)": now - tle_data.timedelta(days=1),
        "OLDSAT": now - tle_data.timedelta(days=10),
    }

    result = tle_data.compute_staleness_warnings(
        _metadata_generated_at("2026-09-28T11:00:00Z"), epochs, now=now
    )

    assert result["stale_satellite_names"] == ["OLDSAT"]


def test_compute_staleness_warnings_boundary_exactly_at_threshold_is_not_stale() -> None:
    now = tle_data.datetime(2026, 9, 28, 12, 0, 0, tzinfo=tle_data.UTC)
    exactly_at_threshold = now - tle_data.timedelta(hours=tle_data.MIRROR_REFRESH_WARNING_HOURS)

    result = tle_data.compute_staleness_warnings(
        _metadata_generated_at(exactly_at_threshold.strftime("%Y-%m-%dT%H:%M:%SZ")),
        {},
        now=now,
    )

    assert result["mirror_stale"] is False
