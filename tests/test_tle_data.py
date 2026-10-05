"""Tests for satellite_pass_predictor.tle_data.

Caching tests use a real directory under tmp_path. requests.get and time.sleep are mocked, so
nothing touches the network and no test waits out a real retry delay.
"""

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from platformdirs import user_cache_dir
from skyfield.api import EarthSatellite

from satellite_pass_predictor import tle_data

FAKE_SATELLITES = {"TESTSAT": 25544}

# Real TLE text, so the fetch -> validate -> cache file -> parse pipeline runs end to end.
_FAKE_TLE_BYTES = (
    b"ISS (ZARYA)\n"
    b"1 25544U 98067A   26263.78762384  .00007766  00000+0  14793-3 0  9996\n"
    b"2 25544  51.6308 186.9475 0004819 163.3027 196.8121 15.49200337586572\n"
)
_OTHER_SATELLITE_TLE_BYTES = _FAKE_TLE_BYTES.replace(b"1 25544U", b"1 99999U").replace(
    b"2 25544 ", b"2 99999 "
)

# Shortly after the fixture TLE's epoch (2026-09-20 18:54 UTC).
FIXED_NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)
_EPOCH = datetime(2026, 9, 20, 18, 54, 0, tzinfo=UTC)


def _fake_response(content: bytes = _FAKE_TLE_BYTES) -> MagicMock:
    """A successful requests.Response stand-in with the given body."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.content = content
    return response


def _http_response(status_code: int) -> requests.Response:
    """A real response with an error status, so raise_for_status() behaves as in production."""
    response = requests.Response()
    response.status_code = status_code
    response._content = b"error"
    return response


def _seed_cache(cache_dir: Path, content: bytes = _FAKE_TLE_BYTES, age_days: float = 0.0) -> Path:
    path = cache_dir / "tle_25544.txt"
    path.write_bytes(content)
    mtime = time.time() - age_days * 86400
    os.utime(path, (mtime, mtime))
    return path


def _load(cache_dir: Path, satellites=FAKE_SATELLITES, **kwargs) -> tle_data.TLELoadResult:
    return tle_data.load_satellites(satellites, cache_dir=cache_dir, now=FIXED_NOW, **kwargs)


def _patch_get(monkeypatch: pytest.MonkeyPatch, get) -> None:
    monkeypatch.setattr(tle_data.requests, "get", get)


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def _iss(ts, line2: str | None = None) -> EarthSatellite:
    name, line1, original_line2 = _FAKE_TLE_BYTES.decode().splitlines()
    return EarthSatellite(line1, line2 or original_line2, name, ts)


# Caching and fetching


def test_uses_fresh_cache_without_fetching(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _seed_cache(tmp_path, age_days=0.1)
    mock_get = MagicMock(side_effect=AssertionError("should not fetch when cache is fresh"))
    _patch_get(monkeypatch, mock_get)

    result = _load(tmp_path)

    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    assert result.failures == {}
    mock_get.assert_not_called()


def test_refetches_when_cache_is_older_than_max_age(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _seed_cache(tmp_path, age_days=5.0)
    mock_get = MagicMock(return_value=_fake_response())
    _patch_get(monkeypatch, mock_get)

    result = _load(tmp_path)

    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    args, kwargs = mock_get.call_args
    assert args[0] == tle_data.CELESTRAK_URL.format(norad_id=25544)
    assert kwargs["timeout"] == tle_data.TLE_FETCH_TIMEOUT_SECONDS
    assert time.time() - path.stat().st_mtime < 60  # the cache was rewritten


def test_fetches_and_writes_the_cache_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response()))

    result = _load(tmp_path)

    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    assert (tmp_path / "tle_25544.txt").read_bytes() == _FAKE_TLE_BYTES


def test_creates_a_missing_cache_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "does" / "not" / "exist"
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response()))

    _load(cache_dir)

    assert (cache_dir / "tle_25544.txt").exists()


@pytest.mark.parametrize(
    "bad_body",
    [
        b"No GP data found",
        b"<html><body>500 Internal Server Error</body></html>",
        b"",
        _OTHER_SATELLITE_TLE_BYTES,
    ],
    ids=["no-gp-data", "html", "empty", "wrong-norad-id"],
)
def test_invalid_200_response_keeps_the_good_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_body: bytes
) -> None:
    """Regression: an invalid 200 response must not overwrite a good (stale) cache."""
    path = _seed_cache(tmp_path, age_days=5.0)
    before = path.stat().st_mtime
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response(bad_body)))

    result = _load(tmp_path)

    assert path.read_bytes() == _FAKE_TLE_BYTES
    assert path.stat().st_mtime == before
    assert result.failures == {}
    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    assert "refresh failed" in result.warnings["TESTSAT"]
    assert list(tmp_path.iterdir()) == [path]  # no leftover temp files


def test_invalid_response_with_no_cache_is_a_failure_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response(b"No GP data found")))

    result = _load(tmp_path)

    assert result.satellites == {}
    assert "invalid TLE" in result.failures["TESTSAT"].reason
    assert list(tmp_path.iterdir()) == []


def test_invalid_cached_file_is_ignored_and_refetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache poisoned earlier is never trusted as fresh."""
    path = _seed_cache(tmp_path, content=b"No GP data found", age_days=0.0)
    mock_get = MagicMock(return_value=_fake_response())
    _patch_get(monkeypatch, mock_get)

    result = _load(tmp_path)

    mock_get.assert_called_once()
    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    assert path.read_bytes() == _FAKE_TLE_BYTES


def test_cache_write_is_atomic_and_a_write_failure_does_not_lose_the_satellite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _seed_cache(tmp_path, age_days=5.0)
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response()))

    def _failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(tle_data.os, "replace", _failing_replace)

    with caplog.at_level(logging.WARNING):
        result = _load(tmp_path)

    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    assert path.read_bytes() == _FAKE_TLE_BYTES
    assert list(tmp_path.iterdir()) == [path]  # the temp file was cleaned up
    assert "Could not cache" in caplog.text


def test_falls_back_to_a_stale_cache_when_the_fetch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_cache(tmp_path, age_days=3.0)
    _no_sleep(monkeypatch)
    _patch_get(monkeypatch, MagicMock(side_effect=requests.ConnectionError("503 unavailable")))

    with caplog.at_level(logging.WARNING):
        result = _load(tmp_path)

    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    assert result.failures == {}
    assert "3.0 day" in result.warnings["TESTSAT"]
    assert "refresh failed" in caplog.text


def test_celestrak_failure_without_cache_names_the_source_and_the_likely_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _no_sleep(monkeypatch)
    _patch_get(monkeypatch, MagicMock(side_effect=requests.ConnectionError("getaddrinfo failed")))

    result = _load(tmp_path)

    failure = result.failures["TESTSAT"]
    assert (failure.name, failure.norad_id) == ("TESTSAT", 25544)
    assert "Celestrak" in failure.reason
    assert "internet connection" in failure.reason


def test_mirror_failure_without_cache_names_the_mirror_and_the_refresh_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _no_sleep(monkeypatch)
    _patch_get(monkeypatch, MagicMock(return_value=_http_response(404)))

    result = _load(tmp_path, source="mirror")

    reason = result.failures["TESTSAT"].reason
    assert "GitHub mirror" in reason
    assert "Celestrak" not in reason
    assert "refresh-tles.yml" in reason
    assert "internet connection" not in reason.lower()


def test_invalid_response_names_the_actual_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response(b"")))

    assert "Celestrak" in _load(tmp_path).failures["TESTSAT"].reason
    assert "GitHub mirror" in _load(tmp_path, source="mirror").failures["TESTSAT"].reason


# Per-satellite results


def test_one_failing_satellite_does_not_affect_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response()))

    result = _load(tmp_path, satellites={"GOOD": 25544, "BOGUS": 99999})

    assert list(result.satellites) == ["GOOD"]
    assert list(result.failures) == ["BOGUS"]
    assert "NORAD ID mismatch" in result.failures["BOGUS"].reason


def test_an_unexpected_error_is_reported_for_that_satellite_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response()))
    real_load_one = tle_data._load_one

    def _flaky(norad_id, *args, **kwargs):
        if norad_id == 99999:
            raise RuntimeError("boom")
        return real_load_one(norad_id, *args, **kwargs)

    monkeypatch.setattr(tle_data, "_load_one", _flaky)

    result = _load(tmp_path, satellites={"GOOD": 25544, "BROKEN": 99999})

    assert list(result.satellites) == ["GOOD"]
    assert "RuntimeError: boom" in result.failures["BROKEN"].reason


def test_default_satellites_are_the_configured_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tle_data, "SATELLITES", {"ISS (ZARYA)": 25544})
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response()))

    result = tle_data.load_satellites(cache_dir=tmp_path, now=FIXED_NOW)

    assert list(result.satellites) == ["ISS (ZARYA)"]


def test_default_source_is_celestrak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_get = MagicMock(return_value=_fake_response())
    _patch_get(monkeypatch, mock_get)

    _load(tmp_path)

    args, _ = mock_get.call_args
    assert args[0] == tle_data.CELESTRAK_URL.format(norad_id=25544)


def test_mirror_source_fetches_from_the_mirror_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_get = MagicMock(return_value=_fake_response())
    _patch_get(monkeypatch, mock_get)

    result = _load(tmp_path, source="mirror")

    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    args, _ = mock_get.call_args
    assert args[0] == tle_data.TLE_MIRROR_URL.format(norad_id=25544)


def test_mirror_source_uses_its_own_much_shorter_cache_age(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An age between the two thresholds is fresh for Celestrak but stale for the mirror."""
    between_the_two_thresholds = (
        tle_data.TLE_MIRROR_CACHE_AGE_HOURS / 24.0 + tle_data.MAX_TLE_AGE_DAYS
    ) / 2
    _seed_cache(tmp_path, age_days=between_the_two_thresholds)
    mock_get = MagicMock(return_value=_fake_response())
    _patch_get(monkeypatch, mock_get)

    _load(tmp_path, source="celestrak")
    mock_get.assert_not_called()

    _load(tmp_path, source="mirror")
    mock_get.assert_called_once()


# Retry policy


def test_retries_a_flaky_fetch_with_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _no_sleep(monkeypatch)
    call_count = 0

    def _get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # fails twice, succeeds on the 3rd attempt
            raise requests.Timeout("Connection timed out")
        return _fake_response()

    _patch_get(monkeypatch, _get)

    assert tle_data.fetch_with_retries("https://example/tle") == _FAKE_TLE_BYTES
    assert call_count == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries_rather_than_retrying_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _no_sleep(monkeypatch)
    mock_get = MagicMock(side_effect=requests.Timeout("Connection timed out"))
    _patch_get(monkeypatch, mock_get)

    with pytest.raises(OSError):
        tle_data.fetch_with_retries("https://example/tle")

    assert mock_get.call_count == tle_data.TLE_FETCH_MAX_RETRIES + 1
    assert len(sleeps) == tle_data.TLE_FETCH_MAX_RETRIES


@pytest.mark.parametrize("status_code", [500, 503, 429])
def test_transient_http_statuses_are_retried(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    sleeps = _no_sleep(monkeypatch)
    mock_get = MagicMock(side_effect=[_http_response(status_code), _fake_response()])
    _patch_get(monkeypatch, mock_get)

    assert tle_data.fetch_with_retries("https://example/tle") == _FAKE_TLE_BYTES
    assert mock_get.call_count == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize("status_code", [404, 403, 400])
def test_permanent_http_statuses_fail_immediately_without_backoff(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    sleeps = _no_sleep(monkeypatch)
    mock_get = MagicMock(return_value=_http_response(status_code))
    _patch_get(monkeypatch, mock_get)

    with pytest.raises(OSError, match=str(status_code)):
        tle_data.fetch_with_retries("https://example/tle")

    assert mock_get.call_count == 1
    assert sleeps == []


def test_connection_errors_and_timeouts_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    for exc in [requests.Timeout("timed out"), requests.ConnectionError("refused")]:
        mock_get = MagicMock(side_effect=exc)
        _patch_get(monkeypatch, mock_get)

        with pytest.raises(OSError):
            tle_data.fetch_with_retries("https://example/tle")

        assert mock_get.call_count == tle_data.TLE_FETCH_MAX_RETRIES + 1


def test_fetch_passes_the_configured_timeout_and_a_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_get = MagicMock(return_value=_fake_response())
    _patch_get(monkeypatch, mock_get)

    tle_data.fetch_with_retries("https://example/tle")

    _, kwargs = mock_get.call_args
    assert kwargs["timeout"] == tle_data.TLE_FETCH_TIMEOUT_SECONDS
    assert kwargs["headers"]["User-Agent"] == tle_data.USER_AGENT


# TLE validity: age limit and SGP4 propagation


def test_a_healthy_tle_passes_the_usability_check(ts) -> None:
    tle_data.check_tle_usable(_iss(ts), FIXED_NOW, ts)


def test_an_old_epoch_loads_with_a_logged_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_cache(tmp_path)
    now = _EPOCH + timedelta(days=tle_data.TLE_EPOCH_WARNING_DAYS + 2)

    with caplog.at_level(logging.WARNING):
        result = tle_data.load_satellites(FAKE_SATELLITES, cache_dir=tmp_path, now=now)

    assert "TESTSAT" in result.satellites
    assert "days old" in caplog.text


def test_an_epoch_within_the_warning_threshold_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_cache(tmp_path)

    with caplog.at_level(logging.WARNING):
        _load(tmp_path)

    assert "days old" not in caplog.text


def test_an_epoch_older_than_the_limit_is_refused_per_satellite(tmp_path: Path) -> None:
    _seed_cache(tmp_path)
    now = _EPOCH + timedelta(days=tle_data.TLE_EPOCH_MAX_DAYS + 5)

    result = tle_data.load_satellites(FAKE_SATELLITES, cache_dir=tmp_path, now=now)

    assert result.satellites == {}
    assert "limit" in result.failures["TESTSAT"].reason


def test_check_tle_usable_refuses_an_epoch_older_than_the_limit(ts) -> None:
    now = _EPOCH + timedelta(days=tle_data.TLE_EPOCH_MAX_DAYS + 1)

    with pytest.raises(tle_data.TLELoadError, match="days old"):
        tle_data.check_tle_usable(_iss(ts), now, ts)


def test_sgp4_propagation_errors_are_a_failure_not_silent_zero_passes(ts, tmp_path: Path) -> None:
    """An orbit with its perigee underground reports an SGP4 error in `position.message`."""
    decayed_line2 = "2 25544  51.6308 186.9475 9000000 163.3027 196.8121 15.49200337586572"

    with pytest.raises(tle_data.TLELoadError, match="SGP4"):
        tle_data.check_tle_usable(_iss(ts, decayed_line2), FIXED_NOW, ts)

    _seed_cache(tmp_path, content=_FAKE_TLE_BYTES.replace(b"0004819", b"9000000"))
    result = _load(tmp_path)
    assert "SGP4" in result.failures["TESTSAT"].reason


def test_nan_positions_are_a_failure(ts) -> None:
    class _FakePosition:
        message = [None, None, None]
        position = MagicMock(km=[[float("nan")] * 3] * 3)

    sat = MagicMock()
    sat.epoch.utc_datetime.return_value = _EPOCH
    sat.at.return_value = _FakePosition()

    with pytest.raises(tle_data.TLELoadError, match="NaN"):
        tle_data.check_tle_usable(sat, FIXED_NOW, ts)


# Cache directory


def test_default_cache_dir_honors_the_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(tle_data.CACHE_DIR_ENV_VAR, str(tmp_path / "custom"))

    assert tle_data.default_cache_dir() == tmp_path / "custom"


def test_default_cache_dir_is_per_user_and_independent_of_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(tle_data.CACHE_DIR_ENV_VAR, raising=False)
    expected = Path(user_cache_dir("satellite-pass-predictor", appauthor=False))

    monkeypatch.chdir(tmp_path)

    assert tle_data.default_cache_dir() == expected
    assert tmp_path not in expected.parents


def test_an_uncreatable_cache_directory_falls_back_to_a_temporary_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory")
    monkeypatch.setattr(tle_data.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    _patch_get(monkeypatch, MagicMock(return_value=_fake_response()))

    result = _load(blocker / "cache")

    assert result.satellites["TESTSAT"].name == "ISS (ZARYA)"
    assert (tmp_path / "tmp" / "satellite-pass-predictor" / "tle_25544.txt").exists()


# is_older_than(): pure staleness logic.


def test_is_older_than_false_when_well_within_max_age() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    reference_time = now - timedelta(hours=1)

    assert tle_data.is_older_than(reference_time, timedelta(hours=24), now=now) is False


def test_is_older_than_true_when_well_past_max_age() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    reference_time = now - timedelta(hours=48)

    assert tle_data.is_older_than(reference_time, timedelta(hours=24), now=now) is True


def test_is_older_than_boundary_exactly_at_max_age_is_not_yet_stale() -> None:
    """The comparison is strict: exactly max_age old is not stale."""
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    reference_time = now - timedelta(hours=24)

    assert tle_data.is_older_than(reference_time, timedelta(hours=24), now=now) is False


def test_is_older_than_boundary_one_second_past_max_age_is_stale() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    reference_time = now - timedelta(hours=24, seconds=1)

    assert tle_data.is_older_than(reference_time, timedelta(hours=24), now=now) is True


def test_is_older_than_uses_the_real_current_time_when_now_not_given() -> None:
    long_ago = datetime(2000, 1, 1, tzinfo=UTC)

    assert tle_data.is_older_than(long_ago, timedelta(days=1)) is True


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

    assert parsed == datetime(2026, 9, 28, 9, 0, 3, tzinfo=UTC)


def test_parse_iso_utc_returns_none_for_garbage() -> None:
    assert tle_data.parse_iso_utc("not a timestamp") is None


# compute_staleness_warnings()


def _metadata_generated_at(timestamp: str) -> tle_data.TLEMirrorMetadata:
    return {"generated_at": timestamp, "satellites": {}}


def test_no_warnings_when_mirror_and_all_epochs_are_fresh() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    metadata = _metadata_generated_at("2026-09-28T11:00:00Z")  # 1h old
    epochs = {"ISS (ZARYA)": now - timedelta(days=1)}  # 1 day old

    result = tle_data.compute_staleness_warnings(metadata, epochs, now=now)

    assert result == {"mirror_stale": False, "stale_satellite_names": []}


def test_mirror_stale_when_generated_at_older_than_warning_threshold() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    metadata = _metadata_generated_at("2026-09-25T12:00:00Z")  # 3 days old

    result = tle_data.compute_staleness_warnings(metadata, {}, now=now)

    assert result["mirror_stale"] is True


def test_mirror_stale_is_false_not_true_when_metadata_is_none() -> None:
    """Unknown provenance is not reported as stale."""
    result = tle_data.compute_staleness_warnings(None, {}, now=datetime.now(UTC))

    assert result["mirror_stale"] is False


def test_satellite_flagged_stale_when_its_epoch_is_old_but_others_are_not() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    epochs = {
        "ISS (ZARYA)": now - timedelta(days=1),
        "OLDSAT": now - timedelta(days=10),
    }

    result = tle_data.compute_staleness_warnings(
        _metadata_generated_at("2026-09-28T11:00:00Z"), epochs, now=now
    )

    assert result["stale_satellite_names"] == ["OLDSAT"]


def test_compute_staleness_warnings_boundary_exactly_at_threshold_is_not_stale() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    exactly_at_threshold = now - timedelta(hours=tle_data.MIRROR_REFRESH_WARNING_HOURS)

    result = tle_data.compute_staleness_warnings(
        _metadata_generated_at(exactly_at_threshold.strftime("%Y-%m-%dT%H:%M:%SZ")),
        {},
        now=now,
    )

    assert result["mirror_stale"] is False
