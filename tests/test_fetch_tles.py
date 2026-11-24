"""Tests for satellite_pass_predictor.fetch_tles, the TLE mirror refresh script.

requests.get is mocked throughout; the frozen ISS TLE is used as realistic response bytes.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import requests

from satellite_pass_predictor import fetch_tles, tle_data

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_ISS_TLE_BYTES = (FIXTURES_DIR / "iss_tle.txt").read_bytes()
_ISS_NORAD_ID = 25544


def _fake_response(content: bytes, status_code: int = 200):
    """A real requests.Response, so raise_for_status() behaves as in production."""
    response = requests.Response()
    response.status_code = status_code
    response._content = content
    return response


# _validate_tle()


def test_validate_tle_accepts_a_real_valid_tle() -> None:
    raw, epoch = tle_data.validate_tle(_ISS_TLE_BYTES, _ISS_NORAD_ID)

    assert raw == _ISS_TLE_BYTES
    assert epoch.year == 2026


def test_validate_tle_rejects_an_html_error_page() -> None:
    html = b"<html><body><h1>500 Internal Server Error</h1></body></html>"

    with pytest.raises(ValueError, match="expected exactly 1 TLE"):
        tle_data.validate_tle(html, _ISS_NORAD_ID)


def test_validate_tle_rejects_an_empty_body() -> None:
    with pytest.raises(ValueError, match="expected exactly 1 TLE"):
        tle_data.validate_tle(b"", _ISS_NORAD_ID)


def test_validate_tle_rejects_a_mismatched_norad_id() -> None:
    """A valid TLE for a different satellite than requested is rejected."""
    wrong_norad_id = 99999

    with pytest.raises(ValueError, match="NORAD ID mismatch"):
        tle_data.validate_tle(_ISS_TLE_BYTES, wrong_norad_id)


# fetch_one(): never raises; failures are returned in the FetchOutcome.


def test_fetch_one_succeeds_for_a_valid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tle_data.requests,
        "get",
        lambda *a, **k: _fake_response(_ISS_TLE_BYTES),
    )

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert outcome.success
    assert outcome.error is None
    assert outcome.tle_text == _ISS_TLE_BYTES
    assert outcome.tle_epoch is not None
    assert outcome.norad_id == _ISS_NORAD_ID


def test_fetch_one_fails_on_a_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert not outcome.success
    assert outcome.tle_text is None
    assert "refused" in (outcome.error or "")


def test_fetch_one_fails_on_an_invalid_response_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad response is a validation failure, not a transient one, so it is fetched once."""
    call_count = 0

    def _get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _fake_response(b"<html>error</html>")

    monkeypatch.setattr(tle_data.requests, "get", _get)

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert not outcome.success
    assert call_count == 1
    assert "expected exactly 1 TLE" in (outcome.error or "")


def test_fetch_one_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    call_count = 0

    def _get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise requests.Timeout("timed out")
        return _fake_response(_ISS_TLE_BYTES)

    monkeypatch.setattr(tle_data.requests, "get", _get)

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert outcome.success
    assert call_count == 3


# load_existing_metadata()


def test_load_existing_metadata_returns_empty_structure_when_file_missing(
    tmp_path: Path,
) -> None:
    """The first run, before the branch has been published, has nothing to carry forward."""
    metadata = fetch_tles.load_existing_metadata(tmp_path)

    assert metadata == {"generated_at": None, "satellites": {}}


def test_load_existing_metadata_returns_empty_structure_for_malformed_json(
    tmp_path: Path,
) -> None:
    (tmp_path / "metadata.json").write_text("{not valid json", encoding="utf-8")

    metadata = fetch_tles.load_existing_metadata(tmp_path)

    assert metadata == {"generated_at": None, "satellites": {}}


def test_load_existing_metadata_reads_a_real_previously_published_file(
    tmp_path: Path,
) -> None:
    previous = {
        "generated_at": "2026-09-28T00:00:00Z",
        "satellites": {
            "25544": {
                "name": "ISS (ZARYA)",
                "fetched_at": "2026-09-28T00:00:00Z",
                "tle_epoch": "2026-09-27T04:10:50Z",
                "source_url": "https://example/",
            }
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(previous), encoding="utf-8")

    metadata = fetch_tles.load_existing_metadata(tmp_path)

    assert metadata == previous


# write_outputs(): a failed satellite keeps its previous file and metadata entry.


def test_write_outputs_writes_file_and_metadata_for_a_successful_outcome(
    tmp_path: Path,
) -> None:
    run_time = fetch_tles.datetime(2026, 9, 28, 12, 0, 0, tzinfo=fetch_tles.UTC)
    outcome = fetch_tles.FetchOutcome(
        name="ISS (ZARYA)",
        norad_id=_ISS_NORAD_ID,
        url="https://example/iss",
        success=True,
        tle_text=_ISS_TLE_BYTES,
        tle_epoch=fetch_tles.datetime(2026, 9, 27, 4, 10, 50, tzinfo=fetch_tles.UTC),
    )

    metadata = fetch_tles.write_outputs(
        tmp_path, [outcome], {"generated_at": None, "satellites": {}}, run_time
    )

    assert (tmp_path / "tle_25544.txt").read_bytes() == _ISS_TLE_BYTES
    assert metadata["generated_at"] == "2026-09-28T12:00:00Z"
    assert metadata["satellites"]["25544"] == {
        "name": "ISS (ZARYA)",
        "fetched_at": "2026-09-28T12:00:00Z",
        "tle_epoch": "2026-09-27T04:10:50Z",
        "source_url": "https://example/iss",
    }
    on_disk = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk == metadata


def test_write_outputs_preserves_existing_file_and_metadata_for_a_failed_satellite(
    tmp_path: Path,
) -> None:
    """A failed fetch leaves the previously published file and entry alone."""
    previous_tle_bytes = b"OLD-BEESAT-1-DATA\n1 ...\n2 ...\n"
    (tmp_path / "tle_35933.txt").write_bytes(previous_tle_bytes)
    previous_metadata: dict[str, Any] = {
        "generated_at": "2026-09-28T00:00:00Z",
        "satellites": {
            "35933": {
                "name": "BEESAT-1",
                "fetched_at": "2026-09-28T00:00:00Z",
                "tle_epoch": "2026-09-26T16:32:10Z",
                "source_url": "https://example/beesat",
            },
        },
    }
    run_time = fetch_tles.datetime(2026, 9, 28, 12, 0, 0, tzinfo=fetch_tles.UTC)
    failed_outcome = fetch_tles.FetchOutcome(
        name="BEESAT-1",
        norad_id=35933,
        url="https://example/beesat",
        success=False,
        error="cannot fetch: Connection timed out",
    )

    metadata = fetch_tles.write_outputs(tmp_path, [failed_outcome], previous_metadata, run_time)

    assert (tmp_path / "tle_35933.txt").read_bytes() == previous_tle_bytes
    assert metadata["satellites"]["35933"] == previous_metadata["satellites"]["35933"]
    # generated_at still advances: the run happened.
    assert metadata["generated_at"] == "2026-09-28T12:00:00Z"


def test_write_outputs_publishes_successes_alongside_preserved_failures(
    tmp_path: Path,
) -> None:
    (tmp_path / "tle_35933.txt").write_bytes(b"OLD-BEESAT-1-DATA\n")
    previous_metadata: dict[str, Any] = {
        "generated_at": "2026-09-28T00:00:00Z",
        "satellites": {
            "35933": {
                "name": "BEESAT-1",
                "fetched_at": "2026-09-28T00:00:00Z",
                "tle_epoch": "2026-09-26T16:32:10Z",
                "source_url": "https://example/beesat",
            },
        },
    }
    run_time = fetch_tles.datetime(2026, 9, 28, 12, 0, 0, tzinfo=fetch_tles.UTC)
    outcomes = [
        fetch_tles.FetchOutcome(
            name="ISS (ZARYA)",
            norad_id=_ISS_NORAD_ID,
            url="https://example/iss",
            success=True,
            tle_text=_ISS_TLE_BYTES,
            tle_epoch=fetch_tles.datetime(2026, 9, 27, 4, 10, 50, tzinfo=fetch_tles.UTC),
        ),
        fetch_tles.FetchOutcome(
            name="BEESAT-1",
            norad_id=35933,
            url="https://example/beesat",
            success=False,
            error="cannot fetch: Connection timed out",
        ),
    ]

    metadata = fetch_tles.write_outputs(tmp_path, outcomes, previous_metadata, run_time)

    assert (tmp_path / "tle_25544.txt").read_bytes() == _ISS_TLE_BYTES
    assert (tmp_path / "tle_35933.txt").read_bytes() == b"OLD-BEESAT-1-DATA\n"
    assert metadata["satellites"]["25544"]["fetched_at"] == "2026-09-28T12:00:00Z"
    assert metadata["satellites"]["35933"] == previous_metadata["satellites"]["35933"]


# main(): exit code is 1 only when a failed satellite has no recent TLE in the mirror.


def test_main_returns_zero_when_every_satellite_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(fetch_tles, "SATELLITES", {"ISS (ZARYA)": _ISS_NORAD_ID})
    monkeypatch.setattr(tle_data.requests, "get", lambda *a, **k: _fake_response(_ISS_TLE_BYTES))

    exit_code = fetch_tles.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "tle_25544.txt").exists()


def test_main_returns_one_when_a_never_fetched_satellite_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        fetch_tles,
        "SATELLITES",
        {"ISS (ZARYA)": _ISS_NORAD_ID, "BEESAT-1": 35933},
    )
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)

    def _get(url, **kwargs):
        if "25544" in url:
            return _fake_response(_ISS_TLE_BYTES)
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(tle_data.requests, "get", _get)

    exit_code = fetch_tles.main(["--output-dir", str(tmp_path)])

    assert exit_code == 1
    # Whatever succeeded is still published.
    assert (tmp_path / "tle_25544.txt").exists()


def _write_previous_metadata(output_dir: Path, norad_id: int, fetched_at: str) -> None:
    """A previously-published metadata.json with one satellite fetched at `fetched_at`."""
    metadata = {
        "generated_at": fetched_at,
        "satellites": {str(norad_id): {"name": "BEESAT-1", "fetched_at": fetched_at}},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _fail_beesat_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fetch_tles,
        "SATELLITES",
        {"ISS (ZARYA)": _ISS_NORAD_ID, "BEESAT-1": 35933},
    )
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)

    def _get(url, **kwargs):
        if "25544" in url:
            return _fake_response(_ISS_TLE_BYTES)
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(tle_data.requests, "get", _get)


def test_main_returns_zero_when_a_failed_satellite_still_has_a_fresh_tle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fail_beesat_only(monkeypatch)
    one_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_previous_metadata(tmp_path, 35933, one_hour_ago)

    exit_code = fetch_tles.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert "::warning::BEESAT-1" in capsys.readouterr().out


def test_main_returns_one_when_a_failed_satellite_has_gone_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fail_beesat_only(monkeypatch)
    two_days_ago = (datetime.now(UTC) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_previous_metadata(tmp_path, 35933, two_days_ago)

    exit_code = fetch_tles.main(["--output-dir", str(tmp_path)])

    assert exit_code == 1
    assert "::error::BEESAT-1" in capsys.readouterr().out


def test_main_respects_max_stale_hours(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fail_beesat_only(monkeypatch)
    three_hours_ago = (datetime.now(UTC) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_previous_metadata(tmp_path, 35933, three_hours_ago)

    exit_code = fetch_tles.main(["--output-dir", str(tmp_path), "--max-stale-hours", "2"])

    assert exit_code == 1


# is_stale()


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"satellites": {}}, True),
        ({"satellites": {"35933": {"name": "BEESAT-1"}}}, True),
        ({"satellites": {"35933": {"fetched_at": "not a date"}}}, True),
        ({"satellites": {"35933": {"fetched_at": "2026-11-22T12:00:00Z"}}}, False),
        ({"satellites": {"35933": {"fetched_at": "2026-11-21T11:59:59Z"}}}, True),
    ],
)
def test_is_stale(metadata: dict[str, Any], expected: bool) -> None:
    run_time = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)

    assert fetch_tles.is_stale(metadata, 35933, run_time, timedelta(hours=24)) is expected
