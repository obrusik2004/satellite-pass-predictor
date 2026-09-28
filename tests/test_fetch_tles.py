"""
Tests for scripts.fetch_tles -- the standalone script
.github/workflows/refresh-tles.yml runs to fetch TLEs from Celestrak and
publish them to the tle-data branch mirror (see that script's module
docstring for why this pipeline exists at all).

No real network, no real Celestrak: requests.get() is mocked throughout,
using the same frozen ISS TLE fixture conftest.py's iss_satellite fixture
uses (real, valid TLE bytes, not a synthetic stand-in) for the "valid
response" cases, and deliberately-broken bytes (HTML, empty, a real TLE
for the wrong NORAD ID) for the rejection cases.

What's worth testing: _validate_tle()'s three rejection cases (not
exactly one TLE -- covers HTML error pages and empty bodies alike --
and a NORAD ID mismatch) alongside the one acceptance case;
fetch_one()'s translation of both fetch and validation failures into a
FetchOutcome rather than a raised exception; write_outputs()'s core
guarantee (a failed satellite's existing file and metadata entry are
left completely untouched, while generated_at still advances and
whatever succeeded still gets published); load_existing_metadata()'s
graceful handling of a missing or malformed file; and main()'s exit
code (0 only when every satellite succeeds).
"""

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from scripts import fetch_tles

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_ISS_TLE_BYTES = (FIXTURES_DIR / "iss_tle.txt").read_bytes()
_ISS_NORAD_ID = 25544


def _fake_response(content: bytes, status_code: int = 200):
    """A requests.Response stand-in -- raise_for_status() raises for a
    non-2xx status, exactly like a real Response would, rather than a
    mock that always succeeds regardless of the status it's given."""
    response = requests.Response()
    response.status_code = status_code
    response._content = content
    return response


# ---------------------------------------------------------------------------
# _validate_tle(): the core accept/reject logic, no network involved.
# ---------------------------------------------------------------------------


def test_validate_tle_accepts_a_real_valid_tle() -> None:
    raw, epoch = fetch_tles._validate_tle(_ISS_TLE_BYTES, _ISS_NORAD_ID)

    assert raw == _ISS_TLE_BYTES
    assert epoch.year == 2026  # sanity check: a real, parsed epoch, not a placeholder


def test_validate_tle_rejects_an_html_error_page() -> None:
    html = b"<html><body><h1>500 Internal Server Error</h1></body></html>"

    with pytest.raises(ValueError, match="expected exactly 1 TLE"):
        fetch_tles._validate_tle(html, _ISS_NORAD_ID)


def test_validate_tle_rejects_an_empty_body() -> None:
    with pytest.raises(ValueError, match="expected exactly 1 TLE"):
        fetch_tles._validate_tle(b"", _ISS_NORAD_ID)


def test_validate_tle_rejects_a_mismatched_norad_id() -> None:
    """A structurally valid TLE, but for a different satellite than the
    one actually requested -- e.g. Celestrak (or a mirror gone stale in
    a way that serves the wrong content) answering the wrong CATNR."""
    wrong_norad_id = 99999

    with pytest.raises(ValueError, match="NORAD ID mismatch"):
        fetch_tles._validate_tle(_ISS_TLE_BYTES, wrong_norad_id)


# ---------------------------------------------------------------------------
# fetch_one(): fetch + validate, translated into a FetchOutcome that
# never raises.
# ---------------------------------------------------------------------------


def test_fetch_one_succeeds_for_a_valid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fetch_tles.requests, "get",
        lambda *a, **k: _fake_response(_ISS_TLE_BYTES),
    )

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert outcome.success
    assert outcome.error is None
    assert outcome.tle_text == _ISS_TLE_BYTES
    assert outcome.tle_epoch is not None
    assert outcome.norad_id == _ISS_NORAD_ID


def test_fetch_one_fails_on_a_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_tles.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        fetch_tles.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert not outcome.success
    assert outcome.tle_text is None
    assert "refused" in (outcome.error or "")


def test_fetch_one_fails_on_an_invalid_response_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structurally bad response (HTML, empty, wrong satellite) is a
    validation problem, not a transient network one -- retrying an
    identical request won't fix it, so fetch_one() should call
    requests.get() exactly once for this case, not
    TLE_FETCH_MAX_RETRIES+1 times."""
    call_count = 0

    def _get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _fake_response(b"<html>error</html>")

    monkeypatch.setattr(fetch_tles.requests, "get", _get)

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert not outcome.success
    assert call_count == 1
    assert "expected exactly 1 TLE" in (outcome.error or "")


def test_fetch_one_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_tles.time, "sleep", lambda seconds: None)
    call_count = 0

    def _get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise requests.Timeout("timed out")
        return _fake_response(_ISS_TLE_BYTES)

    monkeypatch.setattr(fetch_tles.requests, "get", _get)

    outcome = fetch_tles.fetch_one("ISS (ZARYA)", _ISS_NORAD_ID)

    assert outcome.success
    assert call_count == 3


# ---------------------------------------------------------------------------
# load_existing_metadata(): graceful handling of the "seeded output
# directory" refresh-tles.yml prepares before running this script.
# ---------------------------------------------------------------------------


def test_load_existing_metadata_returns_empty_structure_when_file_missing(
    tmp_path: Path,
) -> None:
    """The very first run, before tle-data has ever been published --
    not an error, just nothing to carry forward."""
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
        "satellites": {"25544": {"name": "ISS (ZARYA)", "fetched_at": "2026-09-28T00:00:00Z",
                                  "tle_epoch": "2026-09-27T04:10:50Z", "source_url": "https://example/"}},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(previous), encoding="utf-8")

    metadata = fetch_tles.load_existing_metadata(tmp_path)

    assert metadata == previous


# ---------------------------------------------------------------------------
# write_outputs(): the actual publish step, and its core guarantee --
# a failed satellite's existing file/metadata entry are left alone.
# ---------------------------------------------------------------------------


def test_write_outputs_writes_file_and_metadata_for_a_successful_outcome(
    tmp_path: Path,
) -> None:
    run_time = fetch_tles.datetime(2026, 9, 28, 12, 0, 0, tzinfo=fetch_tles.timezone.utc)
    outcome = fetch_tles.FetchOutcome(
        name="ISS (ZARYA)", norad_id=_ISS_NORAD_ID, url="https://example/iss",
        success=True, tle_text=_ISS_TLE_BYTES,
        tle_epoch=fetch_tles.datetime(2026, 9, 27, 4, 10, 50, tzinfo=fetch_tles.timezone.utc),
    )

    metadata = fetch_tles.write_outputs(tmp_path, [outcome], {"generated_at": None, "satellites": {}}, run_time)

    assert (tmp_path / "tle_25544.txt").read_bytes() == _ISS_TLE_BYTES
    assert metadata["generated_at"] == "2026-09-28T12:00:00Z"
    assert metadata["satellites"]["25544"] == {
        "name": "ISS (ZARYA)",
        "fetched_at": "2026-09-28T12:00:00Z",
        "tle_epoch": "2026-09-27T04:10:50Z",
        "source_url": "https://example/iss",
    }
    # Written to disk, not just returned -- a later run reading this
    # back via load_existing_metadata() must see the same thing.
    on_disk = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk == metadata


def test_write_outputs_preserves_existing_file_and_metadata_for_a_failed_satellite(
    tmp_path: Path,
) -> None:
    """
    The central guarantee this whole publish step exists for: a
    satellite that fails *this* run must not lose its previously
    published TLE file or metadata entry -- the orphan-commit publish
    in refresh-tles.yml republishes exactly what's on disk here, so
    "leave it alone" is what makes a transient failure non-destructive.
    """
    previous_tle_bytes = b"OLD-BEESAT-1-DATA\n1 ...\n2 ...\n"
    (tmp_path / "tle_35933.txt").write_bytes(previous_tle_bytes)
    previous_metadata: dict[str, Any] = {
        "generated_at": "2026-09-28T00:00:00Z",
        "satellites": {
            "35933": {
                "name": "BEESAT-1", "fetched_at": "2026-09-28T00:00:00Z",
                "tle_epoch": "2026-09-26T16:32:10Z", "source_url": "https://example/beesat",
            },
        },
    }
    run_time = fetch_tles.datetime(2026, 9, 28, 12, 0, 0, tzinfo=fetch_tles.timezone.utc)
    failed_outcome = fetch_tles.FetchOutcome(
        name="BEESAT-1", norad_id=35933, url="https://example/beesat",
        success=False, error="cannot fetch: Connection timed out",
    )

    metadata = fetch_tles.write_outputs(tmp_path, [failed_outcome], previous_metadata, run_time)

    # File untouched.
    assert (tmp_path / "tle_35933.txt").read_bytes() == previous_tle_bytes
    # Metadata entry carried forward unchanged.
    assert metadata["satellites"]["35933"] == previous_metadata["satellites"]["35933"]
    # generated_at still advances -- the run happened, even though this
    # satellite's fetch within it failed.
    assert metadata["generated_at"] == "2026-09-28T12:00:00Z"


def test_write_outputs_publishes_successes_alongside_preserved_failures(
    tmp_path: Path,
) -> None:
    """A mixed run: one satellite refreshes successfully, another fails
    and keeps its old file -- both should be reflected correctly in the
    same metadata.json, not one clobbering the other."""
    (tmp_path / "tle_35933.txt").write_bytes(b"OLD-BEESAT-1-DATA\n")
    previous_metadata: dict[str, Any] = {
        "generated_at": "2026-09-28T00:00:00Z",
        "satellites": {
            "35933": {"name": "BEESAT-1", "fetched_at": "2026-09-28T00:00:00Z",
                       "tle_epoch": "2026-09-26T16:32:10Z", "source_url": "https://example/beesat"},
        },
    }
    run_time = fetch_tles.datetime(2026, 9, 28, 12, 0, 0, tzinfo=fetch_tles.timezone.utc)
    outcomes = [
        fetch_tles.FetchOutcome(
            name="ISS (ZARYA)", norad_id=_ISS_NORAD_ID, url="https://example/iss",
            success=True, tle_text=_ISS_TLE_BYTES,
            tle_epoch=fetch_tles.datetime(2026, 9, 27, 4, 10, 50, tzinfo=fetch_tles.timezone.utc),
        ),
        fetch_tles.FetchOutcome(
            name="BEESAT-1", norad_id=35933, url="https://example/beesat",
            success=False, error="cannot fetch: Connection timed out",
        ),
    ]

    metadata = fetch_tles.write_outputs(tmp_path, outcomes, previous_metadata, run_time)

    assert (tmp_path / "tle_25544.txt").read_bytes() == _ISS_TLE_BYTES
    assert (tmp_path / "tle_35933.txt").read_bytes() == b"OLD-BEESAT-1-DATA\n"
    assert metadata["satellites"]["25544"]["fetched_at"] == "2026-09-28T12:00:00Z"
    assert metadata["satellites"]["35933"] == previous_metadata["satellites"]["35933"]


# ---------------------------------------------------------------------------
# main(): end-to-end exit-code contract.
# ---------------------------------------------------------------------------


def test_main_returns_zero_when_every_satellite_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(fetch_tles, "SATELLITES", {"ISS (ZARYA)": _ISS_NORAD_ID})
    monkeypatch.setattr(fetch_tles.requests, "get", lambda *a, **k: _fake_response(_ISS_TLE_BYTES))

    exit_code = fetch_tles.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "tle_25544.txt").exists()


def test_main_returns_one_when_any_satellite_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        fetch_tles, "SATELLITES",
        {"ISS (ZARYA)": _ISS_NORAD_ID, "BEESAT-1": 35933},
    )
    monkeypatch.setattr(fetch_tles.time, "sleep", lambda seconds: None)

    def _get(url, **kwargs):
        if "25544" in url:
            return _fake_response(_ISS_TLE_BYTES)
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(fetch_tles.requests, "get", _get)

    exit_code = fetch_tles.main(["--output-dir", str(tmp_path)])

    assert exit_code == 1
    # The satellite that succeeded is still published even though the
    # run as a whole is reported as failed.
    assert (tmp_path / "tle_25544.txt").exists()
