"""Tests for satellite_pass_predictor.cli.

The real loader runs against a mocked network and a tmp_path cache, with a fixed clock so the
frozen TLE is never "too old".
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from satellite_pass_predictor import cli, tle_data

_ISS_TLE_BYTES = (Path(__file__).parent / "fixtures" / "iss_tle.txt").read_bytes()
_FIXED_NOW = datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC)


def _ok_response() -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.content = _ISS_TLE_BYTES
    return response


@pytest.fixture
def isolated_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Use a tmp cache directory and a fixed clock; returns the output directory."""
    monkeypatch.setenv(tle_data.CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    real_load = tle_data.load_satellites
    monkeypatch.setattr(cli, "load_satellites", lambda **kw: real_load(now=_FIXED_NOW, **kw))
    return tmp_path / "out"


def test_a_failing_satellite_is_logged_and_the_run_continues(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only the ISS answers with a matching TLE; the other satellites fail and are skipped."""
    monkeypatch.setattr(tle_data.requests, "get", lambda *a, **k: _ok_response())

    with caplog.at_level(logging.WARNING):
        exit_code = cli.main(["--output-dir", str(isolated_cli)])

    assert exit_code == 0
    for skipped in ("SWISSCUBE", "BEESAT-1", "MICROSCOPE"):
        assert f"Could not load {skipped}" in caplog.text
    assert "Could not load ISS" not in caplog.text
    assert (isolated_cli / "ground_tracks.png").exists()
    assert (isolated_cli / "report.html").exists()


def test_the_run_fails_cleanly_when_no_satellite_loads(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(tle_data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        tle_data.requests, "get", MagicMock(side_effect=requests.ConnectionError("down"))
    )

    with caplog.at_level(logging.WARNING):
        exit_code = cli.main(["--output-dir", str(isolated_cli)])

    assert exit_code == 1
    assert "No satellites could be loaded" in caplog.text
    assert not isolated_cli.exists()
