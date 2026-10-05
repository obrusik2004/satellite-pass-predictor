"""Tests for satellite_pass_predictor.reporting.

Checks the properties that matter rather than the HTML byte for byte: a file is written, the
image is embedded, the table matches build_pass_rows(), and an empty pass list renders cleanly.
"""

from pathlib import Path

from skyfield.timelib import Time, Timescale

from satellite_pass_predictor.reporting import generate_html_report
from satellite_pass_predictor.visibility import PassDict


def _fake_pass(t: Time) -> PassDict:
    """A PassDict with representative values; all times reuse the instant `t`."""
    return {
        "start_time": t,
        "start_azimuth_deg": 10.0,
        "start_truncated": False,
        "end_time": t,
        "end_azimuth_deg": 200.0,
        "end_truncated": False,
        "max_elevation_deg": 45.0,
        "max_elevation_time": t,
        "max_elevation_truncated": False,
        "duration_minutes": 8.0,
        "low_confidence": False,
    }


def test_generate_html_report_creates_file_and_returns_its_path(
    tmp_path: Path, ts: Timescale
) -> None:
    png_path = tmp_path / "fake_ground_tracks.png"
    png_path.write_bytes(b"not-a-real-png-just-some-bytes")
    output_path = tmp_path / "report.html"

    result = generate_html_report(
        str(png_path), {}, generated_at=ts.utc(2026, 1, 1), output_path=str(output_path)
    )

    assert result == str(output_path)
    assert output_path.exists()


def test_generate_html_report_creates_output_directory_if_missing(
    tmp_path: Path, ts: Timescale
) -> None:
    png_path = tmp_path / "fake_ground_tracks.png"
    png_path.write_bytes(b"not-a-real-png-just-some-bytes")
    output_path = tmp_path / "does" / "not" / "exist" / "yet" / "report.html"
    assert not output_path.parent.exists()

    generate_html_report(
        str(png_path), {}, generated_at=ts.utc(2026, 1, 1), output_path=str(output_path)
    )

    assert output_path.exists()


def test_image_is_embedded_as_base64_not_linked(tmp_path: Path, ts: Timescale) -> None:
    """The report must be one self-contained file, so the PNG is a data URI, not a path."""
    png_bytes = b"some-distinctive-fake-png-content"
    png_path = tmp_path / "fake_ground_tracks.png"
    png_path.write_bytes(png_bytes)
    output_path = tmp_path / "report.html"

    generate_html_report(
        str(png_path), {}, generated_at=ts.utc(2026, 1, 1), output_path=str(output_path)
    )

    document = output_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in document
    assert png_path.name not in document

    import base64

    assert base64.b64encode(png_bytes).decode("ascii") in document


def test_report_contains_the_same_pass_data_as_the_text_table(
    tmp_path: Path, ts: Timescale
) -> None:
    png_path = tmp_path / "fake_ground_tracks.png"
    png_path.write_bytes(b"x")
    output_path = tmp_path / "report.html"

    t = ts.utc(2026, 1, 1, 0, 0, 0)
    passes_by_satellite = {"ISS (ZARYA)": [_fake_pass(t)]}

    generate_html_report(
        str(png_path), passes_by_satellite, generated_at=t, output_path=str(output_path)
    )

    document = output_path.read_text(encoding="utf-8")
    assert "<table>" in document
    assert "ISS (ZARYA)" in document
    assert "10.0" in document  # start azimuth
    assert "45.0" in document  # max elevation
    assert "8.0" in document  # duration


def test_report_with_no_passes_shows_a_clean_message_not_an_empty_table(
    tmp_path: Path, ts: Timescale
) -> None:
    png_path = tmp_path / "fake_ground_tracks.png"
    png_path.write_bytes(b"x")
    output_path = tmp_path / "report.html"

    generate_html_report(
        str(png_path),
        {"ISS (ZARYA)": []},
        generated_at=ts.utc(2026, 1, 1),
        output_path=str(output_path),
    )

    document = output_path.read_text(encoding="utf-8")
    assert "No passes" in document
    assert "<table>" not in document
