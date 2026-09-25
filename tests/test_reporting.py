"""
Tests for satellite_pass_predictor.reporting.

Lighter touch than the logic-heavy modules (propagation/visibility):
this is presentation code, so the tests check the properties that
actually matter if this broke -- a file gets created, the image is
truly embedded (not a link that could go missing), the table shows the
same data build_pass_rows() produces, and the empty-passes case doesn't
render something broken -- rather than exhaustively verifying HTML/CSS
output byte-for-byte.

PURE LOGIC / no Skyfield needed for most of these: generate_html_report()
only reads whatever bytes are at ground_track_png_path and base64-encodes
them, so a tiny fake file stands in for a real PNG. The one exception is
`generated_at`, which must be a real Skyfield Time (for .utc_strftime()),
so the `ts` fixture from conftest.py is used to build one.
"""

from pathlib import Path

from skyfield.timelib import Time, Timescale

from satellite_pass_predictor.reporting import generate_html_report
from satellite_pass_predictor.visibility import PassDict


def _fake_pass(t: Time) -> PassDict:
    """A minimal, valid-shaped PassDict for table-content tests -- the
    actual pass-detection logic is tested in test_visibility.py, this
    just needs *some* representative data to render. All three time
    fields reuse the same instant `t` -- irrelevant for what these
    tests check (the values/formatting rendered, not real pass
    geometry)."""
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
    """
    The whole point of this feature: the report must be one
    self-contained file, so the PNG has to be embedded as a data URI,
    not referenced by a relative path that could go missing if the
    report is moved or emailed on its own.
    """
    png_bytes = b"some-distinctive-fake-png-content"
    png_path = tmp_path / "fake_ground_tracks.png"
    png_path.write_bytes(png_bytes)
    output_path = tmp_path / "report.html"

    generate_html_report(
        str(png_path), {}, generated_at=ts.utc(2026, 1, 1), output_path=str(output_path)
    )

    document = output_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in document
    # not a relative/absolute filesystem link to the PNG
    assert png_path.name not in document

    import base64
    assert base64.b64encode(png_bytes).decode("ascii") in document


def test_report_contains_the_same_pass_data_as_the_text_table(
    tmp_path: Path, ts: Timescale
) -> None:
    """The HTML table should show the same underlying values
    print_passes_table() would print -- same satellite names, same
    formatted numbers -- just as <table> markup instead of text."""
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
    assert "8.0" in document   # duration


def test_report_with_no_passes_shows_a_clean_message_not_an_empty_table(
    tmp_path: Path, ts: Timescale
) -> None:
    png_path = tmp_path / "fake_ground_tracks.png"
    png_path.write_bytes(b"x")
    output_path = tmp_path / "report.html"

    generate_html_report(
        str(png_path), {"ISS (ZARYA)": []}, generated_at=ts.utc(2026, 1, 1),
        output_path=str(output_path),
    )

    document = output_path.read_text(encoding="utf-8")
    assert "No passes" in document
    assert "<table>" not in document
