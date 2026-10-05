"""Tests for satellite_pass_predictor.visualization.

The antimeridian and table tests are pure logic; the figure tests render with the Agg backend
using the frozen ISS TLE fixture.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from skyfield.timelib import Timescale

from satellite_pass_predictor.config import SATELLITE_COLORS, THEME_BACKGROUND_COLOR
from satellite_pass_predictor.visibility import PassDict
from satellite_pass_predictor.visualization import (
    NOTE_CODE_LEGEND,
    _split_at_antimeridian,
    build_ground_tracks_figure,
    build_pass_note_codes,
    plot_ground_tracks,
    print_passes_table,
)


def test_no_crossing_leaves_arrays_unchanged() -> None:
    longitudes = np.array([10.0, 20.0, 30.0, 40.0])
    latitudes = np.array([1.0, 2.0, 3.0, 4.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert np.array_equal(out_lon, longitudes)
    assert np.array_equal(out_lat, latitudes)
    assert not np.any(np.isnan(out_lon))


def test_single_crossing_inserts_nan_at_the_jump() -> None:
    """A NaN is inserted between the samples either side of the crossing; the rest is intact."""
    longitudes = np.array([170.0, 175.0, 179.0, -179.0, -175.0, -170.0])
    latitudes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert len(out_lon) == len(longitudes) + 1
    assert len(out_lat) == len(latitudes) + 1

    assert np.array_equal(out_lon[:3], [170.0, 175.0, 179.0])
    assert np.array_equal(out_lat[:3], [10.0, 11.0, 12.0])
    assert np.isnan(out_lon[3])
    assert np.isnan(out_lat[3])
    assert np.array_equal(out_lon[4:], [-179.0, -175.0, -170.0])
    assert np.array_equal(out_lat[4:], [13.0, 14.0, 15.0])


def test_two_crossings_insert_two_nans() -> None:
    longitudes = np.array([175.0, -175.0, -170.0, 170.0, 175.0])
    latitudes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert len(out_lon) == len(longitudes) + 2
    nan_positions = np.where(np.isnan(out_lon))[0]
    assert len(nan_positions) == 2
    assert np.array_equal(nan_positions, np.where(np.isnan(out_lat))[0])


def test_exactly_180_degree_jump_is_not_treated_as_a_crossing() -> None:
    """Pins the strict `> 180` threshold."""
    longitudes = np.array([0.0, 180.0])
    latitudes = np.array([0.0, 0.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert np.array_equal(out_lon, longitudes)
    assert not np.any(np.isnan(out_lon))


def test_print_passes_table_with_no_passes_prints_a_clean_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A window with no passes is normal, so it prints one line instead of an empty table."""
    print_passes_table({"ISS (ZARYA)": [], "MICROSCOPE": []})

    output = capsys.readouterr().out
    assert "No passes" in output


def _fake_pass(
    ts: Timescale,
    *,
    start_truncated: bool = False,
    end_truncated: bool = False,
    max_elevation_truncated: bool = False,
    low_confidence: bool = False,
) -> PassDict:
    """A PassDict with only the four note flags varied and placeholder values elsewhere."""
    t = ts.utc(2026, 1, 1, 0, 0, 0)
    return {
        "start_time": t,
        "start_azimuth_deg": 0.0,
        "start_truncated": start_truncated,
        "end_time": t,
        "end_azimuth_deg": 0.0,
        "end_truncated": end_truncated,
        "max_elevation_deg": 10.0,
        "max_elevation_time": t,
        "max_elevation_truncated": max_elevation_truncated,
        "duration_minutes": 0.0,
        "low_confidence": low_confidence,
    }


def test_build_pass_note_codes_no_flags_set_returns_empty_string(ts: Timescale) -> None:
    assert build_pass_note_codes(_fake_pass(ts)) == ""


def test_build_pass_note_codes_each_flag_maps_to_its_own_code(ts: Timescale) -> None:
    assert build_pass_note_codes(_fake_pass(ts, start_truncated=True)) == "IP"
    assert build_pass_note_codes(_fake_pass(ts, end_truncated=True)) == "CE"
    assert build_pass_note_codes(_fake_pass(ts, max_elevation_truncated=True)) == "MH"
    assert build_pass_note_codes(_fake_pass(ts, low_confidence=True)) == "LC"


def test_build_pass_note_codes_joins_multiple_flags_in_a_fixed_order(ts: Timescale) -> None:
    pass_ = _fake_pass(ts, low_confidence=True, start_truncated=True, end_truncated=True)

    assert build_pass_note_codes(pass_) == "IP, CE, LC"


def test_note_code_legend_covers_every_code_build_pass_note_codes_can_emit(
    ts: Timescale,
) -> None:
    """The legend and the emitting function must not drift apart."""
    all_flags_pass = _fake_pass(
        ts,
        start_truncated=True,
        end_truncated=True,
        max_elevation_truncated=True,
        low_confidence=True,
    )
    emitted_codes = set(build_pass_note_codes(all_flags_pass).split(", "))
    legend_codes = {code for code, _ in NOTE_CODE_LEGEND}

    assert emitted_codes == legend_codes


def test_plot_ground_tracks_creates_output_directory_if_missing(
    tmp_path: Path, ts: Timescale, iss_satellite
) -> None:
    output_path = tmp_path / "does" / "not" / "exist" / "yet" / "tracks.png"
    assert not output_path.parent.exists()

    t0 = ts.utc(2026, 9, 21, 0, 0, 0)
    result = plot_ground_tracks(
        {"ISS (ZARYA)": iss_satellite},
        ts,
        start_time=t0,
        duration_hours=1,
        step_minutes=30,
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()


def test_build_ground_tracks_figure_returns_a_live_unclosed_figure(
    ts: Timescale, iss_satellite
) -> None:
    """Unlike plot_ground_tracks(), the figure is returned still open, with a line per satellite."""
    t0 = ts.utc(2026, 9, 21, 0, 0, 0)
    satellites = {"ISS (ZARYA)": iss_satellite}

    fig = build_ground_tracks_figure(
        satellites,
        ts,
        start_time=t0,
        duration_hours=1,
        step_minutes=30,
    )

    assert plt.fignum_exists(fig.number)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].lines) == len(satellites)
    plt.close(fig)


def test_each_satellites_line_uses_its_config_color(ts: Timescale, iss_satellite) -> None:
    t0 = ts.utc(2026, 9, 21, 0, 0, 0)
    satellites = {"ISS (ZARYA)": iss_satellite}

    fig = build_ground_tracks_figure(
        satellites,
        ts,
        start_time=t0,
        duration_hours=1,
        step_minutes=30,
    )

    line = fig.axes[0].lines[0]
    assert to_rgb(line.get_color()) == to_rgb(SATELLITE_COLORS["ISS (ZARYA)"])
    plt.close(fig)


def test_figure_and_axes_use_the_dark_theme_background(ts: Timescale, iss_satellite) -> None:
    t0 = ts.utc(2026, 9, 21, 0, 0, 0)
    satellites = {"ISS (ZARYA)": iss_satellite}

    fig = build_ground_tracks_figure(
        satellites,
        ts,
        start_time=t0,
        duration_hours=1,
        step_minutes=30,
    )

    assert to_rgb(fig.get_facecolor()) == to_rgb(THEME_BACKGROUND_COLOR)
    assert to_rgb(fig.axes[0].get_facecolor()) == to_rgb(THEME_BACKGROUND_COLOR)
    plt.close(fig)
