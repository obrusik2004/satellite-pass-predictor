"""
Tests for satellite_pass_predictor.visualization.

Split by what each group needs:

- test_*_crossing_*/test_exactly_180_* : PURE LOGIC. _split_at_antimeridian()
  is plain numpy array manipulation -- no TLE, no EarthSatellite, no
  Skyfield object of any kind.
- test_print_passes_table_* : PURE LOGIC. Constructs its own
  passes_by_satellite dicts directly, no TLE/EarthSatellite needed.
- test_plot_ground_tracks_* : FIXTURE-BASED. Uses the real `iss_satellite`
  fixture (frozen TLE, no live fetch -- see conftest.py) and actually
  renders a matplotlib figure via a non-interactive backend, since that's
  what's needed to exercise plot_ground_tracks() itself (as opposed to
  just its antimeridian-handling logic, tested separately above).
"""

from pathlib import Path

import numpy as np
import pytest
from skyfield.timelib import Timescale

from satellite_pass_predictor.visualization import (
    _split_at_antimeridian,
    plot_ground_tracks,
    print_passes_table,
)


def test_no_crossing_leaves_arrays_unchanged() -> None:
    """A longitude sequence that never approaches +-180 should pass
    through with no NaNs inserted and no values changed."""
    longitudes = np.array([10.0, 20.0, 30.0, 40.0])
    latitudes = np.array([1.0, 2.0, 3.0, 4.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert np.array_equal(out_lon, longitudes)
    assert np.array_equal(out_lat, latitudes)
    assert not np.any(np.isnan(out_lon))


def test_single_crossing_inserts_nan_at_the_jump() -> None:
    """
    Longitude jumps from +179 to -179 between index 2 and 3 -- a real
    ground track crossing the antimeridian, not an actual ~358 degree
    move. A NaN should be inserted between those two samples (so
    matplotlib breaks the line there instead of drawing a streak across
    the whole map), and every other value should be preserved in order.
    """
    longitudes = np.array([170.0, 175.0, 179.0, -179.0, -175.0, -170.0])
    latitudes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert len(out_lon) == len(longitudes) + 1
    assert len(out_lat) == len(latitudes) + 1

    # everything up to and including the last pre-crossing sample is untouched
    assert np.array_equal(out_lon[:3], [170.0, 175.0, 179.0])
    assert np.array_equal(out_lat[:3], [10.0, 11.0, 12.0])
    # the break itself
    assert np.isnan(out_lon[3])
    assert np.isnan(out_lat[3])
    # everything from the first post-crossing sample onward is untouched
    assert np.array_equal(out_lon[4:], [-179.0, -175.0, -170.0])
    assert np.array_equal(out_lat[4:], [13.0, 14.0, 15.0])


def test_two_crossings_insert_two_nans() -> None:
    """A track that wraps, comes back, and wraps again (plausible over a
    24h/multi-orbit window) should get a break at each crossing."""
    longitudes = np.array([175.0, -175.0, -170.0, 170.0, 175.0])
    latitudes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert len(out_lon) == len(longitudes) + 2
    nan_positions = np.where(np.isnan(out_lon))[0]
    assert len(nan_positions) == 2
    # latitude carries NaN at exactly the same positions as longitude
    assert np.array_equal(nan_positions, np.where(np.isnan(out_lat))[0])


def test_exactly_180_degree_jump_is_not_treated_as_a_crossing() -> None:
    """
    Locks in the function's actual threshold semantics (`> 180.0`, not
    `>= 180.0`): a jump of exactly 180 degrees is not split. This isn't
    a physically expected case (real subpoint longitudes won't usually
    land on an exact 180 degree step), but it pins down current,
    deliberate behavior at the boundary so a future change to the
    comparison operator doesn't silently pass unnoticed.
    """
    longitudes = np.array([0.0, 180.0])
    latitudes = np.array([0.0, 0.0])

    out_lon, out_lat = _split_at_antimeridian(longitudes, latitudes)

    assert np.array_equal(out_lon, longitudes)
    assert not np.any(np.isnan(out_lon))


def test_print_passes_table_with_no_passes_prints_a_clean_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Confirms already-correct behavior (not a fix -- verified during the
    robustness audit): a 24h window with zero detected passes for every
    satellite is a real, unremarkable possibility (unlucky timing/
    geometry), not an error. The empty-dict-of-empty-lists case should
    print a clear one-line message, not crash or print an empty/
    confusing table.
    """
    print_passes_table({"ISS (ZARYA)": [], "MICROSCOPE": []})

    output = capsys.readouterr().out
    assert "No passes" in output


def test_plot_ground_tracks_creates_output_directory_if_missing(
    tmp_path: Path, ts: Timescale, iss_satellite
) -> None:
    """
    Confirms already-correct behavior (not a fix -- verified during the
    robustness audit): a completely fresh clone has no output/ directory
    yet. plot_ground_tracks() should create whatever directory its
    output_path lives in rather than assuming it already exists.
    """
    output_path = tmp_path / "does" / "not" / "exist" / "yet" / "tracks.png"
    assert not output_path.parent.exists()

    t0 = ts.utc(2026, 9, 21, 0, 0, 0)
    result = plot_ground_tracks(
        {"ISS (ZARYA)": iss_satellite}, ts, start_time=t0,
        duration_hours=1, step_minutes=30, output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.exists()
