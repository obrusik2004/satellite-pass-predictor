"""Tests for satellite_pass_predictor.geo.find_antimeridian_crossings."""

import numpy as np

from satellite_pass_predictor.geo import find_antimeridian_crossings


def test_no_crossing_returns_empty() -> None:
    longitudes = np.array([10.0, 20.0, 30.0])
    assert len(find_antimeridian_crossings(longitudes)) == 0


def test_single_crossing_is_found_at_the_jump_index() -> None:
    longitudes = np.array([170.0, 179.0, -179.0, -170.0])
    assert list(find_antimeridian_crossings(longitudes)) == [1]  # jump is between index 1 and 2


def test_two_crossings_are_both_found() -> None:
    longitudes = np.array([175.0, -175.0, -170.0, 170.0, 175.0])
    assert list(find_antimeridian_crossings(longitudes)) == [0, 2]


def test_exactly_180_degree_jump_is_not_a_crossing() -> None:
    """Pins the strict `> 180.0` threshold."""
    longitudes = np.array([0.0, 180.0])
    assert len(find_antimeridian_crossings(longitudes)) == 0
