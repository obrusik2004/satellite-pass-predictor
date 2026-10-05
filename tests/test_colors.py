"""Tests for satellite_pass_predictor.colors.hex_to_rgb."""

from satellite_pass_predictor.colors import hex_to_rgb


def test_hex_to_rgb_converts_known_values() -> None:
    assert hex_to_rgb("#FFFFFF") == [255, 255, 255]
    assert hex_to_rgb("#000000") == [0, 0, 0]
    assert hex_to_rgb("#F1666A") == [241, 102, 106]


def test_hex_to_rgb_works_without_the_leading_hash() -> None:
    assert hex_to_rgb("6DCFF6") == hex_to_rgb("#6DCFF6")
