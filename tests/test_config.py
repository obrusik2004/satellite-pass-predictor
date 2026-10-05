"""Tests for the satellite -> color mapping in satellite_pass_predictor.config."""

from satellite_pass_predictor.config import SATELLITE_COLORS, SATELLITES


def test_every_tracked_satellite_has_a_color() -> None:
    assert set(SATELLITE_COLORS.keys()) == set(SATELLITES.keys())


def test_every_satellite_color_is_unique() -> None:
    colors = list(SATELLITE_COLORS.values())
    assert len(colors) == len(set(colors))


def test_every_satellite_color_is_a_well_formed_hex_string() -> None:
    for name, color in SATELLITE_COLORS.items():
        assert color.startswith("#") and len(color) == 7, f"{name}: {color!r}"
        int(color[1:], 16)  # raises ValueError if not valid hex
