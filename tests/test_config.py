"""Tests for satellite_pass_predictor.config: the color mapping and the Streamlit theme sync."""

import tomllib
from pathlib import Path
from typing import Any

from satellite_pass_predictor import config
from satellite_pass_predictor.config import SATELLITE_COLORS, SATELLITES

STREAMLIT_CONFIG = Path(__file__).parent.parent / ".streamlit" / "config.toml"


def test_every_tracked_satellite_has_a_color() -> None:
    assert set(SATELLITE_COLORS.keys()) == set(SATELLITES.keys())


def test_every_satellite_color_is_unique() -> None:
    colors = list(SATELLITE_COLORS.values())
    assert len(colors) == len(set(colors))


def test_every_satellite_color_is_a_well_formed_hex_string() -> None:
    for name, color in SATELLITE_COLORS.items():
        assert color.startswith("#") and len(color) == 7, f"{name}: {color!r}"
        int(color[1:], 16)  # raises ValueError if not valid hex


def _streamlit_theme() -> dict[str, Any]:
    theme: dict[str, Any] = tomllib.loads(STREAMLIT_CONFIG.read_text(encoding="utf-8"))["theme"]
    return theme


def test_theme_constants_match_the_streamlit_theme() -> None:
    """The THEME_* constants duplicate .streamlit/config.toml, so the two must agree."""
    theme = _streamlit_theme()

    assert theme["backgroundColor"].upper() == config.THEME_BACKGROUND_COLOR.upper()
    assert theme["secondaryBackgroundColor"].upper() == config.THEME_PANEL_COLOR.upper()
    assert theme["textColor"].upper() == config.THEME_TEXT_COLOR.upper()
    assert theme["borderColor"].upper() == config.THEME_BORDER_COLOR.upper()


def test_sidebar_theme_matches_the_theme_constants() -> None:
    """The sidebar is the panel color, with the page background as its secondary color."""
    sidebar = _streamlit_theme()["sidebar"]

    assert sidebar["backgroundColor"].upper() == config.THEME_PANEL_COLOR.upper()
    assert sidebar["secondaryBackgroundColor"].upper() == config.THEME_BACKGROUND_COLOR.upper()
    assert sidebar["textColor"].upper() == config.THEME_TEXT_COLOR.upper()
    assert sidebar["borderColor"].upper() == config.THEME_BORDER_COLOR.upper()


def test_visitors_do_not_see_raw_tracebacks() -> None:
    client = tomllib.loads(STREAMLIT_CONFIG.read_text(encoding="utf-8"))["client"]

    assert client["showErrorDetails"] in {"none", "type"}
