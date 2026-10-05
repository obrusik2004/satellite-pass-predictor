"""
Shared color helper: config.py's theme/satellite colors are all plain
"#RRGGBB" hex strings (the form both .streamlit/config.toml and CSS
want), but a couple of renderers need the same color as plain RGB
components instead -- globe.py's pydeck layers (which want [r, g, b]
0-255 triplets, not hex strings) and reporting.py's HTML report (which
needs an rgba() value, for translucent muted text). One conversion,
reused, rather than two separate implementations of the same hex parse.
"""


def hex_to_rgb(hex_color: str) -> list[int]:
    """ "#RRGGBB" -> [r, g, b] (0-255 ints)."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
