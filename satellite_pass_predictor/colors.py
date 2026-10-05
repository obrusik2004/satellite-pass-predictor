"""Color conversion for renderers that need RGB components rather than hex strings."""


def hex_to_rgb(hex_color: str) -> list[int]:
    """ "#RRGGBB" -> [r, g, b] (0-255 ints)."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
