"""Interactive WebGL globe (pydeck GlobeView) of satellite ground tracks and the Kourou station.

Pure rendering: takes already-computed ground tracks and returns a pydeck Deck.
"""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pydeck as pdk
from numpy.typing import NDArray

from .colors import hex_to_rgb
from .config import (
    GLOBE_INITIAL_ZOOM,
    GLOBE_TRACK_WIDTH_M,
    KOUROU_ELEVATION_M,
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    KOUROU_MARKER_COLOR,
    SATELLITE_COLORS,
    THEME_BACKGROUND_COLOR,
    THEME_BORDER_COLOR,
    THEME_PANEL_COLOR,
)
from .geo import find_antimeridian_crossings
from .propagation import GroundTrackDict


def _split_path_at_antimeridian(
    longitudes: NDArray[np.float64], latitudes: NDArray[np.float64]
) -> list[list[list[float]]]:
    """Split a ground track into [lon, lat] sub-paths at each +-180 degree crossing.

    Each cut gets an interpolated point exactly on the meridian at both ends, so the track
    stays visually continuous. Returns one sub-path if there is no crossing.
    """
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    crossings = find_antimeridian_crossings(longitudes)
    if len(crossings) == 0:
        return [[[float(lo), float(la)] for lo, la in zip(longitudes, latitudes, strict=True)]]

    # Per crossing: the meridian being crossed (+180 or -180) and the latitude where the
    # segment reaches it. The far longitude is unwrapped (179 -> -179 becomes 179 -> 181)
    # so the interpolation fraction is plain linear.
    boundaries: list[tuple[float, float]] = []  # (signed boundary longitude, crossing latitude)
    for i in crossings:
        lon0, lon1 = float(longitudes[i]), float(longitudes[i + 1])
        lat0, lat1 = float(latitudes[i]), float(latitudes[i + 1])
        delta = lon1 - lon0
        unwrapped_lon1 = lon1 - 360.0 * round(delta / 360.0)
        unwrapped_delta = unwrapped_lon1 - lon0
        boundary_lon = 180.0 if unwrapped_delta > 0 else -180.0
        t = (boundary_lon - lon0) / unwrapped_delta
        boundaries.append((boundary_lon, lat0 + t * (lat1 - lat0)))

    segment_bounds = [0, *(int(i) + 1 for i in crossings), len(longitudes)]
    paths: list[list[list[float]]] = []
    for j in range(len(segment_bounds) - 1):
        lo_slice = longitudes[segment_bounds[j] : segment_bounds[j + 1]]
        la_slice = latitudes[segment_bounds[j] : segment_bounds[j + 1]]
        path = [[float(lo), float(la)] for lo, la in zip(lo_slice, la_slice, strict=True)]
        if j > 0:
            # Starts after a crossing: begin on the opposite side of the meridian.
            prev_boundary_lon, prev_lat = boundaries[j - 1]
            path.insert(0, [-prev_boundary_lon, prev_lat])
        if j < len(boundaries):
            # Ends before a crossing: finish on the meridian.
            boundary_lon, lat = boundaries[j]
            path.append([boundary_lon, lat])
        paths.append(path)
    return paths


# Natural Earth 1:110m land polygons, geometry only, loaded once at import.
_ASSETS_DIR = Path(__file__).parent / "assets"
_WORLD_LAND_GEOJSON: dict[str, Any] = json.loads(
    (_ASSETS_DIR / "world_land.geojson").read_text(encoding="utf-8")
)

# The basemap reuses theme colors so it stays visually secondary to the tracks.
_LAND_FILL_COLOR = hex_to_rgb(THEME_PANEL_COLOR)
_LAND_BORDER_COLOR = hex_to_rgb(THEME_BORDER_COLOR)
_LAND_BORDER_WIDTH_PIXELS = 1

_KOUROU_COLOR = hex_to_rgb(KOUROU_MARKER_COLOR)
_KOUROU_RADIUS_PIXELS = 10

# Canvas clear color (deck.gl wants 0-1 floats) matches the page so the globe has no backdrop box.
_CANVAS_CLEAR_COLOR = [c / 255.0 for c in hex_to_rgb(THEME_BACKGROUND_COLOR)] + [1.0]

# Only every Nth track point is hoverable; at full density a single point is hard to hit.
_HOVER_POINT_STRIDE = 4


def build_globe_deck(ground_tracks: dict[str, GroundTrackDict]) -> pdk.Deck:
    """Build the globe: a land basemap, a track and hover points per satellite, and Kourou.

    `ground_tracks` is keyed by display name; track colors come from SATELLITE_COLORS. The
    camera starts over Kourou at GLOBE_INITIAL_ZOOM.
    """
    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            data=_WORLD_LAND_GEOJSON,
            filled=True,
            get_fill_color=_LAND_FILL_COLOR,
            stroked=True,
            get_line_color=_LAND_BORDER_COLOR,
            line_width_min_pixels=_LAND_BORDER_WIDTH_PIXELS,
            pickable=False,
        )
    ]
    view_state = pdk.ViewState(
        latitude=KOUROU_LATITUDE_DEG,
        longitude=KOUROU_LONGITUDE_DEG,
        zoom=GLOBE_INITIAL_ZOOM,
        pitch=0,
    )

    for name, track in ground_tracks.items():
        # Color by name, not position, so it doesn't change with selection order.
        color = hex_to_rgb(SATELLITE_COLORS[name])
        # compute_ground_track() always yields arrays, never the scalar half of the union.
        latitude_deg = cast(NDArray[np.float64], track["latitude_deg"])
        longitude_deg = cast(NDArray[np.float64], track["longitude_deg"])
        altitude_km = cast(NDArray[np.float64], track["altitude_km"])
        time_strings = track["time"].utc_strftime("%Y-%m-%d %H:%M:%S UTC")

        layers.append(
            pdk.Layer(
                "PathLayer",
                # PathLayer has no NaN breaks, so a crossing needs separate paths.
                data=[
                    {"path": sub_path, "color": color}
                    for sub_path in _split_path_at_antimeridian(longitude_deg, latitude_deg)
                ],
                get_path="path",
                get_color="color",
                get_width=GLOBE_TRACK_WIDTH_M,
                pickable=False,
            )
        )

        # Built from the unsplit arrays: the split adds synthetic boundary points that have
        # no real time or altitude to show on hover.
        hover_points = [
            {
                "name": name,
                "lon": float(lo),
                "lat": float(la),
                "lat_str": f"{la:.2f}",
                "lon_str": f"{lo:.2f}",
                "alt_str": f"{al:.1f}",
                "time": when,
                "color": color,
            }
            for lo, la, al, when in zip(
                longitude_deg[::_HOVER_POINT_STRIDE],
                latitude_deg[::_HOVER_POINT_STRIDE],
                altitude_km[::_HOVER_POINT_STRIDE],
                time_strings[::_HOVER_POINT_STRIDE],
                strict=True,
            )
        ]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=hover_points,
                get_position=["lon", "lat"],
                get_fill_color="color",
                radius_min_pixels=5,
                pickable=True,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            # Same field names as the hover points so the shared tooltip template fits.
            data=[
                {
                    "name": "Kourou",
                    "time": "Guiana Space Centre (ground station)",
                    "lat_str": f"{KOUROU_LATITUDE_DEG:.4f}",
                    "lon_str": f"{KOUROU_LONGITUDE_DEG:.4f}",
                    "alt_str": f"{KOUROU_ELEVATION_M:.1f}",
                }
            ],
            get_position=[KOUROU_LONGITUDE_DEG, KOUROU_LATITUDE_DEG],
            get_fill_color=_KOUROU_COLOR,
            radius_min_pixels=_KOUROU_RADIUS_PIXELS,
            stroked=True,
            get_line_color=[0, 0, 0],
            line_width_min_pixels=2,
            pickable=True,
        )
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        views=[pdk.View(type="_GlobeView", controller=True)],
        map_provider=None,
        parameters={"clearColor": _CANVAS_CLEAR_COLOR},
        tooltip={
            "html": (
                "<b>{name}</b><br/>{time}<br/>lat {lat_str}°, lon {lon_str}°<br/>alt {alt_str} km"
            ),
        },
    )
