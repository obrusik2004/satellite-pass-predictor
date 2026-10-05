"""
3D globe visualization: an interactive, WebGL-accelerated globe (pydeck /
deck.gl's GlobeView), built from already-computed ground-track data
(GroundTrackDict per satellite, from propagation.compute_ground_track())
-- this module does no propagation of its own, the same principle
visualization.build_ground_tracks_figure() follows for the matplotlib
plot. It's a pure "data in, pydeck Deck out" renderer, which is also
what keeps it simple to test.

This replaces an earlier Plotly Scattergeo (orthographic-projection)
version. Scattergeo renders as SVG, not WebGL, which turned out to have
a hard rendering ceiling: confirmed directly (DOM inspection: 0 <canvas>
elements; with markers on, 98% of all rendered SVG elements were
individual per-point markers) that rotation cost ~195-212ms per frame
(~5fps) even after switching to line-only rendering, and that further
point-density cuts didn't help (an 8x reduction produced no measurable
change) -- the bottleneck was Scattergeo's own SVG geo/projection
redraw, not our data, so no further tuning of that approach could fix
it. pydeck's GlobeView is genuinely WebGL-accelerated (confirmed via
DOM inspection: a real <canvas> element backed by an active WebGL2
context) and measured at a sustained 60fps during a realistic
rAF-paced drag-rotation gesture -- see build_globe_deck()'s docstring
for the rest of that investigation, including a real false start.

A land/coastline basemap (GeoJsonLayer, see _WORLD_LAND_GEOJSON) was
added afterwards so the globe isn't just tracks and a marker on a
plain sphere. Rendered as a proper deck.gl layer, not anything
SVG-based, to stay on the WebGL path above -- see build_globe_deck()'s
docstring for the fill/color design decisions and the (real, then
fixed) performance regression that came with an early, too-detailed
choice of dataset.
"""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pydeck as pdk
from numpy.typing import NDArray

from .colors import hex_to_rgb
from .config import (
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
    """
    Split one satellite's ground track into separate [lon, lat] sub-paths
    at every +-180 degree antimeridian crossing (see build_globe_deck()'s
    docstring for why pydeck's PathLayer needs this, confirmed directly
    by observation, unlike the point-to-point great-circle case it does
    handle correctly on its own).

    Each crossing's own two resulting sub-paths are extended with an
    interpolated point placed exactly on the boundary they were cut at
    (linearly interpolating latitude between the two original straddling
    samples, at the fraction of the segment where longitude actually
    reaches 180 degrees) so the split doesn't leave a visible gap at the
    meridian -- the track still looks continuous, just correctly drawn
    on both sides instead of wrapping the long way round through the
    globe's other hemisphere.

    Returns a list of sub-paths (each a list of [lon, lat] pairs) -- one
    entry if the track never crosses the antimeridian, more if it does
    (a satellite can cross it more than once within one globe window).
    """
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    crossings = find_antimeridian_crossings(longitudes)
    if len(crossings) == 0:
        return [[[float(lo), float(la)] for lo, la in zip(longitudes, latitudes, strict=True)]]

    # For each crossing, the exact boundary longitude (+180 or -180,
    # whichever side the track is moving *towards*) and the latitude at
    # that exact crossing point, linearly interpolated between the two
    # straddling samples. `unwrapped_lon1` re-expresses the far sample's
    # longitude as a continuous (not wrapped-around) value relative to
    # the near one -- e.g. 179 -> -179 becomes 179 -> 181 -- so the
    # fraction-of-segment calculation (`t`) below is a plain linear
    # interpolation rather than needing its own wraparound-aware case.
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
            # This segment starts right after a crossing -- prepend the
            # same crossing's interpolated point, on the *opposite* side
            # of the meridian from where the previous segment ended.
            prev_boundary_lon, prev_lat = boundaries[j - 1]
            path.insert(0, [-prev_boundary_lon, prev_lat])
        if j < len(boundaries):
            # This segment ends right before a crossing -- append that
            # crossing's interpolated point on this segment's own side.
            boundary_lon, lat = boundaries[j]
            path.append([boundary_lon, lat])
        paths.append(path)
    return paths


# Land outlines for visual reference on the globe -- otherwise it's just
# lines and a marker floating on a plain sphere, with nothing to anchor
# what you're looking at. Natural Earth's 1:110m "land" layer (public
# domain, no API key) via their own GitHub repo, the standard source for
# a low-resolution world basemap.
#
# Deliberately land/coastline outlines (ne_110m_land: continents and
# islands merged into one shape each, no internal borders), not full
# ne_110m_admin_0_countries (177 separate country polygons sharing
# political boundary lines): the goal stated for this feature was
# "continents or coastlines to visually anchor" it, not political
# borders, and country boundaries are meaningfully more geometry for no
# benefit toward that goal. This also turned out to matter for
# performance, confirmed by measurement rather than assumed -- see
# build_globe_deck()'s docstring for the full before/after numbers.
# Trimmed further here by dropping Natural Earth's per-feature metadata
# (scale rank, feature class, etc.) this layer has no use for -- it's
# pickable=False, geometry only, loaded once at import time rather than
# on every Streamlit rerun.
_ASSETS_DIR = Path(__file__).parent / "assets"
_WORLD_LAND_GEOJSON: dict[str, Any] = json.loads(
    (_ASSETS_DIR / "world_land.geojson").read_text(encoding="utf-8")
)

# Filled, not outline-only: compared both directly (see
# build_globe_deck()'s docstring) and filled reads immediately as "this
# is Earth" at a glance, which more directly solves the stated problem,
# while a muted, desaturated color keeps it clearly secondary to the
# satellite tracks. Reuses the app's own theme colors (THEME_PANEL_COLOR
# for the fill, THEME_BORDER_COLOR for the coastline) rather than a
# one-off palette, so the basemap sits visually "in" the ESA-inspired
# theme instead of being colored independently of it -- checked directly
# against THEME_BACKGROUND_COLOR: distinctly visible (contrast ratio
# ~1.27:1, matching this same subtle-but-visible relationship the
# original arbitrary colors had against Streamlit's old default dark
# theme) without approaching the saturation of any satellite color or
# Kourou's white marker.
_LAND_FILL_COLOR = hex_to_rgb(THEME_PANEL_COLOR)
_LAND_BORDER_COLOR = hex_to_rgb(THEME_BORDER_COLOR)
_LAND_BORDER_WIDTH_PIXELS = 1

# Kourou is the ground station, not a satellite -- white, larger, and
# outlined in black, distinct from every satellite color (config.
# SATELLITE_COLORS) by both color and treatment (pydeck's
# ScatterplotLayer only draws circles, so "symbol" here means size +
# stroke rather than a literal star shape; see build_globe_deck()'s
# docstring for why that's an acceptable trade-off rather than something
# worth chasing an IconLayer for).
_KOUROU_COLOR = hex_to_rgb(KOUROU_MARKER_COLOR)
_KOUROU_RADIUS_PIXELS = 10

# The WebGL canvas's own clear color -- what shows through in the "void"
# around the globe sphere (map_provider=None means there's no basemap
# tile layer to fill it otherwise). Matched to THEME_BACKGROUND_COLOR so
# the globe blends into the surrounding Streamlit page rather than
# showing deck.gl's own default (a plain white/black rectangle) around
# the sphere. Normalized to 0-1 (deck.gl's WebGL parameter convention),
# not the 0-255 pydeck layer colors use elsewhere in this module.
_CANVAS_CLEAR_COLOR = [c / 255.0 for c in hex_to_rgb(THEME_BACKGROUND_COLOR)] + [1.0]

# Every satellite subpoint gets a line vertex (for a smooth-looking
# track), but only every Nth one also gets a pickable hover point.
# Checked directly, not assumed: at full density (a satellite's full
# ~360 points for a 6h/1min window) individual points sit close enough
# together on screen that precisely hovering one specific point is
# impractical -- verified this by testing hover at that density (it
# essentially never landed on a specific point) versus at this stride
# (reliably hit-testable). Every 4th point still gives a hover sample
# roughly every 4 minutes of track, dense enough to be useful.
_HOVER_POINT_STRIDE = 4


def build_globe_deck(ground_tracks: dict[str, GroundTrackDict]) -> pdk.Deck:
    """
    Build an interactive WebGL globe: one PathLayer per satellite (from
    `ground_tracks`, keyed by display name) for the visible track, a
    thinned ScatterplotLayer per satellite for hover, and a distinct
    marker for the Kourou ground station.

    Antimeridian handling: CONFIRMED needed, by direct observation in the
    running app -- an earlier version of this docstring claimed pydeck's
    GlobeView interpolates each PathLayer segment along the sphere's own
    short 3D chord, making a break at +-180 degrees unnecessary (the way
    it genuinely is unnecessary for the *point-to-point great-circle*
    math). That claim doesn't hold for what PathLayer actually draws:
    rotating the globe to the Pacific/New Zealand view showed every
    satellite's track breaking at the 180 degree meridian with a
    spurious straight line running along a constant latitude connecting
    the two broken ends -- i.e. PathLayer was joining e.g. +179 and -179
    degrees "the long way round" (through 0 degrees), the exact
    wrong-direction streak a flat 2D plot would produce, not the short
    hop across the seam. So this needs the same fix visualization.py's
    flat matplotlib plot needs, just adapted to pydeck's data model:
    pydeck's PathLayer has no NaN-break equivalent (unlike matplotlib),
    so instead of inserting a break value, _split_path_at_antimeridian()
    below splits one long path into multiple separate path entries in
    the layer's `data` list at each crossing (found via the same
    geo.find_antimeridian_crossings() visualization.py's own fix uses,
    not a second, independently-written detector) -- and, so the two
    resulting segments don't leave a visible gap at the meridian, each
    one is extended with an interpolated point placed exactly on the
    boundary it was cut at (+180 for the segment ending there, -180 for
    the segment starting there, or vice versa depending on crossing
    direction), at that crossing's own interpolated latitude.

    Rendering technology, verified rather than assumed: confirmed a
    real <canvas> element backed by an active WebGL2 context (not a 2D
    canvas, not SVG) via direct DOM/context inspection, and measured a
    sustained 60fps over a realistic rAF-paced synthetic drag gesture
    (one simulated pointer move per animation frame, matching how a
    real mouse drag actually arrives) -- no dropped frames, versus the
    Scattergeo version's measured ~195-212ms/frame (~5fps) ceiling.

    Per-point hover tooltips, also verified directly rather than
    assumed to work from pydeck's documentation alone: an initial test
    with a fully-dense point layer (~1444 points across 4 satellites)
    and synthetic pointer events found seemingly zero hits, which
    looked at first like a genuine GlobeView picking limitation --
    until testing the identical layer under a normal flat MapView
    worked immediately, which isolated the real cause to two compounding
    issues with the *test*, not GlobeView itself: synthetic
    canvas.dispatchEvent() calls don't reliably trigger deck.gl's
    GPU-based picking pass the way a genuine mouse hover does, and
    picking one specific point reliably becomes impractical once many
    points sit close together on screen. Retested with genuine hover
    events at a thinned point density (see _HOVER_POINT_STRIDE) and
    tooltips worked correctly and consistently.

    Kourou's marker defaults the camera to center on Kourou/the
    Atlantic (via initial_view_state) so the ground station is visible
    on load without the user needing to rotate to find it first.

    Basemap (GeoJsonLayer, _WORLD_LAND_GEOJSON): added so the globe
    isn't just tracks and a marker floating on a plain sphere. Two
    judgment calls, both settled by looking at the result rather than
    guessing:

    Filled vs. outline-only -- compared both directly in the running
    app. Filled land reads immediately as "this is Earth" at a glance,
    which more directly serves the actual goal ("visually anchor" the
    view) than an outline does; a muted, desaturated fill
    (_LAND_FILL_COLOR) keeps it clearly secondary to the satellite
    tracks rather than competing with them.

    Color against the app's theme -- also checked directly, not assumed:
    _LAND_FILL_COLOR/_LAND_BORDER_COLOR now derive from the app's own
    ESA-inspired theme colors (THEME_PANEL_COLOR/THEME_BORDER_COLOR, see
    config.py) rather than a one-off palette, and read as distinctly
    visible land against THEME_BACKGROUND_COLOR without approaching the
    saturation of any satellite color or Kourou's white marker -- see
    _LAND_FILL_COLOR's own comment for the actual contrast numbers.

    Performance impact -- re-measured with the same rAF-paced
    synthetic-drag methodology as the WebGL verification above, since a
    layer isn't free just because it's WebGL. The first attempt (using
    Natural Earth's full ne_110m_admin_0_countries dataset, 177
    political-border polygons) caused a real, severe regression:
    filled, fps dropped to 1.2 and 11.1 (from the 60fps baseline);
    outline-only was better but still degraded and inconsistent (30.3,
    30.0, then 6.6, 6.5, 29.6, 5.7, 29.6 on repeated measurement) --
    never reliably back to baseline. Root cause was excessive geometry:
    177 fully-detailed country polygons is dramatically more
    triangulated geometry than the sparse satellite tracks the app
    otherwise draws. Switching to ne_110m_land (127 simpler polygons,
    no political borders -- which also happens to be the dataset that
    actually matches what was asked for, see _WORLD_LAND_GEOJSON's
    comment) fixed it: re-measured at 61.3, 60.5, and 58.0 fps, matching
    the pre-basemap baseline within normal run-to-run noise.

    Picking/hover interference -- checked directly rather than assumed
    safe: with the basemap layer present (pickable=False), genuine
    hover was re-tested on both a satellite hover point and Kourou's
    marker and both tooltips fired correctly, confirming the basemap
    sitting at the same visual depth doesn't intercept picking meant
    for the layers above it.
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
        zoom=1.7,
        pitch=0,
    )

    for name, track in ground_tracks.items():
        # Looked up by name (config.SATELLITE_COLORS), not by position in
        # `ground_tracks` -- st.multiselect() (app.py) can return
        # satellites in whatever order the user selected them in, and a
        # positional/index-based color would then reassign a satellite's
        # color every time the selection order changed, which is exactly
        # what a single shared color mapping is meant to prevent.
        color = hex_to_rgb(SATELLITE_COLORS[name])
        # track's lat/lon/alt fields are typed float | NDArray (see
        # propagation.FloatOrArray) since get_subpoint() can also be
        # called with a scalar time -- but compute_ground_track() always
        # builds a vectorized time grid, so these are always arrays in
        # practice. Same cast() pattern used throughout the package
        # (propagation.py, visibility.py, visualization.py).
        latitude_deg = cast(NDArray[np.float64], track["latitude_deg"])
        longitude_deg = cast(NDArray[np.float64], track["longitude_deg"])
        altitude_km = cast(NDArray[np.float64], track["altitude_km"])
        # One batched call across the whole time array rather than
        # formatting each sample's timestamp separately -- Skyfield
        # vectorizes utc_strftime() over a Time array into a plain list
        # of strings, which is materially cheaper than calling it once
        # per point for a satellite with hundreds of samples.
        time_strings = track["time"].utc_strftime("%Y-%m-%d %H:%M:%S UTC")

        layers.append(
            pdk.Layer(
                "PathLayer",
                # One "path" entry per antimeridian-free sub-path (usually
                # just one) rather than a single entry spanning the whole
                # track -- see _split_path_at_antimeridian() and this
                # function's own docstring for why a single entry draws a
                # spurious wrong-direction line at a +-180 degree crossing.
                data=[
                    {"path": sub_path, "color": color}
                    for sub_path in _split_path_at_antimeridian(longitude_deg, latitude_deg)
                ],
                get_path="path",
                get_color="color",
                get_width=15000,
                pickable=False,
            )
        )

        # Deliberately built from the original, unsplit longitude_deg/
        # latitude_deg arrays, not from _split_path_at_antimeridian()'s
        # output: ScatterplotLayer draws independent points, never lines
        # between them, so there's no "wrong direction" to draw at a
        # crossing the way PathLayer has -- a hover point at +179.7 next
        # to one at -179.7 is just two correctly-placed points, not a
        # rendering bug. Using the split output here would also require
        # fabricating hover data (time/altitude) for the synthetic
        # interpolated boundary points _split_path_at_antimeridian() adds
        # for PathLayer's benefit, which don't correspond to a real
        # sample at all.
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
            # Same field names as the per-satellite hover points above (name,
            # time, lat_str, lon_str, alt_str) so the one shared tooltip
            # template renders sensibly for Kourou too, rather than showing
            # unresolved "{time}"/"{alt_str}" placeholders: "time" becomes a
            # descriptive label instead of a timestamp, and "alt_str" reuses
            # config.KOUROU_ELEVATION_M (0m -- coastal, effectively sea
            # level) instead of a made-up value.
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
