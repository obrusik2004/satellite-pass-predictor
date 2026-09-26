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

from .config import KOUROU_ELEVATION_M, KOUROU_LATITUDE_DEG, KOUROU_LONGITUDE_DEG
from .propagation import GroundTrackDict

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
# satellite tracks -- confirmed by looking at it against Streamlit's
# near-black dark theme (background ~rgb(14,17,23)): distinctly visible
# without approaching the saturation of any satellite color or Kourou's
# gold.
_LAND_FILL_COLOR = [30, 38, 50]
_LAND_BORDER_COLOR = [90, 105, 130]
_LAND_BORDER_WIDTH_PIXELS = 1

# Matches matplotlib's default color cycle (the first entries of
# "tab10"), so a satellite's color is at least consistent between this
# globe and the 2D matplotlib plot the CLI/HTML report still use --
# not required, but easy, and there's no reason to gratuitously clash.
# pydeck wants plain [r, g, b] triplets, not hex strings.
_SATELLITE_COLORS = [
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
]

# Kourou is the ground station, not a satellite -- gold, larger, and
# outlined in black, distinct from every satellite color above by both
# color and treatment (pydeck's ScatterplotLayer only draws circles, so
# "symbol" here means size + stroke rather than a literal star shape;
# see build_globe_deck()'s docstring for why that's an acceptable
# trade-off rather than something worth chasing an IconLayer for).
_KOUROU_COLOR = [255, 215, 0]
_KOUROU_RADIUS_PIXELS = 10

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

    Antimeridian handling, re-checked for this geometry rather than
    assumed to carry over from visualization.py's flat matplotlib plot
    (_split_at_antimeridian): that logic exists purely because a flat
    lat/lon plot draws +180 and -180 degrees as two different edges of a
    rectangle, so a track crossing between them jumps straight across
    the image unless deliberately broken. On a true sphere there's no
    such seam -- GlobeView converts each (lat, lon) sample to a 3D
    position on the sphere's actual surface and interpolates along the
    straight (short) 3D chord between consecutive points, not along a
    2D lon/lat line, so a pair like 179 degrees and -179 degrees (2
    degrees apart on the sphere) is never treated as being 358 degrees
    apart. Verified this directly rather than trusting the theory alone:
    fed the globe a path deliberately crossing the antimeridian and
    checked the far/opposite hemisphere for a wrong-way line wrapping
    all the way around the back -- there was none, confirming the short
    path was taken. (There was a barely-visible cosmetic seam of a
    couple of pixels exactly at the crossing vertex under an extreme,
    artificial test at 5-degree point spacing and heavy zoom-in -- a
    tessellation detail, not a wrong-direction line, and not something
    real 1-minute-resolution satellite data is dense enough to make
    visible in practice.) No antimeridian splitting is applied or
    needed here.

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

    Color against Streamlit's dark theme -- also checked by looking,
    not assumed: against the app's near-black background
    (~rgb(14,17,23)), _LAND_FILL_COLOR reads as distinctly visible land
    without approaching the saturation of any satellite color or
    Kourou's gold, and _LAND_BORDER_COLOR gives coastlines a bit of
    extra definition without standing out on its own.

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
    layers = [pdk.Layer(
        "GeoJsonLayer",
        data=_WORLD_LAND_GEOJSON,
        filled=True,
        get_fill_color=_LAND_FILL_COLOR,
        stroked=True,
        get_line_color=_LAND_BORDER_COLOR,
        line_width_min_pixels=_LAND_BORDER_WIDTH_PIXELS,
        pickable=False,
    )]
    view_state = pdk.ViewState(
        latitude=KOUROU_LATITUDE_DEG, longitude=KOUROU_LONGITUDE_DEG,
        zoom=1.7, pitch=0,
    )

    for i, (name, track) in enumerate(ground_tracks.items()):
        color = _SATELLITE_COLORS[i % len(_SATELLITE_COLORS)]
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

        layers.append(pdk.Layer(
            "PathLayer",
            data=[{
                "path": [[float(lo), float(la)] for lo, la in zip(longitude_deg, latitude_deg)],
                "color": color,
            }],
            get_path="path",
            get_color="color",
            get_width=15000,
            pickable=False,
        ))

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
            )
        ]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=hover_points,
            get_position=["lon", "lat"],
            get_fill_color="color",
            radius_min_pixels=5,
            pickable=True,
        ))

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        # Same field names as the per-satellite hover points above (name,
        # time, lat_str, lon_str, alt_str) so the one shared tooltip
        # template renders sensibly for Kourou too, rather than showing
        # unresolved "{time}"/"{alt_str}" placeholders: "time" becomes a
        # descriptive label instead of a timestamp, and "alt_str" reuses
        # config.KOUROU_ELEVATION_M (0m -- coastal, effectively sea
        # level) instead of a made-up value.
        data=[{
            "name": "Kourou",
            "time": "Guiana Space Centre (ground station)",
            "lat_str": f"{KOUROU_LATITUDE_DEG:.4f}",
            "lon_str": f"{KOUROU_LONGITUDE_DEG:.4f}",
            "alt_str": f"{KOUROU_ELEVATION_M:.1f}",
        }],
        get_position=[KOUROU_LONGITUDE_DEG, KOUROU_LATITUDE_DEG],
        get_fill_color=_KOUROU_COLOR,
        radius_min_pixels=_KOUROU_RADIUS_PIXELS,
        stroked=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=2,
        pickable=True,
    ))

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        views=[pdk.View(type="_GlobeView", controller=True)],
        map_provider=None,
        tooltip={
            "html": (
                "<b>{name}</b><br/>{time}<br/>"
                "lat {lat_str}°, lon {lon_str}°<br/>"
                "alt {alt_str} km"
            ),
        },
    )
