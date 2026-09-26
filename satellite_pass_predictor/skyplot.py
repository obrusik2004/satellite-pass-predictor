"""
Sky plot: a polar elevation/azimuth chart of a single visibility pass,
showing how a satellite tracks across the sky as seen from Kourou.

Separate module rather than folded into visualization.py or globe.py:
neither is the right fit -- visualization.py is the flat lat/lon
ground-track plot and the pass-table formatting, globe.py is the 3D
WebGL globe, and this is a third, distinct kind of chart (a 2D polar
plot of a single pass's look angles) with its own concern, matching
this project's one-module-per-concern convention.

Plotly, not pydeck/deck.gl (globe.py's library) or matplotlib
(visualization.py's): this is a small 2D chart, not a 3D geospatial
scene, so deck.gl's WebGL globe machinery doesn't apply here at all.
Plotly was previously removed from this project when the globe was
rebuilt on pydeck for genuine WebGL acceleration (see globe.py's module
docstring) -- that was about a specific rendering-technology problem
(Scattergeo's SVG redraw cost, see globe.py), not a decision against
Plotly everywhere. It's reintroduced here for an unrelated, much
smaller job: one short pass' worth of points (order of 100, not
per-satellite-per-minute over hours), well within what an SVG-based
chart handles instantly -- the performance problem step 8.2 fixed
doesn't apply to this scope.
"""

from typing import cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Timescale
from skyfield.toposlib import GeographicPosition

from .time_utils import build_time_grid
from .visibility import PassDict, compute_altaz

# The pass table (app.py) builds passes from a step_minutes=1 grid --
# fine for detecting whether/when a pass happens, but 1-minute-spaced
# points plotted as a polar arc read as a faceted, kinked line rather
# than a smooth pass track, especially over the short (often just a few
# minutes) window a single pass covers. Recomputing at 5-second
# resolution is cheap: it's one short window for one already-selected
# satellite (typically a few dozen to ~150 points), not a full
# multi-hour, multi-satellite re-propagation like the table itself or
# the globe's ground tracks.
SKY_PLOT_STEP_SECONDS: float = 5.0

_TRACK_COLOR = "#1f77b4"  # matches globe.py/visualization.py's first default color
_RISE_COLOR = "#2ca02c"
_SET_COLOR = "#d62728"


def build_sky_plot_figure(
    sat: EarthSatellite,
    pass_: PassDict,
    observer: GeographicPosition,
    ts: Timescale,
    step_seconds: float = SKY_PLOT_STEP_SECONDS,
) -> go.Figure:
    """
    Build a polar elevation/azimuth chart of `pass_`, recomputed for
    `sat` at `step_seconds` resolution across just that pass's own
    start_time..end_time window (not the whole multi-hour prediction
    window `pass_` was originally detected in).

    Elevation is mapped to radius *inverted* (radius = 90 - elevation),
    so the zenith (looking straight up, elevation 90 deg) sits at the
    plot's center and the horizon (elevation 0 deg) sits at the outer
    edge. This is the standard convention for this kind of chart (the
    same one used by amateur-radio/GNSS pass-tracking tools like
    Heavens-Above, N2YO, and gpredict) -- confirmed rather than assumed,
    since the "obvious" alternative (radius = elevation, horizon at the
    center) would put the least interesting part of the sky in the
    middle and require the reader to read the radial axis backwards
    from how a real dome of sky actually maps to a flat page.

    Azimuth is mapped to the angular axis in compass convention (0 deg
    = N at the top, increasing clockwise through E/S/W), not the
    mathematical convention (0 deg on the right, increasing
    counterclockwise) Plotly's polar chart defaults to -- set via
    angularaxis rotation=90 (rotates the 0-degree position from the
    default "3 o'clock" to "12 o'clock") and direction="clockwise".

    Start (rise) and end (set) are marked with distinct markers/colors
    so the direction of travel across the sky is clear even before
    reading the arrow-less line between them.
    """
    duration_hours = (pass_["end_time"].tt - pass_["start_time"].tt) * 24.0
    t = build_time_grid(
        ts, start_time=pass_["start_time"],
        duration_hours=duration_hours, step_minutes=step_seconds / 60.0,
    )
    altaz = compute_altaz(sat, observer, t)
    # compute_altaz()'s fields are typed float | NDArray (a scalar `t`
    # would produce scalars), but `t` here always comes from
    # build_time_grid(), which always returns a vectorized time -- so
    # these are always arrays in practice, same reasoning as
    # visualization.py/propagation.py's equivalent casts.
    elevation_deg = cast(NDArray[np.float64], altaz["elevation_deg"])
    azimuth_deg = cast(NDArray[np.float64], altaz["azimuth_deg"])
    radius = 90.0 - elevation_deg

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radius, theta=azimuth_deg, mode="lines", name="Pass",
        line={"color": _TRACK_COLOR, "width": 2},
        customdata=elevation_deg,
        hovertemplate="az %{theta:.1f}°, el %{customdata:.1f}°<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[radius[0]], theta=[azimuth_deg[0]], mode="markers", name="Rise",
        marker={"color": _RISE_COLOR, "size": 12, "symbol": "circle"},
        customdata=[elevation_deg[0]],
        hovertemplate="Rise: az %{theta:.1f}°, el %{customdata:.1f}°<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[radius[-1]], theta=[azimuth_deg[-1]], mode="markers", name="Set",
        marker={"color": _SET_COLOR, "size": 12, "symbol": "square"},
        customdata=[elevation_deg[-1]],
        hovertemplate="Set: az %{theta:.1f}°, el %{customdata:.1f}°<extra></extra>",
    ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "range": [0, 90],
                # tickvals are radius (0=center..90=edge); ticktext is
                # the elevation those radii represent (90 deg at the
                # center, 0 deg at the edge) -- the inversion is a
                # display mapping only, so the axis must be relabeled
                # by hand rather than just reversing the range (Plotly's
                # radialaxis has no "autorange=reversed with custom
                # zero" option that would do both at once).
                "tickvals": [0, 30, 60, 90],
                "ticktext": ["90°", "60°", "30°", "0°"],
            },
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickmode": "array",
                "tickvals": [0, 90, 180, 270],
                "ticktext": ["N", "E", "S", "W"],
            },
        },
        showlegend=True,
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
    )
    return fig
