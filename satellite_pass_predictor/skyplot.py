"""Polar sky plot (azimuth/elevation) of a single pass as seen from the ground station."""

from typing import cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Timescale
from skyfield.toposlib import GeographicPosition

from .config import SATELLITE_COLORS, THEME_BACKGROUND_COLOR, THEME_BORDER_COLOR, THEME_TEXT_COLOR
from .time_utils import build_time_grid
from .visibility import PassDict, compute_altaz

# Passes are detected on a 1-minute grid; re-sampling the short window finely keeps the arc smooth.
SKY_PLOT_STEP_SECONDS: float = 5.0

# Neutral marker colors so they can't be mistaken for a satellite's track color; rise and
# set are distinguished by shape.
_RISE_COLOR = "#FFFFFF"
_SET_COLOR = "#B0BEC5"


def build_sky_plot_figure(
    sat: EarthSatellite,
    pass_: PassDict,
    observer: GeographicPosition,
    ts: Timescale,
    satellite_name: str,
    step_seconds: float = SKY_PLOT_STEP_SECONDS,
) -> go.Figure:
    """Plot `pass_` as a polar elevation/azimuth track, re-sampled every `step_seconds`.

    Zenith is at the center and the horizon at the edge; azimuth runs clockwise from north at
    the top. AOS/LOS mark the first and last sample above the elevation mask. The track color
    comes from `satellite_name` (not `sat.name`, whose naming is Celestrak's).
    """
    duration_hours = (pass_["end_time"].tt - pass_["start_time"].tt) * 24.0
    t = build_time_grid(
        ts,
        start_time=pass_["start_time"],
        duration_hours=duration_hours,
        step_minutes=step_seconds / 60.0,
    )
    altaz = compute_altaz(sat, observer, t)
    # The grid is always an array, so the scalar half of the union never applies here.
    elevation_deg = cast(NDArray[np.float64], altaz["elevation_deg"])
    azimuth_deg = cast(NDArray[np.float64], altaz["azimuth_deg"])
    radius = 90.0 - elevation_deg

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=radius,
            theta=azimuth_deg,
            mode="lines",
            name="Pass",
            line={"color": SATELLITE_COLORS[satellite_name], "width": 2},
            customdata=elevation_deg,
            hovertemplate="az %{theta:.1f}°, el %{customdata:.1f}°<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[radius[0]],
            theta=[azimuth_deg[0]],
            mode="markers",
            name="AOS (rise)",
            marker={"color": _RISE_COLOR, "size": 12, "symbol": "circle"},
            customdata=[elevation_deg[0]],
            hovertemplate="AOS (rise): az %{theta:.1f}°, el %{customdata:.1f}°<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[radius[-1]],
            theta=[azimuth_deg[-1]],
            mode="markers",
            name="LOS (set)",
            marker={"color": _SET_COLOR, "size": 12, "symbol": "square"},
            customdata=[elevation_deg[-1]],
            hovertemplate="LOS (set): az %{theta:.1f}°, el %{customdata:.1f}°<extra></extra>",
        )
    )

    fig.update_layout(
        polar={
            "bgcolor": THEME_BACKGROUND_COLOR,
            "radialaxis": {
                "range": [0, 90],
                # Radius is 90 - elevation, so the tick labels are relabeled by hand.
                "tickvals": [0, 30, 60, 90],
                "ticktext": ["90°", "60°", "30°", "0°"],
                "gridcolor": THEME_BORDER_COLOR,
                "linecolor": THEME_BORDER_COLOR,
                "color": THEME_TEXT_COLOR,
            },
            "angularaxis": {
                # Plotly's default is 0 degrees at 3 o'clock, counterclockwise.
                "rotation": 90,
                "direction": "clockwise",
                "tickmode": "array",
                "tickvals": [0, 90, 180, 270],
                "ticktext": ["N", "E", "S", "W"],
                "gridcolor": THEME_BORDER_COLOR,
                "linecolor": THEME_BORDER_COLOR,
                "color": THEME_TEXT_COLOR,
            },
        },
        paper_bgcolor=THEME_BACKGROUND_COLOR,
        font={"color": THEME_TEXT_COLOR},
        showlegend=True,
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
    )
    return fig
