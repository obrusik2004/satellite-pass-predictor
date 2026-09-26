"""
Visualization: turning computed tracks and passes into human-readable
output -- the ground track plot (PNG) and the pass table (printed text).
"""

import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time, Timescale

from .config import OUTPUT_DIR
from .propagation import compute_ground_track
from .visibility import PassDict


def _split_at_antimeridian(
    longitudes: NDArray[np.float64], latitudes: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Insert NaN breaks wherever consecutive longitude samples jump across
    the +180/-180 antimeridian (International Date Line).

    A ground track crossing that line jumps from e.g. +179.9 deg to
    -179.9 deg between two adjacent samples that are actually right next
    to each other on the map. Plotted naively, that reads as one sample
    spanning almost 360 degrees -- a wrong horizontal streak all the way
    across the plot. matplotlib skips over NaN values in a line plot, so
    inserting one at each such jump breaks the line into separate
    segments there instead, without needing to change how the underlying
    lat/lon data was computed.
    """
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)

    jumps = np.where(np.abs(np.diff(longitudes)) > 180.0)[0]
    if len(jumps) == 0:
        return longitudes, latitudes

    insert_at = jumps + 1
    longitudes = np.insert(longitudes, insert_at, np.nan)
    latitudes = np.insert(latitudes, insert_at, np.nan)
    return longitudes, latitudes


def build_ground_tracks_figure(
    satellites: dict[str, EarthSatellite],
    ts: Timescale,
    start_time: Time | None = None,
    duration_hours: float = 24,
    step_minutes: float = 1,
) -> Figure:
    """
    Build (but don't save or close) a matplotlib Figure plotting every
    satellite's ground track over the next `duration_hours` on a plain
    lat/lon grid.

    Split out of plot_ground_tracks() so a caller that needs the live
    Figure object rather than a saved file -- e.g. the Streamlit app,
    via st.pyplot(fig) -- can get one without duplicating this drawing
    code. plot_ground_tracks() itself is now a thin wrapper: build the
    figure, save it, close it.

    Deliberately no coastlines/continent outlines here: cartopy (the
    usual way to get those in matplotlib) can be a pain to install on
    Windows, and a plain 30-degree lat/lon grid with axis labels and a
    legend is a perfectly fine first version. Cartopy remains an option
    for a later polish pass if it installs cleanly.

    All satellites share the same start_time so the tracks are directly
    comparable on one plot.
    """
    if start_time is None:
        start_time = ts.now()

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, sat in satellites.items():
        track = compute_ground_track(
            sat, ts, start_time=start_time,
            duration_hours=duration_hours, step_minutes=step_minutes,
        )
        # track's lat/lon fields are typed float | NDArray (see
        # propagation.FloatOrArray), but compute_ground_track() is always
        # given a vectorized `start_time`/grid here, so these are always
        # arrays in practice -- cast() tells mypy that, as a no-op.
        longitude_deg = cast(NDArray[np.float64], track["longitude_deg"])
        latitude_deg = cast(NDArray[np.float64], track["latitude_deg"])
        lon, lat = _split_at_antimeridian(longitude_deg, latitude_deg)
        ax.plot(lon, lat, linewidth=1, label=name)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(
        "Ground tracks -- next {}h from {}".format(
            duration_hours, start_time.utc_strftime("%Y-%m-%d %H:%M UTC")
        )
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    return fig


def plot_ground_tracks(
    satellites: dict[str, EarthSatellite],
    ts: Timescale,
    start_time: Time | None = None,
    duration_hours: float = 24,
    step_minutes: float = 1,
    output_path: str | None = None,
) -> str:
    """
    Build every satellite's ground-track plot (see
    build_ground_tracks_figure()) and save it as a PNG.

    Returns the path the PNG was saved to.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "ground_tracks.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig = build_ground_tracks_figure(
        satellites, ts, start_time=start_time,
        duration_hours=duration_hours, step_minutes=step_minutes,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


# Shared between print_passes_table() (below) and reporting.py's HTML
# table: (row key, display title, text-column width). The width is only
# meaningful for the fixed-width text rendering here -- an HTML table
# lays its own columns out and simply ignores it -- but keeping one
# authoritative (key, title) list is what actually guarantees the two
# renderings show identical columns in identical order, rather than two
# independently-maintained lists that could quietly drift apart.
PASS_TABLE_COLUMNS: list[tuple[str, str, int]] = [
    ("satellite", "Satellite", 12),
    ("start", "Start (UTC)", 19),
    ("start_az", "Start Az", 8),
    ("max_elev", "Max El", 6),
    ("max_elev_time", "Max El Time", 11),
    ("end", "End (UTC)", 19),
    ("end_az", "End Az", 7),
    ("duration_min", "Dur (min)", 9),
    ("notes", "Notes", 40),
]


def build_pass_rows(
    passes_by_satellite: dict[str, list[PassDict]],
) -> list[dict[str, str | Time | PassDict]]:
    """
    Flatten passes_by_satellite into one row per pass, with every field
    pre-formatted to a display string (matching PASS_TABLE_COLUMNS) and
    sorted by start time -- the shared data prep behind both
    print_passes_table()'s text table and reporting.py's HTML table, so
    the two can't drift out of sync with each other.

    Local dict keys, not a PassDict-style TypedDict: this is a display-
    formatting intermediate (every value coerced to `str`, plus
    "_sort_key"/"_satellite"/"_pass" fields that aren't part of either
    rendering), not one of the fixed data contracts passed between the
    package's logic modules. "_satellite" and "_pass" carry the row's
    original satellite name and raw PassDict through -- app.py's sky
    plot needs the underlying pass (for its exact start/end times) once
    a user selects a table row, and this is the one place that already
    knows which PassDict a given display row came from; recovering that
    mapping independently at the call site would mean re-deriving this
    same sort order there too.
    """
    rows: list[dict[str, str | Time | PassDict]] = []
    for name, passes in passes_by_satellite.items():
        for p in passes:
            notes: list[str] = []
            if p["start_truncated"]:
                notes.append("IN PROGRESS AT START")
            if p["end_truncated"]:
                notes.append("CUT OFF AT END")
            if p["max_elevation_truncated"]:
                notes.append("MAX MAY BE HIGHER (still rising at cutoff)")
            if p["low_confidence"]:
                notes.append("LOW CONFIDENCE (rerun with finer step)")

            rows.append({
                "satellite": name,
                "start": p["start_time"].utc_strftime("%Y-%m-%d %H:%M:%S"),
                "start_az": f"{p['start_azimuth_deg']:.1f}",
                "max_elev": f"{p['max_elevation_deg']:.1f}",
                "max_elev_time": p["max_elevation_time"].utc_strftime("%H:%M:%S"),
                "end": p["end_time"].utc_strftime("%Y-%m-%d %H:%M:%S"),
                "end_az": f"{p['end_azimuth_deg']:.1f}",
                "duration_min": f"{p['duration_minutes']:.1f}",
                "notes": ", ".join(notes),
                "_sort_key": p["start_time"],
                "_satellite": name,
                "_pass": p,
            })

    rows.sort(key=lambda r: cast(Time, r["_sort_key"]))
    return rows


def print_passes_table(passes_by_satellite: dict[str, list[PassDict]]) -> None:
    """
    Print one row per detected pass across all satellites, sorted by
    start time, as a plain fixed-width text table.
    """
    rows = build_pass_rows(passes_by_satellite)

    if not rows:
        print("No passes above threshold in this window.")
        return

    header = "  ".join(f"{title:<{width}}" for _, title, width in PASS_TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{row[key]:<{width}}" for key, _, width in PASS_TABLE_COLUMNS))
