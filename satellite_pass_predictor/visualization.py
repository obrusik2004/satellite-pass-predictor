"""Human-readable output: the flat ground-track plot (PNG) and the pass table (text)."""

import os
from typing import cast

# Select the non-interactive backend before pyplot is imported: auto-detection can fail on
# headless hosts (no display, no GUI toolkit). Only Figures and PNG files are needed here.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Time, Timescale

from .config import (
    OUTPUT_DIR,
    SATELLITE_COLORS,
    THEME_BACKGROUND_COLOR,
    THEME_BORDER_COLOR,
    THEME_TEXT_COLOR,
)
from .geo import find_antimeridian_crossings
from .propagation import compute_ground_track
from .visibility import PassDict


def _split_at_antimeridian(
    longitudes: NDArray[np.float64], latitudes: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Insert NaN breaks at antimeridian crossings so no line is drawn across the whole map."""
    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)

    jumps = find_antimeridian_crossings(longitudes)
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
    """Plot every satellite's ground track on a lat/lon grid and return the open Figure.

    All satellites share `start_time`, so the tracks are comparable. The dark theme keeps the
    pastel satellite colors legible. Used by plot_ground_tracks() for the CLI and HTML report.
    """
    if start_time is None:
        start_time = ts.now()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(THEME_BACKGROUND_COLOR)
    ax.set_facecolor(THEME_BACKGROUND_COLOR)

    for name, sat in satellites.items():
        track = compute_ground_track(
            sat,
            ts,
            start_time=start_time,
            duration_hours=duration_hours,
            step_minutes=step_minutes,
        )
        # The grid is always an array, so the scalar half of the union never applies here.
        longitude_deg = cast(NDArray[np.float64], track["longitude_deg"])
        latitude_deg = cast(NDArray[np.float64], track["latitude_deg"])
        lon, lat = _split_at_antimeridian(longitude_deg, latitude_deg)
        ax.plot(lon, lat, linewidth=1.5, label=name, color=SATELLITE_COLORS[name])

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color=THEME_BORDER_COLOR)
    ax.set_xlabel("Longitude (deg)", color=THEME_TEXT_COLOR)
    ax.set_ylabel("Latitude (deg)", color=THEME_TEXT_COLOR)
    ax.set_title(
        "Ground tracks -- next {}h from {}".format(
            duration_hours, start_time.utc_strftime("%Y-%m-%d %H:%M UTC")
        ),
        color=THEME_TEXT_COLOR,
    )
    ax.tick_params(colors=THEME_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(THEME_BORDER_COLOR)
    legend = ax.legend(
        loc="upper right",
        fontsize=8,
        facecolor=THEME_BACKGROUND_COLOR,
        edgecolor=THEME_BORDER_COLOR,
    )
    for text in legend.get_texts():
        text.set_color(THEME_TEXT_COLOR)
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
    """Save the ground-track figure as a PNG (default: OUTPUT_DIR/ground_tracks.png).

    Returns the path written.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "ground_tracks.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig = build_ground_tracks_figure(
        satellites,
        ts,
        start_time=start_time,
        duration_hours=duration_hours,
        step_minutes=step_minutes,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


# (row key, display title, text-column width). One list drives both the text table and the
# HTML table so their columns can't drift; the HTML table ignores the width.
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
    """Flatten passes into one display row per pass, sorted by start time.

    Shared by the text table, the HTML report and the app. Row values for PASS_TABLE_COLUMNS
    keys are formatted strings; "_sort_key", "_satellite" and "_pass" carry the raw data so
    a selected row can be traced back to its PassDict.
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

            rows.append(
                {
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
                }
            )

    rows.sort(key=lambda r: cast(Time, r["_sort_key"]))
    return rows


# (code, meaning) for each flag build_pass_note_codes() can emit, in its check order.
# Kept in sync with that function by tests.
NOTE_CODE_LEGEND: list[tuple[str, str]] = [
    ("IP", "in progress at start of window -- true rise time unknown"),
    ("CE", "cut off at end of window -- true set time unknown"),
    ("MH", "max elevation may be higher -- was still rising when the window ended"),
    ("LC", "low confidence -- too few samples; rerun with a finer step"),
]


def build_pass_note_codes(pass_: PassDict) -> str:
    """Return the comma-joined note codes (see NOTE_CODE_LEGEND) set on `pass_`.

    A compact alternative to build_pass_rows()'s full-text notes for the app's narrow column.
    """
    codes: list[str] = []
    if pass_["start_truncated"]:
        codes.append("IP")
    if pass_["end_truncated"]:
        codes.append("CE")
    if pass_["max_elevation_truncated"]:
        codes.append("MH")
    if pass_["low_confidence"]:
        codes.append("LC")
    return ", ".join(codes)


def print_passes_table(passes_by_satellite: dict[str, list[PassDict]]) -> None:
    """Print all passes, sorted by start time, as a fixed-width text table."""
    rows = build_pass_rows(passes_by_satellite)

    if not rows:
        print("No passes above threshold in this window.")
        return

    header = "  ".join(f"{title:<{width}}" for _, title, width in PASS_TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{row[key]:<{width}}" for key, _, width in PASS_TABLE_COLUMNS))
