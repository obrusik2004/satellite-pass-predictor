"""
Visualization: turning computed tracks and passes into human-readable
output -- the ground track plot (PNG) and the pass table (printed text).
"""

import os
from typing import cast

# Must run before `import matplotlib.pyplot` (or anything that imports it
# transitively) actually resolves a backend -- matplotlib picks one
# automatically on first pyplot import, and that auto-detection can fail
# outright, not just misbehave, on a truly headless target like Streamlit
# Community Cloud's Linux container (no display server, no GUI toolkit at
# all). This isn't hypothetical: this project already hit a real
# backend problem locally (a broken Tcl/Tk install picked by
# auto-detection on this dev machine -- see conftest.py's own Agg
# override for the test suite), which is exactly the class of failure
# that must not be left to chance for the actual deployed app/CLI, where
# there's no equivalent override in place upstream of this call. Agg is
# the standard non-interactive, render-to-memory/file backend -- all
# this module needs (a Figure returned or saved to PNG), never a
# GUI window.
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
    """
    Insert NaN breaks wherever consecutive longitude samples jump across
    the +180/-180 antimeridian (International Date Line) -- see
    geo.find_antimeridian_crossings() for how a crossing is detected
    (shared with globe.py's own antimeridian fix, which needs a
    different consumption of the same crossing indices -- see that
    module for why).

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

    Dark theme (THEME_BACKGROUND_COLOR etc., see config.py), not
    matplotlib's default white figure -- checked directly, not assumed:
    two of the four SATELLITE_COLORS (SWISSCUBE's light blue, MICROSCOPE's
    yellow) are pastel enough that on white they measure 1.5-1.8:1 WCAG
    contrast, well under even the lenient 3:1 non-text minimum -- visibly
    washed out, not just theoretically under-contrasted. Every satellite
    color instead measures 5.6-11.5:1 against this dark background, so
    the fix is the background (per this function's own docstring
    guidance on this exact trade-off), not the fixed, exact-value
    satellite palette. This is the one shared figure both the Streamlit
    app and main.py/the CLI/reporting.py's (still light-themed) HTML
    report use -- the dark output is a direct consequence of fixing this
    contrast problem for every caller, not a change scoped to the
    Streamlit app alone.
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
        # track's lat/lon fields are typed float | NDArray (see
        # propagation.FloatOrArray), but compute_ground_track() is always
        # given a vectorized `start_time`/grid here, so these are always
        # arrays in practice -- cast() tells mypy that, as a no-op.
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
    """
    Build every satellite's ground-track plot (see
    build_ground_tracks_figure()) and save it as a PNG.

    Returns the path the PNG was saved to.
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


# (code, plain-English meaning) for every flag build_pass_note_codes()
# below can emit -- the single source of truth for both that function
# and the legend caption the Streamlit app shows alongside its table
# (see app.py). Order matches build_pass_note_codes()'s own check
# order, which in turn matches build_pass_rows()'s full-text notes
# above -- kept in sync by test_visualization.py rather than by a
# shared runtime structure, since with only four fixed flags a data-
# driven abstraction here would be more indirection than the four
# straight-line checks below are worth.
NOTE_CODE_LEGEND: list[tuple[str, str]] = [
    ("IP", "in progress at start of window -- true rise time unknown"),
    ("CE", "cut off at end of window -- true set time unknown"),
    ("MH", "max elevation may be higher -- was still rising when the window ended"),
    ("LC", "low confidence -- too few samples; rerun with a finer step"),
]


def build_pass_note_codes(pass_: PassDict) -> str:
    """
    Compact comma-joined codes (IP/CE/MH/LC, see NOTE_CODE_LEGEND) for
    the flags set on `pass_` -- a narrow-column-friendly alternative to
    build_pass_rows()'s full descriptive "notes" text.

    build_pass_rows()'s own "notes" field is left as full text: it's
    shared by print_passes_table()'s plain-text CLI table and
    reporting.py's HTML report, both of which have room for a full
    sentence and no legend to pair codes against. The Streamlit app's
    interactive table is the odd one out -- a fixed-width column that
    was either truncating that full text or, sized to fit it, made the
    rest of the table look lopsided (both tried directly, see app.py) --
    so it uses these codes plus a legend caption instead.
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
