"""
Visualization: turning computed tracks and passes into human-readable
output -- the ground track plot (PNG) and the pass table (printed text).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR
from .propagation import compute_ground_track


def _split_at_antimeridian(longitudes, latitudes):
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


def plot_ground_tracks(satellites, ts, start_time=None, duration_hours=24,
                        step_minutes=1, output_path=None):
    """
    Plot every satellite's ground track over the next `duration_hours` on
    a plain lat/lon grid and save it as a PNG.

    Deliberately no coastlines/continent outlines here (see README) --
    just a 30-degree lat/lon grid, axis labels, and a legend. All
    satellites share the same start_time so the tracks are directly
    comparable on one plot.
    """
    if start_time is None:
        start_time = ts.now()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "ground_tracks.png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, sat in satellites.items():
        track = compute_ground_track(
            sat, ts, start_time=start_time,
            duration_hours=duration_hours, step_minutes=step_minutes,
        )
        lon, lat = _split_at_antimeridian(
            track["longitude_deg"], track["latitude_deg"]
        )
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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def print_passes_table(passes_by_satellite):
    """
    Print one row per detected pass across all satellites, sorted by
    start time, as a plain fixed-width text table.
    """
    rows = []
    for name, passes in passes_by_satellite.items():
        for p in passes:
            notes = []
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
            })

    if not rows:
        print("No passes above threshold in this window.")
        return

    rows.sort(key=lambda r: r["_sort_key"])

    columns = [
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

    header = "  ".join(f"{title:<{width}}" for _, title, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{row[key]:<{width}}" for key, _, width in columns))
