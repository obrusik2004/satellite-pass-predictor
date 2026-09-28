"""
Satellite Pass Predictor
=========================
Thin orchestrator for the satellite_pass_predictor package: runs the
full pipeline end to end -- load TLEs, print each satellite's current
position, save a 24h ground-track plot, compute and print visibility
passes over Kourou, then combine the plot and pass table into a single
self-contained HTML report. All the actual logic lives in the package's
modules (config, tle_data, propagation, time_utils, visibility,
visualization, reporting); this file only sequences calls into them.
"""

import argparse

from skyfield.api import load

from satellite_pass_predictor.config import (
    KOUROU,
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    MIN_PASS_ELEVATION_DEG,
)
from satellite_pass_predictor.propagation import get_subpoint
from satellite_pass_predictor.reporting import generate_html_report
from satellite_pass_predictor.tle_data import load_satellites
from satellite_pass_predictor.visibility import compute_passes
from satellite_pass_predictor.visualization import (
    plot_ground_tracks,
    print_passes_table,
)


def main() -> None:
    """
    Run the full pipeline for the configured satellites, in order: load
    TLEs, print each satellite's current position, save a 24h ground-
    track plot, compute and print Kourou visibility passes, then combine
    the plot and pass table into one HTML report. Deliberately holds no
    logic of its own -- see the package modules (imported above) for
    that.

    `--source` defaults to "celestrak" (fetch directly from Celestrak's
    own endpoint) -- the normal choice here, and the one that actually
    works outside Streamlit Cloud (see tle_data.load_satellites()'s
    `source` parameter and TLE_MIRROR_URL's comment in config.py for why
    the Streamlit app can't use this same default). `--source mirror`
    reads the GitHub-Actions-refreshed tle-data branch mirror instead --
    mainly useful for testing that pipeline locally without deploying.
    """
    parser = argparse.ArgumentParser(description="Satellite Pass Predictor -- CLI")
    parser.add_argument(
        "--source", choices=["celestrak", "mirror"], default="celestrak",
        help="Where to fetch TLEs from (default: celestrak).",
    )
    args = parser.parse_args()

    satellites = load_satellites(source=args.source)

    print(f"Loaded {len(satellites)} satellite(s):\n")
    for name, sat in satellites.items():
        print(f"{name}  (NORAD {sat.model.satnum})")
        print(f"  TLE epoch: {sat.epoch.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

    ts = load.timescale()
    t = ts.now()
    print(f"Current positions at {t.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}:\n")
    for name, sat in satellites.items():
        pos = get_subpoint(sat, t)
        print(f"{name}")
        print(f"  latitude:  {pos['latitude_deg']:+.4f} deg")
        print(f"  longitude: {pos['longitude_deg']:+.4f} deg")
        print(f"  altitude:  {pos['altitude_km']:.1f} km")
        print()

    output_path = plot_ground_tracks(satellites, ts, start_time=t)
    print(f"Ground track plot saved to {output_path}")

    print(
        f"\nVisibility passes over Kourou "
        f"({KOUROU_LATITUDE_DEG:.4f}, {KOUROU_LONGITUDE_DEG:.4f}) "
        f"in the next 24h, elevation >= {MIN_PASS_ELEVATION_DEG:.0f} deg:\n"
    )
    passes_by_satellite = {
        name: compute_passes(sat, KOUROU, ts, start_time=t)
        for name, sat in satellites.items()
    }
    print_passes_table(passes_by_satellite)

    report_path = generate_html_report(output_path, passes_by_satellite, generated_at=t)
    print(f"\nCombined HTML report saved to {report_path}")


if __name__ == "__main__":
    main()
