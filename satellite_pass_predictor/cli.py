"""CLI: runs the full pipeline end to end for the configured satellites --
load TLEs, print positions, save a ground-track plot, compute Kourou
passes, and write a combined HTML report."""

import argparse
import logging
from pathlib import Path

from skyfield.api import load

from .config import (
    KOUROU,
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    MIN_PASS_ELEVATION_DEG,
    OUTPUT_DIR,
)
from .propagation import get_subpoint
from .reporting import generate_html_report
from .tle_data import load_satellites
from .visibility import compute_passes
from .visualization import plot_ground_tracks, print_passes_table

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline once, printing and saving its outputs. Returns the process exit code.

    A satellite that fails to load is logged as a warning and skipped; the run fails (exit
    code 1) only if none load. `--source mirror` reads the GitHub Actions TLE mirror instead
    of Celestrak directly.
    """
    parser = argparse.ArgumentParser(description="Satellite Pass Predictor -- CLI")
    parser.add_argument(
        "--source",
        choices=["celestrak", "mirror"],
        default="celestrak",
        help="Where to fetch TLEs from (default: celestrak).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Directory for the plot and report (default: ./%(default)s).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Failures and stale-TLE warnings are logged by load_satellites().
    satellites = load_satellites(source=args.source).satellites
    if not satellites:
        logger.error("No satellites could be loaded; nothing to do.")
        return 1

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

    output_path = plot_ground_tracks(
        satellites, ts, start_time=t, output_path=str(args.output_dir / "ground_tracks.png")
    )
    print(f"Ground track plot saved to {output_path}")

    print(
        f"\nVisibility passes over Kourou "
        f"({KOUROU_LATITUDE_DEG:.4f}, {KOUROU_LONGITUDE_DEG:.4f}) "
        f"in the next 24h, elevation >= {MIN_PASS_ELEVATION_DEG:.0f} deg:\n"
    )
    passes_by_satellite = {
        name: compute_passes(sat, KOUROU, ts, start_time=t) for name, sat in satellites.items()
    }
    print_passes_table(passes_by_satellite)

    report_path = generate_html_report(
        output_path,
        passes_by_satellite,
        generated_at=t,
        output_path=str(args.output_dir / "report.html"),
    )
    print(f"\nCombined HTML report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
