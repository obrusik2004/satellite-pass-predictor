"""
Satellite Pass Predictor
=========================
Thin orchestrator for the satellite_pass_predictor package: runs the
full pipeline end to end -- load TLEs, print each satellite's current
position, save a 24h ground-track plot, then compute and print
visibility passes over Kourou. All the actual logic lives in the
package's modules (config, tle_data, propagation, time_utils,
visibility, visualization); this file only sequences calls into them.
"""

from skyfield.api import load

from satellite_pass_predictor.config import (
    KOUROU,
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    MIN_PASS_ELEVATION_DEG,
)
from satellite_pass_predictor.propagation import get_subpoint
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
    track plot, then compute and print Kourou visibility passes.
    Deliberately holds no logic of its own -- see the package modules
    (imported above) for that.
    """
    satellites = load_satellites()

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


if __name__ == "__main__":
    main()
