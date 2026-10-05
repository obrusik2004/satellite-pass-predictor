"""Shared fixtures. Nothing here touches the network."""

from pathlib import Path

import matplotlib
import pytest
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Timescale

# Agg renders to memory only, so tests never depend on a GUI toolkit being installed.
matplotlib.use("Agg")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def ts() -> Timescale:
    """A Skyfield timescale, built once."""
    return load.timescale()


@pytest.fixture(scope="session")
def iss_satellite(ts: Timescale) -> EarthSatellite:
    """The ISS from a frozen TLE (epoch 2026-09-20), so tests are deterministic."""
    name, line1, line2 = (FIXTURES_DIR / "iss_tle.txt").read_text().splitlines()
    return EarthSatellite(line1, line2, name.strip(), ts)
