"""
Shared pytest fixtures. Nothing in here touches the network: `ts` uses
Skyfield's bundled leap-second/delta-T data (no download needed for
load.timescale()), and `iss_satellite` builds an EarthSatellite directly
from a frozen TLE fixture file rather than fetching from Celestrak.
"""

from pathlib import Path

import pytest
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Timescale

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def ts() -> Timescale:
    """A Skyfield timescale, shared across tests -- cheap to build once."""
    return load.timescale()


@pytest.fixture(scope="session")
def iss_satellite(ts: Timescale) -> EarthSatellite:
    """
    ISS built from a frozen TLE fixture (tests/fixtures/iss_tle.txt),
    epoch 2026-09-20 ~18:54 UTC -- NOT a live Celestrak fetch. Frozen so
    these tests are deterministic and don't depend on network access or
    on the TLE still being "fresh" whenever someone runs the suite.
    """
    name, line1, line2 = (FIXTURES_DIR / "iss_tle.txt").read_text().splitlines()
    return EarthSatellite(line1, line2, name.strip(), ts)
