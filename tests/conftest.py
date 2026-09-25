"""
Shared pytest fixtures. Nothing in here touches the network: `ts` uses
Skyfield's bundled leap-second/delta-T data (no download needed for
load.timescale()), and `iss_satellite` builds an EarthSatellite directly
from a frozen TLE fixture file rather than fetching from Celestrak.
"""

from pathlib import Path

import matplotlib
import pytest
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite
from skyfield.timelib import Timescale

# Force the non-interactive Agg backend before anything imports
# matplotlib.pyplot, rather than relying on whatever GUI toolkit happens
# to be installed. Discovered the hard way: this machine's default
# backend resolves to TkAgg, and its Tcl/Tk installation is broken
# (missing tcl8.6 library files) -- main.py and the Streamlit app never
# hit this because each only ever creates one figure per process, but
# the test suite creates figures across multiple tests in the same
# pytest process, and a second Tk-backed figure failed to open after
# the first one was closed. Tests should never depend on a GUI toolkit
# being present at all, on any machine -- Agg renders to memory/files
# only, which is all any test here needs. conftest.py is imported by
# pytest before any test module, so this takes effect early enough.
matplotlib.use("Agg")

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
