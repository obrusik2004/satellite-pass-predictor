"""
Configuration: the satellites tracked, the Celestrak endpoint and TLE
caching policy, the Kourou observer location, and the constants that tune
pass detection. No logic lives here -- just the values other modules
import and use.
"""

from skyfield.api import wgs84

# Satellites tracked by this tool, identified by NORAD Catalog Number
# (the stable numeric ID Celestrak, Space-Track etc. all key on -- names
# are not a reliable lookup key since they change over a mission's life).
#
# ISS (ZARYA)  - ~400 km orbit, ~90 min period. Extremely well tracked,
#                which makes it the easiest case to sanity-check against
#                Heavens-Above / N2YO before trusting anything else here.
# SWISSCUBE    - 1U CubeSat, Switzerland's first satellite (launched 2009).
# BEESAT-1     - 1U CubeSat, TU Berlin (launched on the same 2009 rideshare
#                as SwissCube). Two satellites launched together into
#                similar orbits, ~17 years ago -- a natural pair for later
#                comparing how much their orbits have diverged under drag.
# MICROSCOPE   - CNES microsatellite (~300 kg, Myriade-class), launched from
#                Kourou on Soyuz VS14 in April 2016. Flew twin accelerometers
#                to test Einstein's Weak Equivalence Principle to ~1e-15
#                precision -- the most precise test of it ever flown -- and
#                ties directly into this project's Kourou/ESA theme.
#                (NB: EyeSat, NORAD 44877, also launched from Kourou via
#                CNES and would have been a nice small-CubeSat comparison,
#                but it decayed 2023-11-19 and Celestrak has no current
#                elements for it -- confirmed via CATNR lookup before
#                ruling it out.)
SATELLITES = {
    "ISS (ZARYA)": 25544,
    "SWISSCUBE": 35932,
    "BEESAT-1": 35933,
    "MICROSCOPE": 41457,
}

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?CATNR={norad_id}&FORMAT=TLE"
)

# How long a cached TLE is trusted before we bother re-downloading it.
# TLEs are only accurate for a matter of days (drag perturbations aren't
# modeled by SGP4), so we don't want to cache forever -- but we also
# don't want to hit Celestrak's servers on every single run.
MAX_TLE_AGE_DAYS = 1.0

TLE_CACHE_DIR = "data"
OUTPUT_DIR = "output"

# Guiana Space Centre (Centre Spatial Guyanais), Kourou, French Guiana --
# the ESA/CNES/Arianespace launch site this whole project is themed
# around. Coordinates per the Guiana Space Centre's Wikipedia infobox
# (5°10'08"N 52°41'25"W -> 5.169, -52.6903).
#
# Worth being upfront about: the CSG complex isn't a single point, it's
# tens of km of coastline hosting several separate launch pads (Ariane 6,
# Vega, Soyuz), and different official sources cite slightly different
# reference coordinates for "Kourou" as a result (ESA's own spaceport
# page cites 5°3'N for the general area, ~13 km south of the Wikipedia
# point). At the range of a satellite hundreds of km up, moving the
# observer by a few tens of km changes computed elevation by a small
# fraction of a degree -- negligible next to our 10-degree pass
# threshold. So one specific, citable reference point is used for
# reproducibility, not because sub-km precision matters for this
# calculation.
KOUROU_LATITUDE_DEG = 5.169
KOUROU_LONGITUDE_DEG = -52.6903
KOUROU_ELEVATION_M = 0  # coastal, effectively sea level; irrelevant here

KOUROU = wgs84.latlon(
    KOUROU_LATITUDE_DEG, KOUROU_LONGITUDE_DEG, elevation_m=KOUROU_ELEVATION_M
)

# A satellite is considered "visible" for pass-prediction purposes once
# it's at least this many degrees above the horizon -- low elevations are
# usually unusable anyway (obstructions, atmospheric extinction), and
# 10 degrees is the conventional default for both amateur satellite
# tracking and this kind of pass table.
MIN_PASS_ELEVATION_DEG = 10.0

# A detected pass spanning fewer samples than this has its start/end/peak
# resolved only to within one sampling step, since we don't know what
# happened *between* samples -- see find_passes() for how this is used.
MIN_PASS_SAMPLES_FOR_CONFIDENCE = 3
