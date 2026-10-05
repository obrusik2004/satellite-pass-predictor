"""Configuration: tracked satellites, TLE sources and caching, observer location, UI constants."""

from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition

# Tracked satellites by NORAD catalog number (stable, unlike names).
# ISS is the easiest to sanity-check against public trackers; SwissCube and BEESAT-1 are 2009
# CubeSats; MICROSCOPE is a CNES microsatellite launched from Kourou in 2016.
SATELLITES: dict[str, int] = {
    "ISS (ZARYA)": 25544,
    "SWISSCUBE": 35932,
    "BEESAT-1": 35933,
    "MICROSCOPE": 41457,
}

# Single source of truth for a satellite's color in every renderer (see docs/DECISIONS.md).
SATELLITE_COLORS: dict[str, str] = {
    "ISS (ZARYA)": "#F1666A",
    "SWISSCUBE": "#6DCFF6",
    "BEESAT-1": "#76C8AE",
    "MICROSCOPE": "#FFCC4E",
}

# Dark navy theme. Streamlit reads the same values from .streamlit/config.toml; they are
# duplicated here for renderers that bypass it (pydeck, matplotlib, the HTML report).
# Keep the two in sync by hand.
THEME_BACKGROUND_COLOR: str = "#0B1D26"  # page background
THEME_PANEL_COLOR: str = "#003247"  # header bar / sidebar
THEME_BORDER_COLOR: str = "#335E6F"  # dividers / borders
THEME_TEXT_COLOR: str = "#EAF2F5"

# White so the ground station can't be mistaken for a satellite color.
KOUROU_MARKER_COLOR: str = "#FFFFFF"

# Globe view and time-window slider.
GLOBE_INITIAL_ZOOM: float = 1.7
GLOBE_TRACK_WIDTH_M: int = 15000
TIME_WINDOW_MIN_HOURS: int = 1
TIME_WINDOW_MAX_HOURS: int = 72
TIME_WINDOW_DEFAULT_HOURS: int = 24

CELESTRAK_URL: str = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"

# TLE mirror on this repo's `tle-data` branch, refreshed by a GitHub Actions workflow
# (see docs/DECISIONS.md).
TLE_MIRROR_URL: str = (
    "https://raw.githubusercontent.com/obrusik2004/satellite-pass-predictor"
    "/tle-data/tle/tle_{norad_id}.txt"
)
TLE_MIRROR_METADATA_URL: str = (
    "https://raw.githubusercontent.com/obrusik2004/satellite-pass-predictor"
    "/tle-data/tle/metadata.json"
)

# Age after which a cached Celestrak TLE is refetched. TLEs lose accuracy over days.
MAX_TLE_AGE_DAYS: float = 1.0

# Mirror cache age: much shorter, so a refresh (every 6 hours) reaches users within about an
# hour. The app's st.cache_resource TTL uses the same value.
TLE_MIRROR_CACHE_AGE_HOURS: float = 1.0

# Retry policy for TLE fetches: delays of 1s, 2s, 4s between 4 attempts, each bounded by
# the timeout.
TLE_FETCH_MAX_RETRIES: int = 3
TLE_FETCH_RETRY_BASE_DELAY_SECONDS: float = 1.0
TLE_FETCH_TIMEOUT_SECONDS: float = 10.0

# Staleness warning thresholds. The mirror refreshes every 6 hours, so 24h means several missed
# runs. A TLE epoch older than 5 days is stale regardless of the pipeline.
MIRROR_REFRESH_WARNING_HOURS: float = 24.0
TLE_EPOCH_WARNING_DAYS: float = 5.0

# A TLE older than this is refused: after about a month, SGP4 error from unmodeled drag can
# reach tens of kilometres, so predicted pass times are no longer meaningful.
TLE_EPOCH_MAX_DAYS: float = 30.0

# The app retries a failed TLE load after this long, instead of caching it for the full hour.
TLE_FAILED_LOAD_RETRY_SECONDS: int = 60

# Overrides the per-user TLE cache directory.
CACHE_DIR_ENV_VAR: str = "SATPASS_CACHE_DIR"

# Default CLI output directory, relative to the working directory.
OUTPUT_DIR: str = "output"

# Guiana Space Centre, Kourou (Wikipedia coordinates). The site spans tens of km and sources
# differ slightly; at satellite range that shifts elevation by a small fraction of a degree.
KOUROU_LATITUDE_DEG: float = 5.169
KOUROU_LONGITUDE_DEG: float = -52.6903
KOUROU_ELEVATION_M: float = 0  # coastal, effectively sea level

KOUROU: GeographicPosition = wgs84.latlon(
    KOUROU_LATITUDE_DEG, KOUROU_LONGITUDE_DEG, elevation_m=KOUROU_ELEVATION_M
)

# Minimum elevation for a satellite to count as visible (the conventional default mask).
MIN_PASS_ELEVATION_DEG: float = 10.0

# A pass with fewer samples than this is flagged low-confidence (times resolved to ~1 step).
MIN_PASS_SAMPLES_FOR_CONFIDENCE: int = 3

# Footer attribution. Must stay constants: app.py renders them as raw HTML.
AUTHOR_NAME: str = "Aleksander Bruski"
GITHUB_REPO_URL: str = "https://github.com/obrusik2004/satellite-pass-predictor"
LINKEDIN_URL: str = "https://www.linkedin.com/in/aleksander-bruski-b47aa638a/"
COPYRIGHT_YEAR: int = 2026
