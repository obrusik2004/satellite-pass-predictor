"""
Configuration: the satellites tracked, the Celestrak endpoint and TLE
caching policy, the Kourou observer location, and the constants that tune
pass detection. No logic lives here -- just the values other modules
import and use.
"""

from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition

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
SATELLITES: dict[str, int] = {
    "ISS (ZARYA)": 25544,
    "SWISSCUBE": 35932,
    "BEESAT-1": 35933,
    "MICROSCOPE": 41457,
}

# Per-satellite display color -- the single source of truth every
# renderer that draws a satellite reads from (globe.py's pydeck tracks/
# markers, visualization.py's matplotlib ground-track figure,
# skyplot.py's sky track, and app.py's passes table/legend), so the same
# satellite can never show up in a different color in one view than
# another. Hex strings: matplotlib and Plotly both take them directly;
# globe.py and reporting.py convert to RGB where they need it (pydeck
# layers want [r, g, b] triplets, the HTML report wants an rgba() value)
# via the shared colors.hex_to_rgb() helper, rather than this module
# keeping several parallel color representations.
#
# Sampled directly from esa.int, alongside the rest of the ESA-inspired
# palette below ("Visual theme") -- not an arbitrary color-cycle pick.
SATELLITE_COLORS: dict[str, str] = {
    "ISS (ZARYA)": "#F1666A",
    "SWISSCUBE": "#6DCFF6",
    "BEESAT-1": "#76C8AE",
    "MICROSCOPE": "#FFCC4E",
}

# Visual theme -- an ESA-inspired (esa.int) dark navy palette, used by
# the Streamlit app. Deliberately inspired by, not a copy of, ESA's own
# branding: no logo, no "ESA" name/wordmark anywhere (see README).
#
# The *same* values are also set in .streamlit/config.toml, which
# Streamlit itself reads for almost all theming (backgrounds, borders,
# fonts, widget colors). These Python constants exist only for the two
# renderers config.toml can't reach: globe.py's pydeck layers (pydeck
# renders in its own canvas/iframe with no access to the page's theme)
# and visualization.py's matplotlib figure (also used standalone by
# main.py/the CLI and reporting.py's HTML report, entirely outside
# Streamlit). Kept in sync with config.toml by hand -- there's no single
# file both a TOML parser and Python code can read.
THEME_BACKGROUND_COLOR: str = "#0B1D26"  # page background
THEME_PANEL_COLOR: str = "#003247"  # header bar / sidebar
THEME_BORDER_COLOR: str = "#335E6F"  # dividers / borders

# Light text color for both backgrounds above -- checked directly
# (WCAG contrast ratios computed, not eyeballed), not assumed readable
# just because it's light-on-dark: 15.2:1 against THEME_BACKGROUND_COLOR
# and 12.0:1 against THEME_PANEL_COLOR, both far past the 4.5:1 WCAG AA
# minimum for body text.
THEME_TEXT_COLOR: str = "#EAF2F5"

# Ground station marker on the globe -- white, not the gold an earlier
# version of this app used: gold/yellow is now MICROSCOPE's own color
# (SATELLITE_COLORS above), and Kourou is a ground station, not a
# satellite, so it needs a color that can't be mistaken for either a
# satellite track or a selection highlight.
KOUROU_MARKER_COLOR: str = "#FFFFFF"

CELESTRAK_URL: str = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"

# Streamlit Community Cloud's outbound network cannot reach celestrak.org
# at all -- confirmed directly (TCP connect timeouts, not an HTTP error
# or anything our own retry/timeout handling could paper over), while a
# GitHub Actions runner reachability test against the same URL succeeded
# immediately. So the deployed app doesn't fetch from Celestrak itself:
# a scheduled GitHub Actions workflow (.github/workflows/refresh-tles.yml,
# scripts/fetch_tles.py) fetches from Celestrak and republishes the
# results as plain files on the `tle-data` branch of this same repo,
# which Streamlit Cloud *can* reach (raw.githubusercontent.com is a
# generic file host, not Celestrak's own infrastructure). See README's
# "Data pipeline" section for the full picture. main.py/the CLI still
# fetches from Celestrak directly by default (that direct path works
# fine outside Streamlit Cloud); the app and an explicit
# `main.py --source mirror` use this instead -- see
# tle_data.load_satellites()'s `source` parameter.
TLE_MIRROR_URL: str = (
    "https://raw.githubusercontent.com/obrusik2004/satellite-pass-predictor"
    "/tle-data/tle/tle_{norad_id}.txt"
)
TLE_MIRROR_METADATA_URL: str = (
    "https://raw.githubusercontent.com/obrusik2004/satellite-pass-predictor"
    "/tle-data/tle/metadata.json"
)

# How long a cached TLE is trusted before we bother re-downloading it.
# TLEs are only accurate for a matter of days (drag perturbations aren't
# modeled by SGP4), so we don't want to cache forever -- but we also
# don't want to hit Celestrak's servers on every single run.
MAX_TLE_AGE_DAYS: float = 1.0

# The mirror's own on-disk cache needs a *much* shorter age than
# MAX_TLE_AGE_DAYS above: the mirror is refreshed by GitHub Actions every
# 6 hours (see refresh-tles.yml), and a full day-long local cache would
# mean up to ~23 hours of that freshly-published data sitting unused
# before the app would even check for it again. 1 hour keeps the app
# checking often enough that a refresh actually reaches users within
# about an hour of landing on the mirror, without re-fetching the mirror
# file on literally every single Streamlit rerun. app.py's
# st.cache_resource TTL for the loaded satellites is set from this same
# constant (not a separately-chosen number) specifically so the two
# can't drift apart -- see app.py.
TLE_MIRROR_CACHE_AGE_HOURS: float = 1.0

# Retries for the actual network fetch (not the empty/malformed-response
# handling, which is a different failure mode with its own explicit
# ValueError -- see tle_data.py). Originally added on a theory that
# Streamlit Community Cloud's outbound networking was intermittently
# flaky -- since corrected by direct investigation: Streamlit Cloud
# cannot reach celestrak.org *at all* (consistent TCP connect timeouts,
# not an occasional blip), which no amount of retrying from inside that
# environment can fix -- that's the actual reason for the mirror
# pipeline (see TLE_MIRROR_URL above), not this retry loop. This retry
# loop is still worth keeping for genuinely transient failures on
# whichever source is actually reachable -- a dropped connection to the
# mirror, or to Celestrak itself from main.py's direct local fetch --
# just not as a fix for a host that's unreachable outright. 3 retries
# (4 attempts total) with delays doubling from 1s->2s->4s between them,
# plus TLE_FETCH_TIMEOUT_SECONDS bounding each individual attempt (see
# that constant -- without it, a single hung attempt could run far
# longer than this backoff math alone would suggest). Worst case if
# every attempt genuinely hangs the full timeout: 4 *
# TLE_FETCH_TIMEOUT_SECONDS (attempts) + 1+2+4 (backoff between them) =
# 47s -- not fast, but bounded and finite, for what should be a rare
# case; enough attempts to absorb a brief blip without turning this
# into an unbounded retry loop for a genuine, persistent outage.
TLE_FETCH_MAX_RETRIES: int = 3
TLE_FETCH_RETRY_BASE_DELAY_SECONDS: float = 1.0

# Bounds each individual fetch attempt (celestrak.org or the mirror) via
# requests.get(timeout=...) -- a genuinely per-call, thread-safe timeout
# (see tle_data.py's _download_and_cache_tle() docstring for why this
# matters specifically under Streamlit's per-session-thread concurrency:
# an earlier version used socket.setdefaulttimeout(), a process-global
# setting that turned out to be unsafe there). Without an explicit
# timeout at all, a hung connection falls back to whatever the OS/TCP
# stack eventually does on its own (SYN retransmission exhaustion,
# commonly tens of seconds to a couple of minutes), not a short,
# predictable failure. 10s is generous for what this actually has to do
# -- DNS + TCP + TLS handshake plus downloading a TLE file that's only a
# few hundred bytes -- while still failing fast enough that
# TLE_FETCH_MAX_RETRIES retries add a bounded, known worst case rather
# than an open-ended one.
TLE_FETCH_TIMEOUT_SECONDS: float = 10.0

# Thresholds for app.py's st.warning banners -- both pure data-staleness
# checks (see tle_data.is_older_than()), not related to the on-disk
# cache ages above (those govern when to *refetch*; these govern when to
# *warn the user the data itself looks old*, regardless of why).
#
# The mirror is refreshed every 6h (refresh-tles.yml) -- if its last
# successful run is more than 24h old, at least 3-4 consecutive
# scheduled runs have been silently missed (the workflow stopped
# running, or GitHub auto-disabled it after 60 days of repo inactivity
# -- see README), which is worth surfacing rather than quietly serving
# aging data with no indication anything's wrong.
MIRROR_REFRESH_WARNING_HOURS: float = 24.0

# A TLE's epoch (when its orbital elements were valid) more than this
# many days old means the data itself is stale, independent of whether
# the mirror refresh pipeline is running on schedule -- Celestrak simply
# hasn't published a newer set yet (e.g. for a less-actively-tracked
# CubeSat) or the satellite's own tracking has lapsed. Set above
# MAX_TLE_AGE_DAYS (which only governs *our own* refetch cadence, once a
# day) to leave headroom for Celestrak's normal publish cadence -- even
# actively-tracked objects aren't always republished daily -- while still
# catching genuinely old data well before SGP4 accuracy (which degrades
# over days as unmodeled drag accumulates, per MAX_TLE_AGE_DAYS's own
# comment) becomes a real concern.
TLE_EPOCH_WARNING_DAYS: float = 5.0

TLE_CACHE_DIR: str = "data"
OUTPUT_DIR: str = "output"

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
KOUROU_LATITUDE_DEG: float = 5.169
KOUROU_LONGITUDE_DEG: float = -52.6903
KOUROU_ELEVATION_M: float = 0  # coastal, effectively sea level; irrelevant here

KOUROU: GeographicPosition = wgs84.latlon(
    KOUROU_LATITUDE_DEG, KOUROU_LONGITUDE_DEG, elevation_m=KOUROU_ELEVATION_M
)

# A satellite is considered "visible" for pass-prediction purposes once
# it's at least this many degrees above the horizon -- low elevations are
# usually unusable anyway (obstructions, atmospheric extinction), and
# 10 degrees is the conventional default for both amateur satellite
# tracking and this kind of pass table.
MIN_PASS_ELEVATION_DEG: float = 10.0

# A detected pass spanning fewer samples than this has its start/end/peak
# resolved only to within one sampling step, since we don't know what
# happened *between* samples -- see find_passes() for how this is used.
MIN_PASS_SAMPLES_FOR_CONFIDENCE: int = 3

# Attribution shown in the Streamlit app's sidebar footer (app.py).
# Fixed, literal constants -- never derived from any request, session,
# or other runtime input -- specifically because app.py renders them
# into raw HTML via unsafe_allow_html=True: that's only safe when
# everything going into it is a constant like these, not something a
# user could ever influence.
AUTHOR_NAME: str = "Aleksander Bruski"
GITHUB_REPO_URL: str = "https://github.com/obrusik2004/satellite-pass-predictor"
LINKEDIN_URL: str = "https://www.linkedin.com/in/aleksander-bruski-b47aa638a/"
COPYRIGHT_YEAR: int = 2026
