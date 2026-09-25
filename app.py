"""
Satellite Pass Predictor -- Streamlit app (skeleton).

Reuses satellite_pass_predictor's existing package logic end to end --
this file only wires widgets to it and renders the results. No orbital-
mechanics, propagation, or pass-detection logic lives here; see the
package for that.

Deliberately plain for this first version: a 2D matplotlib ground-track
plot and a raw pass table, not yet the target visualization. The point
of this step is validating the whole data flow end to end -- satellite
selection -> TLE loading -> propagation -> pass detection -> display --
recomputed live on every widget interaction (Streamlit reruns this
whole script top to bottom on every interaction; that's expected and is
exactly what makes this different from the static HTML report).
"""

import streamlit as st
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite

from satellite_pass_predictor.config import (
    KOUROU,
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    MAX_TLE_AGE_DAYS,
    MIN_PASS_ELEVATION_DEG,
    SATELLITES,
)
from satellite_pass_predictor.tle_data import load_satellites
from satellite_pass_predictor.visibility import compute_passes
from satellite_pass_predictor.visualization import (
    PASS_TABLE_COLUMNS,
    build_ground_tracks_figure,
    build_pass_rows,
)

st.set_page_config(page_title="Satellite Pass Predictor", layout="wide")


# Streamlit reruns this entire script on every widget interaction, so
# without caching, load_satellites() -- real file I/O, and potentially a
# Celestrak fetch -- would run again every time someone moves a slider.
#
# st.cache_resource, not st.cache_data: cache_data pickles its return
# value (to hand back a safe copy and guard against a caller mutating
# the shared cached object), and EarthSatellite isn't picklable -- it
# wraps a `Satrec` object from the underlying sgp4 C extension, and
# `pickle.dumps()` on one raises "cannot pickle 'Satrec' object"
# (verified directly before writing this). cache_resource is Streamlit's
# decorator for exactly this case: a shared resource that doesn't
# serialize, kept as one live instance in memory rather than copied.
# That's safe here since nothing in this app mutates the returned
# EarthSatellite objects.
#
# TTL matches config.MAX_TLE_AGE_DAYS exactly: that's the point at which
# load_satellites() itself would already consider its on-disk TLE cache
# stale and refetch from Celestrak, so there's nothing to gain from
# expiring this cache any sooner (a call within that window would just
# re-read the same on-disk cache and return an equivalent result), and
# no reason to hold it longer than the data's own staleness policy.
#
# Always loads the full configured satellite set, regardless of which
# ones are currently selected in the UI -- so changing the selection
# below never invalidates this cache or touches Celestrak; selection is
# just an in-memory filter over an already-loaded dict.
@st.cache_resource(ttl=int(MAX_TLE_AGE_DAYS * 24 * 60 * 60))
def _load_all_satellites() -> dict[str, EarthSatellite]:
    return load_satellites()


st.title("Satellite Pass Predictor")
st.caption(
    f"Kourou observer: {KOUROU_LATITUDE_DEG:.4f}°N, "
    f"{abs(KOUROU_LONGITUDE_DEG):.4f}°W · "
    f"elevation threshold {MIN_PASS_ELEVATION_DEG:.0f}°"
)

satellite_names = list(SATELLITES.keys())
selected_names = st.multiselect(
    "Satellites", options=satellite_names, default=satellite_names,
)

duration_hours = st.slider(
    "Time window (hours)", min_value=1, max_value=72, value=24,
    help="How far ahead to compute ground tracks and visibility passes.",
)

if not selected_names:
    st.info("Select at least one satellite to see its ground track and passes.")
    st.stop()

all_satellites = _load_all_satellites()
satellites = {name: all_satellites[name] for name in selected_names}

ts = load.timescale()
now = ts.now()

st.subheader(f"Ground Tracks (next {duration_hours}h)")
fig = build_ground_tracks_figure(
    satellites, ts, start_time=now, duration_hours=duration_hours, step_minutes=1,
)
st.pyplot(fig)

st.subheader("Visibility Passes over Kourou")
passes_by_satellite = {
    name: compute_passes(sat, KOUROU, ts, start_time=now, duration_hours=duration_hours)
    for name, sat in satellites.items()
}
rows = build_pass_rows(passes_by_satellite)
if not rows:
    st.info("No passes above threshold in this window.")
else:
    # Same rows/columns print_passes_table() and the HTML report show --
    # just the display columns (drop "_sort_key", build_pass_rows()'s
    # internal sort key, which isn't meant to be shown).
    display_rows = [
        {title: row[key] for key, title, _ in PASS_TABLE_COLUMNS} for row in rows
    ]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)
