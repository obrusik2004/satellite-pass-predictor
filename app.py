"""
Satellite Pass Predictor -- Streamlit app.

Reuses satellite_pass_predictor's existing package logic end to end --
this file only wires widgets to it and renders the results. No orbital-
mechanics, propagation, or pass-detection logic lives here; see the
package for that.

Primary visualization is an interactive 3D-look globe (globe.py), built
from the same GroundTrackDict data compute_ground_track() already
produces -- not a duplicate computation. The 2D matplotlib plot
(visualization.build_ground_tracks_figure()) is untouched and still used
by main.py/the CLI and the static HTML report; it's just no longer what
this app displays.

Everything here recomputes live on every widget interaction (Streamlit
reruns this whole script top to bottom on every interaction; that's
expected and is exactly what makes this different from the static HTML
report).
"""

from typing import cast

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
from satellite_pass_predictor.globe import build_globe_deck
from satellite_pass_predictor.propagation import compute_ground_track
from satellite_pass_predictor.skyplot import build_sky_plot_figure
from satellite_pass_predictor.tle_data import load_satellites
from satellite_pass_predictor.visibility import PassDict, compute_passes
from satellite_pass_predictor.visualization import PASS_TABLE_COLUMNS, build_pass_rows

st.set_page_config(page_title="Satellite Pass Predictor", layout="wide")

# The globe's *own* display window, independent of the "Time window"
# slider below (which still controls the full pass-detection window).
# Checked directly, not assumed: rendering 72h at 1-minute resolution
# (the slider's max) produces ~4300 points per satellite, and the
# resulting globe is not just slow to build (~2s in pure Python per
# rerun, before the browser even starts drawing) but genuinely
# unreadable -- ~45 overlapping orbits per satellite blur into a solid
# mesh no matter how finely or coarsely sampled, since every orbit's
# ground track largely retraces the last one. Cutting the *sampling*
# density (step_minutes) alone doesn't fix that: it reduces point count
# but not orbit count, so the globe stays just as visually cluttered.
# The only lever that actually helps is showing fewer orbits, i.e. a
# shorter duration -- the same reason the README's hero image uses a 2h
# window instead of 24h. 6h was chosen after checking it directly: ~4
# orbits per satellite, individual tracks still clearly distinguishable,
# ~1440 total points, well under a second to build.
GLOBE_MAX_DURATION_HOURS = 6


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
    help=(
        "How far ahead to compute visibility passes. The globe below "
        f"shows at most the next {GLOBE_MAX_DURATION_HOURS}h regardless "
        "of this setting -- see the note above the globe."
    ),
)

if not selected_names:
    st.info("Select at least one satellite to see its ground track and passes.")
    st.stop()

all_satellites = _load_all_satellites()
satellites = {name: all_satellites[name] for name in selected_names}

ts = load.timescale()
now = ts.now()

globe_duration_hours = min(duration_hours, GLOBE_MAX_DURATION_HOURS)
st.subheader(f"Ground Tracks (next {globe_duration_hours}h)")
if duration_hours > GLOBE_MAX_DURATION_HOURS:
    st.caption(
        f"Showing the next {globe_duration_hours}h rather than the full "
        f"{duration_hours}h time window -- beyond a few orbits, ground "
        "tracks overlap into a solid, unreadable mesh regardless of "
        "sampling detail. The passes table below still covers the full "
        f"{duration_hours}h."
    )
ground_tracks = {
    name: compute_ground_track(sat, ts, start_time=now, duration_hours=globe_duration_hours, step_minutes=1)
    for name, sat in satellites.items()
}
st.pydeck_chart(build_globe_deck(ground_tracks), height=600)

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
    # just the display columns (drop "_sort_key"/"_satellite"/"_pass",
    # build_pass_rows()'s internal bookkeeping fields, which aren't
    # meant to be shown as table columns).
    display_rows = [
        {title: row[key] for key, title, _ in PASS_TABLE_COLUMNS} for row in rows
    ]
    # on_select="rerun" + selection_mode="single-row": confirmed against
    # the actually-pinned streamlit==1.64.0 (not assumed from docs --
    # this dataframe selection API changed across Streamlit versions),
    # via inspect.signature(st.dataframe) and streamlit/elements/arrow.py
    # directly. With on_select="rerun", st.dataframe() returns a
    # DataframeState instead of a DeltaGenerator; event.selection.rows is
    # a list of selected row *positions*, stable against the original
    # data even if the user re-sorts by clicking a column header in the
    # browser (confirmed in arrow.py's own docstring for that field) --
    # exactly what's needed to index back into `rows` below.
    #
    # key="passes_table" is required, not cosmetic: every rerun
    # recomputes passes_by_satellite from now = ts.now(), so the
    # table's cell values shift by a few seconds each time. Without an
    # explicit key, Streamlit derives this widget's identity partly
    # from its data, so a click's data-derived id (pre-rerun) no longer
    # matches the freshly-recomputed table's id (post-rerun) and the
    # selection is silently dropped -- confirmed directly by bisecting
    # against a minimal repro: identical setup, selection worked with
    # static data and failed the instant the data was made to change
    # between reruns like this table's does, and adding a stable key
    # fixed it. A fixed key keeps this widget's identity stable across
    # reruns regardless of what the table displays.
    event = st.dataframe(
        display_rows, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="passes_table",
    )

    st.subheader("Sky Track")
    selected = event.selection.rows
    if not selected:
        st.info("Select a pass above to see its sky track.")
    else:
        # rows and display_rows are the same list, transformed 1:1 in
        # the same order (see the comprehension above) -- so a selected
        # display-row position indexes directly into `rows` to recover
        # the satellite name and raw PassDict build_pass_rows() attached
        # to it (see that function's docstring for why it carries these
        # through rather than having this call site re-derive the same
        # sort order independently).
        selected_row = rows[selected[0]]
        satellite_name = cast(str, selected_row["_satellite"])
        pass_ = cast(PassDict, selected_row["_pass"])
        fig = build_sky_plot_figure(satellites[satellite_name], pass_, KOUROU, ts)
        st.plotly_chart(fig, use_container_width=True)
