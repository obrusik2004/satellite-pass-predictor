"""
Satellite Pass Predictor -- Streamlit app.

Reuses satellite_pass_predictor's existing package logic end to end --
this file only wires widgets to it and renders the results. No orbital-
mechanics, propagation, or pass-detection logic lives here; see the
package for that.

Primary visualization is an interactive 3D-look globe (globe.py), built
from the same GroundTrackDict data compute_ground_track() already
produces -- not a duplicate computation. The 2D matplotlib plot
(visualization.build_ground_tracks_figure()) is still used by main.py/
the CLI and the static HTML report, not this app; it was recolored
during this app's visual redesign purely to fix a contrast problem
shared with this app's own palette (see that function's docstring), not
because it's shown here.

Everything here recomputes live on every widget interaction (Streamlit
reruns this whole script top to bottom on every interaction; that's
expected and is exactly what makes this different from the static HTML
report).
"""

from typing import cast

import pandas as pd
import streamlit as st
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite

from satellite_pass_predictor.config import (
    AUTHOR_NAME,
    COPYRIGHT_YEAR,
    GITHUB_REPO_URL,
    KOUROU,
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    LINKEDIN_URL,
    MIN_PASS_ELEVATION_DEG,
    MIRROR_REFRESH_WARNING_HOURS,
    SATELLITE_COLORS,
    SATELLITES,
    THEME_BORDER_COLOR,
    THEME_PANEL_COLOR,
    THEME_TEXT_COLOR,
    TLE_EPOCH_WARNING_DAYS,
    TLE_MIRROR_CACHE_AGE_HOURS,
)
from satellite_pass_predictor.globe import build_globe_deck
from satellite_pass_predictor.propagation import compute_ground_track
from satellite_pass_predictor.skyplot import build_sky_plot_figure
from satellite_pass_predictor.tle_data import (
    TLEMirrorMetadata,
    compute_staleness_warnings,
    fetch_mirror_metadata,
    load_satellites,
    parse_iso_utc,
)
from satellite_pass_predictor.visibility import PassDict, compute_passes
from satellite_pass_predictor.visualization import (
    NOTE_CODE_LEGEND,
    PASS_TABLE_COLUMNS,
    build_pass_note_codes,
    build_pass_rows,
)

st.set_page_config(page_title="Satellite Pass Predictor", layout="wide")

# Minimal custom CSS -- the one place it's used in this app, for the
# things .streamlit/config.toml's theme genuinely can't express: a
# colored "title band" behind the header (Streamlit's theme has no
# notion of a background scoped to one section of the page), small
# color-swatch dots for the sidebar's satellite legend (no built-in
# widget draws an arbitrary-colored dot), and trimming a few pieces of
# native Streamlit sidebar chrome that reserve real but non-functional
# vertical space (a sidebar's worth of controls plus the provenance/
# footer text otherwise needs to scroll to see on a normal laptop
# viewport). The first two target only classes defined right here
# (pp-header, pp-legend-dot), never Streamlit's own internal, auto-
# generated class names (the "st-emotion-cache-xxxx" kind that changes
# across versions). The sidebar-chrome rules below are the one
# exception: they target Streamlit's own data-testid attributes
# (stSidebarHeader, stSidebarUserContent), which are that different,
# more stable kind of hook -- Streamlit's own documented automation/
# testing contract, not a build-generated hash -- because there's no
# other way to reach chrome this app's own markup doesn't produce.
st.markdown(
    f"""
    <style>
    /* Measured directly (getComputedStyle() in the running app, not
       guessed) before trimming any of this: stSidebarHeader reserved
       60px for just the collapse arrow (this app sets no sidebar logo),
       stSidebarUserContent reserved a 96px bottom padding, and each
       st.divider() line's own default margin took ~49px total -- none
       of that is functional, all of it is exactly the wasted space this
       fixes. Deliberately NOT touching stSidebarUserContent's
       inter-element gap (Streamlit's own default, 16px) -- that also
       separates the legend's own rows, which must keep its current
       look and spacing exactly as it was. */
    [data-testid="stSidebarHeader"] {{
        height: 2rem;
    }}
    [data-testid="stSidebarUserContent"] {{
        padding-bottom: 0.5rem;
    }}
    [data-testid="stSidebar"] hr {{
        margin: 0.4rem 0;
    }}
    .pp-header {{
        background: {THEME_PANEL_COLOR};
        border: 1px solid {THEME_BORDER_COLOR};
        border-radius: 0.5rem;
        padding: 1.1rem 1.75rem;
        margin-bottom: 1.5rem;
        /* Streamlit's own fixed top bar overlaps the very top of the
           main content area -- confirmed directly (the header band's
           top edge rendered underneath it without this). The
           block-container's default top padding isn't reliably enough
           clearance on its own, so this band gets its own explicit
           margin rather than depending on that. client.toolbarMode =
           "minimal" (.streamlit/config.toml) shrinks that top bar, but
           doesn't remove it outright, so this margin stays regardless. */
        margin-top: 2.5rem;
    }}
    .pp-header h1 {{
        margin: 0 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    .pp-header p {{
        margin: 0;
        color: {THEME_TEXT_COLOR};
        opacity: 0.75;
        font-size: 0.95rem;
    }}
    .pp-legend-dot {{
        display: inline-block;
        width: 0.75em;
        height: 0.75em;
        border-radius: 50%;
        margin-right: 0.45em;
        vertical-align: middle;
    }}
    .pp-footer {{
        font-size: 0.75rem;
        line-height: 1.6;
        opacity: 0.6;
    }}
    .pp-footer a {{
        color: inherit;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

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
# network fetch -- would run again every time someone moves a slider.
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
# source="mirror", not the default "celestrak": Streamlit Community
# Cloud cannot reach celestrak.org at all (see TLE_MIRROR_URL's comment
# in config.py) -- this app reads the GitHub-Actions-refreshed mirror
# instead. main.py/the CLI still defaults to "celestrak" directly, which
# works fine outside Streamlit Cloud.
#
# TTL matches config.TLE_MIRROR_CACHE_AGE_HOURS exactly (not
# MAX_TLE_AGE_DAYS, which governs the *Celestrak-source* on-disk cache
# and would be a full day too long here) -- that's the point at which
# load_satellites(source="mirror") itself would already consider its
# on-disk cache stale and re-check the mirror, so there's nothing to
# gain from expiring this cache any sooner, and no reason to hold it
# longer than the data's own staleness policy. Keeping both on the same
# constant is what actually guarantees a 6-hourly mirror refresh reaches
# users within about an hour, rather than the two silently drifting
# apart if chosen independently.
#
# Always loads the full configured satellite set, regardless of which
# ones are currently selected in the UI -- so changing the selection
# below never invalidates this cache or re-checks the mirror; selection
# is just an in-memory filter over an already-loaded dict.
@st.cache_resource(ttl=int(TLE_MIRROR_CACHE_AGE_HOURS * 60 * 60))
def _load_all_satellites() -> dict[str, EarthSatellite]:
    return load_satellites(source="mirror")


# Separate cache (same TTL, so both refresh in step) for the mirror's
# metadata.json -- purely for the provenance caption/warnings below, not
# needed for the TLEs themselves. st.cache_data, not st.cache_resource:
# this returns a plain dict-or-None, not an unpicklable Skyfield object,
# so cache_data's usual copy-on-read behavior is the right (and default)
# choice here.
@st.cache_data(ttl=int(TLE_MIRROR_CACHE_AGE_HOURS * 60 * 60))
def _load_mirror_metadata() -> TLEMirrorMetadata | None:
    return fetch_mirror_metadata()


# Title band -- .pp-header, styled above -- rather than plain st.title()/
# st.caption(): the ESA-style "clean title band" called for here needs a
# background distinct from the page, which st.title() has no option for.
st.markdown(
    f"""
    <div class="pp-header">
      <h1>Satellite Pass Predictor</h1>
      <p>
        Kourou observer: {KOUROU_LATITUDE_DEG:.4f}°N,
        {abs(KOUROU_LONGITUDE_DEG):.4f}°W ·
        elevation threshold {MIN_PASS_ELEVATION_DEG:.0f}°
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Controls live in the sidebar (rather than the main area, as in an
# earlier version of this app) so the main area is dedicated entirely to
# the globe/table/sky-plot output -- standard dashboard layout, and it's
# also where the satellite color legend and data-provenance details
# below naturally belong alongside these same controls.
satellite_names = list(SATELLITES.keys())
with st.sidebar:
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

with st.sidebar:
    # Color legend -- exact hex swatches (config.SATELLITE_COLORS), so
    # the globe's per-satellite track colors are self-explanatory without
    # needing to guess or hover. Only for currently-selected satellites,
    # matching what's actually drawn below.
    st.markdown("**Satellites shown**")
    for name in selected_names:
        st.markdown(
            f'<span class="pp-legend-dot" style="background: {SATELLITE_COLORS[name]};">'
            f"</span>{name}",
            unsafe_allow_html=True,
        )

ts = load.timescale()
now = ts.now()

# Data provenance: where these TLEs actually came from and how fresh
# they are -- worth surfacing explicitly since this app reads a mirror
# rather than fetching Celestrak directly (see _load_all_satellites()'s
# comment for why), so "fresh" here depends on a separate scheduled
# pipeline actually having run recently, not just on this request
# succeeding.
mirror_metadata = _load_mirror_metadata()
satellite_epochs = {name: sat.epoch.utc_datetime() for name, sat in satellites.items()}
staleness = compute_staleness_warnings(mirror_metadata, satellite_epochs)

# Small and muted, at the bottom of the sidebar -- provenance detail
# worth having available, not something that needs main-area prominence
# on every rerun the way an actual staleness problem (below) does.
with st.sidebar:
    st.divider()
    if mirror_metadata is None:
        # fetch_mirror_metadata() already degrades to None for any problem
        # (network failure, missing branch, malformed JSON) rather than
        # raising -- the TLEs themselves already loaded fine above (or
        # load_satellites() would have raised), so this is purely "we
        # can't show provenance details right now," not a reason to stop.
        st.caption("TLEs from Celestrak via GitHub mirror -- refresh provenance unavailable right now.")
    else:
        generated_at = parse_iso_utc(mirror_metadata["generated_at"])
        generated_at_str = (
            generated_at.strftime("%Y-%m-%d %H:%M UTC") if generated_at is not None
            else mirror_metadata["generated_at"]  # malformed timestamp -- show it raw rather than hide it
        )
        st.caption(f"TLEs from Celestrak via GitHub mirror, last refreshed {generated_at_str}")

    epoch_ages = ", ".join(
        f"{name} {(now.utc_datetime() - epoch).total_seconds() / 86400:.1f}d"
        for name, epoch in satellite_epochs.items()
    )
    st.caption(f"TLE epoch age -- {epoch_ages}")

    # Footer -- every value here is a literal constant from config.py
    # (AUTHOR_NAME, GITHUB_REPO_URL, LINKEDIN_URL, COPYRIGHT_YEAR), never
    # anything from a request, session, or other runtime input; that's
    # what makes rendering it as raw HTML via unsafe_allow_html safe (see
    # those constants' own comment in config.py). The LinkedIn link is
    # only included while LINKEDIN_URL is actually set, rather than
    # rendering a broken href, though in practice it always is now.
    footer_links = [f'<a href="{GITHUB_REPO_URL}" target="_blank">GitHub</a>']
    if LINKEDIN_URL:
        footer_links.append(f'<a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>')
    st.divider()
    st.markdown(
        f"""
        <div class="pp-footer">
          Built by {AUTHOR_NAME} &middot; {" &middot; ".join(footer_links)}<br>
          Orbital data: CelesTrak &middot; Basemap: Natural Earth<br>
          Independent project, not affiliated with ESA<br>
          &copy; {COPYRIGHT_YEAR} {AUTHOR_NAME}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Staleness warnings stay in the main area (not the sidebar) -- unlike
# the routine provenance captions above, these mean something is
# actually wrong and should be hard to miss, not tucked away below the
# fold in a panel the user might not scroll.
if staleness["mirror_stale"]:
    st.warning(
        f"The TLE mirror hasn't refreshed in over {MIRROR_REFRESH_WARNING_HOURS:.0f}h -- "
        f"the scheduled GitHub Actions workflow may have stopped running "
        f"(see README's Data pipeline section). Data shown may be outdated."
    )
if staleness["stale_satellite_names"]:
    st.warning(
        f"TLE data for {', '.join(staleness['stale_satellite_names'])} is more than "
        f"{TLE_EPOCH_WARNING_DAYS:.0f} day(s) old -- predictions for that satellite may be less accurate."
    )

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
    # meant to be shown as table columns) -- except Notes, overridden
    # below to short codes rather than build_pass_rows()'s full text:
    # see build_pass_note_codes()'s docstring for why (a fixed-width
    # interactive column, unlike the CLI table/HTML report this data
    # also feeds, has no good way to show a full sentence).
    display_rows = []
    for row in rows:
        display_row = {title: row[key] for key, title, _ in PASS_TABLE_COLUMNS}
        display_row["Notes"] = build_pass_note_codes(cast(PassDict, row["_pass"]))
        display_rows.append(display_row)
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
    # width="stretch" (full container width, matching the globe above
    # and the rest of this wide-layout page) rather than width="content"
    # (sizes the table to its own content) -- content was tried first
    # but overcorrected: it made the whole table shrink to a narrow,
    # oddly floating box unrelated to the rest of the page, since
    # nothing else here is content-sized.
    #
    # width="stretch" does still redistribute leftover space evenly
    # across every column, including Notes -- checked column_types.py
    # directly, and there is no min/max-width or "don't grow" escape
    # hatch for one column in this pinned streamlit==1.64.0's
    # column_config. But measured directly (a canvas pixel probe, since
    # these columns aren't real DOM elements) across 1024-1920px
    # viewports, that redistribution no longer looks lopsided the way
    # it did with full-sentence notes: Notes' configured "small" (75px)
    # inflates to 83-187px depending on viewport, landing squarely in
    # the middle of the pack alongside the other columns (which inflate
    # by the same proportion) rather than dwarfing them the way a
    # 280-400px-wide column of full sentences did. Once the column's
    # own content is short, the redistribution "penalty" stops being
    # visually distinguishable from ordinary column padding.
    # Colored via a pandas Styler, not column_config -- checked directly
    # against this pinned streamlit==1.64.0's column_types.py, and no
    # column type there renders per-cell color in the grid itself
    # (MarkdownColumn only renders markdown in a click-to-expand overlay,
    # not inline -- see that type's own docstring). Styler.map's per-cell
    # CSS *is* rendered inline in the grid (confirmed against
    # elements/lib/pandas_styler_utils.py: it's translated to real CSS
    # rules keyed by row/col, independent of on_select's own row/column
    # selection state) -- so this gives the exact SATELLITE_COLORS hex
    # value per row, not an approximate stand-in like an emoji. Colors
    # only the "Satellite" column's text; every other column is
    # untouched. Selection (on_select="rerun" + key="passes_table" below)
    # was verified by hand to still work with a Styler passed as `data`
    # -- clicking a row still selects it and still drives the sky plot.
    styled_rows = pd.DataFrame(display_rows).style.map(
        lambda satellite_name: f"color: {SATELLITE_COLORS[satellite_name]}; font-weight: 600",
        subset=["Satellite"],
    )
    event = st.dataframe(
        styled_rows, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key="passes_table",
        column_config={"Notes": st.column_config.TextColumn(
            # "small", one of column_config's three named width presets
            # alongside "medium"/"large" -- comfortably fits "LC" alone
            # or a couple of codes comma-joined without truncating.
            width="small",
            # Column-header tooltip (hover the "Notes" header) as a
            # lightweight, always-available second explanation of the
            # codes, alongside the caption below rather than instead of
            # it -- see the caption's own comment for why the caption
            # stays as the primary one.
            help="\n".join(f"- **{code}**: {meaning}" for code, meaning in NOTE_CODE_LEGEND),
        )},
    )
    # No genuine per-cell dynamic tooltip exists for a text column in
    # this pinned streamlit==1.64.0 -- checked directly against
    # column_types.py rather than assumed: every column type's "help"
    # tooltip is documented and implemented identically as a *column
    # header* tooltip (hovering the "Notes" label), never a per-cell
    # one tied to that row's own value. That's a materially different,
    # much simpler mechanism than pydeck's picking-based hover (globe.py),
    # which reads whichever data point the GPU determined is under the
    # cursor -- glide-data-grid (this table's renderer) has no
    # equivalent per-cell hook exposed through Streamlit's API.
    #
    # Given that, the caption stays as the primary explanation (added
    # above the header-help tooltip, not replaced by it): it's visible
    # at a glance without requiring the user to discover and hover a
    # specific header, which matters here since IP/CE/MH/LC are codes
    # invented for this app, not a convention anyone would already
    # know. The header tooltip is a low-cost bonus for anyone who does
    # hover out of habit, not a replacement for it.
    #
    # Only shown when at least one row in the current table actually
    # has a code to explain -- most windows have none, and the legend
    # would just be noise pointing at an all-empty column.
    if any(display_row["Notes"] for display_row in display_rows):
        st.caption(
            "Notes: " + " · ".join(f"**{code}** {meaning}" for code, meaning in NOTE_CODE_LEGEND)
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
        fig = build_sky_plot_figure(satellites[satellite_name], pass_, KOUROU, ts, satellite_name)
        st.plotly_chart(fig, use_container_width=True)
