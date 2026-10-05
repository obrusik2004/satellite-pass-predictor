"""Streamlit front end: wires sidebar controls to the satellite_pass_predictor package.

Shows an interactive globe of ground tracks, a selectable table of passes over Kourou, and a
sky plot of the selected pass. No orbital logic lives here. Streamlit reruns this script on
every interaction, so results are always recomputed from the current time.
"""

import html
from datetime import UTC, datetime, timedelta
from typing import cast

import pandas as pd
import streamlit as st
from skyfield.api import load

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
    TIME_WINDOW_DEFAULT_HOURS,
    TIME_WINDOW_MAX_HOURS,
    TIME_WINDOW_MIN_HOURS,
    TLE_EPOCH_WARNING_DAYS,
    TLE_FAILED_LOAD_RETRY_SECONDS,
    TLE_MIRROR_CACHE_AGE_HOURS,
)
from satellite_pass_predictor.globe import build_globe_deck
from satellite_pass_predictor.propagation import GroundTrackDict, compute_ground_track
from satellite_pass_predictor.skyplot import build_sky_plot_figure
from satellite_pass_predictor.tle_data import (
    TLELoadResult,
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

# Custom CSS for what the theme can't express: the title band, the legend swatches and trimming
# sidebar chrome. Selectors use our own classes/keys or Streamlit's data-testid hooks, never its
# generated class names.
st.markdown(
    f"""
    <style>
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
        background: {html.escape(THEME_PANEL_COLOR)};
        border: 1px solid {html.escape(THEME_BORDER_COLOR)};
        border-radius: 0.5rem;
        padding: 1.1rem 1.75rem;
        margin-bottom: 1.5rem;
        /* Clears Streamlit's fixed top bar. */
        margin-top: 2.5rem;
    }}
    .pp-header h1 {{
        margin: 0 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    .pp-header p {{
        margin: 0;
        color: {html.escape(THEME_TEXT_COLOR)};
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
    /* Neutralize the subheader's asymmetric padding so the button centers on the heading text. */
    .st-key-globe_header {{
        margin-top: 0.75rem;
    }}
    .st-key-globe_header [data-testid="stMarkdownContainer"] {{
        margin-bottom: 0;
    }}
    .st-key-globe_header h3 {{
        padding: 0;
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

# The globe shows at most this many hours: beyond a few orbits the tracks blur into a mesh,
# and build time grows. The "Time window" slider still sets the pass-detection window.
GLOBE_MAX_DURATION_HOURS = 6

# Changing the key of the container holding the globe remounts the chart, which restores its
# initial view; the "Reset view" button bumps the counter.
GLOBE_VIEW_COUNTER_KEY = "globe_view_counter"
st.session_state.setdefault(GLOBE_VIEW_COUNTER_KEY, 0)


def _reset_globe_view() -> None:
    st.session_state[GLOBE_VIEW_COUNTER_KEY] += 1


# A fragment, so "Reset view" reruns only the globe and not the whole script.
@st.fragment
def _render_globe(
    ground_tracks: dict[str, GroundTrackDict], globe_duration_hours: int, duration_hours: int
) -> None:
    with st.container(
        key="globe_header",
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        st.subheader(f"Ground Tracks (next {globe_duration_hours}h)")
        st.button(
            "Reset view",
            key="globe_reset",
            icon=":material/explore:",
            help="Return the globe to its initial view over Kourou.",
            on_click=_reset_globe_view,
        )
    if duration_hours > GLOBE_MAX_DURATION_HOURS:
        st.caption(
            f"Showing the next {globe_duration_hours}h rather than the full "
            f"{duration_hours}h time window -- beyond a few orbits, ground "
            "tracks overlap into a solid, unreadable mesh regardless of "
            "sampling detail. The passes table below still covers the full "
            f"{duration_hours}h."
        )
    with st.container(key=f"globe_{st.session_state[GLOBE_VIEW_COUNTER_KEY]}"):
        st.pydeck_chart(build_globe_deck(ground_tracks), height=600)


# cache_resource, not cache_data: EarthSatellite wraps an sgp4 Satrec, which can't be pickled.
# The mirror, not Celestrak, since Streamlit Cloud can't reach Celestrak (docs/DECISIONS.md).
# The TTL matches the mirror's own cache age so a refresh reaches users within about an hour.
# Always loads every satellite so changing the selection never reloads.
@st.cache_resource(ttl=int(TLE_MIRROR_CACHE_AGE_HOURS * 60 * 60))
def _load_all_satellites() -> TLELoadResult:
    return load_satellites(source="mirror")


# Same TTL, so both refresh together.
@st.cache_data(ttl=int(TLE_MIRROR_CACHE_AGE_HOURS * 60 * 60))
def _load_mirror_metadata() -> TLEMirrorMetadata | None:
    return fetch_mirror_metadata()


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

satellite_names = list(SATELLITES.keys())
with st.sidebar:
    selected_names = st.multiselect(
        "Satellites",
        options=satellite_names,
        default=satellite_names,
    )

    duration_hours = st.slider(
        "Time window (hours)",
        min_value=TIME_WINDOW_MIN_HOURS,
        max_value=TIME_WINDOW_MAX_HOURS,
        value=TIME_WINDOW_DEFAULT_HOURS,
        help=(
            "How far ahead to compute visibility passes. The globe below "
            f"shows at most the next {GLOBE_MAX_DURATION_HOURS}h regardless "
            "of this setting -- see the note above the globe."
        ),
    )

if not selected_names:
    st.info("Select at least one satellite to see its ground track and passes.")
    st.stop()

load_result = _load_all_satellites()
# A failed load is retried after a short delay instead of being cached for the full TTL.
if load_result.failures and datetime.now(UTC) - load_result.loaded_at > timedelta(
    seconds=TLE_FAILED_LOAD_RETRY_SECONDS
):
    _load_all_satellites.clear()
    load_result = _load_all_satellites()

failure_list = "\n".join(f"- **{f.name}**: {f.reason}" for f in load_result.failures.values())
if not load_result.satellites:
    st.error(f"No satellite data could be loaded, so there is nothing to show:\n\n{failure_list}")
    st.stop()
if load_result.failures:
    st.warning(f"Some satellites could not be loaded and are not shown:\n\n{failure_list}")
for notice in load_result.warnings.values():
    st.warning(notice)

selected_names = [name for name in selected_names if name in load_result.satellites]
if not selected_names:
    st.info("None of the selected satellites could be loaded.")
    st.stop()
satellites = {name: load_result.satellites[name] for name in selected_names}

with st.sidebar:
    st.markdown("**Satellites shown**")
    for name in selected_names:
        st.markdown(
            f'<span class="pp-legend-dot" style="background: '
            f'{html.escape(SATELLITE_COLORS[name])};"></span>{html.escape(name)}',
            unsafe_allow_html=True,
        )

ts = load.timescale()
now = ts.now()

mirror_metadata = _load_mirror_metadata()
satellite_epochs = {name: sat.epoch.utc_datetime() for name, sat in satellites.items()}
staleness = compute_staleness_warnings(mirror_metadata, satellite_epochs)

with st.sidebar:
    st.divider()
    if mirror_metadata is None:
        st.caption(
            "TLEs from Celestrak via GitHub mirror -- refresh provenance unavailable right now."
        )
    else:
        generated_at = parse_iso_utc(mirror_metadata["generated_at"])
        generated_at_str = (
            generated_at.strftime("%Y-%m-%d %H:%M UTC")
            if generated_at is not None
            else mirror_metadata["generated_at"]  # malformed: show it raw
        )
        st.caption(f"TLEs from Celestrak via GitHub mirror, last refreshed {generated_at_str}")

    epoch_ages = ", ".join(
        f"{name} {(now.utc_datetime() - epoch).total_seconds() / 86400:.1f}d"
        for name, epoch in satellite_epochs.items()
    )
    st.caption(f"TLE epoch age -- {epoch_ages}")

    github_url = html.escape(GITHUB_REPO_URL)
    footer_links = [f'<a href="{github_url}" target="_blank">GitHub</a>']
    if LINKEDIN_URL:
        footer_links.append(f'<a href="{html.escape(LINKEDIN_URL)}" target="_blank">LinkedIn</a>')
    author = html.escape(AUTHOR_NAME)
    st.divider()
    st.markdown(
        f"""
        <div class="pp-footer">
          Built by {author} &middot; {" &middot; ".join(footer_links)}<br>
          Orbital data: CelesTrak &middot; Basemap: Natural Earth<br>
          Independent project, not affiliated with ESA<br>
          &copy; {COPYRIGHT_YEAR} {author}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Staleness warnings go in the main area, where they are hard to miss.
if staleness["mirror_stale"]:
    st.warning(
        f"The TLE mirror hasn't refreshed in over {MIRROR_REFRESH_WARNING_HOURS:.0f}h -- "
        f"the scheduled GitHub Actions workflow may have stopped running "
        f"(see README's Data pipeline section). Data shown may be outdated."
    )
if staleness["stale_satellite_names"]:
    st.warning(
        f"TLE data for {', '.join(staleness['stale_satellite_names'])} is more than "
        f"{TLE_EPOCH_WARNING_DAYS:.0f} day(s) old -- predictions for that satellite "
        f"may be less accurate."
    )

globe_duration_hours = min(duration_hours, GLOBE_MAX_DURATION_HOURS)
ground_tracks = {
    name: compute_ground_track(
        sat, ts, start_time=now, duration_hours=globe_duration_hours, step_minutes=1
    )
    for name, sat in satellites.items()
}
_render_globe(ground_tracks, globe_duration_hours, duration_hours)

st.subheader("Visibility Passes over Kourou")
passes_by_satellite = {
    name: compute_passes(sat, KOUROU, ts, start_time=now, duration_hours=duration_hours)
    for name, sat in satellites.items()
}
rows = build_pass_rows(passes_by_satellite)
if not rows:
    st.info("No passes above threshold in this window.")
else:
    # Display columns only; Notes are shortened to codes to fit the narrow column.
    display_rows = []
    for row in rows:
        display_row = {title: row[key] for key, title, _ in PASS_TABLE_COLUMNS}
        display_row["Notes"] = build_pass_note_codes(cast(PassDict, row["_pass"]))
        display_rows.append(display_row)
    # A Styler colors the Satellite text; column_config has no per-cell color.
    styled_rows = pd.DataFrame(display_rows).style.map(
        lambda satellite_name: f"color: {SATELLITE_COLORS[satellite_name]}; font-weight: 600",
        subset=["Satellite"],
    )
    # The explicit key keeps the widget identity (and so the row selection) stable across
    # reruns, whose recomputed data would otherwise give it a new identity. Selected row
    # positions index into `rows` even after the user re-sorts the table.
    event = st.dataframe(
        styled_rows,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="passes_table",
        column_config={
            "Notes": st.column_config.TextColumn(
                width="small",
                # Header tooltip; per-cell tooltips aren't available, hence the caption below.
                help="\n".join(f"- **{code}**: {meaning}" for code, meaning in NOTE_CODE_LEGEND),
            )
        },
    )
    # Only worth showing when some row actually has a code.
    if any(display_row["Notes"] for display_row in display_rows):
        st.caption(
            "Notes: " + " · ".join(f"**{code}** {meaning}" for code, meaning in NOTE_CODE_LEGEND)
        )

    st.subheader("Sky Track")
    selected = event.selection.rows
    if not selected:
        st.info("Select a pass above to see its sky track.")
    else:
        selected_row = rows[selected[0]]
        satellite_name = cast(str, selected_row["_satellite"])
        pass_ = cast(PassDict, selected_row["_pass"])
        fig = build_sky_plot_figure(satellites[satellite_name], pass_, KOUROU, ts, satellite_name)
        st.plotly_chart(fig, width="stretch")
