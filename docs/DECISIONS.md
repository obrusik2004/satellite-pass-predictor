# Engineering decisions

Short records of the choices that shape the code: context, decision, consequence.

## 1. SGP4 only, because the input is TLEs

**Context.** Celestrak TLEs hold mean elements fitted to the SGP4 model, not osculating state vectors.
**Decision.** Propagate with SGP4 (via Skyfield) only; a numerical propagator fed TLE elements adds no accuracy and introduces a systematic error.
**Consequence.** Accuracy is bounded by TLE age, not the propagator, so the app warns when an epoch is older than `TLE_EPOCH_WARNING_DAYS`.

## 2. TLE mirror on a separate branch, refreshed by GitHub Actions

**Context.** Streamlit Community Cloud cannot reach celestrak.org, but it can reach `raw.githubusercontent.com`.
**Decision.** A workflow runs every 6 hours, validates each TLE and force-pushes `tle/` as one orphan commit to `tle-data`; the app reads that branch and the CLI still defaults to Celestrak.
**Consequence.** `main` gets no refresh commits or redeploys and a failed satellite keeps its last good file. Freshness depends on the workflow, which GitHub pauses after 60 days of repository inactivity, so the app shows mirror age and warns past `MIRROR_REFRESH_WARNING_HOURS`.

## 3. The mirror has its own, shorter cache age

**Context.** Celestrak TLEs are cached for a day, but the mirror changes every 6 hours.
**Decision.** `TLE_MIRROR_CACHE_AGE_HOURS` (1 hour) sets both the on-disk cache age and the app's `st.cache_resource` TTL.
**Consequence.** A refresh reaches users within about an hour and the two expiries can't drift. `EarthSatellite` isn't picklable, hence `cache_resource` rather than `cache_data`.

## 4. A pydeck GlobeView instead of Plotly for ground tracks

**Context.** Plotly's orthographic `Scattergeo` renders as SVG and rotated at only a few frames per second.
**Decision.** Use pydeck's `_GlobeView` (WebGL) for the globe; keep Plotly for the small 2D sky plot, where the point count is tiny.
**Consequence.** Rotation is smooth, at the cost of a second charting library. The globe shows at most 6 hours, since more orbits blur into an unreadable mesh whatever the sampling.

## 5. Land polygons only for the basemap

**Context.** The globe needs an anchor for the eye, but Natural Earth's country polygons (177 shapes with borders) were heavy enough to slow rotation.
**Decision.** Ship only the 1:110m land layer (127 polygons, geometry only), loaded once at import and drawn in muted theme colors.
**Consequence.** Rotation stays smooth and tracks stay visually primary; there are no borders or labels.

## 6. Split ground tracks at the antimeridian

**Context.** A line between samples at +179 and -179 degrees is drawn the long way across the map; matplotlib can break a line with NaN, but pydeck's `PathLayer` cannot.
**Decision.** Detect crossings once (`geo.find_antimeridian_crossings`); the flat plot inserts NaNs, and the globe splits the track into paths that end on the meridian at an interpolated latitude.
**Consequence.** No wrong-way streaks and no gap at the seam. Hover points use the unsplit samples, since the boundary points have no real time or altitude.

## 7. One color per satellite, defined once

**Context.** A satellite appears on the globe, in the table, the sky plot, the flat plot and the legend.
**Decision.** `SATELLITE_COLORS` in `config.py` is the only definition, looked up by name (selection order varies); the station and rise/set markers are neutral so they can't read as a satellite.
**Consequence.** Colors stay consistent everywhere. Theme colors are duplicated in `.streamlit/config.toml` for renderers Streamlit can't theme, and kept in sync by hand.

## 8. Lock file plus ranges, split into core / app / dev

**Context.** Streamlit Cloud installs from `requirements.txt`, while development and CI install the package.
**Decision.** `requirements.txt` is the exact lock; `pyproject.toml` declares direct dependencies as ranges in a core set, an `app` extra and a `dev` extra, and CI and the refresh job use the lock as constraints.
**Consequence.** The refresh job installs only core dependencies, so an unrelated PyPI problem can't fail it, and there is no second hand-maintained list.

## 9. Validate before caching; report failures per satellite

**Context.** A 200 response with an error body used to overwrite a good cache and then count as fresh, and one bad satellite aborted the whole load.
**Decision.** A response is parsed and checked (one TLE, matching NORAD ID) before an atomic write, a failed refresh falls back to the cache, and `load_satellites` returns the loaded satellites plus a per-satellite failure report. A TLE older than `TLE_EPOCH_MAX_DAYS` or one SGP4 cannot propagate is a failure, not a silent empty result.
**Consequence.** One satellite's problem shows as a warning naming the satellite and the source while the rest still render, and a poisoned cache heals itself. Only transient errors (connection, timeout, 5xx, 429) are retried.
