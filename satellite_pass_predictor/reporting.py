"""
Reporting: combines the ground-track plot and the visibility-pass table
into one self-contained HTML report -- a single file (the plot embedded
as a base64 data URI, not a linked image) that works if emailed or
opened from anywhere, with nothing that can go missing.

No templating library here (Jinja2 etc.): the page is one mostly-static
shell with a handful of substitutions and one small loop over table
rows, built with plain f-strings and a loop -- exactly the case plain
string templating is good enough for, without pulling in a dependency
whose actual template-language features (control flow, filters,
template inheritance) this page has no use for.
"""

import base64
import html
import os

from skyfield.timelib import Time

from .config import (
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    MIN_PASS_ELEVATION_DEG,
    OUTPUT_DIR,
)
from .visibility import PassDict
from .visualization import PASS_TABLE_COLUMNS, build_pass_rows

# Table columns that hold quantities (degrees, minutes) rather than
# labels/timestamps -- right-aligned in the HTML table, unlike the
# fixed-width text table where every column is left-aligned. Plain text
# output has no real notion of alignment beyond padding; an HTML table
# can and should distinguish "a number" from "a label" the way a
# spreadsheet would.
_NUMERIC_COLUMNS = {"start_az", "max_elev", "end_az", "duration_min"}

_CSS = """
  :root {
    --bg: #f5f6f8;
    --card-bg: #ffffff;
    --text: #1a1a2e;
    --muted: #63707e;
    --border: #e2e5ea;
    --accent: #2563eb;
    --stripe: #f7f9fb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 2.5rem 1rem;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }
  .container {
    max-width: 960px;
    margin: 0 auto;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2rem 2.5rem 2.5rem;
  }
  header {
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    padding-bottom: 1.1rem;
  }
  header h1 { margin: 0 0 0.3rem 0; font-size: 1.5rem; }
  header p { margin: 0; color: var(--muted); font-size: 0.9rem; }
  h2 {
    font-size: 1.05rem;
    margin: 2rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }
  img {
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    display: block;
  }
  /* Wraps the table only -- so a wide table (long Notes text, many
     columns) scrolls within its own box instead of forcing the whole
     page to scroll horizontally. */
  .table-wrapper { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td {
    text-align: left;
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  /* Notes can genuinely be long (multiple comma-joined flags) -- letting
     it wrap keeps that readable instead of truncated at the edge of the
     table-wrapper's scroll area. No min-width: the column is empty in
     the common case (no flagged passes) and shouldn't reserve space it
     isn't using; when it does have content, wrapping plus the table's
     own natural layout is enough to keep it readable. */
  td.notes { white-space: normal; }
  th {
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
  }
  td.num, th.num { text-align: right; }
  tbody tr:nth-child(even) { background: var(--stripe); }
  .no-passes { color: var(--muted); font-style: italic; margin: 0; }
  footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.78rem;
  }
"""


def _column_css_class(key: str) -> str:
    """CSS class for a pass-table column: right-aligned for quantities,
    wrapping (rather than truncated) for the potentially-long Notes
    column, plain for everything else (labels/timestamps)."""
    if key in _NUMERIC_COLUMNS:
        return "num"
    if key == "notes":
        return "notes"
    return ""


def _passes_table_html(passes_by_satellite: dict[str, list[PassDict]]) -> str:
    """
    Render the exact same rows/columns print_passes_table() prints (via
    the shared build_pass_rows() helper), as an HTML <table> rather
    than a <pre>-dumped copy of the terminal text.
    """
    rows = build_pass_rows(passes_by_satellite)
    if not rows:
        return '<p class="no-passes">No passes above threshold in this window.</p>'

    def _cell(tag: str, key: str, text: str) -> str:
        css_class = _column_css_class(key)
        class_attr = f' class="{css_class}"' if css_class else ""
        return f"<{tag}{class_attr}>{html.escape(text)}</{tag}>"

    header_cells = "".join(
        _cell("th", key, title) for key, title, _ in PASS_TABLE_COLUMNS
    )

    body_rows = []
    for row in rows:
        cells = "".join(
            _cell("td", key, str(row[key])) for key, _, _ in PASS_TABLE_COLUMNS
        )
        body_rows.append(f"<tr>{cells}</tr>")

    # Wrapped in .table-wrapper so a wide table (long Notes text, this
    # many columns) scrolls within its own box on a narrow viewport
    # instead of forcing the whole page to scroll horizontally.
    return (
        '<div class="table-wrapper">\n'
        "<table>\n"
        f"  <thead><tr>{header_cells}</tr></thead>\n"
        f"  <tbody>\n    " + "\n    ".join(body_rows) + "\n  </tbody>\n"
        "</table>\n"
        "</div>"
    )


def generate_html_report(
    ground_track_png_path: str,
    passes_by_satellite: dict[str, list[PassDict]],
    generated_at: Time,
    duration_hours: float = 24,
    min_elevation_deg: float = MIN_PASS_ELEVATION_DEG,
    output_path: str | None = None,
) -> str:
    """
    Build a single, self-contained HTML report combining the ground-
    track plot and the visibility-pass table, and save it.

    Takes the *path* of an already-saved ground-track PNG (from
    plot_ground_tracks()) rather than the satellites/timescale needed to
    draw one -- the plot has already been computed once for the
    standalone PNG, and re-propagating and re-rendering it here just to
    embed it would be duplicate work for an identical image. The PNG
    bytes are read back and base64-encoded into a `data:` URI, so the
    resulting HTML file has no external image reference that could go
    missing if moved, emailed, or opened somewhere else.

    Returns the path the HTML report was saved to.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "report.html")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(ground_track_png_path, "rb") as f:
        png_base64 = base64.b64encode(f.read()).decode("ascii")

    table_html = _passes_table_html(passes_by_satellite)
    generated_at_str = generated_at.utc_strftime("%Y-%m-%d %H:%M:%S UTC")

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Satellite Pass Predictor Report</title>
<style>
{_CSS}
</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Satellite Pass Predictor</h1>
      <p>
        Generated {html.escape(generated_at_str)} &middot;
        Kourou observer: {KOUROU_LATITUDE_DEG:.4f}&deg;N, {abs(KOUROU_LONGITUDE_DEG):.4f}&deg;W
      </p>
    </header>

    <section>
      <h2>Ground Tracks (next {duration_hours:.0f}h)</h2>
      <img src="data:image/png;base64,{png_base64}" alt="Satellite ground tracks over the next {duration_hours:.0f} hours">
    </section>

    <section>
      <h2>Visibility Passes over Kourou (elevation &ge; {min_elevation_deg:.0f}&deg;)</h2>
      {table_html}
    </section>

    <footer>
      <p>satellite-pass-predictor &middot; SGP4 propagation via Skyfield &middot; TLE data from Celestrak</p>
    </footer>
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(document)

    return output_path
