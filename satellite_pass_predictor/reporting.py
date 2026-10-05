"""Self-contained HTML report: the ground-track plot (embedded as base64) and the pass table.

Built with plain f-strings; the page is too small to justify a template engine.
"""

import base64
import html
import os

from skyfield.timelib import Time

from .colors import hex_to_rgb
from .config import (
    KOUROU_LATITUDE_DEG,
    KOUROU_LONGITUDE_DEG,
    MIN_PASS_ELEVATION_DEG,
    OUTPUT_DIR,
    THEME_BACKGROUND_COLOR,
    THEME_BORDER_COLOR,
    THEME_PANEL_COLOR,
    THEME_TEXT_COLOR,
)
from .visibility import PassDict
from .visualization import PASS_TABLE_COLUMNS, build_pass_rows

# Quantity columns are right-aligned in the HTML table.
_NUMERIC_COLUMNS = {"start_az", "max_elev", "end_az", "duration_min"}

# "r, g, b" for use inside rgba(...), since CSS variables can't add alpha to a hex color.
_TEXT_RGB = ", ".join(str(c) for c in hex_to_rgb(THEME_TEXT_COLOR))

# Same dark theme as the app. Only :root needs f-string substitution; the rest of the
# stylesheet is a plain string so its braces needn't be escaped.
_ROOT_VARS = f"""
  :root {{
    --bg: {THEME_BACKGROUND_COLOR};
    --card-bg: {THEME_PANEL_COLOR};
    --text: {THEME_TEXT_COLOR};
    --muted: rgba({_TEXT_RGB}, 0.65);
    --border: {THEME_BORDER_COLOR};
    --stripe: rgba({_TEXT_RGB}, 0.04);
  }}
"""

_CSS = (
    _ROOT_VARS
    + """
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
  /* A wide table scrolls in its own box instead of the whole page. */
  .table-wrapper { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td {
    text-align: left;
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  /* Notes can be long and is usually empty, so let it wrap rather than reserve width. */
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
)


def _column_css_class(key: str) -> str:
    """CSS class for a pass-table column: "num" for quantities, "notes" for Notes, else none."""
    if key in _NUMERIC_COLUMNS:
        return "num"
    if key == "notes":
        return "notes"
    return ""


def _passes_table_html(passes_by_satellite: dict[str, list[PassDict]]) -> str:
    """Render the same rows and columns as print_passes_table() as an HTML table."""
    rows = build_pass_rows(passes_by_satellite)
    if not rows:
        return '<p class="no-passes">No passes above threshold in this window.</p>'

    def _cell(tag: str, key: str, text: str) -> str:
        css_class = _column_css_class(key)
        class_attr = f' class="{css_class}"' if css_class else ""
        return f"<{tag}{class_attr}>{html.escape(text)}</{tag}>"

    header_cells = "".join(_cell("th", key, title) for key, title, _ in PASS_TABLE_COLUMNS)

    body_rows = []
    for row in rows:
        cells = "".join(_cell("td", key, str(row[key])) for key, _, _ in PASS_TABLE_COLUMNS)
        body_rows.append(f"<tr>{cells}</tr>")

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
    """Write the HTML report (default: OUTPUT_DIR/report.html) and return its path.

    `ground_track_png_path` is an already-rendered plot; it is embedded as a base64 data URI so
    the report is a single file. `duration_hours` and `min_elevation_deg` are used for the
    headings only; they should match what the passes were computed with.
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
      <img src="data:image/png;base64,{png_base64}"
           alt="Satellite ground tracks over the next {duration_hours:.0f} hours">
    </section>

    <section>
      <h2>Visibility Passes over Kourou (elevation &ge; {min_elevation_deg:.0f}&deg;)</h2>
      {table_html}
    </section>

    <footer>
      <p>satellite-pass-predictor &middot; SGP4 propagation via Skyfield &middot;
      TLE data from Celestrak</p>
    </footer>
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(document)

    return output_path
