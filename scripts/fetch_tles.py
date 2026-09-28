"""
Fetch current TLEs from Celestrak and publish them as a flat file mirror
(tle_<norad_id>.txt per satellite, plus metadata.json) -- run by
.github/workflows/refresh-tles.yml on a schedule, not by the app itself.

Why this exists at all: Streamlit Community Cloud cannot reach
celestrak.org (confirmed directly -- consistent TCP connect timeouts,
not an HTTP error our own retry/timeout handling could work around),
while a GitHub Actions runner reaches it fine. So a GitHub Actions
workflow runs this script on a schedule and republishes the results on
this repo's `tle-data` branch, which Streamlit Cloud *can* reach
(raw.githubusercontent.com is a generic file host, unrelated to
Celestrak's own infrastructure) -- see satellite_pass_predictor.tle_data
's `source="mirror"` and config.py's TLE_MIRROR_URL.

Reuses SATELLITES and CELESTRAK_URL from satellite_pass_predictor.config
(the single source of truth for which satellites are tracked and where
Celestrak's endpoint is -- this script has no NORAD list or URL of its
own to drift out of sync with the app's) and the same
TLE_FETCH_MAX_RETRIES/TLE_FETCH_RETRY_BASE_DELAY_SECONDS/
TLE_FETCH_TIMEOUT_SECONDS retry/timeout constants tle_data.py's own
fetch uses -- but not tle_data.py's actual fetch functions themselves,
since those are private (underscore-prefixed) internals of the app's
local-caching flow, and this script's validate-before-write, keep-the-
previous-file-on-failure responsibilities are different enough to
warrant their own small implementation rather than reusing that one.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Run as a plain script (`python scripts/fetch_tles.py`, from any CWD --
# this is how refresh-tles.yml invokes it), Python only adds this file's
# own directory (scripts/) to sys.path, not the repo root -- so
# `satellite_pass_predictor` (a sibling of scripts/, not a child) isn't
# importable without this. Inserted before that import specifically so
# it takes effect first; harmless if the repo root happens to already be
# on sys.path some other way (e.g. `python -m scripts.fetch_tles`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from skyfield.iokit import parse_tle_file

from satellite_pass_predictor.config import (
    CELESTRAK_URL,
    SATELLITES,
    TLE_FETCH_MAX_RETRIES,
    TLE_FETCH_RETRY_BASE_DELAY_SECONDS,
    TLE_FETCH_TIMEOUT_SECONDS,
)

USER_AGENT = (
    "satellite-pass-predictor-tle-mirror-refresh/1.0 "
    "(+https://github.com/obrusik2004/satellite-pass-predictor)"
)

_ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class FetchOutcome:
    """
    One satellite's fetch-and-validate result -- either a success (with
    the validated raw TLE bytes and its epoch, ready to write) or a
    failure (with a message, and nothing to write -- the caller must
    leave that satellite's existing published file and metadata entry
    untouched).
    """
    name: str
    norad_id: int
    url: str
    success: bool
    error: str | None = None
    tle_text: bytes | None = None
    tle_epoch: datetime | None = None


def _fetch_with_retries(url: str) -> bytes:
    """
    GET `url` with exponential backoff on request failures, reusing the
    same TLE_FETCH_MAX_RETRIES/TLE_FETCH_RETRY_BASE_DELAY_SECONDS/
    TLE_FETCH_TIMEOUT_SECONDS constants tle_data.py's own fetch uses (a
    single shared retry/timeout *policy*, even though the actual fetch
    code here is separate -- see this module's docstring for why).
    Raises OSError with the last underlying error if every attempt
    fails; never silently returns a response that raise_for_status()
    rejected.
    """
    last_error: Exception | None = None
    for attempt in range(TLE_FETCH_MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, timeout=TLE_FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            last_error = e
            if attempt < TLE_FETCH_MAX_RETRIES:
                time.sleep(TLE_FETCH_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
    raise OSError(f"cannot fetch {url}: {last_error}") from last_error


def _validate_tle(raw: bytes, expected_norad_id: int) -> tuple[bytes, datetime]:
    """
    Confirm `raw` is exactly one valid, correctly-identified TLE before
    anything gets written to disk -- an HTML error page, an empty body,
    or a response for the wrong satellite must never overwrite a good
    published file, so this is checked *before* write_outputs() ever
    sees this satellite's bytes, not after.

    Returns (raw, epoch) on success. Raises ValueError naming the exact
    problem otherwise: not exactly one TLE (covers both "0 TLEs" --
    empty body, HTML error page, "No GP data found" -- and "more than
    1", which shouldn't happen for a single-CATNR request but would
    indicate something unexpected about the response), or a NORAD ID
    that doesn't match what was actually requested.
    """
    entries = list(parse_tle_file(raw.splitlines()))
    if len(entries) != 1:
        raise ValueError(
            f"expected exactly 1 TLE, got {len(entries)} -- the response may "
            f"be an HTML error page, an empty/\"No GP data found\" body, or "
            f"contain more entries than expected"
        )
    sat = entries[0]
    actual_norad_id = sat.model.satnum
    if actual_norad_id != expected_norad_id:
        raise ValueError(
            f"NORAD ID mismatch: requested {expected_norad_id}, response is "
            f"for {actual_norad_id}"
        )
    return raw, sat.epoch.utc_datetime()


def fetch_one(name: str, norad_id: int) -> FetchOutcome:
    """Fetch and validate one satellite's TLE from Celestrak. Never
    raises -- any failure (network or validation) is captured in the
    returned FetchOutcome instead, so one satellite's failure can't
    stop the rest of the run."""
    url = CELESTRAK_URL.format(norad_id=norad_id)
    try:
        raw = _fetch_with_retries(url)
        tle_text, tle_epoch = _validate_tle(raw, norad_id)
    except (OSError, ValueError) as e:
        return FetchOutcome(name=name, norad_id=norad_id, url=url, success=False, error=str(e))
    return FetchOutcome(
        name=name, norad_id=norad_id, url=url, success=True,
        tle_text=tle_text, tle_epoch=tle_epoch,
    )


def load_existing_metadata(output_dir: Path) -> dict[str, Any]:
    """
    Read a previously-published metadata.json from `output_dir` (the
    checked-out tle-data branch contents, per refresh-tles.yml) if one
    exists, so write_outputs() can carry forward per-satellite entries
    for whatever fails *this* run rather than losing that history.

    Returns an empty-but-valid structure -- never raises -- for a
    missing file (the very first run, before tle-data has ever been
    published), unreadable file, or malformed JSON: any of those just
    means "nothing to carry forward," not a reason to fail the run.
    """
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return {"generated_at": None, "satellites": {}}
    try:
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"generated_at": None, "satellites": {}}
    if not isinstance(data, dict) or not isinstance(data.get("satellites"), dict):
        return {"generated_at": None, "satellites": {}}
    return data


def write_outputs(
    output_dir: Path,
    outcomes: list[FetchOutcome],
    previous_metadata: dict[str, Any],
    run_time: datetime,
) -> dict[str, Any]:
    """
    Write every successful outcome's TLE file and build the final
    metadata.json, then write that too. Always publishes whatever
    succeeded this run -- it does not wait to see whether every
    satellite succeeded before writing anything, since a partial
    failure should still publish the satellites that *did* refresh.

    For a satellite that failed this run: neither its tle_<norad_id>.txt
    nor its metadata.json entry are touched -- `previous_metadata`'s
    entry for it (if any) is carried forward unchanged, and its on-disk
    file (already sitting in `output_dir`, seeded from the current
    tle-data branch by refresh-tles.yml before this script ever runs)
    is simply left alone.

    `generated_at` is always updated to `run_time`, independent of
    per-satellite success -- it records "the workflow ran at this
    time," which is worth knowing even when some individual fetches
    failed (e.g. to notice the pipeline is alive and running on
    schedule at all).

    Returns the final metadata dict (mainly so tests can assert on it
    directly rather than re-reading the file this also writes).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    satellites_metadata: dict[str, Any] = dict(previous_metadata.get("satellites", {}))

    for outcome in outcomes:
        if not outcome.success:
            continue
        assert outcome.tle_text is not None and outcome.tle_epoch is not None  # guaranteed by fetch_one() on success
        (output_dir / f"tle_{outcome.norad_id}.txt").write_bytes(outcome.tle_text)
        satellites_metadata[str(outcome.norad_id)] = {
            "name": outcome.name,
            "fetched_at": run_time.strftime(_ISO_FORMAT),
            "tle_epoch": outcome.tle_epoch.strftime(_ISO_FORMAT),
            "source_url": outcome.url,
        }

    metadata: dict[str, Any] = {
        "generated_at": run_time.strftime(_ISO_FORMAT),
        "satellites": satellites_metadata,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch current TLEs from Celestrak and publish them as a flat "
            "file mirror (for the tle-data branch -- see "
            ".github/workflows/refresh-tles.yml)."
        )
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tle"),
        help="Directory to write tle_<norad_id>.txt and metadata.json into "
             "(default: tle/). Should already contain the tle-data branch's "
             "current contents, if any, so a failed satellite's previous "
             "file/metadata entry can be preserved.",
    )
    args = parser.parse_args(argv)

    run_time = datetime.now(timezone.utc)
    previous_metadata = load_existing_metadata(args.output_dir)

    outcomes = [fetch_one(name, norad_id) for name, norad_id in SATELLITES.items()]

    for outcome in outcomes:
        if outcome.success:
            assert outcome.tle_epoch is not None
            print(f"OK   {outcome.name} (NORAD {outcome.norad_id}): epoch {outcome.tle_epoch.isoformat()}")
        else:
            print(f"FAIL {outcome.name} (NORAD {outcome.norad_id}): {outcome.error}")

    # Publish whatever succeeded (and whatever was preserved from the
    # previous run for anything that failed) *before* deciding the exit
    # code -- a partial failure still gets partially published.
    write_outputs(args.output_dir, outcomes, previous_metadata, run_time)

    failures = [o for o in outcomes if not o.success]
    if failures:
        print(
            f"\n{len(failures)}/{len(outcomes)} satellite(s) failed to "
            f"refresh this run -- previously-published files for those "
            f"satellites were preserved and republished as-is."
        )
        return 1
    print(f"\nAll {len(outcomes)} satellite(s) refreshed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
