"""Fetches current TLEs from Celestrak and publishes them as a flat file
mirror (tle_<norad_id>.txt per satellite plus metadata.json) -- run by
.github/workflows/refresh-tles.yml on a schedule, since Streamlit
Community Cloud cannot reach celestrak.org directly. See
tle_data.load_satellites()'s `source="mirror"` and config.TLE_MIRROR_URL.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from skyfield.iokit import parse_tle_file

from .config import (
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
    """One satellite's fetch-and-validate result: a success (with the
    validated TLE bytes and epoch) or a failure (with a message and
    nothing to write, so the caller leaves that satellite's last
    published file and metadata entry untouched)."""

    name: str
    norad_id: int
    url: str
    success: bool
    error: str | None = None
    tle_text: bytes | None = None
    tle_epoch: datetime | None = None


def _fetch_with_retries(url: str) -> bytes:
    """GET `url` with exponential backoff. Raises OSError with the last
    error if every attempt fails."""
    last_error: Exception | None = None
    for attempt in range(TLE_FETCH_MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=TLE_FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            last_error = e
            if attempt < TLE_FETCH_MAX_RETRIES:
                time.sleep(TLE_FETCH_RETRY_BASE_DELAY_SECONDS * (2**attempt))
    raise OSError(f"cannot fetch {url}: {last_error}") from last_error


def _validate_tle(raw: bytes, expected_norad_id: int) -> tuple[bytes, datetime]:
    """Confirm `raw` is exactly one valid TLE for `expected_norad_id`
    before anything is written to disk. Returns (raw, epoch); raises
    ValueError naming the problem otherwise."""
    entries = list(parse_tle_file(raw.splitlines()))
    if len(entries) != 1:
        raise ValueError(
            f"expected exactly 1 TLE, got {len(entries)} -- the response may "
            f'be an HTML error page, an empty/"No GP data found" body, or '
            f"contain more entries than expected"
        )
    sat = entries[0]
    actual_norad_id = sat.model.satnum
    if actual_norad_id != expected_norad_id:
        raise ValueError(
            f"NORAD ID mismatch: requested {expected_norad_id}, response is for {actual_norad_id}"
        )
    return raw, sat.epoch.utc_datetime()


def fetch_one(name: str, norad_id: int) -> FetchOutcome:
    """Fetch and validate one satellite's TLE. Never raises -- any
    failure is captured in the returned FetchOutcome instead."""
    url = CELESTRAK_URL.format(norad_id=norad_id)
    try:
        raw = _fetch_with_retries(url)
        tle_text, tle_epoch = _validate_tle(raw, norad_id)
    except (OSError, ValueError) as e:
        return FetchOutcome(name=name, norad_id=norad_id, url=url, success=False, error=str(e))
    return FetchOutcome(
        name=name,
        norad_id=norad_id,
        url=url,
        success=True,
        tle_text=tle_text,
        tle_epoch=tle_epoch,
    )


def load_existing_metadata(output_dir: Path) -> dict[str, Any]:
    """Read a previously-published metadata.json from `output_dir`, if
    any, so write_outputs() can carry forward entries for whatever fails
    this run. Never raises -- a missing or malformed file just means
    nothing to carry forward."""
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
    """Write every successful outcome's TLE file and the combined
    metadata.json. A satellite that failed this run keeps its previous
    file and metadata entry untouched; `generated_at` always advances.
    Returns the final metadata dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    satellites_metadata: dict[str, Any] = dict(previous_metadata.get("satellites", {}))

    for outcome in outcomes:
        if not outcome.success:
            continue
        assert outcome.tle_text is not None and outcome.tle_epoch is not None
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
        "--output-dir",
        type=Path,
        default=Path("tle"),
        help="Directory to write tle_<norad_id>.txt and metadata.json into "
        "(default: tle/). Should already contain the tle-data branch's "
        "current contents, if any, so a failed satellite's previous "
        "file/metadata entry can be preserved.",
    )
    args = parser.parse_args(argv)

    run_time = datetime.now(UTC)
    previous_metadata = load_existing_metadata(args.output_dir)

    outcomes = [fetch_one(name, norad_id) for name, norad_id in SATELLITES.items()]

    for outcome in outcomes:
        if outcome.success:
            assert outcome.tle_epoch is not None
            print(
                f"OK   {outcome.name} (NORAD {outcome.norad_id}): "
                f"epoch {outcome.tle_epoch.isoformat()}"
            )
        else:
            print(f"FAIL {outcome.name} (NORAD {outcome.norad_id}): {outcome.error}")

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
