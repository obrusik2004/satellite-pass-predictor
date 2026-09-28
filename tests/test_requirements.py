"""
Tests that requirements-fetch.txt (the minimal dependency set
.github/workflows/refresh-tles.yml installs -- see that file's own header
comment for why it exists as a separate file from requirements.txt) can't
silently drift out of sync with requirements.txt (what Streamlit Cloud
actually installs). Both files are expected to pin the *same* version for
any package they share; nothing here enforces which packages belong in
either file, only that a shared package's pin can't quietly diverge.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _parse_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        pins[name.strip().lower()] = version.strip()
    return pins


def test_fetch_requirements_pins_match_main_requirements() -> None:
    main_pins = _parse_pins(REPO_ROOT / "requirements.txt")
    fetch_pins = _parse_pins(REPO_ROOT / "requirements-fetch.txt")

    assert fetch_pins, "requirements-fetch.txt should list at least one package"

    missing = sorted(set(fetch_pins) - set(main_pins))
    assert not missing, (
        f"requirements-fetch.txt pins package(s) not found in "
        f"requirements.txt at all: {missing}"
    )

    mismatched = sorted(
        name for name, version in fetch_pins.items() if main_pins[name] != version
    )
    assert not mismatched, (
        f"requirements-fetch.txt pins a different version than "
        f"requirements.txt for: {mismatched} -- keep the two in sync by hand"
    )
