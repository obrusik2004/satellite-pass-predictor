"""Repo-root shim: `python main.py` runs the CLI without installing the
package. The installed console script (`satpass`) calls the same
satellite_pass_predictor.cli.main()."""

from satellite_pass_predictor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
