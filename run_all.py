#!/usr/bin/env python3
"""
RUN ALL VILLAGES
================

This is the script graders will use to regenerate your predictions from scratch.

USAGE:
    uv run run_all.py                # all villages, local (Silver/Gold) mode
    uv run run_all.py --global       # all villages, Bronze mode

This finds every subdirectory of data/ that contains input.geojson + imagery.tif,
runs the method, writes predictions.geojson, and self-scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bhume import load, score, write_predictions
from method import correct_village


def village_dirs(root: Path):
    """Yield every village folder (has input.geojson + imagery.tif)."""
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "input.geojson").exists() and (d / "imagery.tif").exists():
            yield d


def main(argv):
    # Parse mode
    mode = "global" if "--global" in argv else "local"
    root = Path("data")

    # Find all villages
    found = list(village_dirs(root))
    if not found:
        print("No village folders under data/ (need input.geojson + imagery.tif)")
        return

    print(f"Found {len(found)} villages, mode={mode}\n")

    # Process each
    for d in found:
        print(f"{'=' * 70}")
        print(f"{d.name}")
        print(f"{'=' * 70}")

        try:
            village = load(d)
            preds = correct_village(village, mode=mode)

            gs = preds.attrs.get("global_shift", (0, 0))
            n_corr = int((preds.status == "corrected").sum())
            n_flag = int((preds.status == "flagged").sum())
            print(f"Drift: dx={gs[0]:.1f}m dy={gs[1]:.1f}m")
            print(f"Output: {n_corr} corrected · {n_flag} flagged")

            write_predictions(d / "predictions.geojson", preds)
            print(f"Wrote: {d.name}/predictions.geojson")

            if village.example_truths is not None:
                print()
                print(score(preds, village))
        except Exception as e:
            print(f"ERROR: {e}")

        print()


if __name__ == "__main__":
    main(sys.argv[1:])
