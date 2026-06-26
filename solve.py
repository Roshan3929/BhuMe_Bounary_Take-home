#!/usr/bin/env python3
"""
SOLVE ONE VILLAGE
=================

Load a village bundle, correct it, write predictions.geojson, and self-score.

USAGE:
    uv run solve.py data/<village_slug>           # local (Silver/Gold): per-plot registration
    uv run solve.py data/<village_slug> --global  # global (Bronze): village shift only

ARGUMENTS:
    Positional: path to village folder (must contain input.geojson + imagery.tif)
    --global: use Bronze mode (global shift) instead of default local mode
    --search-m: override search window radius (default 45)
    --conf-floor: override confidence threshold (default 0.35)
"""

from __future__ import annotations

import sys
from pathlib import Path

from bhume import load, score, write_predictions
from method import correct_village


def main(argv):
    # Parse arguments
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}

    village_dir = args[0] if args else "data/34855_vadnerbhairav_chandavad_nashik"
    mode = "global" if "--global" in flags else "local"

    # LOAD: Read the village bundle
    village = load(village_dir)
    n_truth = 0 if village.example_truths is None else len(village.example_truths)
    print(f"\nLoaded {village.slug}:")
    print(f"  {len(village.plots)} plots · {n_truth} example truths · "
          f"boundaries={'yes' if village.boundaries_path else 'no'}")
    print(f"  mode={mode}")

    # CORRECT: Run the method
    print()
    preds = correct_village(village, mode=mode)

    # Report global shift
    gs = preds.attrs.get("global_shift", (0, 0))
    gq = preds.attrs.get("global_quality", 0)
    print()
    print(f"Global shift estimate: dx={gs[0]:.1f}m dy={gs[1]:.1f}m (quality={gq:.2f})")

    # Count outputs
    n_corr = int((preds.status == "corrected").sum())
    n_flag = int((preds.status == "flagged").sum())
    print(f"Output: {n_corr} corrected · {n_flag} flagged · {n_corr + n_flag} total")

    # WRITE: Save predictions.geojson
    out = write_predictions(Path(village_dir) / "predictions.geojson", preds)
    print(f"Wrote: {out}")

    # SCORE: Self-score against example truths (if available)
    if village.example_truths is not None:
        print()
        print(score(preds, village))
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
