# BhuMe Cadastral Boundary Correction

Corrects drifted cadastral plot outlines in Maharashtra by registering them against
satellite imagery using Chamfer matching, then scoring each correction with a
calibrated confidence estimate.

**Key results on Vadnerbhairav village (6 example truths):**

| Metric | Official baseline | This method |
|---|---|---|
| Median IoU | 0.612 | **0.874** |
| Centroid error | — | **3.5 m** |
| Spearman (conf vs IoU) | — | **0.943** |
| Truths corrected | — | 6 / 6 |

---

## How it works

Old paper maps were georeferenced onto satellite imagery, causing plot outlines to
drift 5–25 metres from their real positions. This method detects where the real
field boundaries are in the satellite image, finds the shift that best aligns each
plot's outline to those edges, and decides whether the correction is trustworthy
enough to apply.

The pipeline has five stages:

```
imagery.tif ─┐
             ├─► Sobel + coherence filter ─┐
boundaries.tif ──► resample + normalize ───┴─► fused edge map
                                                      │
input.geojson ──► area-ratio triage ──────────────────┤
                                                      ▼
                                          Chamfer registration
                                          (search ±38 px window)
                                                      │
                                          Confidence fusion
                                          (geometric mean of 5 signals)
                                                      │
                              ┌───────────┬───────────┘
                              ▼           ▼           ▼
                          leave alone   flag       correct
                          (shift<1.5px) (conf<0.35) (apply shift)
```

---

## Project layout

```
BhuMe_BoundaryCorrection/
│
├── pyproject.toml          # uv project config — add dependencies here
├── uv.lock                 # locked dependency versions (auto-generated)
├── .gitignore              # excludes *.tif (large rasters); keeps predictions.geojson
│
├── solve.py                # entry point: run on one village
├── run_all.py              # entry point: run on every village under data/
│
├── bhume/                  # PROVIDED starter kit — do not modify
│   ├── __init__.py         # loads a village into a Village dataclass
│   ├── geo.py              # CRS helpers, geometry utilities
│   ├── io.py               # GeoJSON read/write, predictions contract
│   ├── baseline.py         # reference baseline (global-shift only)
│   └── score.py            # IoU, centroid error, Spearman, AUC scoring
│
├── method/                 # YOUR implementation
│   ├── __init__.py         # public exports: triage, register_plot, confidence, correct_village
│   ├── register.py         # Sobel edges, edge fusion, Chamfer registration
│   ├── decide.py           # area-ratio triage, confidence signal fusion
│   └── pipeline.py         # global shift estimation, per-plot orchestration
│
└── data/
    ├── 34855_vadnerbhairav_chandavad_nashik/
    │   ├── imagery.tif              # satellite RGB raster (gitignored)
    │   ├── boundaries.tif           # pre-computed edge hints (gitignored)
    │   ├── input.geojson            # official (drifted) plot outlines
    │   ├── example_truths.geojson   # ground-truth outlines for 6 plots
    │   └── predictions.geojson      # written by solve.py / run_all.py
    └── Malatavadi/
        ├── imagery.tif
        ├── boundaries.tif
        ├── input.geojson
        ├── example_truths.geojson
        └── predictions.geojson
```

---

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Add a new dependency (e.g. matplotlib for visualisation)
# Edit pyproject.toml → add "matplotlib>=3.7" to the dependencies list
uv sync
```

---

## Running

```bash
# Single village
uv run solve.py data/34855_vadnerbhairav_chandavad_nashik

# All villages under data/
uv run run_all.py
```

Output is written to `data/<village>/predictions.geojson`. If `example_truths.geojson`
exists, scores are printed to stdout immediately after.

---

## File reference

### `solve.py`

Entry point for a single village. Accepts one positional argument — the path to a
village directory. Loads the village, calls `correct_village()`, writes
`predictions.geojson`, and scores against example truths if available.

```bash
uv run solve.py data/Malatavadi
```

---

### `run_all.py`

Discovers every village under `data/` and runs `solve.py` logic on each in sequence.
Prints a per-village summary block. Use this for the final submission run.

```bash
uv run run_all.py
```

---

### `method/register.py`

Contains all image-processing logic: edge detection, edge fusion, and the Chamfer
registration search.

**`_sobel_edges(rgb)`**
Converts RGB to grayscale, computes horizontal (`gx`) and vertical (`gy`) Sobel
gradients, applies a coherence filter (keeps pixels where both `gx` and `gy` are
strong — this removes tree noise while keeping real field boundaries), then
normalises by the 99th-percentile magnitude.

The coherence filter is adaptive: sparse-edge patches (edge coverage < 5%) skip
the filter and use plain magnitude so there are enough edges to register against.

**`edge_evidence(patch_rgb, boundaries_path, w_image, w_hint)`**
Runs `_sobel_edges` on the image patch, resamples `boundaries.tif` to match the
patch grid, and fuses the two with:

```
fused = max( w_image × sobel_edges,  w_hint × boundaries_norm )
```

Returns `(edge_bool, edge_density)` — a binary edge mask at the 88th-percentile
threshold and the fraction of the patch that is edge.

**`register_plot(src, geometry, patch, boundaries_path, search_px)`**
The core registration function. Steps:

1. Rasterise the plot outline into a binary mask.
2. Run `edge_evidence` on the patch.
3. Compute a distance transform on the inverted edge mask (distance of each pixel to
   the nearest edge).
4. Slide the outline mask across a `±search_px` window and compute mean distance at
   each candidate shift — this is the Chamfer cost.
5. Find the shift with minimum cost.
6. Compute `peak_sharpness`: how isolated the minimum is relative to the cost surface.
   A sharp, narrow minimum means a confident registration; a flat surface means
   ambiguous edges.

Returns a `Registration` dataclass containing `dx_crs`, `dy_crs` (shift in metres),
`shift_px`, `cost_min`, `peak_sharpness`, `edge_density`, and `ok`.

**Tunable parameters in `register.py`:**

| Parameter | Default | Effect |
|---|---|---|
| `w_image` | 1.2 | Weight for Sobel edges from imagery. Increase if `boundaries.tif` is sparse. |
| `w_hint` | 0.9 | Weight for pre-computed boundary hints. Decrease if hints are unreliable. |
| Edge threshold percentile | 88th | Fraction of fused pixels that become "edge" in the binary mask. Lower = more edges. |
| Sobel normalisation percentile | 99th | Robust ceiling for Sobel magnitude. Lower = more edges detected. |

---

### `method/decide.py`

Handles two decisions: whether to attempt a correction, and how confident to be.

**`Triage` dataclass / `triage(plot_row)`**
Checks the area ratio: `drawn_area / recorded_area`. If this falls outside
`[RATIO_LOW, RATIO_HIGH]` (default `[0.70, 1.40]`), the plot's shape is probably
wrong (not just shifted) and it should be flagged without attempting registration.

**`confidence(reg, tri, global_shift_crs, weights)`**
Fuses five independent signals into a single 0–1 confidence score using a weighted
geometric mean. Geometric mean semantics mean a single low signal tanks the overall
score — you cannot compensate a bad `peak` with a good `area` score.

| Signal | Default weight | What it measures |
|---|---|---|
| `peak` | 1.0 | Sharpness of the registration cost minimum. High = unambiguous best shift. |
| `residual` | 1.0 | How close the outline lands to edges after shifting. High = good fit. |
| `edge` | 0.7 | Edge density in the patch. High = enough evidence to trust the registration. |
| `area` | 0.8 | How well drawn and recorded areas match. High = shape is geometrically sane. |
| `agree` | 0.8 | How close this plot's shift is to the village-wide median shift. |

The `agree` weight was reduced from 1.2 to 0.8 during tuning. At 1.2 it dominated
the geometric mean, causing confidence to reflect "does this plot agree with the
village average" rather than "did this plot's own registration succeed". Lowering it
to 0.8 let `peak` and `residual` — the per-plot quality signals — drive confidence,
which pushed Spearman from 0.600 to 0.943.

Weights adapt by edge density: sparse-edge villages (density < 0.05) use
`peak=0.6, edge=0.9` to be more tolerant of ambiguous registrations where edges are
scarce.

**Tunable parameters in `decide.py`:**

| Parameter | Default | Effect |
|---|---|---|
| `RATIO_LOW` | 0.70 | Lower bound for sane area ratio. Decrease to attempt more plots. |
| `RATIO_HIGH` | 1.40 | Upper bound for sane area ratio. Increase to attempt more plots. |
| `peak` weight | 1.0 | Raise to penalise ambiguous registrations more heavily. |
| `agree` weight | 0.8 | Raise if village drift is uniform. Lower if plots drift independently. |
| Edge density threshold | 0.05 | Below this, the sparse-edge weight set is applied. |

---

### `method/pipeline.py`

Orchestrates the full correction loop.

**`estimate_global_shift(village, src, search_px)`**
Samples ~120–150 plots, runs `register_plot` on each with a minimum quality filter
(`peak_sharpness ≥ 0.30`), and takes the robust median of their shifts as the
village-wide prior. This prior is used as:

- The search window centre for per-plot registration.
- The reference for the `agree` confidence signal.

Does not use `example_truths` — the prior is estimated from imagery alone so it
generalises to the hidden test set.

**`correct_village(village, mode, search_m, conf_floor, leave_alone_px)`**
Main correction loop. For each plot:

1. Run area-ratio triage — skip if outside band.
2. Register the plot against the fused edge map.
3. Compute confidence.
4. If `shift_px < leave_alone_px`: already aligned, keep original.
5. If `confidence < conf_floor`: too uncertain, flag.
6. Otherwise: apply shift and mark as corrected.

**Tunable parameters in `pipeline.py`:**

| Parameter | Default | Effect |
|---|---|---|
| `search_m` | 45.0 m | Registration search radius in metres. Increase if shifts saturate at window edge. |
| `conf_floor` | 0.35 | Minimum confidence to apply a correction. Lower = more corrections, higher risk. |
| `leave_alone_px` | 1.5 px | Plots with shifts smaller than this are considered already aligned. |

---

### `bhume/` — provided starter kit (read-only)

**`bhume/__init__.py`** — `load(village_dir)` returns a `Village` dataclass
containing paths to all data files and loaded GeoDataFrames for plots and truths.

**`bhume/geo.py`** — CRS conversion helpers (`geom_to_imagery_crs`), bounding box
utilities, and coordinate maths used across the codebase.

**`bhume/io.py`** — Reads and writes `predictions.geojson` in the contract format
expected by the grader. Each feature carries `plot_number`, `status`
(`corrected` / `flagged`), `confidence`, and `method_note`.

**`bhume/score.py`** — Scoring utilities: IoU (intersection over union), centroid
error in metres, Spearman rank correlation between confidence and IoU (calibration),
and AUC. Called automatically at the end of each `solve.py` run when truths exist.

**`bhume/baseline.py`** — Reference implementation using a single global shift
with no per-plot registration and flat confidence. Useful for sanity-checking that
your method actually improves over the naive approach.

---

## Data files

**`imagery.tif`** — Satellite RGB raster, EPSG:3857, ~1.2 m/pixel. This is the
primary ground truth. Sobel gradients computed from this image are the main signal
for edge detection and registration.

**`boundaries.tif`** — Pre-computed edge hints, same CRS, ~2.4 m/pixel (half
resolution). Binary: 0 or 255. Sparse and unreliable on vegetated villages (Malatavadi:
2.3% coverage). Acts as a secondary signal, fused with Sobel using `max()`.

**`input.geojson`** — Official plot outlines in EPSG:4326. These are the drifted
boundaries to be corrected. Each feature has `plot_number`, `map_area_sqm`
(drawn area), `recorded_area_sqm` (revenue record area), and `pot_kharaba_ha`
(wasteland area from revenue records).

**`example_truths.geojson`** — Ground-truth outlines for a small subset of plots
(3–6 per village). Used only for scoring during development. Not used by the
correction algorithm itself.

**`predictions.geojson`** — Written by `solve.py`. One feature per plot with the
corrected or flagged geometry, confidence score, and a method note explaining why
each decision was made.

---

## Key design decisions

**Why geometric mean for confidence?**
A geometric mean with per-signal weights implements "weakest-link" semantics: a
single very low signal (e.g. near-zero edge density) cannot be compensated by high
scores elsewhere. This makes confidence conservative, which is appropriate for a
system where incorrect corrections are worse than flags.

**Why lower the `agree` weight?**
At weight 1.2, the agreement signal dominated confidence, making the score reflect
village conformity rather than per-plot registration quality. Plots with perfect
local registrations were sometimes flagged because they disagreed slightly with the
village median. Lowering to 0.8 let `peak` and `residual` lead, which is what
pushed Spearman from 0.600 to 0.943.

**Why coherence filtering on Sobel?**
Plain Sobel detects all brightness gradients: road edges, tree branches, building
shadows. Real field boundaries produce a consistent gradient direction (perpendicular
to the boundary). By requiring both `gx` and `gy` to be strong (coherent), tree
noise (many conflicting directions) is removed while true boundaries are preserved.
This sharpens the Chamfer cost surface, raising `peak_sharpness` and improving both
registration accuracy and calibration.

**Why is Malatavadi mostly flagged?**
Malatavadi has 2.3% edge coverage in `boundaries.tif`, tiny plots (median 874 m²),
and non-uniform drift — each plot shifts in a different direction. The global shift
prior is meaningless on this village. Flagging ambiguous plots is the correct,
honest response; forcing corrections produced IoU 0.000 with 43-metre centroid
errors in testing.

---

## Dependencies

Managed via `uv` and `pyproject.toml`.

```
geopandas>=1.0
rasterio>=1.3
shapely>=2.0
numpy>=1.24
scipy>=1.10
pillow>=10.0
matplotlib>=3.7   # for visualisation scripts only — not required for solve.py
```

---

## Visualisation

Quick commands for inspecting results (require matplotlib):

```bash

# Fused edge map: Sobel vs boundaries vs fused side-by-side
uv run --with matplotlib python3 view_fused_edges.py


```

---

## Author

Roshan John
BhuMe Take-Home Challenge — June 2026
