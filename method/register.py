"""
IMAGE REGISTRATION CORE
=======================

This file finds the translation (shift in x, y) that best lands a plot's outline onto
the real field edges visible in satellite imagery.

KEY IDEA:
The drift between official boundaries and real fields is mostly a TRANSLATION (a uniform
shift across the whole plot). We recover it by:

  1. Detect edges in the image (Sobel gradient) + use the pre-computed boundary hints
  2. Build a "distance to nearest edge" map (distance transform)
  3. Rasterize the plot outline and slide it across the search window
  4. For each candidate shift, compute how close the outline pixels land to edges
  5. Pick the shift that minimizes that distance

This is called CHAMFER MATCHING in computer vision. It's a classical, parameter-light
approach: no ML, no training, just geometry.

WHY NOT OPENCV/SKIMAGE?
The kit doesn't have these deps, so we use numpy + scipy primitives. Same algorithm,
just lower-level.

WHY DO WE GET CONFIDENCE FOR FREE?
The cost surface (how good each shift is) tells us how *confident* we should be:
  - Sharp, isolated minimum → confident (one clear field boundary)
  - Flat surface → not confident (no visible edges, maybe canopy)
  - Low residual after the shift → confident (outline actually sits on edges)
  - High residual → not confident (outline still far from edges)

These signals flow into the confidence fusion in decide.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt, sobel

from bhume.geo import geom_to_imagery_crs


# =============================================================================
# PART 1: The result container
# =============================================================================

@dataclass
class Registration:
    """
    The result of registering one plot's outline against real edges.

    When we slide the outline over the search window and test each shift, we get back
    a bunch of numbers. This container holds all of them, so the caller (decide.py)
    can fuse them into a single confidence score.

    Attributes:
        dcol, drow: the best pixel shift we found (column, row deltas in the image grid)
        dx_crs, dy_crs: the same shift converted to real-world metres (EPSG:3857 units)
        shift_px: magnitude of the shift in pixels (sqrt(drow² + dcol²))
        cost_min: residual at the best shift — mean distance from outline pixels to edges
                  0 = on edges, high = far from edges
        cost_zero: residual at NO shift (the official position) — used to measure improvement
        peak_sharpness: 0..1, how isolated and deep the minimum is
                  1 = sharp, clear valley; 0 = flat surface (no edge evidence)
        edge_density: fraction of patch pixels that are edges (how much evidence overall)
        n_outline_px: how many pixels the outline spans (sanity check)
        ok: False if there was nothing to register against (empty edge map)
    """
    # Pixel shifts
    dcol: int
    drow: int
    # Real-world shifts (metres in EPSG:3857, which is web-mercator)
    dx_crs: float
    dy_crs: float
    # Magnitudes and costs
    shift_px: float
    cost_min: float       # residual at best shift
    cost_zero: float      # residual at no shift
    # Confidence raw signals
    peak_sharpness: float  # how isolated the minimum is
    edge_density: float    # how many edges in the patch
    n_outline_px: int
    # Success flag
    ok: bool


# =============================================================================
# PART 2: Edge evidence — what we trust as "the real field boundary"
# =============================================================================

def _sobel_edges(rgb: np.ndarray) -> np.ndarray:
    """
    Compute edge map from satellite imagery using Sobel gradient WITH COHERENCE FILTERING.

    WHAT IS SOBEL?
    Sobel is a gradient operator: it measures how fast pixel brightness changes at each
    location. Field boundaries have sharp brightness changes (dark field → bright sky),
    so high gradient = likely an edge.

    THE PROBLEM WITH PLAIN SOBEL:
    Trees have many internal edges (branches, leaves) pointing in many directions.
    Sobel detects ALL of them, creating noise. Trees look like edges everywhere.

    THE SOLUTION: COHERENCE FILTERING
    Real field boundaries have consistent gradient direction: the gradient points
    perpendicular to the boundary (one clear direction).
    Trees have INCOHERENT gradients: many directions, conflicting signals.

    By requiring BOTH horizontal (gx) AND vertical (gy) gradients to be strong,
    we filter out tree noise and keep real boundaries.

    ALGORITHM:
    1. Convert RGB to grayscale
    2. Compute horizontal gradient (gx) and vertical gradient (gy) using Sobel
    3. Find where BOTH gx AND gy are strong (coherent edge)
    4. Apply coherence filter: mag × coherence
    5. Normalize by percentile

    RETURNS:
    A (H, W) array, values in [0, 1], higher = more edge-like AND coherent.
    """
    # Convert RGB to grayscale: average the three colour channels
    if rgb.ndim == 3:
        gray = rgb.mean(axis=2).astype(np.float32)
    else:
        gray = rgb.astype(np.float32)

    # Compute Sobel gradients: rate of change in x and y directions
    gx = sobel(gray, axis=1)  # horizontal (left-right) gradient
    gy = sobel(gray, axis=0)  # vertical (top-bottom) gradient

    # Magnitude: how much the brightness is changing at this pixel
    mag = np.hypot(gx, gy)
    
    # ============================================================================
    # ADAPTIVE COHERENCE: Adjust based on edge sparsity
    # ============================================================================
    
    # First pass: estimate edge coverage
    mag_norm_trial = np.clip(mag / (np.percentile(mag, 99.0) or 1.0), 0.0, 1.0)
    initial_coverage = (mag_norm_trial > 0.3).sum() / mag_norm_trial.size
    
    if initial_coverage < 0.05:
        # SPARSE EDGES (Malatavadi): Plain Sobel, NO coherence filter
        # Use all edges, don't filter anything
        mag_filtered = mag       
    
    else:
        # DENSE EDGES (Vadnerbhairav-like): be strict
        # Use AND logic: need BOTH gx AND gy strong
        # This filters tree noise while keeping real boundaries
        
        gx_pct = np.percentile(np.abs(gx), 75)  # Top 25%
        gy_pct = np.percentile(np.abs(gy), 75)
        
        gx_strong = np.abs(gx) > gx_pct
        gy_strong = np.abs(gy) > gy_pct
        
        # AND: multiply (need both strong)
        coherence = (gx_strong.astype(float) * gy_strong.astype(float))
    
        # Apply coherence filter
        mag_filtered = mag * coherence
    
    # Robust normalization
    hi = np.percentile(mag_filtered, 99.0) or 1.0
    return np.clip(mag_filtered / hi, 0.0, 1.0)


def _aligned_boundaries(boundaries_path, patch) -> np.ndarray | None:
    """
    Resample boundaries.tif onto the imagery patch's exact pixel grid.

    THE PROBLEM:
    boundaries.tif and imagery.tif live on different CRS/resolution:
      - imagery: 1.19 m/px in EPSG:3857
      - boundaries: 2.39 m/px in EPSG:3857 (exactly half resolution)

    When we overlay them, pixels don't line up. This function reprojects the boundary
    raster onto the imagery's grid so we can fuse them pixel-for-pixel.

    RETURNS:
    A (H, W) array in [0, 1] of boundary confidence, or None if unavailable.
    """
    if boundaries_path is None:
        return None

    H, W = patch.image.shape[:2]
    dst = np.zeros((H, W), dtype=np.float32)

    try:
        with rasterio.open(boundaries_path) as bsrc:
            band = bsrc.read(1).astype(np.float32)
            # Reproject: read from boundaries_path's grid, write to patch's grid
            reproject(
                source=band,
                destination=dst,
                src_transform=bsrc.transform,       # boundaries.tif's affine transform
                src_crs=bsrc.crs,                    # boundaries.tif's CRS (3857)
                dst_transform=patch.transform,      # patch's affine transform
                dst_crs=patch.crs,                   # patch's CRS (3857)
                resampling=Resampling.bilinear,    # smooth interpolation (not nearest)
            )
    except Exception:
        return None

    # Normalize to [0, 1]
    m = float(dst.max())
    return dst / m if m > 0 else None


def edge_evidence(patch, boundaries_path, w_image: float = 1.2, w_hint: float = 0.9) -> tuple[np.ndarray, float]:
    """
    Fuse two independent edge sources: satellite image + boundary hints.g

    FUSION STRATEGY:
    Both are confidence maps in [0, 1]. We take the *maximum* at each pixel.
    Why maximum, not average? Because either signal being high is good enough.
    If the image shows an edge OR the hints show an edge, we trust that location.

    WEIGHTS:
    w_image, w_hint scale the two signals before fusion. Start with both at 1.0.
    If hints are too noisy later, lower w_hint. If image edges are too noisy, lower w_image.

    EDGE BINARY MASK:
    We also return a binary mask (True = edge, False = not edge) by thresholding.
    The threshold is the 88th percentile of fused confidence — keep the top 12% as
    "strong enough to be a real edge". This is per-patch, adaptive.

    RETURNS:
    (edge_bool, density) where:
      - edge_bool is (H, W) boolean, True = edge pixel
      - density is fraction of pixels that are edges (0..1)
    """
    img_e = _sobel_edges(patch.image)
    hint = _aligned_boundaries(boundaries_path, patch)

    if hint is not None:
        fused = np.maximum(w_image * img_e, w_hint * hint)
    else:
        fused = img_e

    fused = np.clip(fused, 0.0, 1.0)

    # Threshold: 88th percentile means "top 12% of values"
    thr = np.percentile(fused, 88.0)
    edge_bool = fused >= max(thr, 1e-6)  # at least 1e-6 to handle all-zeros case
    density = float(edge_bool.mean())

    return edge_bool, density


# =============================================================================
# PART 3: Plot outline rasterization
# =============================================================================

def _outline_pixels(src, geom_4326, patch) -> np.ndarray:
    """
    Burn the plot's *outline* (boundary, not filled) into the patch's image grid.

    WHAT IS AN OUTLINE?
    The plot's official geometry is a Polygon. The outline is just the ring of points
    that form its perimeter — the boundary, not the interior.

    RASTERIZATION:
    Convert this ring into pixels. Each pixel that intersects the ring gets marked.
    We use all_touched=True so even thin touches count (important for small plots).

    RETURNS:
    A (N, 2) array of [row, col] pixel coordinates where the outline touches.
    """
    geom_img = geom_to_imagery_crs(src, geom_4326)

    H, W = patch.image.shape[:2]
    try:
        mask = rio_features.rasterize(
            [(geom_img.boundary, 1)],  # geom_img.boundary = the ring, not filled
            out_shape=(H, W),
            transform=patch.transform,
            all_touched=True,
            dtype="uint8",
        )
    except Exception:
        return np.empty((0, 2), dtype=np.int64)

    rows, cols = np.nonzero(mask)
    return np.stack([rows, cols], axis=1)


# =============================================================================
# PART 4: The chamfer matching search
# =============================================================================

def register_plot(src, geom_4326, patch, boundaries_path, search_px: int = 25,
                  prior_shift_px: tuple[int, int] = (0, 0)) -> Registration:
    """
    Find the translation that best lands the plot outline on real field edges.

    THIS IS THE CORE ALGORITHM.

    STEPS:
    1. Build edge evidence (Sobel + boundary hints)
    2. Rasterize the plot outline
    3. Compute distance-transform: for each pixel, "how far is the nearest edge?"
    4. For each candidate shift in the search window:
       - Move the outline by that shift
       - Sample the distance values under the moved outline pixels
       - Average those distances = cost at that shift
    5. Pick the shift with the lowest average distance (outline closest to edges)

    SEARCH WINDOW:
    We don't try all possible shifts (-∞ to +∞). Instead, we search a window around
    the prior_shift_px (which defaults to 0,0 but in practice is the village-wide drift).
    This keeps computation fast and keeps the solution stable.

    prior_shift_px = (drow_prior, dcol_prior):
      The search window is centered at this point. If the true shift is outside the
      window, we'll miss it, but the global prior usually gets us close.

    RETURNS:
    A Registration with the shift, costs, and confidence raw signals.
    """
    # Get the edge evidence
    edge_bool, density = edge_evidence(patch, boundaries_path)
    # Get the outline pixels to match
    outline = _outline_pixels(src, geom_4326, patch)

    H, W = edge_bool.shape

    # Distance transform: for each pixel, distance to nearest edge
    if edge_bool.any():
        dist = distance_transform_edt(~edge_bool).astype(np.float32)
    else:
        # No edges at all (e.g., entirely under canopy) -> all pixels are far from edge
        dist = np.full((H, W), float(max(H, W)), dtype=np.float32)

    # Sanity check: do we have anything to match?
    if outline.shape[0] == 0 or not edge_bool.any():
        return Registration(0, 0, 0.0, 0.0, 0.0, float(dist.mean()), float(dist.mean()),
                            0.0, density, int(outline.shape[0]), ok=False)

    # Extract outline pixel coordinates
    r0, c0 = outline[:, 0], outline[:, 1]
    pr, pc = prior_shift_px

    # SEARCH: try shifts in a window around the prior
    def cost_at(drow: int, dcol: int) -> float:
        """Chamfer cost: mean distance from outline pixels (at shifted location) to edges."""
        rr = np.clip(r0 + drow, 0, H - 1)
        cc = np.clip(c0 + dcol, 0, W - 1)
        return float(dist[rr, cc].mean())

    # Generate all candidate shifts in the window
    offsets = range(-search_px, search_px + 1)
    surface = np.empty((len(offsets), len(offsets)), dtype=np.float32)

    for i, drow in enumerate(offsets):
        for j, dcol in enumerate(offsets):
            # Test shift: prior + this offset
            surface[i, j] = cost_at(pr + drow, pc + dcol)

    # Find the minimum
    flat_idx = int(np.argmin(surface))
    bi, bj = divmod(flat_idx, surface.shape[1])
    best_drow = pr + (bi - search_px)
    best_dcol = pc + (bj - search_px)
    cost_min = float(surface[bi, bj])
    cost_zero = cost_at(0, 0)

    # PEAK SHARPNESS: is the minimum deep and isolated, or is the surface flat?
    # Compare the minimum to the median of the surface.
    s_med = float(np.median(surface))
    s_std = float(surface.std()) + 1e-6
    # Formula: (median - min) / (median + std) normalized to [0, 1]
    # Intuition: sharp minimum -> numerator is large, denominator small -> high sharpness
    peak_sharpness = float(np.clip((s_med - cost_min) / (s_med + s_std), 0.0, 1.0))

    # Convert pixel shifts to real-world distances (metres)
    a = patch.transform  # Affine transform: pixel (col, row) -> (x, y) in EPSG:3857
    dx_crs = a.a * best_dcol + a.b * best_drow
    dy_crs = a.d * best_dcol + a.e * best_drow

    return Registration(
        dcol=best_dcol,
        drow=best_drow,
        dx_crs=dx_crs,
        dy_crs=dy_crs,
        shift_px=float(np.hypot(best_drow, best_dcol)),
        cost_min=cost_min,
        cost_zero=cost_zero,
        peak_sharpness=peak_sharpness,
        edge_density=density,
        n_outline_px=int(outline.shape[0]),
        ok=True,
    )
