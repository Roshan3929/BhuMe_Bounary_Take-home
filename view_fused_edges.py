"""
View the fused edge map from imagery.tif + boundaries.tif
Run: python3 view_fused_edges.py
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from rasterio.warp import reproject, Resampling

# Configure
village = '34855_vadnerbhairav_chandavad_nashik'  # Change this to test different villages
w_image = 1.2           # Change to your current weights
w_hint = 0.9

print(f"Loading {village}...")

# STEP 1: Load imagery
with rasterio.open(f'data/{village}/imagery.tif') as src:
    rgb = src.read([1, 2, 3])
    rgb = np.transpose(rgb, (1, 2, 0))
    transform = src.transform
    crs = src.crs

print(f"  Size: {rgb.shape[0]}×{rgb.shape[1]}")

# STEP 2: Compute Sobel edges (exactly as method/register.py does)
gray = rgb.mean(axis=2).astype(np.float32)
gx = sobel(gray, axis=1)
gy = sobel(gray, axis=0)
mag = np.hypot(gx, gy)
hi = np.percentile(mag, 99.0) or 1.0
sobel_edges = np.clip(mag / hi, 0.0, 1.0)

# STEP 3: Load and resample boundaries to imagery grid
H, W = rgb.shape[:2]
boundaries_resampled = np.zeros((H, W), dtype=np.float32)

with rasterio.open(f'data/{village}/boundaries.tif') as bsrc:
    band = bsrc.read(1).astype(np.float32)
    reproject(
        source=band,
        destination=boundaries_resampled,
        src_transform=bsrc.transform,
        src_crs=bsrc.crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )

m = float(boundaries_resampled.max())
boundaries_norm = boundaries_resampled / m if m > 0 else boundaries_resampled

# STEP 4: Fuse edges (this is YOUR fused map)
fused = np.maximum(w_image * sobel_edges, w_hint * boundaries_norm)

# STEP 5: Binary threshold (88th percentile, as in edge_evidence())
thr = np.percentile(fused, 88.0)
edge_bool = fused >= max(thr, 1e-6)

# STEP 6: Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1
axes[0, 0].imshow(rgb.astype(np.uint8))
axes[0, 0].set_title('Original Imagery')
axes[0, 0].axis('off')

im1 = axes[0, 1].imshow(sobel_edges, cmap='hot')
axes[0, 1].set_title(f'Sobel Edges (weight={w_image})')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(boundaries_norm, cmap='hot')
axes[0, 2].set_title(f'Boundaries (weight={w_hint})')
plt.colorbar(im2, ax=axes[0, 2])

# Row 2 (THE KEY ONE)
im3 = axes[1, 0].imshow(fused, cmap='hot')
axes[1, 0].set_title(f'✓ FUSED EDGES ✓\nmax({w_image}*sobel, {w_hint}*boundaries)')
plt.colorbar(im3, ax=axes[1, 0])

axes[1, 1].imshow(edge_bool, cmap='gray')
axes[1, 1].set_title(f'Binary Mask (threshold={thr:.4f})')
axes[1, 1].axis('off')

# Overlay
ax = axes[1, 2]
ax.imshow(rgb.astype(np.uint8), alpha=0.7)
fused_colored = plt.cm.hot(fused)
ax.imshow(fused_colored, alpha=0.5)
ax.set_title('Imagery + Fused Edges')
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{village}_fused_edges.png', dpi=150)
print(f"\nSaved: {village}_fused_edges.png\n")

# Print stats
print("STATISTICS:")
print(f"  Sobel edge pixels (>0.1):    {(sobel_edges > 0.1).sum()/sobel_edges.size*100:.1f}%")
print(f"  Boundaries pixels (>0):      {(boundaries_norm > 0).sum()/boundaries_norm.size*100:.1f}%")
print(f"  Fused high-conf (>0.5):      {(fused > 0.5).sum()/fused.size*100:.1f}%")
print(f"  Binary edge mask (>thresh):  {edge_bool.sum()/edge_bool.size*100:.1f}%")

# Activate interactive zoom mode if available
if hasattr(fig.canvas.manager, "toolbar") and fig.canvas.manager.toolbar is not None:
    try:
        fig.canvas.manager.toolbar.zoom()
    except Exception:
        pass

plt.show()