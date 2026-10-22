"""
OLSС refinement of YOLO instance-segmentation masks — reference implementation.

This is the core algorithm of the paper's framework: a conservative-union
bidirectional fusion of YOLO masks with LSC superpixels, gated by a multi-band
spectral test.

Pipeline
--------
1. LSC over-segmentation of the visible bands.
   n_segments  = int(felz_count * 0.75 + (H*W/150) * 0.25) * density_multiplier
   region_size = max(10, int(sqrt(H*W / n_segments)))
   cv2.ximgproc.createSuperpixelLSC(img, region_size, ratio=lsc_ratio)
2. Cotton Set: a superpixel is "cotton" iff the mean of EVERY normalised band
   exceeds t_cotton (default 0.17).
3. REMOVE pass (Step A): a superpixel that is NOT in the Cotton Set and whose
   overlap with the YOLO mask exceeds tau_rm is zeroed.
4. ADD pass (Step B): for each detected instance, a superpixel that IS in the
   Cotton Set and whose overlap with that instance exceeds tau_add is set to 1.
5. Post-processing: remove_small_objects(min_size) + binary_fill_holes.

IMPORTANT
---------
Do NOT apply morphological opening or dilation after step 4. The pixel-level
F1/IoU used in the paper is sensitive to mask thickness: a 3x3 dilation lowers
F1 from 0.8085 to 0.7477 on the 24-image test split. The archive's
post-processing is only remove_small_objects + binary_fill_holes.

Default parameters (paper v6 / "Ours" configuration)
----------------------------------------------------
    t_cotton = 0.17   tau_add = 0.85   tau_rm = 0.95
    lsc_ratio = 0.075 density_multiplier = 1.3
    min_size = 2

Verified: with the trained YOLOv12m five-band checkpoint this reproduces
F1 = 0.8085 / IoU = 0.6793 on the 24-image test split (archived reference
0.8082 / 0.6790; baseline YOLO 0.8048 / 0.6741).

Requires: numpy, opencv-contrib-python (for cv2.ximgproc), scikit-image, scipy.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.segmentation import felzenszwalb


# --------------------------------------------------------------------------- #
# 1. Input normalisation
# --------------------------------------------------------------------------- #
def normalize_bands(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Per-band percentile stretch to [0, 1].

    Parameters
    ----------
    img : (H, W, C) array, integer or float. C is normally 5 (B, G, R, RE, NIR).
    p_low, p_high : percentile bounds.

    Returns
    -------
    (H, W, C) float32 in [0, 1].
    """
    f = np.asarray(img, dtype=np.float32)
    lo = np.percentile(f, p_low, axis=(0, 1))
    hi = np.percentile(f, p_high, axis=(0, 1))
    den = hi - lo
    den[den == 0] = 1e-8
    return np.clip((f - lo) / den, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 2. LSC superpixels
# --------------------------------------------------------------------------- #
def estimate_n_segments(bands: np.ndarray, density_multiplier: float = 1.3) -> int:
    """Segment budget: 0.75 * Felzenszwalb region count + 0.25 * area term."""
    h, w = bands.shape[:2]
    total = h * w
    visible = (bands[:, :, :3] * 255).astype(np.uint8)
    felz = felzenszwalb(visible, scale=100, sigma=0.8, min_size=5, channel_axis=-1)
    felz_count = len(np.unique(felz))
    n = int(felz_count * 0.75 + int(total / 150) * 0.25)
    n = int(n * density_multiplier)
    return max(100, min(n, total // 20))


def lsc_superpixels(bands: np.ndarray, density_multiplier: float = 1.3,
                    lsc_ratio: float = 0.075, iterations: int = 20) -> np.ndarray:
    """LSC over-segmentation of the normalised visible bands.

    Returns an (H, W) int32 label map.
    """
    h, w = bands.shape[:2]
    n_seg = estimate_n_segments(bands, density_multiplier)
    region_size = max(10, int(np.sqrt(h * w / n_seg)))
    visible = (bands[:, :, :3] * 255).astype(np.uint8)
    lsc = cv2.ximgproc.createSuperpixelLSC(visible, region_size=region_size,
                                           ratio=lsc_ratio)
    lsc.iterate(iterations)
    lsc.enforceLabelConnectivity(max(3, region_size // 8))
    return lsc.getLabels()


# --------------------------------------------------------------------------- #
# 3. Cotton Set (multi-band spectral test)
# --------------------------------------------------------------------------- #
def cotton_set(segments: np.ndarray, bands: np.ndarray,
               t_cotton: float = 0.17) -> dict:
    """Map superpixel id -> bool: True iff every band mean exceeds t_cotton."""
    out = {}
    n_bands = bands.shape[2]
    for sid in np.unique(segments):
        sid = int(sid)
        if sid < 0:
            continue
        mask = segments == sid
        out[sid] = all(float(bands[:, :, i][mask].mean()) > t_cotton
                       for i in range(n_bands))
    return out


# --------------------------------------------------------------------------- #
# 4. Bidirectional fusion
# --------------------------------------------------------------------------- #
def olsc_refine(yolo_mask: np.ndarray, instances: list, segments: np.ndarray,
                is_cotton: dict, tau_add: float = 0.85, tau_rm: float = 0.95,
                min_size: int = 2) -> np.ndarray:
    """Apply the REMOVE pass, the ADD pass and post-processing.

    Parameters
    ----------
    yolo_mask : (H, W) merged binary mask of all detected instances.
    instances : list of (H, W) binary masks, one per detected instance.
    segments  : (H, W) superpixel label map.
    is_cotton : dict from :func:`cotton_set`.
    tau_add, tau_rm, min_size : see module docstring.

    Returns
    -------
    (H, W) uint8 refined mask.
    """
    final = (np.asarray(yolo_mask) > 0).astype(np.uint8)

    # Step A — REMOVE: prune non-cotton superpixels almost fully covered by the mask
    covered = np.unique(segments[final > 0])
    for sid in covered:
        sid = int(sid)
        if is_cotton.get(sid, False):
            continue
        sp = segments == sid
        area = sp.sum()
        if area and (sp & (final > 0)).sum() / area > tau_rm:
            final[sp] = 0

    # Step B — ADD: absorb cotton superpixels that largely overlap an instance
    for inst in instances:
        inst_b = np.asarray(inst) > 0
        for sid in np.unique(segments[inst_b]):
            sid = int(sid)
            if not is_cotton.get(sid, False):
                continue
            sp = segments == sid
            area = sp.sum()
            if area and (sp & inst_b).sum() / area > tau_add:
                final[sp] = 1

    # Post-processing — do NOT add morphology here (see module docstring)
    final = remove_small_objects(final.astype(bool), min_size=min_size)
    final = ndimage.binary_fill_holes(final)
    return final.astype(np.uint8)


# --------------------------------------------------------------------------- #
# 5. Convenience wrapper
# --------------------------------------------------------------------------- #
def refine_mask(yolo_mask: np.ndarray, instances: list, raw_image: np.ndarray,
                t_cotton: float = 0.17, tau_add: float = 0.85,
                tau_rm: float = 0.95, density_multiplier: float = 1.3,
                lsc_ratio: float = 0.075, min_size: int = 2) -> np.ndarray:
    """Run the full refinement from a raw multispectral image and YOLO output."""
    bands = normalize_bands(raw_image)
    segments = lsc_superpixels(bands, density_multiplier, lsc_ratio)
    is_cotton = cotton_set(segments, bands, t_cotton)
    return olsc_refine(yolo_mask, instances, segments, is_cotton,
                       tau_add, tau_rm, min_size)


# --------------------------------------------------------------------------- #
# 6. Metrics (pixel level, as used in the paper)
# --------------------------------------------------------------------------- #
def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """F1, IoU, recall and precision on the merged foreground mask."""
    p = np.asarray(pred) > 0
    g = np.asarray(gt) > 0
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "F1": 2 * precision * recall / (precision + recall + 1e-8),
        "IoU": tp / (tp + fp + fn + 1e-8),
        "recall": recall,
        "precision": precision,
    }


def polygons_to_mask(label_path: str, height: int, width: int) -> np.ndarray:
    """Rasterise a YOLO polygon-segmentation label file to a binary mask."""
    mask = np.zeros((height, width), np.uint8)
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            xy = np.asarray(parts[1:], dtype=np.float32)
            pts = np.stack([xy[0::2] * width, xy[1::2] * height], -1).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask
