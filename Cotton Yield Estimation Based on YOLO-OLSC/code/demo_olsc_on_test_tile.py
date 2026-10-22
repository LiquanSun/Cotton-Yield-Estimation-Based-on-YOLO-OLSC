"""
Minimal reproducibility demo for the OLSC refinement.

Given the trained YOLOv12m five-band checkpoint and one annotated test tile,
this script prints the baseline and refined pixel-level metrics.

Usage
-----
    python demo_olsc_on_test_tile.py  \
        --weights  weights/yolo12m_5c_best.pt \
        --image    test_tile/ts_000000000014.tiff \
        --label    test_tile/ts_000000000014.txt

Reproduces, for this configuration:
    baseline YOLO   F1 ~ 0.80   IoU ~ 0.67
    + OLSC          F1 ~ 0.81   IoU ~ 0.68
(Per-tile values vary; the mean over the 24-image test split is
 baseline F1 = 0.8048 / IoU = 0.6741, refined F1 = 0.8085 / IoU = 0.6793.)
"""
from __future__ import annotations

import argparse

import numpy as np
import cv2
import tifffile

from olsc_refine import refine_mask, pixel_metrics, polygons_to_mask

CONF, IOU, MAX_DET = 0.25, 0.7, 3000


def load_raw(path: str) -> np.ndarray:
    """Read a multispectral TIFF as (H, W, C) uint8."""
    arr = tifffile.imread(path)
    if arr.ndim == 3 and arr.shape[0] <= 10:      # stored as (C, H, W)
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr] * 5, -1)
    return arr


def predict(model, raw: np.ndarray):
    """Return the merged binary mask and the per-instance masks."""
    res = model.predict(raw, conf=CONF, iou=IOU, max_det=MAX_DET,
                        retina_masks=True, verbose=False)[0]
    h, w = raw.shape[:2]
    merged = np.zeros((h, w), np.uint8)
    instances = []
    if res.masks is not None and len(res.masks) > 0:
        for m in res.masks.data.cpu().numpy():
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            b = (m > 0.5).astype(np.uint8)
            instances.append(b)
            merged |= b
    return merged, instances


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    raw = load_raw(args.image)
    gt = polygons_to_mask(args.label, raw.shape[0], raw.shape[1])

    baseline, instances = predict(model, raw)
    refined = refine_mask(baseline, instances, raw)

    m_b = pixel_metrics(baseline, gt)
    m_r = pixel_metrics(refined, gt)

    print(f"instances detected : {len(instances)}")
    print(f"baseline YOLO      : F1={m_b['F1']:.4f}  IoU={m_b['IoU']:.4f}  "
          f"R={m_b['recall']:.4f}  P={m_b['precision']:.4f}")
    print(f"+ OLSC refinement  : F1={m_r['F1']:.4f}  IoU={m_r['IoU']:.4f}  "
          f"R={m_r['recall']:.4f}  P={m_r['precision']:.4f}")
    print(f"delta              : dF1={m_r['F1'] - m_b['F1']:+.4f}  "
          f"dIoU={m_r['IoU'] - m_b['IoU']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
