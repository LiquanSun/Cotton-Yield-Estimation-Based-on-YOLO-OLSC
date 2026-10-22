"""
在测试图 ts_000000000014 上跑基线 YOLO 与 OLSC 精修，并输出可视化对比图。

输出（位于 code_package/data/ 或指定目录）：
    ts_000000000014_result.png   1x5 面板：
       (a) 假彩色原图  (b) 真值掩膜  (c) YOLO 基线  (d) + OLSC 精修  (e) 误差图
       误差图配色：TP 绿 / FP 红 / FN 蓝
    同时打印三个区域的逐区域指标与全图指标。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from olsc_refine import refine_mask, pixel_metrics, polygons_to_mask  # noqa: E402

CONF, IOU, MAX_DET = 0.25, 0.7, 3000


def load_raw(p: str) -> np.ndarray:
    a = tifffile.imread(p)
    if a.ndim == 3 and a.shape[0] <= 10:
        a = np.transpose(a, (1, 2, 0))
    return a


def stretch(b: np.ndarray) -> np.ndarray:
    f = b.astype(np.float32)
    lo, hi = np.percentile(f, 2), np.percentile(f, 98)
    if hi <= lo:
        return np.zeros(f.shape, np.uint8)
    return np.clip((f - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def label_panel(shape, text, scale=0.45):
    h, w = shape[:2]
    bar = np.full((22, w, 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def error_panel(gt, pred):
    h, w = gt.shape
    out = np.full((h, w, 3), 255, np.uint8)
    tp = (pred > 0) & (gt > 0)
    fp = (pred > 0) & (gt == 0)
    fn = (pred == 0) & (gt > 0)
    out[tp] = (60, 180, 75)     # green  TP
    out[fp] = (220, 50, 45)     # red    FP
    out[fn] = (40, 80, 220)     # blue   FN
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="ts_000000000014_result.png")
    ap.add_argument("--ratio", type=float, default=3.0, help="放大倍数（最近邻，保留像素块）")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    raw = load_raw(args.image)
    h, w = raw.shape[:2]
    gt = polygons_to_mask(args.label, h, w)

    res = model.predict(raw, conf=CONF, iou=IOU, max_det=MAX_DET,
                        retina_masks=True, verbose=False)[0]
    merged = np.zeros((h, w), np.uint8)
    instances = []
    if res.masks is not None and len(res.masks) > 0:
        for m in res.masks.data.cpu().numpy():
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            b = (m > 0.5).astype(np.uint8)
            instances.append(b)
            merged |= b

    refined = refine_mask(merged, instances, raw)

    m_b = pixel_metrics(merged, gt)
    m_r = pixel_metrics(refined, gt)

    # 面板
    fc = np.stack([stretch(raw[:, :, 4]), stretch(raw[:, :, 2]), stretch(raw[:, :, 1])], -1)
    gt_rgb = np.zeros((h, w, 3), np.uint8)
    gt_rgb[gt > 0] = (255, 214, 0)
    base_rgb = np.zeros((h, w, 3), np.uint8)
    base_rgb[merged > 0] = (60, 200, 90)
    ref_rgb = np.zeros((h, w, 3), np.uint8)
    ref_rgb[refined > 0] = (60, 200, 90)
    err_rgb = error_panel(gt, refined)

    panels = [
        (fc, "(a) NIR-R-G false colour"),
        (gt_rgb, "(b) ground truth"),
        (base_rgb, f"(c) YOLO  F1={m_b['F1']:.4f}"),
        (ref_rgb, f"(d) + OLSC  F1={m_r['F1']:.4f}"),
        (err_rgb, "(e) error of (d): TP / FP / FN"),
    ]
    strip = []
    for img, txt in panels:
        big = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        big = np.vstack([label_panel(big.shape, txt), big])
        big = cv2.copyMakeBorder(big, 0, 0, 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        strip.append(big)
    grid = np.hstack(strip[:-1]) if len(strip) > 1 else strip[0]
    # 误差图单独一行放不下，改为两行
    row1 = np.hstack(strip[:3])
    row2 = np.hstack([strip[3], strip[4]])
    if row1.shape[1] >= row2.shape[1]:
        pad = row1.shape[1] - row2.shape[1]
        row2 = cv2.copyMakeBorder(row2, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        pad = row2.shape[1] - row1.shape[1]
        row1 = cv2.copyMakeBorder(row1, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    grid = np.vstack([row1, row2])

    Image.fromarray(grid).save(args.out)

    print(f"image            : {Path(args.image).name}  ({h}x{w})")
    print(f"instances        : {len(instances)}")
    print(f"GT positive px   : {int(gt.sum()):,}")
    print(f"baseline YOLO    : F1={m_b['F1']:.4f}  IoU={m_b['IoU']:.4f}  "
          f"R={m_b['recall']:.4f}  P={m_b['precision']:.4f}  pred_px={int((merged>0).sum()):,}")
    print(f"+ OLSC           : F1={m_r['F1']:.4f}  IoU={m_r['IoU']:.4f}  "
          f"R={m_r['recall']:.4f}  P={m_r['precision']:.4f}  pred_px={int((refined>0).sum()):,}")
    print(f"delta            : dF1={m_r['F1']-m_b['F1']:+.4f}  dIoU={m_r['IoU']-m_b['IoU']:+.4f}  "
          f"dRecall={m_r['recall']-m_b['recall']:+.4f}  dPrecision={m_r['precision']-m_b['precision']:+.4f}")
    print(f"saved            : {args.out}  {grid.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
