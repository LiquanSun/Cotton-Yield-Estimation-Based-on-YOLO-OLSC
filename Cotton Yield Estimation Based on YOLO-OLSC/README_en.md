# Code and data for: Cotton Yield Estimation Based on YOLO-OLSC Segmentation Method Using UAV Multispectral Imagery

Manuscript ATECH-D-26-01983, *Smart Agricultural Technology*.

This package contains the reference implementation of the OLSC refinement, one annotated
test tile, and the trained YOLOv12m five-band checkpoint, so that the main segmentation
result can be re-run on a single machine.

---

## 1. Package contents

```
.
├── code/
│   ├── olsc_refine.py             core algorithm (LSC superpixels, Cotton Set,
│   │                              REMOVE/ADD passes, metrics, label rasterisation)
│   ├── demo_olsc_on_test_tile.py  minimal command-line demo, prints metrics
│   ├── run_test_tile_14.py        reproduces the five-panel figure (Fig. A1)
│   └── requirements.txt
├── data/
│   ├── ts_000000000014.tiff       256 × 256 × 5 uint8, band order B, G, R, RE, NIR
│   ├── ts_000000000014.txt        YOLO polygon-segmentation label, one class
│   ├── ts_000000000014_bands.png  per-band view of the tile
│   ├── ts_000000000014_falsecolor.png   NIR-R-G false colour
│   └── ts_000000000014_result.png       output of run_test_tile_14.py
└── weights/
    └── yolo12m_5c_best.pt         YOLOv12m-seg, five-channel input (~45 MB)
```

---

## 2. Requirements

Python 3.9 or newer. The versions used for the results reported in the paper were:

| Package | Version |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.5.1+cu121 |
| Ultralytics | 8.3.233 |
| OpenCV | 4.12.0 (`opencv-contrib-python`) |
| NumPy | 2.2.6 |
| scikit-image | 0.25.2 |
| SciPy | 1.14 |
| tifffile | 2025.10.16 |

`opencv-contrib-python` is required, not `opencv-python`: the LSC superpixel implementation
lives in `cv2.ximgproc`.

### Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r code/requirements.txt
```

If a CUDA build of PyTorch is wanted, install it first following the official selector,
then run the command above; `requirements.txt` will not downgrade it.

---

## 3. Running

All commands below assume the package root as the working directory.

### 3.1 Minimal demo — metrics only

```bash
cd code
python demo_olsc_on_test_tile.py \
    --weights ../weights/yolo12m_5c_best.pt \
    --image   ../data/ts_000000000014.tiff \
    --label   ../data/ts_000000000014.txt
```

Expected output on tile `ts_000000000014` (numbers vary slightly with GPU/BLAS):

```
instances detected : ~40
baseline YOLO      : F1≈0.80  IoU≈0.67
+ OLSC refinement  : F1≈0.81  IoU≈0.68
delta              : dF1>0    dIoU>0
```

### 3.2 Five-panel figure — Fig. A1 of the appendix

```bash
cd code
python run_test_tile_14.py \
    --weights ../weights/yolo12m_5c_best.pt \
    --image   ../data/ts_000000000014.tiff \
    --label   ../data/ts_000000000014.txt \
    --out     ../data/ts_000000000014_result.png
```

Panels: (a) NIR-R-G false colour, (b) ground truth, (c) YOLO baseline,
(d) + OLSC refinement, (e) error map with TP green / FP red / FN blue.
The per-tile F1/IoU/recall/precision of both masks are printed to stdout.

### 3.3 Using the refinement from your own code

```python
import tifffile
from olsc_refine import refine_mask, pixel_metrics, polygons_to_mask

raw   = tifffile.imread("data/ts_000000000014.tiff")   # (H, W, 5)
raw   = raw.transpose(1, 2, 0) if raw.shape[0] <= 10 else raw
yolo  = ...            # (H, W) uint8, merged mask of all detected instances
insts = [...]          # list of (H, W) uint8, one mask per instance
refined = refine_mask(yolo, insts, raw)                # (H, W) uint8
```

`refine_mask` accepts the raw multispectral tile and performs the percentile stretch
internally; pass already-stretched data only if you also pass `t_cotton` accordingly.

---

## 4. Algorithm

Implemented in `olsc_refine.py`; the constants below are the "Ours" configuration of
the paper.

1. **Normalisation.** Per-band 2 %–98 % percentile stretch to [0, 1]
   (`normalize_bands`).
2. **LSC over-segmentation** of the three visible bands (`lsc_superpixels`).
   The segment budget is a mix of a Felzenszwalb region count and an area term:

   ```
   n_segments  = int(felz_count * 0.75 + int(H*W/150) * 0.25) * density_multiplier
   region_size = max(10, int(sqrt(H*W / n_segments)))
   ```

   with `density_multiplier = 1.3`, `lsc_ratio = 0.075`, `iterate(20)`.
3. **Cotton Set** (`cotton_set`). A superpixel counts as cotton when the mean of
   *every* normalised band exceeds `t_cotton = 0.17`.
4. **REMOVE pass (Step A).** A superpixel that is **not** in the Cotton Set and whose
   overlap with the merged YOLO mask exceeds `tau_rm = 0.95` is set to 0.
5. **ADD pass (Step B).** For each detected instance, a superpixel that **is** in the
   Cotton Set and whose overlap with that instance exceeds `tau_add = 0.85` is set to 1.
6. **Post-processing.** `remove_small_objects(min_size=2)` followed by
   `binary_fill_holes`.

The two thresholds are deliberately asymmetric: removal is conservative (0.95),
addition is permissive (0.85). This raises recall while lowering precision only slightly,
which is the behaviour reported in Table A2 and discussed in the response to Comment 12.

### Note on morphology

Do **not** apply morphological opening or dilation after step 5. The pixel-level metric
used in the paper is sensitive to mask thickness: a single 3 × 3 dilation lowers pixel-level
F1 by about 0.06 on the 24-image test split. The archive contains only
`remove_small_objects` and `binary_fill_holes`.

### Parameters

| Symbol | Code argument | Value |
|---|---|---|
| Cotton threshold | `t_cotton` | 0.17 |
| ADD threshold | `tau_add` | 0.85 |
| REMOVE threshold | `tau_rm` | 0.95 |
| LSC compactness | `lsc_ratio` | 0.075 |
| Density multiplier | `density_multiplier` | 1.3 |
| Minimum object size | `min_size` | 2 |

`T_cotton = 0.17` is the argmax of a grid search over
{0.10, 0.13, 0.15, 0.17, 0.20, 0.25} on the validation split. It acts on
percentile-stretched digital numbers, not reflectance, so it depends on sensor and
illumination and must be re-derived for a different camera or site.

---

## 5. Reproduction values

Detection and refinement use Ultralytics defaults `conf = 0.25`, `iou = 0.7`,
`max_det = 3000`, `retina_masks = True`, with masks binarised at 0.5.

| Setting | F1 | IoU |
|---|---|---|
| Baseline YOLO, 24-image test split | 0.8048 | 0.6741 |
| YOLO + OLSC, 24-image test split | 0.8082 | 0.6790 |

These are pixel-level merged-mask values on the 24 test tiles of the held-out source
region, averaged over the tiles. The last digit varies slightly with GPU and library
build.

The full 129-tile corpus (84 / 21 / 24 train–validation–test), the 36-plot yield table and
the analysis scripts that generate every table and figure of the article are deposited in
the public repository linked in the Data availability statement.

---

## 6. Label format

`ts_000000000014.txt` follows the YOLO segmentation convention, one instance per line:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

with coordinates normalised to [0, 1] and a single class, `cotton` (class id 0).
Polygons contain 4–195 vertices; `polygons_to_mask()` rasterises them with
`cv2.fillPoly`.

---

## 7. Licence and citation

Code is released under the MIT licence. The annotated data, the test tile and the trained
weights are released under CC BY 4.0.

If you use this package, please cite the article:

> Ma, H., Sun, L., Chen, Z., Li, C., Liu, H. Cotton Yield Estimation Based on YOLO-OLSC
> Segmentation Method Using UAV Multispectral Imagery. *Smart Agricultural Technology*
> (under review, ATECH-D-26-01983).

---

## 8. Contact

Liquan Sun — lq_sun@xju.edu.cn
College of Geography and Remote Sensing Sciences, Xinjiang University, Urumqi 830017, China
