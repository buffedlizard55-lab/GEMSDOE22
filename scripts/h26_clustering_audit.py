"""H26 — fit the fractal-clustering statistic to the real traces and audit every
published submission with it (the brief's second required use).

Two statistics are reported, because they answer different questions:

1. **Centroid Ripley K** (``src/gems/clustering.audit_predicted_clustering``).
   Treats each connected trace as one point located at its centre of mass.  It is
   the statistic the trunk already uses; it is density-normalised, so populations
   with different trace counts are comparable.

2. **Area-based normalised correlation count (NCC)** — added here.  It counts
   pairs of *occupied pixels* in annuli around each sampled pixel and divides by
   the count expected at the same density, so it does not change when a prediction
   is fragmented into many small components.  The centroid version is confounded
   by fragmentation (a prediction split into 10 pieces has 10 centroids spread
   over the same line); the NCC is not.  This is the "normalized correlation
   count" of the later structural-geology literature the brief cites.

Both are normalised so that 1.0 = random (Poisson) spacing, > 1 = clustered,
< 1 = regularly spaced.  The audit flag fires when a *prediction's* structure
diverges from the structure actually measured in this region's catalogue, which
is the signature the brief asks to catch (survey-line aliasing, acquisition-block
edges) rather than genuine geology.

Run:
    python3 scripts/h26_clustering_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.clustering import (  # noqa: E402
    audit_predicted_clustering, fit_clustering_dimension, grid_alignment_audit,
)
from gems.metric import dti_score_fast  # noqa: E402
from gems.paths import DATA_DIR, DOWNLOADS_DIR  # noqa: E402

EVI = ROOT / "evidence"
SCALES_PX = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
N_SAMPLE = 60_000


def ncc_area(mask: np.ndarray, scales_px=SCALES_PX, n_sample: int = N_SAMPLE,
             seed: int = 26) -> dict:
    """Density-normalised correlation count of a binary raster, by pixel pairs."""
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n < 100:
        return {"n_px": n, "scales_px": list(scales_px), "ncc": [float("nan")] * len(scales_px)}
    pts_all = np.argwhere(m).astype(np.float64)
    rng = np.random.default_rng(seed)
    if pts_all.shape[0] > n_sample:
        idx = rng.choice(pts_all.shape[0], n_sample, replace=False)
        pts = pts_all[idx]
    else:
        pts = pts_all
    tree = cKDTree(pts)
    area = float(m.size)
    dens = n / area
    out = []
    for r in scales_px:
        lo, hi = max(r - 0.5, 0.0), r + 0.5
        pairs = (tree.count_neighbors(tree, hi) - tree.count_neighbors(tree, lo)) / 2.0
        expected = 0.5 * pts.shape[0] * (pts.shape[0] - 1) * (
            np.pi * (hi ** 2 - lo ** 2)) / area
        out.append(float(pairs / expected) if expected > 0 else float("nan"))
    return {"n_px": n, "scales_px": list(scales_px), "ncc": out}


def main() -> int:
    t0 = time.time()
    EVI.mkdir(exist_ok=True)
    with rasterio.open(DATA_DIR / "labels.tif") as s:
        lab = s.read(1)
    with rasterio.open(DATA_DIR / "sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    cat = (lab > 0) & fp

    fit = fit_clustering_dimension(cat, fp, 100.0)
    fit_d = {k: (v if not isinstance(v, (np.ndarray, list)) else list(v))
             for k, v in fit.__dict__.items()}
    ref_ncc = ncc_area(cat)

    files = sorted(p for p in DOWNLOADS_DIR.rglob("*.tif") if p.is_file())
    audits = {}
    for p in files:
        with rasterio.open(p) as s:
            a = s.read(1)
        finite = np.isfinite(a) | np.isnan(a)
        val = np.nan_to_num(a, nan=0.0)
        inside = fp
        out_of_range = int(((val < 0) | (val > 1))[inside].sum())
        neg_outside = int((val != 0).sum() - (val[inside] != 0).sum())
        pred = (val > 0.5) & fp
        ca = audit_predicted_clustering(pred, fp, fit)
        pred_ncc = ncc_area(pred)
        lr = [float(np.log(x / y)) if (x > 0 and y > 0) else float("nan")
              for x, y in zip(pred_ncc["ncc"], ref_ncc["ncc"])]
        finite_lr = [v for v in lr if np.isfinite(v)]
        div = float(np.mean(np.abs(finite_lr))) if finite_lr else float("nan")
        flag = ("CONSISTENT" if div < 0.40 else "WARN_DIVERGENT" if div < 0.80
                else "FLAG_ARTIFACT_LIKELY") if np.isfinite(div) else "UNDETERMINED"
        aliasing = grid_alignment_audit(pred, fp)
        audits[p.name] = {
            "path": str(p.relative_to(ROOT)),
            "n_emitted_footprint": int(pred.sum()),
            "n_emitted_outside_footprint_gt0": neg_outside,
            "n_out_of_range_inside_footprint": out_of_range,
            "centroid_audit": ca,
            "ncc_area": pred_ncc,
            "ncc_area_reference_catalogue": ref_ncc,
            "ncc_log_ratio": [round(v, 4) if np.isfinite(v) else None for v in lr],
            "ncc_divergence": round(div, 4) if np.isfinite(div) else None,
            "ncc_flag": flag,
            "grid_alignment_audit": aliasing,
        }
        print(f"  {p.name[:60]:60s} n={pred.sum():>8,} div={div:.3f} {flag:20s} "
              f"max_straight_run={aliasing.get('max_straight_run_px')}", flush=True)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/h26_clustering_audit.py",
        "status": "COMPUTED_ON_REAL_DATA",
        "clustering_fit_labels_tif": fit_d,
        "reference_ncc_catalogue": ref_ncc,
        "flag_thresholds": {"CONSISTENT": "<0.40", "WARN_DIVERGENT": "0.40-0.80",
                            "FLAG_ARTIFACT_LIKELY": ">0.80"},
        "n_files_audited": len(audits),
        "audits": audits,
        "seconds": round(time.time() - t0, 1),
    }
    (EVI / "h26_clustering_audit.json").write_text(json.dumps(out, indent=1))
    print(f"wrote evidence/h26_clustering_audit.json ({len(audits)} files, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
