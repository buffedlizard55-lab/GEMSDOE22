#!/usr/bin/env python3
"""Step 03 -- build the streamed feature cache (competition bands + external +
structural priors) into data/derived/features.f16."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import clustering as cl
from gems22.features import build_cache
from gems22.spec import REPO, load_features, load_labels

OUT = REPO / "data/derived/features.f16"


def structural_layers(cat: np.ndarray, footprint: np.ndarray,
                      cfit: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    d = ndimage.distance_transform_edt(~cat)
    out["struct_dist_cat_log"] = np.log1p(d).astype(np.float32)
    out["struct_dist_cat_inv"] = (1.0 / (1.0 + d / 3.0)).astype(np.float32)
    out["struct_footprint"] = footprint.astype(np.float32)
    out["struct_cat_dilate3"] = ndimage.binary_dilation(
        cat, np.ones((3, 3), bool), iterations=3).astype(np.float32)

    # --- Bour & Davy (1999) nearest-larger-neighbour geometric prior --------
    pop = None
    for p in cfit.get("populations", []):
        if p["population"].startswith("A_"):
            pop = p
            break
    if pop is not None:
        xl = pop.get("nearest_larger", {})
        lab, n = ndimage.label(cat, structure=cl.STRUCT8)
        t = cl.extract_traces(cat, skeleton=True)
        # align trace ids with the label map produced above
        t = cl.Traces(length_px=t.length_px, centroid_row=t.centroid_row,
                      centroid_col=t.centroid_col, label_ids=np.arange(1, n + 1), n=n)
        prior = cl.bour_davy_prior_field(lab, t, xl, pixel_m=100.0, top_k=200,
                                         r_max_px=60, log_sigma=1.0)
        out["struct_bour_davy_prior"] = prior
        out["struct_bour_davy_prior_valid"] = (prior > 0).astype(np.float32)
    return out


def main() -> None:
    t0 = time.time()
    A, names = load_features()
    print(f"loaded features {A.shape} bands={names}")
    cat = load_labels()
    import rasterio
    with rasterio.open(REPO / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    cfit = {}
    p = REPO / "evidence/clustering_fit.json"
    if p.exists():
        cfit = json.loads(p.read_text())
    else:
        print("WARNING: evidence/clustering_fit.json missing -> run 02_fit_clustering.py")
    extra = structural_layers(cat, fp, cfit)
    print("structural layers:", {k: (v.dtype.str, float(np.mean(v)), int((v > 0).sum()))
                                 for k, v in extra.items()})
    out, fnames = build_cache(A, names, OUT, external_dir=REPO / "assets/external",
                              structural=extra, verbose=True)
    print(f"wrote {out} ({out.stat().st_size/1e9:.2f} GB) with {len(fnames)} layers "
          f"in {time.time()-t0:.0f}s")
    (REPO / "data/derived/feature_names.json").write_text(json.dumps(fnames, indent=1))


if __name__ == "__main__":
    main()
