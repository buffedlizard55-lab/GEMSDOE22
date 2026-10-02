#!/usr/bin/env python3
"""Step 02 -- fit the fault-population spatial statistics BEFORE touching a model.

Populations fitted (all inside the official GeoDAWN footprint):
  A. `labels.tif` connected traces  -- the competition's USGS QFD + INGENIOUS
     raster catalogue (60,988 positive pixels).
  B. GDR 1391 INGENIOUS *vector* Quaternary fault traces
     (assets/external/gdr_qfaults_traces.csv, 1,125 traces with an explicit
      surveyed `full_length_m`, so no rasterisation bias).

Statistics fitted, with primary literature:
  * length-frequency exponent a  from N(>=L) = C L^-a
  * nearest-larger-neighbour exponent x from <d(l)> = A l^x
  * correlation dimension D from the two-point correlation integral C2(r) ~ r^D
  * Bour & Davy (1999) consistency test  x  vs  (a-1)/D
        https://doi.org/10.1029/1999GL900419
  * Marrett et al. (2018) normalised correlation count (NCC) on scanlines,
    with a Monte-Carlo CSR null and a clustered/random/regular verdict
        https://doi.org/10.1016/j.jsg.2017.06.012
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import clustering as cl
from gems22.spec import REPO, Spec, load_labels

PIXEL_M = 100.0


def fit_population(name: str, lengths_m: np.ndarray, pts_row: np.ndarray,
                   pts_col: np.ndarray, mask_for_ncc: np.ndarray | None,
                   l_min_candidates=(800.0, 1200.0, 1500.0, 1800.0, 2400.0)) -> dict:
    rep: dict = {"population": name, "n_traces": int(len(lengths_m))}
    L = np.asarray(lengths_m, float)
    L = L[np.isfinite(L) & (L > 0)]
    rep["length_m"] = {"min": float(L.min()), "median": float(np.median(L)),
                       "p90": float(np.percentile(L, 90)), "max": float(L.max()),
                       "total_km": float(L.sum() / 1000.0), "n": int(L.size)}
    rep["length_frequency"] = {f"l_min_{int(lm)}": cl.fit_length_frequency(L, lm)
                               for lm in l_min_candidates if (L >= lm).sum() >= 20}
    # Bour & Davy nearest-larger-neighbour
    t = cl.Traces(length_px=L / PIXEL_M, centroid_row=pts_row.astype(float),
                  centroid_col=pts_col.astype(float),
                  label_ids=np.arange(1, len(L) + 1), n=len(L))
    Ls, d = cl.nearest_larger_neighbour(t, PIXEL_M)
    rep["nearest_larger"] = cl.fit_nearest_larger(Ls, d)
    pts = np.vstack([pts_row, pts_col]).T.astype(float)
    rep["correlation_dimension"] = cl.correlation_dimension(
        pts, np.array([1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140.0]), PIXEL_M)
    # consistency test x vs (a-1)/D
    best = None
    for k, v in rep["length_frequency"].items():
        if np.isfinite(v.get("a", np.nan)) and (best is None or v["r2"] > best[1]["r2"]):
            best = (k, v)
    a = best[1]["a"] if best else float("nan")
    D = rep["correlation_dimension"]["D"]
    x = rep["nearest_larger"].get("x", float("nan"))
    rep["bour_davy_consistency"] = {
        "best_a_source": best[0] if best else None, "a": a,
        "a_r2": best[1]["r2"] if best else float("nan"),
        "D_correlation_dimension": D,
        "D_r2": rep["correlation_dimension"].get("r2"),
        "x_measured": x, "x_r2": rep["nearest_larger"].get("r2"),
        "x_predicted_(a-1)/D": float((a - 1.0) / D) if np.isfinite(a) and np.isfinite(D) and D else None,
        "ratio_x_measured/x_predicted": (
            float(x / ((a - 1.0) / D)) if np.isfinite(x) and np.isfinite(a) and np.isfinite(D) and D and a != 1 else None),
        "interpretation": (
            "Bour & Davy (1999) GRL 26(13):2001-2004 predict x=(a-1)/D. "
            "A ratio near 1 means the catalogue's spatial clustering and its "
            "length-frequency scaling are mutually consistent, i.e. the "
            "population behaves like a self-similar fractal fault network and "
            "the fitted <d(l)> scaling may be extrapolated to place unmapped "
            "short faults. A ratio far from 1 means mapping incompleteness "
            "(not fractal geometry) dominates the length distribution."
        ),
    }
    if mask_for_ncc is not None:
        t0 = time.time()
        ncc = cl.ncc_1d_from_raster(mask_for_ncc, angles_deg=(0, 90), n_random=400, seed=7)
        pooled = ncc.get("pooled", {})
        verdict = cl.classify_arrangement(np.array(pooled.get("ncc", [])),
                                          np.array(pooled.get("lag_centres_px", [])),
                                          lo_px=2.0, hi_px=40.0)
        rep["ncc_marrett_2018"] = {"angles_used": [0, 90], "n_scanlines": pooled.get("n_scanlines_total"),
                                   "verdict_2_to_40_px": verdict,
                                   "lag_centres_px": pooled.get("lag_centres_px"),
                                   "ncc": pooled.get("ncc"),
                                   "seconds": round(time.time() - t0, 1)}
    return rep


def main() -> None:
    spec = Spec.load()
    spec.verify()
    cat = load_labels()
    print(f"catalogue positive px = {int(cat.sum())}")

    lab, n = ndimage.label(cat, structure=cl.STRUCT8)
    print(f"catalogue connected traces = {n}")

    # --- Population A: raster catalogue traces -------------------------------
    sk = None
    try:
        from skimage.morphology import skeletonize
        sk = skeletonize(cat)
    except Exception:
        sk = cat
    sk_lab = np.where(lab > 0, lab, 0)
    lens_px = np.bincount(sk_lab.ravel(), minlength=n + 1)[1:].astype(float)
    comp_px = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(float)
    lens_px = np.where(lens_px > 0, lens_px, np.maximum(comp_px, 1.0))
    ids = np.arange(1, n + 1)
    rows = ndimage.sum(np.indices(cat.shape)[0], lab, index=ids) / np.maximum(comp_px, 1)
    cols = ndimage.sum(np.indices(cat.shape)[1], lab, index=ids) / np.maximum(comp_px, 1)
    repA = fit_population("A_labels_tif_raster_catalogue", lens_px * PIXEL_M, rows, cols, cat)

    # --- Population B: GDR 1391 INGENIOUS vector traces ----------------------
    p = REPO / "assets/external/gdr_qfaults_traces.csv"
    Lb, rb, cb = [], [], []
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                L = float(row["clipped_length_m"] or "nan")
                if not np.isfinite(L) or L <= 0:
                    L = float(row["full_length_m"] or "nan")
                r = float(row["centroid_row"]); c = float(row["centroid_col"])
            except Exception:
                continue
            if not (np.isfinite(L) and np.isfinite(r) and np.isfinite(c)):
                continue
            if not (0 <= r < cat.shape[0] and 0 <= c < cat.shape[1]):
                continue
            if str(row.get("centroid_in_footprint", "")).lower() not in ("true", "1", "yes"):
                continue
            Lb.append(L); rb.append(r); cb.append(c)
    print(f"INGENIOUS vector traces inside footprint = {len(Lb)}")
    repB = fit_population("B_gdr1391_ingenious_vector_traces", np.array(Lb),
                          np.array(rb), np.array(cb), None)

    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pixel_m": PIXEL_M,
           "sources": {
               "labels_tif_sha256": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
               "gdr_qfaults_traces_csv": "https://gdr.openei.org/submissions/1391 (DOI 10.15121/1881483)",
               "bour_davy_1999": "https://doi.org/10.1029/1999GL900419",
               "marrett_2018_ncc": "https://doi.org/10.1016/j.jsg.2017.06.012",
               "wang_2019_ncc_on_faults": "https://doi.org/10.1144/petgeo2018-146",
               "bonnet_2001_review": "https://doi.org/10.1029/1999RG000074",
           },
           "populations": [repA, repB]}
    (REPO / "evidence").mkdir(exist_ok=True)
    (REPO / "evidence/gems22_clustering_fit.json").write_text(json.dumps(out, indent=1, default=str))

    for rep in (repA, repB):
        print("\n" + "=" * 78)
        print(f"POPULATION {rep['population']}  n_traces={rep['n_traces']}")
        print("  length (m):", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in rep["length_m"].items()})
        for k, v in rep["length_frequency"].items():
            print(f"  {k:14s} a={v['a']:.4f} R2={v['r2']:.4f} C={v['C']:.3g} n={v['n_traces']}")
        xl = rep["nearest_larger"]
        print(f"  nearest-larger: x={xl.get('x')} A={xl.get('A')} R2={xl.get('r2')} n={xl.get('n')} bins={xl.get('n_bins_used')}")
        print(f"  correlation dimension D={rep['correlation_dimension']['D']} R2={rep['correlation_dimension'].get('r2')}")
        bc = rep["bour_davy_consistency"]
        print(f"  BOUR&DAVY: x_meas={bc['x_measured']}  (a-1)/D={bc['x_predicted_(a-1)/D']}  ratio={bc['ratio_x_measured/x_predicted']}")
        if "ncc_marrett_2018" in rep:
            v = rep["ncc_marrett_2018"]["verdict_2_to_40_px"]
            print(f"  NCC(Marrett 2018) 2-40px: verdict={v.get('verdict')} mean={v.get('mean_ncc')} "
                  f"range=[{v.get('min_ncc')},{v.get('max_ncc')}] slope={v.get('loglog_slope')} "
                  f"D_ncc={v.get('implied_correlation_dimension')}")
    print("\nwrote evidence/gems22_clustering_fit.json")


if __name__ == "__main__":
    main()
