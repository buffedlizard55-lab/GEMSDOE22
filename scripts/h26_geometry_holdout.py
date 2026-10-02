"""H26 — spatially-blocked (trace-cluster) holdout test of geometric placement policies.

The brief requires that no submission slot is spent on an idea that has not beaten
the current holdout best, and specifically that the *clustering statistic* be used
as a geometric prior (favouring a candidate lying along the extrapolated
clustering pattern of a known larger fault over an equally-scored but spatially
isolated one).  This script measures whether that actually helps on the only
honest offline instrument available here.

Design (per fold, all of it fold-safe):

    visible V_k  = (catalogue | gap) minus fold k     (what a solver could see)
    hidden  H_k  = fold k                            (the stand-in for "new faults")
    domain  D_k  = footprint minus V_k               (the live scored domain analogue)
    GT           = H_k (subset of D_k)

Two heads are scored, exactly as ``evidence/holdout_union.json`` does:

    head A : GT = fold traces that are in ``labels.tif`` (the competition catalogue)
    head B : GT = fold traces that are in the SGMC-gap proxy (USGS state-map faults
             that are pixel-disjoint from the catalogue) -- an independent mapping
             effort, therefore a strictly better stand-in for "faults missing from
             the catalogue" than head A is.

Nothing used to build a score touches H_k: the priors are functions of V_k only,
and the detector is retrained on V_k inside the fold.  That is invariant I-2.

Arms (each emitted at matched budget n on D_k):
    random          control, no skill
    halo            H22-1's isotropic Gaussian halo on d(V_k)  (lambda = 1.8 km)
    bour_davy       ring prior at the measured nearest-larger-neighbour median
    strike_tip      anisotropic continuation beyond both tips of V_k traces
    strike_tip_dilate   + one-pixel metric-matching band
    det             HGB detector retrained inside the fold (fold-safe base)
    det+halo_w0.5 / det+strike_tip_w0.5      detector boosted in rank space
    det+halo_w0.5+dilate / det+strike_tip_w0.5+dilate

Metric: the official distance-weighted Tversky index (alpha 0.2, beta 0.8, R = 3 px).

Run:
    python3 scripts/h26_geometry_holdout.py            # ~10-15 min on 2 CPU
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.geometry_prior import (  # noqa: E402
    bour_davy_ring, dilate_band, distance_to, emit_topn, halo_isotropic,
    strike_tip_field,
)
from gems.metric import ALPHA, BETA, RADIUS_PX  # noqa: E402
from gems.paths import DATA_DIR  # noqa: E402
from gems22.holdout import trace_cluster_folds  # noqa: E402

EVI = ROOT / "evidence"
BUDGET_FRAC = (0.005, 0.01, 0.0245, 0.05, 0.10)
LAMBDA_PX = 18.0          # H22-1 ships lambda = 1.8 km
BANDS = ["det_elev", "det_elev_slope", "tmi", "tmi_hg", "tmi_vg",
         "iso_grav_anom", "iso_grav_anom_slope", "cond_surf",
         "depth_to_base_surf", "tc"]


def load_grid():
    with rasterio.open(DATA_DIR / "labels.tif") as s:
        lab = s.read(1)
    with rasterio.open(DATA_DIR / "sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(ROOT / "assets" / "external" / "derived_sgmc_faults_100m_u8.tif") as s:
        sg = s.read(1) > 0
    cat = (lab > 0) & fp
    gap = sg & ~cat & fp
    return cat, gap, fp


def load_features(names: list[str]) -> np.ndarray:
    """(len(names), H, W) float32, sentinel-sanitised, in the given band order."""
    with rasterio.open(DATA_DIR / "training_features.tif") as s:
        desc = list(s.descriptions)
        idx = []
        for n in names:
            hit = [i for i, d in enumerate(desc) if d.split(" - ")[0].strip() == n]
            if not hit:
                raise KeyError(f"band {n!r} not found among {desc}")
            idx.append(hit[0] + 1)
        arr = s.read(idx).astype(np.float32)
    bad = ~np.isfinite(arr) | (arr <= -3.0e38)
    if bad.any():
        med = np.array([np.median(a[~b]) if (~b).any() else 0.0
                        for a, b in zip(arr, bad)], dtype=np.float32)
        arr = np.where(bad, med[:, None, None], arr)
    return arr


def rank01(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ranks of ``x`` over ``mask`` mapped to [0, 1]; 0 elsewhere."""
    out = np.zeros(x.shape, dtype=np.float32)
    vals = x[mask]
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="stable")
    r = np.empty(vals.size, dtype=np.float32)
    r[order] = np.linspace(0.0, 1.0, vals.size, dtype=np.float32)
    out[mask] = r
    return out


def fit_detector(X: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )
    clf.fit(X, y)
    return clf


def predict_blocks(clf, Xall: np.ndarray, rows_per_block: int = 128) -> np.ndarray:
    nB, H, W = Xall.shape
    out = np.empty(H * W, dtype=np.float32)
    step = rows_per_block * W
    for start in range(0, H * W, step):
        stop = min(H * W, start + step)
        out[start:stop] = clf.predict_proba(Xall.reshape(nB, -1)[:, start:stop].T)[:, 1]
    return out.reshape(H, W)


def dti_with_gt_weights(pred: np.ndarray, gt: np.ndarray, domain: np.ndarray,
                        fp_cost: np.ndarray) -> dict:
    """Exact DTI, reusing a precomputed ``fp_cost = 1 - k(d(., GT))`` raster."""
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool) & domain
    n_gt = int(g.sum())
    if n_gt == 0 or not p.any():
        return {"TP_w": 0.0, "FP_w": float(p.sum()), "FN_w": float(n_gt),
                "dti": 0.0, "n_gt": n_gt, "n_pred": int(p.sum()), "coverage": 0.0}
    d_to_p = ndimage.distance_transform_edt(~p)
    tp_w = float(np.maximum(1.0 - d_to_p[g] / RADIUS_PX, 0.0).sum())
    fn_w = float(n_gt) - tp_w
    fp_w = float(fp_cost[p & domain].sum())
    den = tp_w + ALPHA * fp_w + BETA * fn_w + 1e-7
    return {"TP_w": tp_w, "FP_w": fp_w, "FN_w": fn_w, "dti": float(tp_w / den),
            "n_gt": n_gt, "n_pred": int(p.sum()), "coverage": float(tp_w / n_gt)}


def main() -> int:
    t0 = time.time()
    EVI.mkdir(exist_ok=True)
    cat, gap, fp = load_grid()
    union = cat | gap
    folds = trace_cluster_folds(union, n_folds=4, n_clusters=48, seed=22)
    heads = {"A": cat, "B": gap}
    fold_counts = {k: {h: int((v & m).sum()) for h, m in heads.items()} for k, v in folds.items()}
    print("fold gt counts", fold_counts, flush=True)

    print("loading features...", flush=True)
    F = load_features(BANDS)
    print("features", F.shape, flush=True)

    records: list[dict] = []
    rng = np.random.default_rng(26)

    for fname in sorted(folds):
        t_f = time.time()
        V_k = union & ~folds[fname]
        D_k = fp & ~V_k
        n_dom = int(D_k.sum())
        fold_traces = folds[fname]

        # ---- fold-safe geometric priors (functions of V_k only) --------------
        halo = halo_isotropic(V_k, LAMBDA_PX)
        bd = bour_davy_ring(V_k, peak_px=16.3, sigma_px=24.0)
        tip = strike_tip_field(V_k, lengths_m=None, max_reach_px=20)

        # ---- fold-safe detector --------------------------------------------
        Xtr, ytr = sample_training(F, V_k, D_k, n_pos=100_000, n_neg=100_000, rng=rng)
        clf = fit_detector(Xtr, ytr, seed=26)
        del Xtr, ytr
        det = np.where(D_k, predict_blocks(clf, F), 0.0).astype(np.float32)
        del clf
        print(f"  {fname}: detector trained ({time.time()-t_f:.0f}s)", flush=True)

        flat = np.zeros(n_dom, dtype=np.float32)
        sub = rng.choice(n_dom, min(n_dom, 400_000), replace=False)
        flat[sub] = rng.random(sub.size).astype(np.float32)
        rank_rand = np.zeros(D_k.shape, np.float32)
        rank_rand[D_k] = flat

        r_halo = rank01(halo, D_k)
        r_bd = rank01(bd, D_k)
        r_tip = rank01(tip, D_k)
        r_det = rank01(det, D_k)

        arms = {
            "random": rank_rand,
            "halo": r_halo,
            "bour_davy": r_bd,
            "strike_tip": r_tip,
            "det": r_det,
            "det+halo_w0.5": r_det + 0.5 * r_halo,
            "det+strike_tip_w0.5": r_det + 0.5 * r_tip,
        }
        del halo, bd, tip, det

        for hname, hmask in heads.items():
            gt = fold_traces & hmask & D_k
            if int(gt.sum()) < 50:
                continue
            d_gt = ndimage.distance_transform_edt(~gt)
            fp_cost = 1.0 - np.maximum(1.0 - d_gt / RADIUS_PX, 0.0)
            del d_gt
            DILATABLE = {"halo", "strike_tip", "det", "det+halo_w0.5", "det+strike_tip_w0.5"}
            for aname, score in arms.items():
                for frac in BUDGET_FRAC:
                    n = int(round(frac * n_dom))
                    pred = emit_topn(score, D_k, n)
                    variants = [("", pred)]
                    if aname in DILATABLE:
                        variants.append(("+dilate1", dilate_band(pred.astype(np.float32), D_k, 1)))
                    for suffix, pr in variants:
                        c = dti_with_gt_weights(pr, gt, D_k, fp_cost)
                        records.append({
                            "fold": fname, "head": hname, "arm": aname + suffix,
                            "budget_frac": frac, "n": c["n_pred"], "A": round(c["TP_w"], 2),
                            "B": round(c["FP_w"], 2), "n_gt": c["n_gt"],
                            "dti": round(c["dti"], 6), "coverage": round(c["coverage"], 6),
                        })
            del fp_cost, gt
        del arms, r_halo, r_bd, r_tip, r_det, rank_rand, V_k, D_k, fold_traces
        print(f"  {fname}: done ({time.time()-t_f:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

    # ---- summarise -----------------------------------------------------------
    summary = []
    keys = sorted({(r["arm"], r["head"], r["budget_frac"]) for r in records})
    for arm, head, frac in keys:
        vals = [r["dti"] for r in records
                if r["arm"] == arm and r["head"] == head and r["budget_frac"] == frac]
        rand_vals = [r["dti"] for r in records
                     if r["arm"] == "random" and r["head"] == head and r["budget_frac"] == frac]
        summary.append({
            "arm": arm, "head": head, "budget_frac": frac,
            "mean_dti": round(float(np.mean(vals)), 6),
            "min_dti": round(float(np.min(vals)), 6),
            "folds_gaining_vs_det": None,
            "delta_vs_random": (round(float(np.mean(vals)) - float(np.mean(rand_vals)), 6)
                                if rand_vals else None),
        })
    # per-arm paired comparison against the detector arm (same fold/head/budget)
    det_map = {(r["head"], r["budget_frac"], r["fold"]): r["dti"]
               for r in records if r["arm"] == "det"}
    for row in summary:
        gains = []
        for r in records:
            if r["arm"] != row["arm"] or r["head"] != row["head"] or r["budget_frac"] != row["budget_frac"]:
                continue
            k = (r["head"], r["budget_frac"], r["fold"])
            if k in det_map:
                gains.append(r["dti"] - det_map[k])
        if gains:
            row["mean_delta_vs_det"] = round(float(np.mean(gains)), 6)
            row["folds_det_beaten"] = f"{sum(1 for g in gains if g > 0)}/{len(gains)}"

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/h26_geometry_holdout.py",
        "status": "COMPUTED_ON_REAL_DATA",
        "protocol": {
            "folds": "gems22.holdout.trace_cluster_folds(cat|gap, 4, 48, seed=22) — same partition as evidence/holdout_union.json",
            "heads": {"A": "labels.tif catalogue traces in the fold",
                      "B": "SGMC-gap traces in the fold (independent mapping effort)"},
            "domain": "footprint & ~visible (= cat|gap minus the fold)",
            "metric": "official distance-weighted Tversky, alpha=0.2, beta=0.8, R=300 m",
            "detector": "sklearn HistGradientBoostingClassifier retrained inside every fold on 10 feature bands",
            "leak_control": "every prior and every detector is a function of the visible mask only (invariant I-2)",
        },
        "fold_gt_pixel_counts": fold_counts,
        "bands": BANDS,
        "budget_fracs": list(BUDGET_FRAC),
        "per_row": records,
        "summary": summary,
        "seconds": round(time.time() - t0, 1),
    }
    (EVI / "h26_geometry_holdout.json").write_text(json.dumps(out, indent=1))
    print(f"wrote evidence/h26_geometry_holdout.json ({len(records)} rows, {time.time()-t0:.0f}s)")
    return 0


def sample_training(Xall: np.ndarray, positive: np.ndarray, domain: np.ndarray,
                    n_pos: int, n_neg: int, rng: np.random.Generator):
    pos_idx = np.flatnonzero(positive.ravel())
    neg_idx = np.flatnonzero(domain.ravel())
    if pos_idx.size > n_pos:
        pos_idx = rng.choice(pos_idx, n_pos, replace=False)
    if neg_idx.size > n_neg:
        neg_idx = rng.choice(neg_idx, n_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    X = Xall.reshape(Xall.shape[0], -1)[:, idx].T.astype(np.float32)
    y = np.concatenate([np.ones(pos_idx.size, np.uint8), np.zeros(neg_idx.size, np.uint8)])
    return X, y


if __name__ == "__main__":
    raise SystemExit(main())
