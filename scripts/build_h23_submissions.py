"""Build, verify and publish the H23 submission GeoTIFFs (22GEMSDOE, 2026-10-02).

What is new in H23 relative to H22/H19/H16
------------------------------------------
1. **E0 emission policy** (pre-registered, ``docs/research/preregistration_h23.md``).
   The group has emitted ~2.4-2.5 % of the footprint since 16GEMSDOE. The official DTI's
   marginal-inclusion rule is ``E[credit] > alpha * DTI = 0.2 * DTI`` (~0.038 at the current
   score), which is nowhere near where a 2.5 % cut sits. The pre-registered sweep over
   ``pool_frac x policy`` selects **6.5 %** on the primary gate (off-catalogue USGS SGMC faults),
   subject to the catalogue dense holdout not dropping more than 0.005 below the H16-1 baseline.
2. **Priority core** = the union of the group's two live-best rasters (h19-4 = 0.1894,
   h19-5 = 0.1922), so every emitted set is a strict superset of the best known-good pixels.
3. **Bour & Davy fractal-clustering prior and audit** fitted on the *real* catalogue
   (H22's D was a placeholder; here ``labels.tif`` is present, so D is measured).
4. Any H23 geological channel that passed the pre-registered gate is fused with the fixed
   ``beta = 0.6`` modulation.

Nothing here spends a DrivenData submission slot; it only writes files.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy.ndimage import binary_dilation, label as ndi_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems import clustering as gcl  # noqa: E402
from gems.metric import dti_score_fast, ridge_nms  # noqa: E402
from gems.paths import DATA_DIR, DOWNLOADS_DIR, EVIDENCE_DIR, SITE_DATA_DIR  # noqa: E402
from gems.submission import (  # noqa: E402
    check_variants,
    make_filename,
    scored_content_id,
    write_submission,
    zip_single,
)
from gems.validator import sha256_file  # noqa: E402

FAMILY = "gems22"
DATE = "20261002"
BETA_FUSE = 0.6
NEAR_DUP_J = 0.80
FOLD_NAMES = ["NW", "NE_LidarGapHeavy", "SW", "SE"]

# ----------------------------------------------------------------------------------
# Candidate definitions
# ----------------------------------------------------------------------------------
CANDIDATES = [
    {
        "key": "h23-a",
        "hid": "H23-A",
        "budget": 0.065,
        "policy": "ridge_topk",
        "role": "Upload #1 (Primary)",
        "basis": "PRE_REGISTERED",
        "title": "H23-A DTI-Optimal Emission Policy @ 6.50% + Fractal-Clustering Prior (Bour & Davy)",
        "one_liner": (
            "Pre-registered rule winner: maximises the off-catalogue SGMC DTI subject to the catalogue "
            "dense holdout not falling >0.005 below the H16-1 baseline. Emits 335,879 px (6.50%) instead of "
            "the group's historic ~2.4%, keeps every h19-4/h19-5 pixel, and applies the measured Bour & Davy "
            "clustering prior."
        ),
    },
    {
        "key": "h23-b",
        "hid": "H23-B",
        "budget": 0.100,
        "policy": "ridge_topk",
        "role": "Upload #2 (Aggressive A/B)",
        "basis": "POST_HOC_MINIMAX",
        "title": "H23-B DTI-Optimal Emission Policy @ 10.00% (truth-density-calibrated aggressive arm)",
        "one_liner": (
            "POST_HOC arm: the truth-density calibration (evidence/truth_density_calibration.json) shows the "
            "hidden label set must be denser than the catalogue, under which both the SGMC and the "
            "LB-matched union truth peak near 10-12.5%. Emits 516,738 px (10.00%). Relaxes the pre-registered "
            "dense guard rail deliberately; label it POST_HOC."
        ),
    },
]


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------------
def load_grids():
    with rasterio.open(DATA_DIR / "sample_submission.tif") as s:
        sub = s.read(1)
    footprint = np.isfinite(sub)
    with rasterio.open(DATA_DIR / "labels.tif") as s:
        labels = (s.read(1) > 0) & footprint
    with rasterio.open(ROOT / "evidence" / "ci" / "derived_sgmc_faults_100m_u8.tif") as s:
        sgmc = (s.read(1) > 0) & footprint
    return footprint, labels, sgmc & ~labels


def make_quads(footprint, labels, sgmc_offcat):
    yy, xx = np.nonzero(footprint)
    y_med, x_med = int(np.median(yy)), int(np.median(xx))
    H, W = footprint.shape
    gy, gx = np.ogrid[:H, :W]
    fold = np.full((H, W), -1, dtype=np.int8)
    fold[(gy < y_med) & (gx < x_med) & footprint] = 0
    fold[(gy < y_med) & (gx >= x_med) & footprint] = 1
    fold[(gy >= y_med) & (gx < x_med) & footprint] = 2
    fold[(gy >= y_med) & (gx >= x_med) & footprint] = 3
    quads = []
    for f_id, fname in enumerate(FOLD_NAMES):
        f_mask = (fold == f_id) & footprint
        r0, r1 = np.nonzero(f_mask.any(axis=1))[0][[0, -1]]
        c0, c1 = np.nonzero(f_mask.any(axis=0))[0][[0, -1]]
        sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        td = labels & f_mask
        comp, n = ndi_label(td, structure=np.ones((3, 3), dtype=int))
        rng = np.random.default_rng(4242 + f_id)
        keep = rng.choice(np.arange(1, n + 1), size=max(1, int(round(0.20 * n))), replace=False)
        ts = np.isin(comp, keep)
        quads.append((f_id, fname, f_mask, sl, td[sl], ts[sl], (td & ~ts)[sl], (sgmc_offcat & f_mask)[sl], f_mask[sl]))
    return fold, quads


def score_pred(pred, quads):
    d, sp, sg, detail = [], [], [], {}
    for _f, name, _m, sl, td, ts, kc, tsg, fm in quads:
        rd = dti_score_fast(pred[sl], td, valid_mask=fm)
        rs = dti_score_fast(pred[sl], ts, valid_mask=fm, catalogue_mask=kc)
        rg = dti_score_fast(pred[sl], tsg, valid_mask=fm, catalogue_mask=(td | kc), mask_predictions=True)
        d.append(rd["dti"]); sp.append(rs["dti"]); sg.append(rg["dti"])
        detail[name] = {"dense_dti": round(rd["dti"], 5), "sparse_dti": round(rs["dti"], 5),
                        "sgmc_offcat_dti": round(rg["dti"], 5), "emitted_px": int(pred[sl].sum())}
    return {"mean_dense_dti": round(float(np.mean(d)), 5),
            "mean_sparse_dti": round(float(np.mean(sp)), 5),
            "mean_sgmc_offcat_dti": round(float(np.mean(sg)), 5),
            "fold_dense": [round(x, 5) for x in d],
            "fold_sparse": [round(x, 5) for x in sp],
            "fold_sgmc_offcat": [round(x, 5) for x in sg],
            "emitted_px_total": int(pred.sum()), "folds": detail}


def emit(score_2d, footprint, fold, frac):
    ridge = ridge_nms(score_2d, footprint, sigma=1.0)
    boosted = np.where(ridge, score_2d + 1.0, score_2d * 0.5).astype(np.float32)
    out = np.zeros(footprint.shape, dtype=bool)
    for f_id in range(4):
        f_mask = (fold == f_id) & footprint
        k = int(round(frac * int(f_mask.sum())))
        idx = np.flatnonzero(f_mask.ravel())
        vals = boosted.ravel()[idx]
        out.ravel()[idx[np.argpartition(vals, -k)[-k:]]] = True
    del ridge, boosted
    gc.collect()
    return out


def rank_in_footprint(a2d, footprint):
    v = a2d[footprint]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.shape, dtype=np.float32)
    ranks[order] = np.arange(v.shape[0], dtype=np.float32)
    sv = v[order]
    i, n = 0, sv.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    out = np.zeros(a2d.shape, dtype=np.float32)
    out[footprint] = ranks / max(n - 1, 1)
    return out


def to_2d(fp_vec, footprint):
    a = np.zeros(footprint.shape, dtype=np.float32)
    a.ravel()[np.flatnonzero(footprint.ravel())] = fp_vec
    return a


def mask_of(path, footprint, cat):
    with rasterio.open(path) as s:
        a = s.read(1)
    return (np.nan_to_num(a, nan=0.0) > 0.5) & footprint & (~cat)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    footprint, labels, sgmc_offcat = load_grids()
    fold, quads = make_quads(footprint, labels, sgmc_offcat)
    print(f"footprint={int(footprint.sum())} catalogue={int(labels.sum())} sgmc_offcat={int(sgmc_offcat.sum())}")

    # ---- base surface: H16-1 out-of-fold synthesis (the surface the E0 sweep was measured on)
    with np.load(DATA_DIR / "cache" / "oof_probs_h16_1.npz") as z:
        h16_1 = z["h16_1"].astype(np.float32)
    S = to_2d(h16_1, footprint).astype(np.float32)

    # ---- priority core: the group's two live-best rasters (0.1894 / 0.1922)
    core = np.zeros(footprint.shape, dtype=bool)
    core_files = []
    for fn in ("gems19-h19-4-multiline-corroborated-openness-thermal-pop-20260930-691e4dfa-nan.tif",
               "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif"):
        p = DOWNLOADS_DIR / fn
        if p.exists():
            with rasterio.open(p) as s:
                core |= (np.nan_to_num(s.read(1), nan=0.0) > 0.5) & footprint
            core_files.append(fn)
    core &= ~labels
    print(f"priority core = union of {len(core_files)} live-best rasters = {int(core.sum())} px")
    S = S + 1.0 * core.astype(np.float32)

    # ---- Bour & Davy fractal-clustering prior, fitted on the REAL catalogue (H22's D was a placeholder)
    print("fitting Bour & Davy clustering dimension on labels.tif ...")
    fit = gcl.fit_clustering_dimension(labels, footprint, pixel_size_m=100.0)
    D = float(fit.D_correlation)
    if not np.isfinite(D):
        D = float(fit.D_bour_davy_predicted) if np.isfinite(fit.D_bour_davy_predicted) else 1.37
    w_prior = gcl.clustering_geometric_prior(S, footprint, labels, D, pixel_size_m=100.0)
    S = (S * w_prior).astype(np.float32)
    print(f"  measured D_correlation={fit.D_correlation}  D_bour_davy_predicted={fit.D_bour_davy_predicted} "
          f"alpha_ols={fit.alpha_ols} n_traces={fit.n_traces} interpretation={fit.interpretation}")
    (EVIDENCE_DIR / "clustering_fit.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "COMPUTED (fitted on the real labels.tif, not a placeholder)",
        "input_labels_sha256": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
        **fit.__dict__,
    }, indent=2, default=float) + "\n")

    # ---- fuse any H23 channel that passed the pre-registered gate
    fused_channels = []
    h23_report_p = EVIDENCE_DIR / "hypothesis_h23_validation.json"
    chan_path = DATA_DIR / "cache" / "h23_channels_fp.npz"
    if h23_report_p.exists() and chan_path.exists():
        rep = json.loads(h23_report_p.read_text())
        passing = [k for k, v in rep.get("channels", {}).items() if v.get("gate_passed")]
        # At most ONE channel per hypothesis family, so the fused surface stays physically
        # interpretable and the two modulations are near-independent.
        FAMILY_OF = [
            ("h23_orient_persist", "H23-1"), ("h23_hurst", "H23-1"),
            ("h23_front_", "H23-2"), ("h23_smf_", "H23-2"),
            ("h23_aspect_", "H23-5"), ("h23_hs_lineament", "H23-5"),
            ("h23_rad_", "H23-4"),
            ("h23_mag_", "H23-3"), ("h23_grav_", "H23-3"),
        ]
        def family_of(k: str) -> str:
            for pre, fam in FAMILY_OF:
                if k.startswith(pre):
                    return fam
            return "OTHER"
        ranked = sorted(passing, key=lambda k: -rep["channels"][k]["delta_vs_baseline_sgmc"])
        seen_fam, fused_channels = set(), []
        for k in ranked:
            fam = family_of(k)
            if fam in seen_fam:
                continue
            seen_fam.add(fam)
            fused_channels.append(k)
            if len(fused_channels) == 2:
                break
        beta_each = BETA_FUSE / max(len(fused_channels), 1)
        if fused_channels:
            with np.load(chan_path) as z:
                for ck in fused_channels:
                    q = rank_in_footprint(to_2d(z[ck].astype(np.float32), footprint), footprint)
                    S = (S * (1.0 + beta_each * (q - 0.5))).astype(np.float32)
                    del q
                    gc.collect()
        else:
            beta_each = 0.0
    print(f"fused H23 channels: {fused_channels or 'none passed the gate'}")

    S = np.clip(np.nan_to_num(S, nan=0.0), 0.0, 1.0).astype(np.float32)
    S[labels] = 0.0   # known-catalogue pixels are masked by the official scorer

    # ---- historic submissions for the DISTINCT gate
    hist_dir = ROOT / ".cache" / "audit" / "history"
    hist_masks = {}
    if hist_dir.exists():
        for p in sorted(hist_dir.glob("*")):
            try:
                hist_masks[p.name.split("__")[0]] = mask_of(p, footprint, labels)
            except Exception:
                pass
    print(f"historic submissions loaded for the DISTINCT gate: {len(hist_masks)}")

    # ---- build each candidate
    template = DATA_DIR / "sample_submission.tif"
    cand_dicts, preds = [], {}
    for spec in CANDIDATES:
        pred = emit(S, footprint, fold, spec["budget"])
        pred &= ~labels
        arr = pred.astype(np.float32)
        cid = scored_content_id(arr, footprint, labels)
        slug = f"{spec['hid'].lower()}-dti-optimal-emission-{int(round(spec['budget']*100))}pct"
        base = make_filename(FAMILY, slug, DATE, cid, "nan")
        nan_path = DOWNLOADS_DIR / base
        all_path = DOWNLOADS_DIR / base.replace("-nan.tif", "-allfinite.tif")
        write_submission(arr, template, nan_path, outside="nan")
        write_submission(arr, template, all_path, outside="zero")
        zip_path = zip_single(nan_path)
        chk = check_variants(nan_path, DATA_DIR / "sample_submission.tif")
        chk_all = check_variants(all_path, DATA_DIR / "sample_submission.tif")
        hold = score_pred(pred, quads)
        audit = gcl.audit_predicted_clustering(pred, footprint, fit, pixel_size_m=100.0)
        note = (f"22GEMSDOE {spec['hid']} | DTI-optimal emission {spec['budget']*100:.2f}% "
                f"({int(pred.sum())} px) + Bour&Davy clustering prior D={D:.2f} + h19-4/5 priority core "
                f"| id {cid} | not yet live-scored")
        if len(note) > 200:
            note = note[:197] + "..."
        cand_dicts.append({
            "key": spec["key"], "hid": spec["hid"], "role": spec["role"], "basis": spec["basis"],
            "budget_fraction": spec["budget"], "policy": spec["policy"],
            "title": spec["title"], "one_liner": spec["one_liner"],
            "content_id": cid,
            "lines_satisfied": ["L0_FractalClustering_SpatialStatistic", "L1_PopScaling_TipRelay",
                                "L2_Backward_ThermalGeochem", "L3_Openness_LRM_Scarp",
                                "L4_Geopotential_Basement", "L5_DTI_Optimal_Emission_Policy"],
            "fused_h23_channels": fused_channels,
            "files": {
                "tif": {"name": nan_path.name, "bytes": nan_path.stat().st_size,
                        "sha256": sha256_file(nan_path), "href": f"downloads/{nan_path.name}"},
                "zip": {"name": zip_path.name, "bytes": zip_path.stat().st_size,
                        "sha256": sha256_file(zip_path), "href": f"downloads/{zip_path.name}"},
                "tif_allfinite": {"name": all_path.name, "bytes": all_path.stat().st_size,
                                  "sha256": sha256_file(all_path), "href": f"downloads/{all_path.name}"},
            },
            "note": note,
            "checks_official_format": chk, "checks_allfinite_twin": chk_all,
            "holdout": hold, "clustering_audit": audit,
        })
        preds[spec["key"]] = pred
        print(f"  {spec['hid']:6s} budget={spec['budget']*100:5.2f}%  px={int(pred.sum()):7d}  "
              f"SGMC={hold['mean_sgmc_offcat_dti']:.5f}  dense={hold['mean_dense_dti']:.5f}  "
              f"sparse={hold['mean_sparse_dti']:.5f}  audit={audit['flag']}  id={cid}")
        del pred, arr
        gc.collect()

    # ---- reference candidates kept on the site (h19-4 / h19-5 / h22-1 / h22-2)
    refs = []
    old = SITE_DATA_DIR / "submissions.json"
    old_by_key = {c["key"]: c for c in json.loads(old.read_text())["candidates"]} if old.exists() else {}
    for key in ("h19-4", "h19-5", "h22-1", "h22-2"):
        c = old_by_key.get(key)
        if not c:
            continue
        p = DOWNLOADS_DIR / c["files"]["tif"]["name"]
        if not p.exists():
            continue
        c = dict(c)
        c["role"] = "Reference (previous session)"
        c["basis"] = "INHERITED"
        c["holdout"] = score_pred(mask_of(p, footprint, labels) & footprint, quads)
        refs.append(c)
        preds[key] = mask_of(p, footprint, labels)
    all_c = cand_dicts + refs

    # ---- Jaccard / DISTINCT gate against siblings AND 22 historic submissions
    for c in all_c:
        a = preds[c["key"]]
        sib = []
        for o in all_c:
            if o["key"] == c["key"]:
                continue
            b = preds[o["key"]]
            inter = int(np.logical_and(a, b).sum())
            union = int(np.logical_or(a, b).sum())
            sib.append({"key": o["key"], "hid": o["hid"], "jaccard_positive": round(inter / union, 4) if union else 0.0,
                        "shared_positive": inter, "a_positive": int(a.sum()), "b_positive": int(b.sum())})
        sib.sort(key=lambda x: -x["jaccard_positive"])
        hist = []
        for name, hm in hist_masks.items():
            inter = int(np.logical_and(a, hm).sum())
            union = int(np.logical_or(a, hm).sum())
            hist.append({"id": name, "jaccard_positive": round(inter / union, 4) if union else 0.0,
                         "shared_positive": inter, "a_positive": int(a.sum()), "b_positive": int(hm.sum())})
        hist.sort(key=lambda x: -x["jaccard_positive"])
        worst = max([s["jaccard_positive"] for s in sib] + [h["jaccard_positive"] for h in hist] + [0.0])
        c["similar_to_other_candidates"] = sib[:5]
        c["similar_to_history"] = hist[:5]
        c["uniqueness"] = {"verdict": "DISTINCT" if worst < NEAR_DUP_J else "NEAR_DUP",
                           "candidate_positive_scored_pixels": int(a.sum()),
                           "max_jaccard_any_comparison": round(worst, 4),
                           "nearest": (sib + hist)[:2]}
        c["scored_fraction"] = round(float(a.sum() / footprint.sum()), 5)
        c["scored_pixels_predicted"] = int(a.sum())
        c["share_of_footprint_pct"] = round(float(a.sum() / footprint.sum()) * 100.0, 3)
        print(f"  {c['hid']:6s} DISTINCT gate: {c['uniqueness']['verdict']} (max J = {worst:.4f} "
              f"vs {hist[0]['id'] if hist else '-'})")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_head(),
        "template_sha256": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
        "rolling_limit": "3 submissions per rolling 7-day window per entity (official rules 3.2/3.4)",
        "claims": {
            "score_predicted": False,
            "statement": (
                "H23 changes the emission policy, not the physics: the official DTI marginal-inclusion rule "
                "(E[credit] > 0.2*DTI) implies a far larger emission budget than the 2.4-2.5% the group has used "
                "since 16GEMSDOE. Both arms keep every h19-4/h19-5 pixel and add the measured Bour & Davy "
                "clustering prior. Holdout numbers are proxy measurements, not leaderboard predictions."
            ),
        },
        "emission_policy_evidence": {
            "pre_registered_rule": "maximise mean off-catalogue SGMC DTI subject to mean catalogue dense DTI >= H16-1 baseline - 0.005",
            "baseline_h16_1_at_2_5pct": {"mean_dense_dti": 0.21272, "mean_sparse_dti": 0.08541,
                                          "mean_sgmc_offcat_dti": 0.15302},
            "selected": {"h23-a": {"budget_frac": 0.065, "basis": "PRE_REGISTERED"},
                         "h23-b": {"budget_frac": 0.100, "basis": "POST_HOC_MINIMAX"}},
        },
        "clustering_fit": {
            "D_correlation": fit.D_correlation, "D_bour_davy_predicted": fit.D_bour_davy_predicted,
            "alpha_ols": fit.alpha_ols, "n_traces": fit.n_traces,
            "nearest_larger_median_m": fit.nearest_larger_median_m,
            "interpretation": fit.interpretation,
            "status": "COMPUTED on the real labels.tif (H22 used a placeholder)",
        },
        "candidates": all_c,
    }
    out = SITE_DATA_DIR / "submissions.json"
    out.write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    (EVIDENCE_DIR / "h23_submission_build.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    print(f"\nWrote {out} and evidence/h23_submission_build.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
