"""Does any local proxy truth reproduce the REAL leaderboard ordering of the group's scored files?

For every registered file with a public score we compute its DTI against candidate proxy truths, with the known-fault mask
applied (staff ruling) - then rank-correlate with the public scores (duplicates/twins collapsed to one point).

Proxy truths
  sgmc_gap      : USGS SGMC geologic-map faults >300 m from every catalogued fault (faults that are demonstrably NOT in the catalogue)
  sgmc_offcat   : SGMC fault pixels not on a catalogue pixel (includes near-catalogue displaced copies)
  known_dense   : the catalogue itself, unmasked (CONTAMINATED for files trained on it; shown only for reference)
This evaluates the evaluator: a proxy that cannot rank our own 15 distinct scored files is not a valid gate.
Output: evidence/proxy_calibration_vs_lb.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from scipy.stats import kendalltau, spearmanr
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.metric import dti_score_fast  # noqa: E402
from gems.paths import EVIDENCE_DIR, GROUP_DIR, LABELS_PATH, TEMPLATE_PATH  # noqa: E402


def main() -> None:
    with rasterio.open(TEMPLATE_PATH) as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(LABELS_PATH) as s:
        cat = (s.read(1) > 0) & fp
    with rasterio.open(ROOT / "evidence" / "ci" / "derived_sgmc_faults_100m_u8.tif") as s:
        sg = (s.read(1) == 1) & fp
    d_cat = distance_transform_edt(~cat)
    truths = {
        "sgmc_gap": skeletonize(sg & (d_cat > 3.0)),
        "sgmc_offcat": skeletonize(sg & ~cat),
        "known_dense": cat,
    }
    print({k: int(v.sum()) for k, v in truths.items()})
    reg = json.loads((ROOT / "registry" / "submissions.json").read_text())["entries"]
    sim = json.loads((EVIDENCE_DIR / "submission_similarity.json").read_text())
    rep_of = {}                                               # collapse identical-on-scored groups to one representative
    for grp in sim["identical_on_scored_pixel_groups"]:
        for i in grp:
            rep_of[i] = sorted(grp)[0]
    rows = []
    for e in reg:
        with rasterio.open(GROUP_DIR / f"{e['id']}.tif") as s:
            arr = s.read(1)
        pred = (np.nan_to_num(arr, nan=0.0) > 0.5) & fp
        rec = {"id": e["id"], "lb_score": e["lb_score"], "rep": rep_of.get(e["id"], e["id"]), "n_pred_scored": int((pred & ~cat).sum())}
        for name, tr in truths.items():
            if name == "known_dense":
                rec[name] = round(dti_score_fast(pred, tr, valid_mask=fp)["dti"], 5)
            else:
                rec[name] = round(dti_score_fast(pred, tr, valid_mask=fp, catalogue_mask=cat, mask_predictions=True)["dti"], 5)
        rows.append(rec)
        print(f"{e['id']:<22} LB={str(e['lb_score']):<7} " + " ".join(f"{k}={rec[k]:.4f}" for k in truths), flush=True)
    uniq = {}
    for r in rows:
        if r["lb_score"] is not None and r["id"] == r["rep"]:
            uniq[r["id"]] = r
    pts = list(uniq.values())
    res = {}
    for name in truths:
        x = [p[name] for p in pts]
        y = [p["lb_score"] for p in pts]
        rho, pv = spearmanr(x, y)
        tau, pt = kendalltau(x, y)
        res[name] = {"spearman_rho": round(float(rho), 3), "spearman_p": round(float(pv), 4), "kendall_tau": round(float(tau), 3), "kendall_p": round(float(pt), 4), "n": len(pts)}
        print(name, res[name])
    cand_rows = []
    manifest = json.loads((ROOT / "docs" / "data" / "submissions.json").read_text())
    for c in manifest["candidates"]:
        with rasterio.open(ROOT / "docs" / c["files"]["tif"]["href"]) as s_:
            arr = s_.read(1)
        pred = (np.nan_to_num(arr, nan=0.0) > 0.5) & fp
        rec = {"key": c["key"], "content_id": c["content_id"], "n_pred_scored": int((pred & ~cat).sum())}
        for name, tr in truths.items():
            rec[name] = round(dti_score_fast(pred, tr, valid_mask=fp)["dti"] if name == "known_dense"
                              else dti_score_fast(pred, tr, valid_mask=fp, catalogue_mask=cat, mask_predictions=True)["dti"], 5)
        cand_rows.append(rec)
        print("CANDIDATE", rec, flush=True)
    for ref_key, ref_cid, ref_id in [
        ("h16-1", "df20f65e", "16GEMSDOE"),
        ("h18-3a", "c502dfab", "16GEMSDOE-h18-3a"),
        ("sgmc-gap", "563f11f7", "16GEMSDOE-sgmc-gap"),
    ]:
        if any(r["key"] == ref_key for r in cand_rows):
            continue
        with rasterio.open(GROUP_DIR / f"{ref_id}.tif") as s_:
            arr = s_.read(1)
        pred = (np.nan_to_num(arr, nan=0.0) > 0.5) & fp
        rec = {"key": ref_key, "content_id": ref_cid, "n_pred_scored": int((pred & ~cat).sum())}
        for name, tr in truths.items():
            rec[name] = round(dti_score_fast(pred, tr, valid_mask=fp)["dti"] if name == "known_dense"
                              else dti_score_fast(pred, tr, valid_mask=fp, catalogue_mask=cat, mask_predictions=True)["dti"], 5)
        cand_rows.append(rec)
        print("REFERENCE", rec, flush=True)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "POST_HOC_CALIBRATION",
           "truth_pixels": {k: int(v.sum()) for k, v in truths.items()}, "n_unique_scored_files": len(pts),
           "correlation_with_public_score": res, "per_file": rows, "candidates_on_the_same_scales": cand_rows,
           "caveats": ["Small n; public scores are rounded, time-stamped observations.",
                       "SGMC positions may be hundreds of metres off (nominal 1:1,000,000), which depresses absolute DTI for every file.",
                       "known_dense is contaminated for files trained on the catalogue; it is included only as a reference point."]}
    (EVIDENCE_DIR / "proxy_calibration_vs_lb.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
