"""EXPLORATORY: which signal families are the group's scored files enriched in, and does that track the public score?

For every distinct scored file: the mean percentile rank (over the footprint) of each feature channel at the file's scored
positive pixels (0.5 = no enrichment), then Spearman correlation across files with the public score.
n = 15 and ~55 channels, so expect a few false positives: this generates hypotheses, it does not test them.
Output: evidence/lb_signal_attribution.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.paths import DATA_DIR, EVIDENCE_DIR, GROUP_DIR, LABELS_PATH, TEMPLATE_PATH  # noqa: E402


def main() -> None:
    with rasterio.open(TEMPLATE_PATH) as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(LABELS_PATH) as s:
        cat = (s.read(1) > 0) & fp
    fp_idx = np.flatnonzero(fp.ravel())
    d_cat = distance_transform_edt(~cat).ravel()[fp_idx]
    reg = json.loads((ROOT / "registry" / "submissions.json").read_text())["entries"]
    sim = json.loads((EVIDENCE_DIR / "submission_similarity.json").read_text())
    rep_of = {i: sorted(g)[0] for g in sim["identical_on_scored_pixel_groups"] for i in g}
    files = [e for e in reg if e["lb_score"] is not None and rep_of.get(e["id"], e["id"]) == e["id"]]
    pos = {}
    for e in files:
        with rasterio.open(GROUP_DIR / f"{e['id']}.tif") as s:
            a = s.read(1)
        p2 = (np.nan_to_num(a, nan=0.0) > 0.5) & fp & ~cat
        pos[e["id"]] = p2.ravel()[fp_idx]
    feats = np.load(DATA_DIR / "cache" / "features_fp.npz")
    names = [n for n in feats.files if not n.startswith("ctx_oof")]           # ctx_oof_* are other models' outputs, not physical signals
    lb = np.array([e["lb_score"] for e in files])
    table, rows = {}, []
    for n in names:
        v = feats[n].astype(np.float64)
        pct = (rankdata(v, method="average") - 0.5) / len(v)                    # percentile rank over the footprint
        enr = np.array([float(pct[pos[e["id"]]].mean()) for e in files])
        table[n] = enr
        if np.ptp(enr) > 1e-9:
            rho, p = spearmanr(enr, lb)
            rows.append({"feature": n, "spearman_rho": round(float(rho), 3), "p_uncorrected": round(float(p), 4),
                         "enrichment_range": [round(float(enr.min()), 3), round(float(enr.max()), 3)]})
    dmed = np.array([float(np.median(d_cat[pos[e["id"]]])) for e in files])
    rho, p = spearmanr(dmed, lb)
    rows.append({"feature": "median_distance_to_catalogue_px", "spearman_rho": round(float(rho), 3), "p_uncorrected": round(float(p), 4), "enrichment_range": [round(float(dmed.min()), 1), round(float(dmed.max()), 1)]})
    rows.sort(key=lambda r: -abs(r["spearman_rho"]))
    m = len(rows)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "EXPLORATORY_HYPOTHESIS_GENERATION",
           "n_files": len(files), "files": [e["id"] for e in files], "n_comparisons": m, "bonferroni_p_threshold_0_05": round(0.05 / m, 5),
           "expected_false_positives_at_p_0_05": round(0.05 * m, 1),
           "top_by_abs_rho": rows[:15], "bottom_by_abs_rho": rows[-5:],
           "caveat": "n=15 files, many correlated channels; nothing here survives a multiple-comparison correction unless p < the Bonferroni threshold."}
    (EVIDENCE_DIR / "lb_signal_attribution.json").write_text(json.dumps(out, indent=2) + "\n")
    print("comparisons", m, "bonferroni p <", out["bonferroni_p_threshold_0_05"], "| expected false positives at p<.05:", out["expected_false_positives_at_p_0_05"])
    for r in rows[:14]:
        print(f"  {r['feature']:<34} rho={r['spearman_rho']:+.3f} p={r['p_uncorrected']:.4f} range={r['enrichment_range']}")
    print("  ...")
    for r in rows[-3:]:
        print(f"  {r['feature']:<34} rho={r['spearman_rho']:+.3f} p={r['p_uncorrected']:.4f}")


if __name__ == "__main__":
    main()
