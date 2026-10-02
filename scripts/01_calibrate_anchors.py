#!/usr/bin/env python3
"""Calibrate offline proxy ground truths against the THREE live-leaderboard anchors.

We possess the exact GeoTIFFs that DrivenData scored 0.1855 / 0.1894 / 0.1922.
Any offline proxy that cannot reproduce that ordering is not usable for gating.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22.spec import Spec, REPO
from gems22.metric import components

ANCHORS = {
    "h16-1": (REPO / "assets/lb_anchors/lb_anchor_h16-1_0.1855.tif", 0.1855),
    "h19-4": (REPO / "assets/lb_anchors/lb_anchor_h19-4_0.1894.tif", 0.1894),
    "h19-5": (REPO / "assets/lb_anchors/lb_anchor_h19-5_0.1922.tif", 0.1922),
}


def build_proxy_truths(spec: Spec) -> dict:
    """Three honest proxies for 'real faults absent from the competition catalogue'."""
    sgmc = (rasterio.open(REPO / "assets/external/derived_sgmc_faults_100m_u8.tif").read(1) > 0)
    assert sgmc.shape == spec.shape
    cat = spec.catalogue
    dcat = ndimage.distance_transform_edt(~cat)
    out = {}
    # P1: SGMC faults not in catalogue, strict pixel-disjoint
    out["sgmc_gap_strict"] = sgmc & ~cat & spec.footprint
    # P2: also drop SGMC pixels within the DTI kernel radius of the catalogue
    #     (those are plausibly the *same* fault, digitised 1-3 px differently)
    out["sgmc_gap_d3"] = sgmc & (dcat > 3.0) & spec.footprint
    out["sgmc_gap_d5"] = sgmc & (dcat > 5.0) & spec.footprint
    out["sgmc_all"] = sgmc & spec.footprint
    return out


def main() -> None:
    spec = Spec.load()
    checks = spec.verify()
    print("Spec verified:", json.dumps(checks, indent=1))
    print("Spec summary:", json.dumps(spec.summary(), indent=1, default=str))

    truths = build_proxy_truths(spec)
    preds = {}
    for k, (p, lb) in ANCHORS.items():
        a = rasterio.open(p).read(1)
        preds[k] = (np.nan_to_num(a, nan=0.0) > 0.5, lb)

    rows = []
    for tname, T in truths.items():
        for pname, (P, lb) in preds.items():
            c = components(P.astype(np.float64), T, evaluate_mask=spec.evaluate_mask)
            rows.append({"truth": tname, "pred": pname, "lb": lb, "n_truth": c.n_gt,
                         "n_pred": c.n_pred_pos, "tp_w": round(c.tp_w, 2),
                         "fp_w": round(c.fp_w, 2), "proxy_dti": round(c.dti, 6)})
    # rank correlation of proxy_dti with lb within each truth
    from scipy.stats import spearmanr
    summary = []
    for tname in truths:
        sub = [r for r in rows if r["truth"] == tname]
        rho, p = spearmanr([r["lb"] for r in sub], [r["proxy_dti"] for r in sub])
        ordlb = [r["pred"] for r in sorted(sub, key=lambda r: -r["lb"])]
        ordpx = [r["pred"] for r in sorted(sub, key=lambda r: -r["proxy_dti"])]
        summary.append({"truth": tname, "n_truth_px": int(truths[tname].sum()),
                        "spearman_rho": float(rho), "spearman_p": float(p),
                        "order_matches_lb": ordlb == ordpx,
                        "lb_order": ordlb, "proxy_order": ordpx})
    out = {"generated_utc": "2026-10-01", "spec_checks": checks,
           "spec_summary": spec.summary(), "rows": rows, "summary": summary}
    (REPO / "evidence").mkdir(exist_ok=True)
    (REPO / "evidence/anchor_calibration.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n=== per-truth ranking vs live LB ===")
    for s in summary:
        print(f"  {s['truth']:18s} n={s['n_truth_px']:7d} rho={s['spearman_rho']:+.3f} "
              f"order_match={s['order_matches_lb']} proxy_order={s['proxy_order']}")
    print("\n=== detail ===")
    for r in rows:
        print(f"  {r['truth']:18s} {r['pred']:6s} lb={r['lb']:.4f} npred={r['n_pred']:7d} "
              f"tp_w={r['tp_w']:9.1f} fp_w={r['fp_w']:9.1f} proxyDTI={r['proxy_dti']:.6f}")


if __name__ == "__main__":
    main()
