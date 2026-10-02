#!/usr/bin/env python3
"""Step 05 -- choose the emission budget under the LIVE ground-truth size, build
the submission, run the Marrett NCC artefact audit, and write both variants.

WHAT THE EVIDENCE SUPPORTS (all measured in this repo, nothing assumed)
----------------------------------------------------------------------
1. `scripts/04_train_and_optimize.py` + `04c_head_to_head.py`: a from-scratch
   gradient-boosted head over a 156-layer filter bank scored eff = 0.004-0.008
   against held-out SGMC-gap traces, versus 0.028-0.042 for the three GeoTIFFs
   whose LIVE public scores we know. It therefore FAILS the pre-registered gate
   ("do not spend a submission slot on an idea that hasn't beaten the current
   holdout best") and is NOT used to build the submission. Recorded, not hidden.

2. The best maps available in this sandbox are the three live-scored anchors in
   `assets/lb_anchors/` (0.1855 / 0.1894 / 0.1922). They are the only candidate
   whose quality is authenticated by an actual DrivenData score.

3. The one lever that can still be applied to them without new data is the
   EMISSION BUDGET, chosen by the exact DTI marginal rule. Measured on
   contamination-free held-out SGMC-gap traces, the offline optimum sits at
   60k-90k px -- but that offline ground truth is 4-12x sparser than the live
   one, and the metric's floor is 0.8*|G|. Rescaling by coverage preservation
   (validated: it reproduces the observed live 0.1855 to within 0.010) moves the
   optimum the other way. `scripts/04b_infer_G_and_rescale.py` does that
   arithmetic; this script applies its answer.

BUILD
-----
Base score map = the best live-proven anchor. Ranked by
    in-mask   : 2 + local-mask-density      (trim the least-corroborated first)
    out-of-mask: (1/(1+d)) * (0.7 + 0.3 * Bour-Davy prior)
so trimming drops isolated specks and extending grows the ridge outward in
distance order, biased toward pixels that lie along the extrapolated clustering
pattern of a known larger fault (Bour & Davy 1999) rather than spatially
isolated ones. Emission is binary 1.0 -- optimal for a fixed support, proved in
`gems22.metric` identity (2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import clustering as cl
from gems22 import holdout as ho
from gems22 import submission as sub
from gems22.metric import budget_curve, components, tau
from gems22.spec import REPO, load_labels

ANCHORS = {
    "h16-1": ("assets/lb_anchors/lb_anchor_h16-1_0.1855.tif", 0.1855),
    "h19-4": ("assets/lb_anchors/lb_anchor_h19-4_0.1894.tif", 0.1894),
    "h19-5": ("assets/lb_anchors/lb_anchor_h19-5_0.1922.tif", 0.1922),
}
BUDGETS = [30_000, 45_000, 60_000, 80_000, 100_000, 120_000, 150_000, 180_000,
           220_000, 260_000, 300_000, 360_000, 420_000, 500_000, 600_000,
           750_000, 900_000, 1_100_000, 1_400_000]


def rank_score(mask: np.ndarray, bd_prior: np.ndarray, dens_r: int = 5) -> np.ndarray:
    """Continuous re-ranking of a proven binary ridge mask (see module docstring)."""
    m = mask.astype(np.float32)
    dens = ndimage.uniform_filter(m, 2 * dens_r + 1)
    dmax = float(np.percentile(dens[m > 0], 99)) if m.any() else 1.0
    dens_n = np.clip(dens / max(dmax, 1e-9), 0, 1)
    d = ndimage.distance_transform_edt(~mask)
    outside = (1.0 / (1.0 + d)) * (0.7 + 0.3 * bd_prior.astype(np.float32))
    return np.where(mask, 2.0 + dens_n, outside).astype(np.float32)


def live_rescale(curve: list[dict], G_hold: int, G_live: float) -> list[dict]:
    s = G_live / max(G_hold, 1)
    out = []
    for r in curve:
        # A rescales with |G| (coverage preservation); B is the emitted pixel
        # count and does NOT -- it is a property of the submission, not of G.
        A = s * r["A"]
        B = r["B"]
        den = (1.0 - 0.8) * A + 0.2 * B + 0.8 * G_live      # = 0.2A + 0.2B + 0.8|G|
        d = A / den if den > 0 else 0.0
        out.append({"n": r["n"], "A_hold": r["A"], "A_live_scaled": round(A, 1),
                    "B": r["B"], "dti_hold": r["dti"], "dti_live_scaled": round(d, 6),
                    "tau_live": round(tau(d), 6),
                    "marg_eff_live": None,
                    "coverage_of_G_live": round(A / G_live, 4)})
    for i in range(1, len(out)):
        dn = out[i]["n"] - out[i - 1]["n"]
        if dn > 0:
            out[i]["marg_eff_live"] = round(
                (out[i]["A_live_scaled"] - out[i - 1]["A_live_scaled"]) / dn, 5)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=0, help="0 = take from live-rescaled analysis")
    ap.add_argument("--base", default="h19-5", choices=list(ANCHORS))
    ap.add_argument("--prior-w", type=float, default=0.30)
    ap.add_argument("--bd-topk", type=int, default=200)
    ap.add_argument("--tag", default="h22-value-emit")
    ap.add_argument("--force-budget", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    cat = load_labels()
    with rasterio.open(REPO / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    sg = (rasterio.open(REPO / "assets/external/derived_sgmc_faults_100m_u8.tif").read(1) > 0)
    gap = sg & ~cat & fp
    scored = fp & ~cat
    cfit = json.loads((REPO / "evidence/clustering_fit.json").read_text())
    ggeo = json.loads((REPO / "registry/group_geometry.json").read_text()) \
        if (REPO / "registry/group_geometry.json").exists() else {}
    G_live = float(ggeo.get("G_used") or 125_000)
    print(f"scored domain {int(scored.sum()):,} | SGMC-gap {int(gap.sum()):,} | "
          f"|G|_live used = {G_live:,.0f}", flush=True)

    # ---- Bour & Davy geometric prior from the FULL catalogue ---------------
    lab, n = ndimage.label(cat, structure=cl.STRUCT8)
    t = cl.extract_traces(cat, skeleton=True)
    pop = next((p for p in cfit["populations"] if p["population"].startswith("A_")), None)
    bd = cl.bour_davy_prior_field(lab, t, (pop or {}).get("nearest_larger", {}),
                                  pixel_m=100.0, top_k=args.bd_topk, r_max_px=60)
    print(f"Bour&Davy prior: nonzero={int((bd>0).sum()):,} mean={bd.mean():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- live-rescaled budget optimum for every anchor ---------------------
    tfolds = ho.trace_cluster_folds(cat | gap, n_folds=4, n_clusters=48, seed=22)
    analysis: dict[str, dict] = {}
    print("\n=== live-rescaled DTI-vs-budget for each anchor ===", flush=True)
    for an, (relp, lb) in ANCHORS.items():
        A_mask = (np.nan_to_num(rasterio.open(REPO / relp).read(1), nan=0.0) > 0.5)
        per_fold, votes_off, votes_live = [], [], []
        for fname, gt_all in tfolds.items():
            tgt = gt_all & gap & fp
            trained = gap & ~ndimage.binary_dilation(tgt, np.ones((3, 3), bool), 1)
            dom = scored & ~trained
            Gh = int((tgt & dom).sum())
            if Gh < 200:
                continue
            sc = rank_score(A_mask & dom, bd)
            curve = budget_curve(sc, tgt, dom, BUDGETS)
            rs = live_rescale(curve, Gh, G_live)
            bo = max(curve, key=lambda r: r["dti"])
            bl = max(rs, key=lambda r: r["dti_live_scaled"])
            native = min(curve, key=lambda r: abs(r["n"] - A_mask.sum()))
            native_l = min(rs, key=lambda r: abs(r["n"] - A_mask.sum()))
            votes_off.append(bo["n"]); votes_live.append(bl["n"])
            per_fold.append({"fold": fname, "G_hold": Gh, "domain_px": int(dom.sum()),
                             "offline_optimum": bo, "live_rescaled_optimum": bl,
                             "at_native_budget": {"offline": native, "live": native_l},
                             "live_gain_vs_native": round(
                                 bl["dti_live_scaled"] - native_l["dti_live_scaled"], 6),
                             "curve_offline": curve, "curve_live_scaled": rs})
            print(f"  {an} {fname}: G_hold={Gh:6,d} native n={native['n']:7,d} "
                  f"DTI_off={native['dti']:.5f} DTI_live={native_l['dti_live_scaled']:.5f} | "
                  f"off-opt n={bo['n']:7,d} live-opt n={bl['n']:7,d} "
                  f"DTI_live={bl['dti_live_scaled']:.5f} "
                  f"(gain {bl['dti_live_scaled']-native_l['dti_live_scaled']:+.5f})", flush=True)
        analysis[an] = {"live_score": lb, "folds": per_fold,
                        "median_offline_optimum": int(np.median(votes_off)) if votes_off else None,
                        "median_live_optimum": int(np.median(votes_live)) if votes_live else None,
                        "mean_live_gain_vs_native": float(np.mean(
                            [f["live_gain_vs_native"] for f in per_fold])) if per_fold else None,
                        "folds_gaining": int(sum(f["live_gain_vs_native"] > 0 for f in per_fold)),
                        "n_folds": len(per_fold)}
        print(f"  {an} SUMMARY: median offline optimum={analysis[an]['median_offline_optimum']:,} "
              f"median LIVE-rescaled optimum={analysis[an]['median_live_optimum']:,} "
              f"mean live gain={analysis[an]['mean_live_gain_vs_native']:+.5f} "
              f"({analysis[an]['folds_gaining']}/{analysis[an]['n_folds']} folds gain)", flush=True)

    # ---- pick base + budget ------------------------------------------------
    best_base = max(analysis, key=lambda a: (analysis[a]["mean_live_gain_vs_native"] or -9))
    base = args.base
    budget = args.budget or args.force_budget
    src = "cli"
    if budget <= 0:
        cand = [analysis[a]["median_live_optimum"] for a in analysis
                if analysis[a]["median_live_optimum"]]
        if cand:
            budget = int(np.median(cand))
            src = f"median_live_rescaled_optimum_over_{len(cand)}_anchors_x_4_folds"
        else:
            budget, src = 120_000, "fallback_native"
    # never emit outside [1% , 25%] of the scored domain without explicit intent
    lo, hi = int(0.010 * scored.sum()), int(0.25 * scored.sum())
    clipped = min(max(budget, lo), hi)
    if clipped != budget:
        src += f" (clipped from {budget:,} to the sane band [{lo:,},{hi:,}])"
        budget = clipped
    print(f"\nBASE = {base} (live {ANCHORS[base][1]}) | BUDGET = {budget:,} scored px "
          f"({100*budget/scored.sum():.3f}% of domain) [{src}]", flush=True)
    print(f"  (live-rescaled analysis preferred base {best_base}; --base kept at {base} "
          f"because it holds the highest authenticated live score)", flush=True)

    # ---- build ------------------------------------------------------------
    A_mask = (np.nan_to_num(rasterio.open(REPO / ANCHORS[base][0]).read(1), nan=0.0) > 0.5)
    sc = rank_score(A_mask, bd)
    elig = scored
    order = np.argsort(-sc[elig].ravel(), kind="stable")
    elig_idx = np.flatnonzero(elig.ravel())
    S = np.zeros(cat.shape, bool)
    S.reshape(-1)[elig_idx[order[:budget]]] = True
    n_from_base = int((S & A_mask).sum())
    n_new = int((S & ~A_mask).sum())
    n_dropped = int((A_mask & scored & ~S).sum())
    print(f"support={int(S.sum()):,} px  = {n_from_base:,} retained from the proven base "
          f"+ {n_new:,} newly emitted; {n_dropped:,} base px trimmed", flush=True)

    # ---- post-hoc Marrett NCC artefact audit -------------------------------
    audit = {}
    for nm, mask in (("known_catalogue", cat), ("base_anchor", A_mask & scored),
                     ("predicted", S)):
        ncc = cl.ncc_1d_from_raster(mask, angles_deg=(0, 90), n_random=300, seed=11)
        pooled = ncc.get("pooled", {})
        audit[nm] = cl.classify_arrangement(np.array(pooled.get("ncc", [])),
                                            np.array(pooled.get("lag_centres_px", [])),
                                            lo_px=2.0, hi_px=40.0)
        audit[nm]["ncc_pooled"] = {"lag_centres_px": pooled.get("lag_centres_px"),
                                   "ncc": pooled.get("ncc"),
                                   "n_scanlines": pooled.get("n_scanlines_total")}
    k = audit["known_catalogue"]["mean_ncc"]; p = audit["predicted"]["mean_ncc"]
    b = audit["base_anchor"]["mean_ncc"]
    ratio = p / k if k else None
    flags = []
    if ratio is not None:
        audit["ratio_predicted_over_catalogue_mean_ncc"] = round(float(ratio), 4)
        audit["ratio_predicted_over_base_mean_ncc"] = round(float(p / b), 4) if b else None
        if ratio < 0.5:
            flags.append(f"PREDICTED POPULATION FAR LESS CLUSTERED than the measured regional "
                         f"population (mean NCC ratio {ratio:.2f}) -- consistent with survey-line "
                         f"aliasing or acquisition-block edges rather than genuine geology.")
        elif ratio > 2.0:
            flags.append(f"PREDICTED POPULATION FAR MORE CLUSTERED than the measured regional "
                         f"population (mean NCC ratio {ratio:.2f}) -- emission is collapsing onto "
                         f"a few hot spots and leaving most of the province unsampled.")
    if audit["known_catalogue"]["verdict"] != audit["predicted"]["verdict"]:
        flags.append(f"VERDICT MISMATCH: catalogue={audit['known_catalogue']['verdict']} vs "
                     f"predicted={audit['predicted']['verdict']}.")
    audit["flags"] = flags
    audit["verdict"] = "PASS" if not flags else "REVIEW"
    audit["method_citation"] = ("Marrett, Gale, Gomez & Laubach (2018), J. Struct. Geol. 108:16-33, "
                                "doi:10.1016/j.jsg.2017.06.012; applied to faults in Wang, Laubach, "
                                "Gale & Ramos (2019), Petrol. Geosci. 25:415-428, "
                                "doi:10.1144/petgeo2018-146")
    print(f"\nNCC AUDIT: catalogue mean={k:.3f} ({audit['known_catalogue']['verdict']}) | "
          f"base anchor mean={b:.3f} ({audit['base_anchor']['verdict']}) | "
          f"predicted mean={p:.3f} ({audit['predicted']['verdict']}) -> {audit['verdict']}",
          flush=True)
    for f in flags:
        print("  FLAG:", f, flush=True)

    # ---- write ------------------------------------------------------------
    cid = sub.content_id(S.astype(np.float32), fp, cat)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    note = (f"GEMSDOE22 {args.tag} | value-based emission n={budget:,} "
            f"({100*budget/int(scored.sum()):.2f}% scored domain) from live-proven base {base} "
            f"(live {ANCHORS[base][1]}) | DTI marginal rule at |G|={G_live:,.0f} | "
            f"Bour&Davy(1999) prior w={args.prior_w} | NCC audit {audit['verdict']} | id {cid}")
    written = []
    for variant in ("allfinite", "nan"):
        nm = sub.make_submission_name(args.tag, cid, variant, stamp)
        w = sub.write_submission(S.astype(np.float32), REPO / "docs/downloads" / nm,
                                 fp, variant, cat)
        sub.write_submission(S.astype(np.float32), REPO / "submissions" / nm, fp, variant, cat)
        sub.zip_submission(REPO / "docs/downloads" / nm)
        sub.zip_submission(REPO / "submissions" / nm)
        written.append(w)
        print(f"  wrote {nm}  sha256={w.sha256[:16]} pos_fp={w.n_positive_footprint:,} "
              f"pos_scored={w.n_positive_scored:,} hard_all_pass={w.checks['hard_all_pass']}",
              flush=True)

    rec = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tag": args.tag, "content_id": cid, "base_anchor": base,
           "base_anchor_live_score": ANCHORS[base][1],
           "budget": budget, "budget_source": src, "G_live_used": G_live,
           "prior_weight": args.prior_w, "bd_topk": args.bd_topk,
           "scored_domain_px": int(scored.sum()),
           "budget_frac_of_scored_domain": round(budget / int(scored.sum()), 6),
           "support_composition": {"retained_from_base": n_from_base,
                                   "newly_emitted": n_new, "trimmed_from_base": n_dropped},
           "live_rescaled_budget_analysis": analysis,
           "best_base_by_live_gain": best_base,
           "drivendata_note": note, "audit": audit,
           "gate": {"rule": "do not spend a submission slot on an idea that has not "
                            "beaten the current holdout best",
                    "gbm_heads_passed": False,
                    "gbm_heads_reason": "held-out SGMC-gap efficiency 0.004-0.008 vs "
                                        "0.028-0.042 for the live-scored anchors (0/6 folds won)",
                    "this_file_basis": "value-based re-emission of the highest authenticated "
                                       "live map, using no unvalidated detector",
                    "expected_live_direction": ("live-rescaled DTI improves over the base's own "
                                                "budget in "
                                                f"{analysis[base]['folds_gaining']}/"
                                                f"{analysis[base]['n_folds']} folds; the offline "
                                                "target does NOT reproduce the live ordering of "
                                                "the three anchors (flag F-01), so this is a "
                                                "directional expectation, not a guarantee")},
           "files": [sub.registry_record(w, note) for w in written],
           "seconds": round(time.time() - t0, 1)}
    (REPO / "evidence/submission_build.json").write_text(json.dumps(rec, indent=1, default=str))
    reg = REPO / "registry/submissions.json"
    ex = json.loads(reg.read_text()) if reg.exists() else {"schema_version": 1, "entries": []}
    ex["entries"] = [e for e in ex.get("entries", []) if e.get("content_id") != cid]
    ex["entries"].append(rec)
    ex["generated_utc"] = rec["generated_utc"]
    ex["score_source"] = ("live public scores for the three anchors are the owner-reported "
                          "numbers in the project brief, cross-checked against the live "
                          "leaderboard on 2026-10-01")
    reg.write_text(json.dumps(ex, indent=1, default=str))
    print(f"\nwrote evidence/submission_build.json + registry/submissions.json "
          f"({time.time()-t0:.0f}s)")
    print("\nDRIVENDATA NOTE:\n  " + note)


if __name__ == "__main__":
    main()
