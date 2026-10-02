#!/usr/bin/env python3
"""Step 04c -- the decisive head-to-head: our out-of-fold maps versus the three
GeoTIFFs whose LIVE public scores we know, on targets the anchors never trained on.

WHY THIS COMPARISON IS THE ONLY FAIR ONE AVAILABLE
--------------------------------------------------
* The live private ground truth is unobservable from this sandbox.
* Held-out catalogue traces are contaminated IN THE ANCHORS' FAVOUR: every anchor
  was built with the full catalogue as an input, so its ridge detectors can sit
  directly on a trace we are calling "held out".
* The SGMC-gap target is external. It was never a label for h16-1, h19-4 or h19-5
  (16GEMSDOE produced one SGMC-gap-derived file but it is NOT one of the three
  anchors), and it is a real, public, independently-mapped population of faults
  that the competition catalogue missed -- structurally the same object as the
  private test set.

So: score BOTH the anchors and our out-of-fold maps against SGMC-gap, inside the
SAME scored domain, at the SAME emitted-pixel budget.  That is a like-for-like
comparison with no contamination in either direction.

The anchors are also scored at their own native budget (121k-124k px) so the
familiar live numbers can be sanity-checked against the offline ones.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import holdout as ho
from gems22.metric import budget_curve, components
from gems22.spec import REPO, load_labels

DERIV = REPO / "data/derived"
ANCHORS = {
    "h16-1 (live 0.1855)": REPO / "assets/lb_anchors/lb_anchor_h16-1_0.1855.tif",
    "h19-4 (live 0.1894)": REPO / "assets/lb_anchors/lb_anchor_h19-4_0.1894.tif",
    "h19-5 (live 0.1922)": REPO / "assets/lb_anchors/lb_anchor_h19-5_0.1922.tif",
}
BUDGETS = [25_000, 50_000, 75_000, 100_000, 120_000, 150_000, 200_000, 250_000,
           300_000, 400_000, 500_000, 700_000, 1_000_000, 1_500_000, 2_000_000,
           3_000_000, 4_000_000]


def main() -> None:
    t0 = time.time()
    cat = load_labels()
    with rasterio.open(REPO / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    sg = (rasterio.open(REPO / "assets/external/derived_sgmc_faults_100m_u8.tif").read(1) > 0)
    gap = sg & ~cat & fp
    scored = fp & ~cat                      # live scored domain (forum 11516 post 2)
    print(f"scored domain {int(scored.sum()):,} px | SGMC-gap target {int(gap.sum()):,} px "
          f"({100*gap.sum()/scored.sum():.3f}% of domain)", flush=True)

    hr = REPO / "evidence/holdout_union.json"
    folds = json.loads(hr.read_text()).get("folds", {}) if hr.exists() else {}
    SEED = 22
    tfolds = ho.trace_cluster_folds(cat | gap, n_folds=4, n_clusters=48, seed=SEED)
    print(f"reconstructed the same trace-cluster partition used for training "
          f"(seed={SEED}): {sorted(tfolds)}", flush=True)

    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "target": "HELD-OUT SGMC-gap traces only (the fold's own share of the "
                     "USGS state geologic map compilation faults that are "
                     "pixel-disjoint from the competition catalogue). Head B is "
                     "trained on the complement, so scoring against the full gap "
                     "would be leakage; measured, that leakage inflates DTI from "
                     "0.09 to 0.31.",
           "scored_domain_px": int(scored.sum()), "target_px": int(gap.sum()),
           "budgets": BUDGETS, "anchors": {}, "oof": {}, "verdict": {}}

    # ---- anchors, whole-domain, at their native budget ---------------------
    print("\n=== ANCHORS on SGMC-gap over the WHOLE scored domain ===")
    for name, p in ANCHORS.items():
        a = np.nan_to_num(rasterio.open(p).read(1), nan=0.0)
        S = (a > 0.5) & scored
        c = components(S.astype(np.float64), gap, evaluate_mask=scored)
        out["anchors"][name] = {"file": p.name, "n": int(c.n_pred_pos),
                                "A": round(c.tp_w, 2), "B": round(c.fp_w, 2),
                                "eff": round(c.tp_w / max(c.n_pred_pos, 1), 5),
                                "coverage_of_target": round(c.tp_w / max(int(gap.sum()), 1), 5),
                                "dti": round(c.dti, 6)}
        print(f"  {name:22s} n={c.n_pred_pos:9,d} A={c.tp_w:9,.1f} eff={c.tp_w/max(c.n_pred_pos,1):.4f} "
              f"cov={c.tp_w/gap.sum():.4f} DTI={c.dti:.5f}", flush=True)

    # ---- our out-of-fold maps, per fold, on the same target -----------------
    print("\n=== OUR OUT-OF-FOLD MAPS on SGMC-gap inside each fold's scored domain ===")
    anchor_ref = None
    for fname, rec in folds.items():
        for h in ("A", "B", "C"):
            f = DERIV / f"oof_{h}_{fname}.f32"
            if not f.exists():
                continue
            P = np.fromfile(f, dtype=np.float32).reshape(cat.shape)
            # ---- CONTAMINATION CONTROL --------------------------------------
            # Head B was TRAINED on the SGMC-gap traces that are not in this
            # fold, so the target must be ONLY this fold's held-out gap traces
            # and the scored domain must exclude everything head B was trained
            # on.  The anchors never saw the gap at all, so this domain/target
            # pair is contamination-free in BOTH directions and identical for
            # every model being compared.
            if fname not in tfolds:
                continue
            gt_all = tfolds[fname]
            tgt = gt_all & gap & fp                      # held-out gap traces
            if h == "B":
                trained = gap & ~ndimage.binary_dilation(tgt, np.ones((3, 3), bool), 1)
                dom = scored & ~trained
            else:
                dom = scored
            if int(tgt.sum()) < 200:
                print(f"  fold {fname} head {h}: held-out target too small "
                      f"({int(tgt.sum())} px), skipped", flush=True)
                continue
            cur = budget_curve(P, tgt, dom, BUDGETS)
            if not cur:
                continue
            best = max(cur, key=lambda r: r["dti"])
            at120 = min(cur, key=lambda r: abs(r["n"] - 120_000))
            # anchor scored on the SAME domain, at the SAME budget, for fairness
            anch = {}
            for aname, ap in ANCHORS.items():
                a = np.nan_to_num(rasterio.open(ap).read(1), nan=0.0)
                # anchors are binary masks: score them directly with components()
                S = (a > 0.5) & dom
                cc = components(S.astype(np.float64), tgt, evaluate_mask=dom)
                anch[aname] = {"n": int(cc.n_pred_pos), "A": round(cc.tp_w, 2),
                               "eff": round(cc.tp_w / max(cc.n_pred_pos, 1), 5),
                               "dti": round(cc.dti, 6)}
            out["oof"][f"{fname}/{h}"] = {"domain_px": int(dom.sum()),
                                          "target_px_in_domain": int((tgt & dom).sum()),
                                          "head": h, "fold": fname,
                                          "at_120k": at120, "best": best,
                                          "gain_best_vs_120k": round(best["dti"] - at120["dti"], 6),
                                          "anchors_on_same_domain": anch,
                                          "curve": cur}
            best_anchor = max(anch.items(), key=lambda kv: kv[1]["dti"])
            beat = at120["dti"] > best_anchor[1]["dti"]
            print(f"  fold {fname} head {h}: dom={int(dom.sum()):,} "
                  f"HELD-OUT tgt={int((tgt&dom).sum()):,}")
            print(f"      ours   n={at120['n']:,} A={at120['A']:,.0f} eff={at120['avg_eff']:.4f} DTI={at120['dti']:.5f}")
            for an, av in anch.items():
                print(f"      {an:22s} n={av['n']:,} A={av['A']:,.0f} eff={av['eff']:.4f} DTI={av['dti']:.5f}")
            print(f"      -> ours beats best anchor at matched budget? {'YES' if beat else 'NO'}"
                  f"   | ours at DTI-optimal n={best['n']:,} DTI={best['dti']:.5f}", flush=True)
            if anchor_ref is None:
                anchor_ref = best_anchor

    # ---- verdict -----------------------------------------------------------
    wins = []
    for k, v in out["oof"].items():
        ba = max(v["anchors_on_same_domain"].items(), key=lambda kv: kv[1]["dti"])
        wins.append({"key": k, "ours_120k": v["at_120k"]["dti"],
                     "ours_best": v["best"]["dti"], "ours_best_n": v["best"]["n"],
                     "best_anchor": ba[0], "anchor_dti": ba[1]["dti"],
                     "beats_at_matched_budget": bool(v["at_120k"]["dti"] > ba[1]["dti"]),
                     "beats_at_own_optimum": bool(v["best"]["dti"] > ba[1]["dti"])})
    out["verdict"]["per_fold"] = wins
    if wins:
        out["verdict"]["beats_at_matched_budget"] = \
            f"{sum(w['beats_at_matched_budget'] for w in wins)}/{len(wins)}"
        out["verdict"]["beats_at_own_optimum"] = \
            f"{sum(w['beats_at_own_optimum'] for w in wins)}/{len(wins)}"
        nb = [w["ours_best_n"] for w in wins]
        out["verdict"]["median_dti_optimal_budget"] = int(np.median(nb)) if nb else None
        out["verdict"]["gate"] = (
            "PASS - our map beats the best live-scored anchor on the clean external "
            "target at matched budget" if all(w["beats_at_matched_budget"] for w in wins)
            else ("PARTIAL - our map beats the anchors only at a larger budget"
                  if any(w["beats_at_own_optimum"] for w in wins)
                  else "FAIL - do not spend a submission slot on this map"))
    out["seconds"] = round(time.time() - t0, 1)
    (REPO / "evidence/head_to_head.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nVERDICT:", json.dumps(out["verdict"].get("gate"), indent=1))
    print("beats at matched budget:", out["verdict"].get("beats_at_matched_budget"),
          "| at own optimum:", out["verdict"].get("beats_at_own_optimum"),
          "| median DTI-optimal budget:", out["verdict"].get("median_dti_optimal_budget"))
    print(f"wrote evidence/head_to_head.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
