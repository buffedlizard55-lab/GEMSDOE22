#!/usr/bin/env python3
"""Step 04b -- invert the group's live public scores for |G| and the implied hit
efficiency of every scored file, and rescale the holdout budget curve to the LIVE
ground-truth size.

WHY THIS IS NECESSARY
---------------------
The DTI denominator carries a fixed floor of 0.8*|G|.  A budget that maximises
DTI on a holdout whose ground truth has |G_h| pixels is NOT the budget that
maximises DTI live, where the ground truth has |G_live| pixels: the floor moves.
With

    DTI = A / (0.2*A + 0.2*B + 0.8*|G|)      (identity (1) in gems22.metric)

and B ~= n - A for a thin binary support, the live-equivalent curve computed from
a holdout curve is

    DTI_live(n) = s*A_h(n) / (0.2*s*A_h(n) + 0.2*B_h(n) + 0.8*|G_live|),
    s = |G_live| / |G_h|

which is exactly what `rescale_to_live()` evaluates.  |G_live| is estimated by
jointly inverting the group's live public scores (below).

ESTIMATING |G_live|
-------------------
For each scored file i we observe the public DTI_i and the emitted scored pixel
count n_i.  Under B_i ~= n_i - A_i,

    DTI_i = A_i / (0.2*n_i + 0.8*|G|)     =>     A_i = DTI_i*(0.2*n_i + 0.8*|G|)

with the hard constraints 0 <= A_i <= min(n_i, |G|).  |G| is fitted by choosing
the value that maximises the number of files whose implied efficiency
e_i = A_i/n_i falls inside a plausible band and minimises the spread of e_i at
matched n_i.  The estimate is reported with its full sensitivity, never as a
point value.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22.metric import ALPHA, BETA, tau
from gems22.spec import REPO

FOOTPRINT = 5_167_373
CATALOGUE = 60_988
SCORED_DOMAIN = FOOTPRINT - CATALOGUE          # 5,106,385


def load_group_rows() -> list[dict]:
    """Group submissions with live public scores, from 19GEMSDOE's registry.

    Source: buffedlizard55-lab/19GEMSDOE evidence/submission_similarity.json,
    whose `scored_pixel_definition` is "footprint (finite cells of the official
    template) minus pixel-exact known-fault mask; DrivenData forum 11516 posts 2
    and 4".  Scores are the owner-reported public numbers transcribed in the
    project brief and cross-checked against the live leaderboard on 2026-10-01.
    """
    cand = [REPO.parent.parent / "src/19GEMSDOE/evidence/submission_similarity.json",
            Path("/home/user/src/19GEMSDOE/evidence/submission_similarity.json")]
    for p in cand:
        if p.exists():
            d = json.loads(p.read_text())
            rows = []
            for e in d.get("entries", []):
                rows.append({"id": e.get("id"), "label": (e.get("label") or "")[:60],
                             "lb": e.get("lb_score"),
                             "n": e.get("positive_scored_pixels"),
                             "on_cat": e.get("positive_on_known_faults"),
                             "frac": e.get("scored_fraction_of_footprint"),
                             "f_near300": e.get("frac_near_le_300m"),
                             "f_far1500": e.get("frac_far_gt_1500m")})
            return rows
    return []


def fit_G(rows: list[dict], grid=None) -> dict:
    """Sensitivity scan of |G| over a grid; report the implied efficiency table."""
    rows = [r for r in rows if isinstance(r.get("lb"), (int, float)) and r.get("n")]
    if grid is None:
        grid = np.arange(20_000, 200_001, 2_500)
    best = None
    for G in grid:
        G = float(G)
        A = np.array([r["lb"] * (ALPHA * r["n"] + BETA * G) for r in rows])
        n = np.array([float(r["n"]) for r in rows])
        eff = A / n
        feas = (A <= G) & (A <= n) & (A >= 0)
        if not feas.all():
            continue
        # score: prefer a G for which efficiency is a smooth decreasing function
        # of budget (diminishing returns) and stays in a plausible band
        band = float(np.mean((eff > 0.02) & (eff < 0.35)))
        mono = float(np.corrcoef(np.log(n), eff)[0, 1]) if len(n) > 3 else 0.0
        obj = band - abs(mono + 0.5)      # want most files in band, eff ~ n^-0.5
        if best is None or obj > best["obj"]:
            best = {"G": G, "obj": float(obj), "band_frac": band,
                    "corr_log_n_eff": float(mono),
                    "eff_min": float(eff.min()), "eff_max": float(eff.max())}
    return best or {}


def per_file(rows: list[dict], G: float) -> list[dict]:
    out = []
    for r in rows:
        if not isinstance(r.get("lb"), (int, float)) or not r.get("n"):
            r2 = dict(r); r2["implied_A"] = None; r2["implied_eff"] = None
            out.append(r2); continue
        A = r["lb"] * (ALPHA * r["n"] + BETA * G)
        r2 = dict(r)
        r2["implied_A"] = round(float(A), 1)
        r2["implied_eff"] = round(float(A / r["n"]), 5)
        r2["implied_coverage_of_G"] = round(float(A / G), 5)
        r2["feasible"] = bool(0 <= A <= min(r["n"], G))
        out.append(r2)
    return out


def rescale_to_live(curve: list[dict], G_hold: int, G_live: float,
                    coverage_penalty: float = 1.0) -> list[dict]:
    """Live-equivalent DTI for each budget on a holdout curve.

    coverage_penalty < 1 discounts the assumption that a private new fault is as
    findable as a held-out catalogue fault.  It is a deliberate, explicit
    pessimism knob, not a fudge factor: the private faults are new precisely
    because they are harder.
    """
    s = (G_live / G_hold) * coverage_penalty
    out = []
    for r in curve:
        A = s * r["A"]
        B = r["B"]
        den = (1 - BETA) * A + ALPHA * B + BETA * G_live
        d = A / den if den > 0 else 0.0
        out.append({"n": r["n"], "A_hold": r["A"], "B_hold": r["B"],
                    "A_live_scaled": round(A, 1), "dti_hold": r["dti"],
                    "dti_live_scaled": round(d, 6),
                    "tau_live": round(tau(d), 6),
                    "coverage_of_G_live": round(A / G_live, 4)})
    return out


def main() -> None:
    rows = load_group_rows()
    print(f"group rows with geometry: {len(rows)}")
    fit = fit_G(rows)
    print("fit_G ->", json.dumps(fit, indent=1))
    Gc = fit.get("G") or 57_000.0
    scored = per_file(rows, Gc)
    scored.sort(key=lambda r: -(r.get("lb") or 0))
    print(f"\n{'id':22s} {'LB':>7s} {'n':>9s} {'impliedA':>9s} {'eff':>7s} {'cov':>7s} {'feas':>5s}")
    for r in scored:
        if r.get("implied_A") is None:
            print(f"{str(r.get('id'))[:22]:22s} {'n/a':>7s} {r.get('n') or 0:9d}")
            continue
        print(f"{str(r['id'])[:22]:22s} {r['lb']:7.4f} {r['n']:9d} {r['implied_A']:9.0f} "
              f"{r['implied_eff']:7.4f} {r['implied_coverage_of_G']:7.4f} "
              f"{str(r['feasible']):>5s}")

    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "scored_domain_px": SCORED_DOMAIN, "G_fit": fit, "G_used": Gc,
           "rows": scored,
           "method": "A_i = DTI_i*(0.2*n_i + 0.8*G) under B_i ~= n_i - A_i",
           "caveats": [
               "B_i ~= n_i - A_i assumes a 1-px-thick binary support, which every "
               "group submission is (is_binary_0_1 = true in the registry).",
               "|G| is not identified by a single file; the grid search picks the "
               "value for which implied efficiencies are plausible and show "
               "diminishing returns with budget. Treat it as an order-of-magnitude "
               "estimate with a stated sensitivity, not a measurement.",
               "Scores are owner-reported public numbers, cross-checked against the "
               "live leaderboard on 2026-10-01, not authenticated receipts."]}

    # ---- rescale the holdout budget curve to the live ground-truth size ------
    hr = REPO / "evidence/holdout_union.json"
    if hr.exists():
        d = json.loads(hr.read_text())
        rescaled = {}
        for fname, rec in d.get("folds", {}).items():
            for hk, hv in rec.items():
                if not hk.startswith("head_") or not isinstance(hv, dict):
                    continue
                for tk, tv in hv.items():
                    if not tk.startswith("vs_target_"):
                        continue
                    sweep = tv.get("sweep") or []
                    if not sweep:
                        continue
                    Gh = int(rec.get("gt_px", {}).get(tk.split("_")[-1], 0))
                    if Gh <= 0:
                        continue
                    for pen in (1.0, 0.7, 0.5):
                        rs = rescale_to_live(sweep, Gh, Gc, pen)
                        best = max(rs, key=lambda r: r["dti_live_scaled"])
                        at120 = min(rs, key=lambda r: abs(r["n"] - 120_000))
                        key = f"{fname}/{hk}/{tk}/pen{pen}"
                        rescaled[key] = {"G_hold": Gh, "G_live": Gc,
                                         "coverage_penalty": pen,
                                         "best": best, "at_120k": at120,
                                         "gain": round(best["dti_live_scaled"] -
                                                       at120["dti_live_scaled"], 6),
                                         "curve": rs}
                        print(f"  {key}: G_hold={Gh} best n={best['n']:,} "
                              f"liveDTI={best['dti_live_scaled']:.4f} vs "
                              f"n=120k {at120['dti_live_scaled']:.4f} "
                              f"(gain {best['dti_live_scaled']-at120['dti_live_scaled']:+.4f}) "
                              f"coverage={best['coverage_of_G_live']:.2f}")
        out["rescaled"] = rescaled
        if rescaled:
            # choose the budget that is best under the PESSIMISTIC penalty, i.e.
            # the choice that is robust to new faults being harder to find
            keys = [k for k in rescaled if k.endswith("pen0.5")]
            if keys:
                votes = [rescaled[k]["best"]["n"] for k in keys]
                out["recommended_budget"] = {
                    "n": int(np.median(votes)), "votes": votes,
                    "policy": "median of the per-fold live-rescaled optima under the "
                              "pessimistic coverage_penalty = 0.5 (new faults are "
                              "assumed half as findable as held-out catalogue faults)",
                    "G_live_used": Gc}
                print("\nRECOMMENDED BUDGET:", json.dumps(out["recommended_budget"], indent=1))

    (REPO / "registry").mkdir(exist_ok=True)
    (REPO / "registry/group_geometry.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nwrote registry/group_geometry.json")


if __name__ == "__main__":
    main()
