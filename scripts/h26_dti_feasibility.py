"""H26 — why the group's best file scores 0.1922, and what 0.30+ would require.

The official metric (page 967, fetched and quoted verbatim in
``src/gems22/metric.py``) is

    TP_w = sum_{g in G} max_{x: d(x,g) <= R} p(x) k(d(x,g))
    FP_w = sum_{x: p(x) > 0} p(x) [1 - max_{g in G} k(d(x,g))]
    FN_w = sum_{g in G} [1 - max_{x: d(x,g) <= R} p(x) k(d(x,g))]
    DTI  = TP_w / (TP_w + 0.2 FP_w + 0.8 FN_w + eps)

Because the same max term appears in TP_w and FN_w, ``FN_w = |G| - TP_w``
exactly, so

    DTI = A / (0.2 A + 0.2 B + 0.8 |G|),   A = TP_w, B = FP_w.

This script turns that closed form into the three numbers that actually decide
strategy, using only quantities that are (a) measured here or (b) derived from
the group's own live scores:

1. **Implied A, precision and coverage for every live-scored file** -- i.e. what
   the leaderboard says the file actually achieved, in metric units.
2. **The no-skill floor** -- the DTI a *random* binary emission of the same size
   achieves.  Any claimed gain must be stated net of this.
3. **The (n, precision) pairs that reach a target DTI** -- the feasibility map for
   0.30 and for the current public leader 0.3195.

It also re-verifies identity (1) numerically on a real published raster.

Run:
    python3 scripts/h26_dti_feasibility.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.metric import ALPHA, BETA, KERNEL_OFFSETS, RADIUS_PX  # noqa: E402
from gems.paths import DATA_DIR, DOWNLOADS_DIR  # noqa: E402

EVI = ROOT / "evidence"
ALPHA_, BETA_ = 0.2, 0.8

# live public scores reported by the project owner (registry/gems22_submissions.json
# and registry/submissions.json).  Kept as data, not hard-coded into the prose.
LIVE = [
    ("h19-5", 121131, 0.1922),
    ("h19-4", 123779, 0.1894),
    ("h16-1", 123939, 0.1855),
    ("h28-dotted-ridge", 120966, 0.1839),
    ("ens12 (GEMSDOE1/5/8)", 166519, 0.1563),
    ("pindrop-v4-nodes", 155021, 0.1193),
    ("r7-nms3-dem10", 103347, 0.1294),
    ("h25-ctx-ridge", 143657, 0.1280),
]
G_QUANTILES = {"p16": 96255.0, "p50": 116106.0, "p84": 134963.0,
               "g_mle": 107000.0, "g_gridfit": 125000.0}


def implied_A(dti: float, n: int, G: float) -> float:
    """A under the convention B = n - A (the one `fit_G_mle` uses)."""
    return dti * (ALPHA_ * n + BETA_ * G)


def implied_A_bounds(dti: float, n: int, G: float) -> tuple[float, float]:
    """Exact bounds on A consistent with an observed (dti, n, G) and 0 <= B <= n.

    A(1 - alpha*dti) = alpha*dti*B + beta*dti*G  =>  A is linear in B, so the
    bounds are attained at B = 0 and B = n.  Both are *inferences*; the truth
    lies between them and is not observable from here.
    """
    lo = (BETA_ * dti * G) / (1.0 - ALPHA_ * dti)
    hi = (ALPHA_ * dti * n + BETA_ * dti * G) / (1.0 - ALPHA_ * dti)
    return lo, hi


def no_skill_floor(n: int, G: float, domain_px: int) -> dict:
    """Expected DTI of a random binary emission of n pixels in domain_px."""
    ksum = float(np.sum([k for _dy, _dx, k in KERNEL_OFFSETS]))
    f = n / domain_px
    A = G * f * ksum                     # expected total weighted credit
    A = min(A, G)
    B = n - A                            # emitted mass minus the credited part
    dti = A / (ALPHA_ * A + ALPHA_ * B + BETA_ * G + 1e-12)
    return {"n": n, "A_expected": A, "FP_expected": B, "dti_expected": dti,
            "kernel_offset_sum": ksum}


def required_precision(dti: float, n: int, G: float) -> dict:
    """Precision q = A/n needed to reach ``dti`` at budget n (binary emission)."""
    A = dti * (ALPHA_ * n + BETA_ * G) / (1.0 - ALPHA_ * dti)
    return {"n": n, "required_A": A, "required_precision_q": A / n,
            "required_coverage_A_over_G": A / G,
            "feasible": bool(A <= n and A <= G)}


def main() -> int:
    EVI.mkdir(exist_ok=True)
    with rasterio.open(DATA_DIR / "sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(DATA_DIR / "labels.tif") as s:
        cat = (s.read(1) > 0) & fp
    with rasterio.open(DATA_DIR / "existing_faults.tif") as s:
        ex = s.read(1) > 0
    domain_px = int((fp & ~cat).sum())
    print(f"footprint {int(fp.sum()):,}  catalogue {int(cat.sum()):,}  scored domain {domain_px:,}")

    # ---- identity (1) on a real published raster -----------------------------
    p = DOWNLOADS_DIR / "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif"
    identity = {}
    if p.exists():
        from gems.metric import dti_components_exact
        with rasterio.open(p) as s:
            a = s.read(1)
        c = dti_components_exact(a, cat, valid_mask=fp & ~cat, catalogue_mask=cat)
        identity = {
            "file": p.name,
            "TP_w": round(c["TP_w"], 4), "FP_w": round(c["FP_w"], 4),
            "FN_w": round(c["FN_w"], 4), "n_gt": c["n_truth"], "dti": round(c["dti"], 6),
            "residual_FN_minus_absG_minus_TP": round(c["FN_w"] - (c["n_truth"] - c["TP_w"]), 9),
            "note": "computed against the PUBLIC catalogue as a stand-in for the scored set",
        }
        print("identity check:", identity["residual_FN_minus_absG_minus_TP"])

    # ---- what the live scores imply -----------------------------------------
    rows = []
    for name, n, lb in LIVE:
        r = {"id": name, "n_emitted": n, "live_public_dti": lb}
        for gname, G in G_QUANTILES.items():
            A = implied_A(lb, n, G)
            lo, hi = implied_A_bounds(lb, n, G)
            r[f"implied_A_conventionB_eq_n_minus_A_at_{gname}"] = round(A, 1)
            r[f"implied_A_bounds_at_{gname}"] = [round(lo, 1), round(hi, 1)]
            r[f"implied_precision_bounds_at_{gname}"] = [round(lo / n, 4), round(hi / n, 4)]
            r[f"implied_coverage_bounds_at_{gname}"] = [round(lo / G, 4), round(hi / G, 4)]
        rows.append(r)

    # ---- no-skill floor -----------------------------------------------------
    floors = [no_skill_floor(n, G_QUANTILES["p50"], domain_px)
              for n in (60_000, 121_131, 250_000, 550_000, 1_000_000, 2_000_000)]
    for f in floors:
        f["A_expected"] = round(f["A_expected"], 1)
        f["FP_expected"] = round(f["FP_expected"], 1)
        f["dti_expected"] = round(f["dti_expected"], 5)

    # ---- what 0.30 / the leader would require -------------------------------
    feas = []
    for target in (0.20, 0.25, 0.30, 0.3195):
        for n in (121_131, 250_000, 550_000, 1_000_000, 2_000_000):
            req = required_precision(target, n, G_QUANTILES["p50"])
            req["target_dti"] = target
            req["required_A"] = round(req["required_A"], 1)
            req["required_precision_q"] = round(req["required_precision_q"], 4)
            req["required_coverage_A_over_G"] = round(req["required_coverage_A_over_G"], 4)
            feas.append(req)

    # ---- marginal rule at a few operating points ----------------------------
    marg = []
    for dti in (0.19, 0.20, 0.25, 0.30, 0.3195):
        tau = ALPHA_ * dti / (1 - ALPHA_ * dti)
        marg.append({"dti": dti, "tau": round(tau, 6),
                     "pi_star_break_even_posterior": round(tau / (1 + tau), 6),
                     "one_over_tau": round(1.0 / tau, 3)})

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/h26_dti_feasibility.py",
        "status": "COMPUTED_ON_REAL_DATA",
        "metric": {"alpha": ALPHA, "beta": BETA, "R_px": RADIUS_PX,
                   "closed_form": "DTI = A / (0.2A + 0.2B + 0.8|G|), FN_w = |G| - A exactly"},
        "grid": {"footprint_px": int(fp.sum()), "catalogue_px": int(cat.sum()),
                 "catalogue_px_via_existing_faults_tif": int(ex.sum()),
                 "scored_domain_px": domain_px},
        "identity_check": identity,
        "G_quantiles": G_QUANTILES,
        "live_scores_implied": rows,
        "no_skill_floor": floors,
        "feasibility_map": feas,
        "marginal_rule": marg,
        "caveat": ("A here is inferred from the closed form and a fitted |G|; the hidden "
                   "private ground truth is not observable from this repository. The "
                   "implied precision is therefore an inference, not a measurement, and it "
                   "carries the full |G| uncertainty band quoted above."),
    }
    (EVI / "h26_dti_feasibility.json").write_text(json.dumps(out, indent=1))
    print("wrote evidence/h26_dti_feasibility.json")
    for r in rows[:4]:
        print(f"  {r['id']:22s} n={r['n_emitted']:>8,} lb={r['live_public_dti']:.4f} "
              f"A(p50)={r['implied_A_conventionB_eq_n_minus_A_at_p50']:>8,.0f} "
              f"bounds={r['implied_A_bounds_at_p50']} "
              f"q_bounds={r['implied_precision_bounds_at_p50']}")
    for f in floors:
        print(f"  no-skill n={f['n']:>9,} expected DTI={f['dti_expected']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
