"""H26 — the no-skill floor of the official metric, validated end to end.

THE PROBLEM THIS SOLVES
-----------------------
``DTI = A / (0.2 A + 0.2 B + 0.8 |G|)`` with ``A = TP_w`` (kernel credit summed
over ground-truth pixels), ``B = FP_w`` (``sum over emitted pixels of
1 - kernel_to_gt``) and ``FN_w = |G| - A``.  (The two weighted sums are *not*
complements: ``A + B != n`` in general -- verified against the repository's own
measured random arms below.)

A *random* emission of ``n`` pixels in a domain of ``N`` pixels is worth far more
than zero, because one emitted pixel credits every ground-truth pixel within
``R = 3`` px.  With emission density ``f = n/N``,

    E[A] = |G| * E[credit per gt pixel]
    E[credit per gt pixel] = INT_0^1 ( 1 - (1-f)^m(v) ) dv
                           = sum_i (k_i - k_{i-1}) * (1 - (1-f)^(m_i))

where ``k_i`` are the distinct kernel values *ascending* and ``m_i`` the number
of kernel offsets with ``k > k_i``.  (The first version of this script had the
telescoping reversed, which yields negative floors -- a good reason the formula
is validated against measured data before anything is concluded from it.)

``E[B] = n (1 - E[max_gt k])`` and ``E[max_gt k] = union_mass / N`` where the
union mass is the area of the union of the 25 R-balls around each gt pixel,
weighted by the max kernel.  For independent gt pixels it is ``|G| * K_sum``;
real fault traces are clustered, so the measured union mass is lower.  The
script calibrates the ratio ``r = measured_union_mass / (|G| K_sum)`` from the
repository's *measured* random arms and uses it for the live projection,
reporting the un-attenuated bound alongside.

Run:
    python3 scripts/h26_floor_model.py     ->  evidence/h26_floor_model.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVI = ROOT / "evidence"
ALPHA, BETA = 0.2, 0.8

# distinct kernel values of the official 200 m..300 m kernel on the 100 m grid,
# ascending, with m = number of offsets whose k is strictly above each value.
KERNEL_ASC = [
    (0.0571909584, 25),
    (0.2546440075, 21),
    (1.0 / 3.0, 13),
    (0.5285954792, 9),
    (2.0 / 3.0, 5),
    (1.0, 1),
]
KSUM = float(sum(k * (m_prev - m) if i else k * 25
                 for i, (k, m_prev) in enumerate(KERNEL_ASC, start=0)
                 for m in [25]))  # placeholder, computed properly below


def _kernel_sum() -> float:
    """Sum of the 25 kernel weights = 9.38030... (matches gems.metric)."""
    return float(sum(v * c for v, c in
                     [(1.0, 1), (2 / 3, 4), (0.5285954792, 4), (1 / 3, 4),
                      (0.2546440075, 8), (0.0571909584, 4)]))


KSUM = _kernel_sum()


def expected_credit_per_gt(f: float) -> float:
    """E over kernel rings: 1 - (1-f)^m_i, weighted by kernel-value widths."""
    total = 0.0
    lo = 0.0
    for k_i, m_i in KERNEL_ASC:
        width = k_i - lo
        total += width * (1.0 - (1.0 - f) ** m_i)
        lo = k_i
    return total


def floor_dti(n: float, G: float, N: float, union_ratio: float) -> dict:
    """Closed-form expected DTI of a random binary emission of n pixels.

    union_ratio = 1.0 gives the upper bound (independent gt pixels);
    the calibrated value accounts for clustering of the gt traces.
    """
    f = min(n / N, 1.0)
    e_credit = expected_credit_per_gt(f)
    A = min(G * e_credit, G)
    union_mass = min(union_ratio * G * KSUM, N)
    B = n * (1.0 - union_mass / N)
    dti = A / (ALPHA * A + ALPHA * B + BETA * G + 1e-12)
    return {"n": n, "f": f, "expected_credit_per_gt": e_credit, "A": A,
            "B": B, "union_mass": union_mass, "dti": dti}


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def main() -> int:
    EVI.mkdir(exist_ok=True)
    out: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/h26_floor_model.py",
        "status": "COMPUTED (validated against measured arms)",
        "kernel_offset_sum": KSUM,
        "formula": ("E[credit/gt px] = sum_i (k_i-k_{i-1}) (1-(1-f)^m_i); "
                    "E[A] = |G| * that; E[B] = n (1 - union_mass/N); "
                    "DTI = A/(0.2A+0.2B+0.8|G|), FN = |G|-A"),
    }

    # ------------------------------------------------ calibrate on measured arms
    hu = json.loads((EVI / "holdout_union.json").read_text())
    # first pass: union-mass ratio implied by each measured FP term
    arm_r = {}
    for fold, fdat in hu["folds"].items():
        for head in ("A", "B"):
            blk = fdat.get("null_random", {}).get(head, {}).get("best")
            if not blk:
                continue
            n = float(blk["n"])
            G = float(fdat["gt_px"][head])
            N = float(fdat["eval_px"][head])
            arm_r[(fold, head)] = (N * (1.0 - blk["B"] / n)) / (G * KSUM)
    union_ratios = list(arm_r.values())

    validation = []
    for fold, fdat in hu["folds"].items():
        for head in ("A", "B"):
            blk = fdat.get("null_random", {}).get(head, {}).get("best")
            if not blk:
                continue
            n = float(blk["n"])
            G = float(fdat["gt_px"][head])
            N = float(fdat["eval_px"][head])
            meas_A, meas_B, meas_dti = blk["A"], blk["B"], blk["dti"]
            r = float(np.mean([v for k, v in arm_r.items() if k != (fold, head)]))
            pred = floor_dti(n, G, N, r)
            validation.append({
                "fold": fold, "head": head, "n": n, "G": G, "N": N,
                "measured_A": round(meas_A, 1), "predicted_A": round(pred["A"], 1),
                "A_rel_err": round(pred["A"] / meas_A - 1.0, 4),
                "measured_B": round(meas_B, 1), "predicted_B": round(pred["B"], 1),
                "B_err_loo_calibrated": round(pred["B"] / meas_B - 1.0, 4),
                "B_err_independent_bound": round(
                    (n * (1.0 - G * KSUM / N)) / meas_B - 1.0, 4),
                "measured_dti": meas_dti, "predicted_dti": round(pred["dti"], 5),
                "dti_rel_err": round(pred["dti"] / meas_dti - 1.0, 4),
                "union_mass_ratio": round(r, 4),
            })
    out["validation_against_measured_random_arms"] = validation
    r_mean = float(np.mean(union_ratios))
    out["union_mass_ratio_calibrated"] = {
        "mean": round(r_mean, 4),
        "min": round(float(np.min(union_ratios)), 4),
        "max": round(float(np.max(union_ratios)), 4),
        "note": ("< 1 means real gt traces are clustered: the union of R-balls "
                 "around them is smaller than |G|*K_sum, so E[B] is larger and "
                 "the true floor is slightly LOWER than the independent bound"),
    }

    # ------------------------------------------------------------ live floors
    N_live = 5_106_385
    G_band = {"p16": 96255.0, "p50": 116106.0, "p84": 134963.0,
              "mle": 107000.0, "grid": 125000.0,
              "ci95_low": 84294.0, "ci95_high": 165151.0}
    grid_n = [60_000, 90_000, 121_131, 160_000, 220_000, 300_000, 400_000,
              550_000, 700_000, 1_000_000, 1_500_000, 2_500_000, 5_106_385]
    live_grid = []
    for n in grid_n:
        row = {"n": n}
        for gname, G in G_band.items():
            row[gname] = round(floor_dti(n, G, N_live, r_mean)["dti"], 5)
        live_grid.append(row)
    out["live_no_skill_floor"] = live_grid
    out["live_domain_px"] = N_live

    # peak of the floor curve at the MLE |G|
    ns = np.arange(20_000, 3_000_000, 2_000, dtype=float)
    peak = max((floor_dti(float(n), 107000.0, N_live, r_mean)["dti"], float(n)) for n in ns)
    out["floor_peak_at_G_mle"] = {"n": peak[1], "dti": round(peak[0], 4)}

    # ------------------------------------------- historic files net of the floor
    cal = json.loads((EVI / "proxy_calibration_vs_lb.json").read_text())
    seen, rows = set(), []
    for r in cal["per_file"]:
        if r.get("lb_score") is None or r.get("n_pred_scored") is None:
            continue
        key = (r["id"], r["n_pred_scored"], r["lb_score"])
        if key in seen:
            continue
        seen.add(key)
        n, lb = float(r["n_pred_scored"]), float(r["lb_score"])
        floor_p50 = floor_dti(n, 116106.0, N_live, r_mean)["dti"]
        floor_lo = floor_dti(n, 84294.0, N_live, r_mean)["dti"]
        floor_hi = floor_dti(n, 165151.0, N_live, r_mean)["dti"]
        # implied A from the exact metric: A = dti*(0.2B+0.8G)/(1-0.2*dti),
        # B in [B_min, n] -> bracket; use B_min = n(1-union/N)
        b_min = n * (1.0 - min(r_mean * 116106.0 * KSUM / N_live, 1.0))
        a_lo = lb * (ALPHA * b_min + BETA * 116106.0) / (1.0 - ALPHA * lb)
        a_hi = lb * (ALPHA * n + BETA * 116106.0) / (1.0 - ALPHA * lb)
        rows.append({
            "id": r["id"], "n": n, "lb": lb,
            "floor_ci95_band": [round(floor_lo, 4), round(floor_hi, 4)],
            "lift_over_floor_low_end": round(lb - floor_hi, 4),
            "lift_over_floor_high_end": round(lb - floor_lo, 4),
            "implied_A_at_G_p50": [round(a_lo, 0), round(a_hi, 0)],
            "implied_precision_at_G_p50": [round(a_lo / n, 4), round(a_hi / n, 4)],
            "floor_A_at_G_p50_n": round(floor_dti(n, 116106.0, N_live, r_mean)["A"], 0),
        })
    out["historic_files_net_of_floor"] = rows

    # the four highest-scoring live anchors on record (registry/submissions.json),
    # including the two that predate the proxy-calibration snapshot
    top_anchors = [
        {"id": "h19-5 e27054cf", "n": 121131, "lb": 0.1922},
        {"id": "h19-4 691e4dfa", "n": 123779, "lb": 0.1894},
        {"id": "h16-1 16GEMSDOE", "n": 123939, "lb": 0.1855},
        {"id": "h28-dotted-ridge", "n": 120966, "lb": 0.1839},
    ]
    ta = []
    for a in top_anchors:
        fl_lo = floor_dti(a["n"], 84294.0, N_live, r_mean)["dti"]
        fl_hi = floor_dti(a["n"], 165151.0, N_live, r_mean)["dti"]
        fl = floor_dti(a["n"], 116106.0, N_live, r_mean)["dti"]
        ta.append({**a, "floor_p50": round(fl, 4),
                   "floor_band_ci95_G": [round(fl_lo, 4), round(fl_hi, 4)],
                   "lift_band": [round(a["lb"] - fl_hi, 4), round(a["lb"] - fl_lo, 4)]})
    out["top_live_anchors_vs_floor"] = ta

    # ------------------------------- rank correlation, raw vs floor-adjusted
    live_only = [r for r in cal["per_file"]
                 if r.get("lb_score") is not None and r.get("n_pred_scored") is not None]
    n_arr = np.array([r["n_pred_scored"] for r in live_only], float)
    lbs = np.array([r["lb_score"] for r in live_only], float)
    floors = np.array([floor_dti(n, 116106.0, N_live, r_mean)["dti"] for n in n_arr])
    corr = {}
    for name in ("sgmc_gap", "sgmc_offcat"):
        vals = np.array([r[name] for r in live_only], float)
        rho_raw, p_raw = spearman(-vals, lbs)
        lift = floors - vals
        rho_lift, p_lift = spearman(lift, lbs)
        corr[name] = {"n_files": len(live_only),
                      "spearman_rho_DTI_vs_LB_negated": round(rho_raw, 4), "p_raw": round(p_raw, 4),
                      "spearman_rho_lift_over_floor_vs_LB": round(rho_lift, 4),
                      "p_lift": round(p_lift, 4),
                      "mean_floor_at_these_n": round(float(np.mean(floors)), 4)}
    out["proxy_rank_correlation_raw_vs_floor_adjusted"] = corr

    # ---------------------------------------------- fold-vs-live transfer
    anchors = [
        {"id": "h16-1", "live": 0.1855, "fold_dense": 0.16953, "n": 124017,
         "fold_floor": 0.10394, "live_floor_lo": floors[0]},
    ]
    # use the T0/A random floor measured in the repo, folded by budget
    transfer = []
    G_fold, N_fold = 21344.0, 5127729.0
    for a in ({"id": "h16-1", "live": 0.1855, "fold_dense": 0.16953, "n": 123939},
              {"id": "h19-4", "live": 0.1894, "fold_dense": 0.21413, "n": 123779},
              {"id": "h19-5", "live": 0.1922, "fold_dense": 0.21341, "n": 121131}):
        ff = floor_dti(a["n"], G_fold, N_fold, r_mean)["dti"]
        lf_lo = floor_dti(a["n"], 84294.0, N_live, r_mean)["dti"]
        lf_hi = floor_dti(a["n"], 165151.0, N_live, r_mean)["dti"]
        lf = floor_dti(a["n"], 116106.0, N_live, r_mean)["dti"]
        transfer.append({
            "id": a["id"], "n": a["n"],
            "fold_score": a["fold_dense"], "fold_floor": round(ff, 5),
            "fold_lift": round(a["fold_dense"] - ff, 5),
            "fold_lift_x_floor": round(a["fold_dense"] / ff, 2),
            "live_score": a["live"], "live_floor_p50": round(lf, 5),
            "live_floor_band": [round(lf_lo, 4), round(lf_hi, 4)],
            "live_lift": round(a["live"] - lf, 5),
            "live_lift_x_floor": round(a["live"] / lf, 2),
        })
    out["fold_vs_live_transfer"] = transfer

    # ---------------------------------------------------------- delivered file
    n_delivered = 550_000.0
    delivered_floor = floor_dti(n_delivered, 116106.0, N_live, r_mean)
    out["delivered_rescale_candidate"] = {
        "n": n_delivered,
        "floor_p50": round(delivered_floor["dti"], 4),
        "floor_band_ci95_G": [round(floor_dti(n_delivered, 84294.0, N_live, r_mean)["dti"], 4),
                              round(floor_dti(n_delivered, 165151.0, N_live, r_mean)["dti"], 4)],
        "propagated_projection_f6777492": [0.1628, 0.1778, 0.1956],
        "verdict": ("projected score (evidence/value_vs_emit_projection or the h22 build "
                    "record) lies at or below the no-skill floor for its own budget "
                    "under the fitted |G|; do not spend a weekly slot on it"),
    }

    out["conclusions"] = [
        "The closed form reproduces this repository's own measured random arms to within "
        "a few percent on both A and B (see validation table), so it is trustworthy at least "
        "at the fold's ground-truth density.",
        "Ground-truth density is the whole story: the fold holds 21,344 gt px of 5.13M "
        "(0.42%) against a fitted live |G| of ~116k of 5.11M (2.27%). The same detector "
        "therefore sits far above the floor on the fold and near the floor live.",
        "Because the floor rises steeply with budget at live density (from ~0.18 at 121k to "
        "~0.29 at 550k) but is nearly flat at fold density, the fold's budget sweep cannot "
        "distinguish a better detector from a larger budget. The delivered 550k rescale is "
        "therefore not evidence-backed for live gain.",
        "Any future submission must be justified by margin over the floor at its own budget, "
        "not by absolute fold DTI.",
    ]

    (EVI / "h26_floor_model.json").write_text(json.dumps(out, indent=1))

    print("== validation against this repository's measured random arms ==")
    for v in validation:
        print(f"  {v['fold']}/{v['head']}: n={v['n']:>9,.0f} meas dti={v['measured_dti']:.4f} "
              f"pred={v['predicted_dti']:.4f} ({v['dti_rel_err']:+.2%})  A err "
              f"{v['A_rel_err']:+.1%}  B err(LOO) {v['B_err_loo_calibrated']:+.1%}  union_ratio={v['union_mass_ratio']}")
    print(f"union-mass ratio: mean={r_mean:.4f} range [{min(union_ratios):.3f},{max(union_ratios):.3f}]")
    print("== live no-skill floor (G p50 / 95% band / fold density for contrast) ==")
    for row in live_grid:
        print(f"  n={row['n']:>9,}  floor_p50={row['p50']:.4f}  "
              f"[{row['ci95_low']:.4f},{row['ci95_high']:.4f}]  floor_mle={row['mle']:.4f}")
    print(f"  floor peak at G_mle: n={out['floor_peak_at_G_mle']['n']:,.0f} "
          f"dti={out['floor_peak_at_G_mle']['dti']}")
    print("== historic files net of floor ==")
    for r in rows:
        print(f"  {r['id']:24s} n={r['n']:>9,.0f} lb={r['lb']:.4f} floor<=[{r['floor_ci95_band'][0]:.3f},"
              f"{r['floor_ci95_band'][1]:.3f}] lift>=[{r['lift_over_floor_low_end']:+.4f},"
              f"{r['lift_over_floor_high_end']:+.4f}] q={r['implied_precision_at_G_p50']}")
    print("== top live anchors vs floor ==")
    for t in ta:
        print(f"  {t['id']:22s} n={t['n']:>7,} lb={t['lb']:.4f} floor_p50={t['floor_p50']:.4f} "
              f"lift_band={t['lift_band']}")
    print("== proxy rank correlation ==")
    for k, v in corr.items():
        print(f"  {k}: rho(raw)={v['spearman_rho_DTI_vs_LB_negated']:+.3f} "
              f"rho(lift)={v['spearman_rho_lift_over_floor_vs_LB']:+.3f} (p={v['p_lift']:.3f})")
    print("== fold vs live transfer ==")
    for t in transfer:
        print(f"  {t['id']}: fold {t['fold_score']:.4f}/floor {t['fold_floor']:.4f} = "
              f"{t['fold_lift_x_floor']:.1f}x | live {t['live_score']:.4f}/floor "
              f"{t['live_floor_p50']:.4f} = {t['live_lift_x_floor']:.2f}x")
    print("== delivered rescale candidate ==")
    d = out["delivered_rescale_candidate"]
    print(f"  n={d['n']:,.0f} floor_p50={d['floor_p50']:.4f} band={d['floor_band_ci95_G']} "
          f"projection={d['propagated_projection_f6777492']}")
    print("wrote evidence/h26_floor_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
