"""Tests for the official DTI metric implementation.

Every assertion here is traceable to the published formula at
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
(section "Performance metric" -> "Mathematical representation").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22.metric import (ALPHA, BETA, RADIUS_PX, break_even_posterior,
                           components, confidence_scale_monotone, kernel_offsets,
                           optimal_support, solve_fixed_point, tau)


# ---------------------------------------------------------------------------
# An INDEPENDENT transcription of the published formula: no scipy, no shared
# helpers, triple loop over pixels.  Slow, but it shares no code with the fast
# implementation, so agreement is real evidence.
# ---------------------------------------------------------------------------
def brute_force(pred, gt, alpha=ALPHA, beta=BETA, R=RADIUS_PX):
    pred = np.asarray(pred, float)
    gt = np.asarray(gt).astype(bool)
    G = list(zip(*np.where(gt)))
    P = list(zip(*np.where(pred > 0)))

    def k(d):
        return max(1.0 - d / R, 0.0)

    tp = 0.0
    for gy, gx in G:
        best = 0.0
        for py, px in P:
            d = float(np.hypot(gy - py, gx - px))
            if d <= R + 1e-12:
                best = max(best, pred[py, px] * k(d))
        tp += best
    fn = len(G) - tp
    fp = 0.0
    for py, px in P:
        bk = 0.0
        for gy, gx in G:
            d = float(np.hypot(gy - py, gx - px))
            if d <= R + 1e-12:
                bk = max(bk, k(d))
        fp += pred[py, px] * (1.0 - bk)
    den = tp + alpha * fp + beta * fn
    return tp, fp, fn, (tp / den if den > 0 else 0.0)


def test_official_constants():
    assert ALPHA == 0.2 and BETA == 0.8 and RADIUS_PX == 3.0


def test_kernel_offsets():
    dy, dx, d, k = kernel_offsets()
    assert (k >= 0).all() and k.max() == 1.0
    assert (d <= RADIUS_PX + 1e-12).all()
    # offsets with d <= 3 on an integer grid: 29 of them
    assert len(dy) == 29
    by = {round(float(a), 4): round(float(b), 4) for a, b in zip(d, k)}
    # keys are rounded to 4 dp so lookups are exact; compare with matching tol
    assert by[0.0] == pytest.approx(1.0)
    assert by[1.0] == pytest.approx(2 / 3, abs=1e-4)
    assert by[2.0] == pytest.approx(1 / 3, abs=1e-4)
    assert by[3.0] == pytest.approx(0.0, abs=1e-9)
    ksqrt2 = 1 - np.sqrt(2) / 3
    assert any(abs(v - ksqrt2) < 1e-4 for v in by.values()), "d=sqrt(2) kernel weight missing"


@pytest.mark.parametrize("seed", range(60))
def test_matches_independent_brute_force(seed):
    rng = np.random.default_rng(seed)
    H = int(rng.integers(4, 12)); W = int(rng.integers(4, 12))
    gt = rng.random((H, W)) < rng.uniform(0.05, 0.4)
    p = np.where(rng.random((H, W)) < rng.uniform(0.0, 0.5), rng.random((H, W)), 0.0)
    c = components(p, gt)
    b = brute_force(p, gt)
    assert abs(c.tp_w - b[0]) < 1e-11
    assert abs(c.fp_w - b[1]) < 1e-11
    assert abs(c.fn_w - b[2]) < 1e-11
    assert abs(c.dti - b[3]) < 1e-11


def test_identity_fn_equals_n_gt_minus_tp():
    """FN_w = |G| - TP_w exactly, because the same max term appears in both."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        gt = rng.random((40, 40)) < 0.05
        p = np.where(rng.random((40, 40)) < 0.1, rng.random((40, 40)), 0.0)
        c = components(p, gt)
        assert c.identity_fn < 1e-11
        assert abs(c.dti - c.closed_form) < 1e-11


def test_closed_form_denominator_has_fixed_floor():
    """DTI = A / (0.2A + 0.2B + 0.8|G|) -- the 0.8|G| floor is unavoidable."""
    gt = np.zeros((30, 30), bool); gt[15, 10:20] = True
    p = np.zeros((30, 30)); p[15, 10:20] = 1.0
    c = components(p, gt)
    assert c.tp_w == 10.0 and c.fp_w == 0.0 and c.fn_w == 0.0
    assert c.dti == pytest.approx(1.0)


def test_perfect_prediction_scores_one_and_empty_scores_zero():
    gt = np.zeros((20, 20), bool); gt[5:15, 10] = True
    assert components(gt.astype(float), gt).dti == pytest.approx(1.0)
    assert components(np.zeros((20, 20)), gt).dti == 0.0


def test_one_pixel_offset_costs_exactly_a_third():
    """A 1-px (100 m) misalignment drops DTI from 1.0 to 2/3 on a long line."""
    n = 200
    gt = np.zeros((n, n), bool); gt[n // 2, 20:n - 20] = True
    p = np.zeros((n, n)); p[n // 2 + 1, 20:n - 20] = 1.0
    c = components(p, gt)
    assert c.tp_w / c.n_gt == pytest.approx(2 / 3, abs=1e-9)
    assert c.dti == pytest.approx(2 / 3, abs=1e-6)


def test_dti_monotone_in_uniform_confidence():
    """Identity (2): for a fixed support, binary p=1 dominates every soft scale."""
    rng = np.random.default_rng(3)
    gt = rng.random((70, 70)) < 0.04
    sup = (rng.random((70, 70)) < 0.15).astype(float)
    tr = confidence_scale_monotone(sup, gt, qs=(0.02, 0.1, 0.3, 0.6, 0.9, 1.0))
    vals = [d for _, d in tr]
    assert all(vals[i] <= vals[i + 1] + 1e-15 for i in range(len(vals) - 1))
    assert vals[-1] == pytest.approx(components(sup, gt).dti)


def test_evaluate_mask_excludes_masked_pixels_from_both_terms():
    """Forum 11516 post 2: known-fault pixels are excluded from evaluation."""
    gt = np.zeros((20, 20), bool); gt[10, 5:15] = True
    p = np.zeros((20, 20)); p[10, 5:15] = 1.0
    mask = np.ones((20, 20), bool); mask[10, 5:10] = False
    c = components(p, gt, evaluate_mask=mask)
    assert c.n_gt == 5 and c.tp_w == 5.0 and c.fp_w == 0.0 and c.dti == pytest.approx(1.0)


def test_masked_prediction_pixels_cost_nothing():
    """Predicting ON a masked pixel is score-neutral (staff: 'should not matter')."""
    gt = np.zeros((20, 20), bool); gt[10, 5:15] = True
    p = np.zeros((20, 20)); p[10, 5:15] = 1.0
    p[0, :] = 1.0                                   # junk on masked rows
    mask = np.ones((20, 20), bool); mask[0, :] = False
    assert components(p, gt, evaluate_mask=mask).dti == pytest.approx(1.0)


def test_break_even_posterior_is_small_at_competition_operating_point():
    """At our live anchor DTI = 0.1894 the break-even posterior is ~3.9%."""
    pi = break_even_posterior(0.1894)
    assert 0.03 < pi < 0.05
    assert break_even_posterior(0.3168) > pi          # leader can afford to be pickier
    assert tau(0.0) == 0.0


def test_optimal_support_uses_the_break_even_rule():
    rng = np.random.default_rng(11)
    prob = rng.random((50, 50))
    elig = np.ones((50, 50), bool)
    s = optimal_support(prob, 0.2, elig)
    pi = break_even_posterior(0.2)
    assert s.sum() == int((prob >= pi).sum())


def test_solve_fixed_point_converges():
    rng = np.random.default_rng(5)
    gt = rng.random((60, 60)) < 0.03
    prob = np.where(gt, rng.uniform(0.4, 1.0, gt.shape), rng.uniform(0.0, 0.3, gt.shape))
    elig = np.ones(gt.shape, bool)
    r = solve_fixed_point(prob, gt, elig, dti0=0.2)
    assert 0.0 < r["dti_fixed_point"] <= 1.0
    assert r["n_support"] > 0
    assert len(r["trajectory"]) >= 1


def test_official_worked_example_arithmetic():
    """The page's example: TP=3.00, FP=1.89, FN=2.00 -> 0.60.

    We cannot recover the exact 5-pixel geometry from the prose alone (the two
    schematic PNGs live on drivendata-public-assets.s3.amazonaws.com, which is
    unreachable from this sandbox -- see evidence/irregularities.json F-02), so
    we assert the part that IS checkable: the published components are
    arithmetically consistent with the published formula under alpha=0.2,
    beta=0.8, and they satisfy FN = |G| - TP with |G| = 5.
    """
    tp, fp, fn = 3.00, 1.89, 2.00
    dti = tp / (tp + 0.2 * fp + 0.8 * fn)
    assert round(dti, 2) == 0.60
    assert tp + fn == 5.0                              # identity (1) implies |G| = 5


def test_fit_G_mle_and_uncertainty_propagation():
    from gems22.metric import fit_G_mle, propagate_G_uncertainty

    obs = [
        {"id": "h19-5", "n": 121131, "lb": 0.1922},
        {"id": "h19-4", "n": 123779, "lb": 0.1894},
        {"id": "h16-1", "n": 123939, "lb": 0.1855},
        {"id": "h28", "n": 65236, "lb": 0.1839},
        {"id": "ens12", "n": 166519, "lb": 0.1563},
        {"id": "ens12_dup", "n": 166519, "lb": 0.1563},  # duplicate collapsed
        {"id": "g7", "n": 76859, "lb": 0.1461},
        {"id": "g12", "n": 103347, "lb": 0.1294},
        {"id": "h25", "n": 161366, "lb": 0.1280},
        {"id": "g13", "n": 319377, "lb": 0.0904},
    ]
    res = fit_G_mle(obs)
    assert res["n_unique_binary_observations"] == 9
    q = res["G_quantiles"]
    assert 40_000 <= q["p025"] < q["p16"] < q["p50"] < q["p84"] < q["p975"] <= 190_000
    u = propagate_G_uncertainty(A_hold=3395.0, B_hold=116756.5, G_hold=19897.0, G_quantiles=q, coverage_penalty=1.10)
    assert 0.10 < u["dti_live_p16"] <= u["dti_live_p50"] <= u["dti_live_p84"] < 0.30
    assert len(u["tau_p16_p50_p84"]) == 3 and len(u["pi_star_p16_p50_p84"]) == 3


def test_gems22_thermal_conduit_inversion_layers():
    from gems22.hypotheses import HYPOTHESES, build_thermal_conduit_layers

    assert len(HYPOTHESES) >= 3
    fp = np.ones((40, 40), dtype=bool)
    cat = np.zeros((40, 40), dtype=bool)
    cat[20, 10:30] = True
    layers, rep = build_thermal_conduit_layers((40, 40), fp, cat)
    assert set(layers.keys()) == {
        "thermal_wellspring_conduit",
        "thermal_probe2m_conduit",
        "thermal_paleo_vent_conduit",
        "thermal_backward_composite",
    }
    for arr in layers.values():
        assert arr.shape == (40, 40) and arr.dtype == np.float32
    assert rep["wellspring_in_footprint"] == 27092

