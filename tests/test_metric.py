"""Metric primitives: official worked example, catalogue masking, marginal-inclusion bar, ridge thinning."""
from __future__ import annotations

import numpy as np

from gems.metric import dti_components_exact, dti_score_fast, marginal_inclusion_threshold, ridge_nms, verify_organizer_worked_example


def test_organizer_worked_example_matches_published_numbers():
    r = verify_organizer_worked_example()          # DrivenData page 967: TP_w 3.00, FP_w 1.89, FN_w 2.00 -> 0.60
    assert (r["TP_w"], r["FP_w"], r["FN_w"]) == (3.0, 1.89, 2.0)
    assert r["matches_0_60"] is True


def test_known_fault_pixels_are_neutral_when_masked():
    """Staff ruling (forum 11516): predictions on known-fault pixels cannot change the score."""
    valid = np.ones((64, 64), bool)
    cat = np.zeros_like(valid); cat[10, 10:50] = True
    new = np.zeros_like(valid); new[40, 10:50] = True
    off = np.zeros_like(valid); off[41, 15:45] = True
    a = dti_score_fast(off, new, valid_mask=valid, catalogue_mask=cat)["dti"]
    b = dti_score_fast(off | cat, new, valid_mask=valid, catalogue_mask=cat)["dti"]
    assert abs(a - b) < 1e-12


def test_a_prediction_near_but_not_on_a_known_fault_is_penalised():
    """Staff ruling (forum 11516 #4): the buffer does not apply to known faults."""
    valid = np.ones((64, 64), bool)
    cat = np.zeros_like(valid); cat[10, 10:50] = True
    new = np.zeros_like(valid); new[40, 10:50] = True
    far = np.zeros_like(valid); far[41, 15:45] = True
    near_known = far.copy(); near_known[11, 15:45] = True          # 1 px beside the known fault, far from the new one
    assert dti_score_fast(near_known, new, valid_mask=valid, catalogue_mask=cat)["dti"] < dti_score_fast(far, new, valid_mask=valid, catalogue_mask=cat)["dti"]


def test_exact_and_fast_agree_for_binary_predictions():
    rng = np.random.default_rng(0)
    truth = np.zeros((40, 40), bool); truth[20, 5:35] = True
    pred = rng.random((40, 40)) < 0.05
    pred[21, 8:30] = True
    f = dti_score_fast(pred, truth)
    e = dti_components_exact(pred.astype(float), truth)
    assert abs(f["dti"] - e["dti"]) < 1e-6 and abs(f["TP_w"] - e["TP_w"]) < 1e-6


def test_marginal_inclusion_bar_values_used_on_the_site():
    assert round(marginal_inclusion_threshold(0.1563), 4) == 0.0323
    assert round(marginal_inclusion_threshold(0.3168), 4) == 0.0676


def test_recall_dominated_form_of_the_score():
    """DTI = T / (0.2 T + 0.2 F + 0.8 G) for alpha=0.2, beta=0.8 (derived on the knowledge page)."""
    T, F, G = 18_000.0, 160_000.0, 100_000.0
    direct = T / (T + 0.2 * F + 0.8 * (G - T))
    assert abs(direct - T / (0.2 * T + 0.2 * F + 0.8 * G)) < 1e-12
    assert abs(T / (0.8 * G) - direct) > 0 and direct < T / (0.8 * G)


def test_ridge_nms_keeps_a_continuous_centre_line_and_suppresses_flanks():
    yy, xx = np.ogrid[:50, :50]
    score = np.exp(-((yy - 25.0) ** 2) / (2 * 1.5 ** 2)).astype(np.float32) * np.ones((50, 50), np.float32)
    r = ridge_nms(score, np.ones((50, 50), bool), sigma=1.0)
    assert r[25, 5:45].sum() >= 38 and r[23, 5:45].sum() == 0 and r[27, 5:45].sum() == 0


def test_minimum_recall_bound_quoted_on_the_knowledge_page():
    """Even with zero false-positive mass, score s needs TP_w/G >= 0.8 s / (1 - 0.2 s)."""
    for s, expected_pct in ((0.1563, 12.9), (0.3168, 27.1)):
        t = 0.8 * s / (1 - 0.2 * s)
        assert abs(100 * t - expected_pct) < 0.06
        assert abs(t / (0.2 * t + 0.8) - s) < 1e-12              # G = 1, F = 0 reproduces the target score
    assert abs(0.35 / (0.2 * 0.35 + 0.2 * 1.17 + 0.8) - 0.3168) < 1e-3   # the page's illustration
