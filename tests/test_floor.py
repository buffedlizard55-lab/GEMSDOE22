"""Validate the closed-form no-skill floor (``gems.floor``) against the real metric.

The point of these tests is that the floor claim ("a random emission of 121k
pixels scores about 0.18 on this task") is a *computed* statement that must be
reproducible from the repository's own metric implementation, not a narrative.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems import floor  # noqa: E402
from gems.metric import dti_components_exact  # noqa: E402

EVIDENCE = ROOT / "evidence"


def test_closed_form_matches_exact_metric_on_padded_grid() -> None:
    """Monte-Carlo a random emission on the real metric, mean over 6 trials."""
    rng = np.random.default_rng(20261002)
    H = W = 400
    pad = 6                      # keep all kernels inside the grid: no edge bias
    interior = H - 2 * pad
    N = interior * interior
    n_emitted, n_truth = 7_000, 3_000
    yy, xx = np.mgrid[pad:H - pad, pad:W - pad]
    yy, xx = yy.ravel(), xx.ravel()

    acc = {"A": 0.0, "B": 0.0, "dti": 0.0}
    trials = 6
    for _ in range(trials):
        truth = np.zeros((H, W), dtype=np.uint8)
        pred = np.zeros((H, W), dtype=np.float32)
        i_gt = rng.choice(N, size=n_truth, replace=False)
        i_pr = rng.choice(N, size=n_emitted, replace=False)
        truth[yy[i_gt], xx[i_gt]] = 1
        pred[yy[i_pr], xx[i_pr]] = 1.0
        c = dti_components_exact(pred, truth)
        acc["A"] += c["TP_w"] / trials
        acc["B"] += c["FP_w"] / trials
        acc["dti"] += c["dti"] / trials

    exp = floor.expected_components(n_emitted, n_truth, N, union_ratio=1.0)
    assert abs(acc["A"] / exp["A"] - 1.0) < 0.08, (acc["A"], exp["A"])
    assert abs(acc["B"] / exp["B"] - 1.0) < 0.05, (acc["B"], exp["B"])
    assert abs(acc["dti"] / exp["dti"] - 1.0) < 0.06, (acc["dti"], exp["dti"])


def test_live_floor_regression_numbers() -> None:
    """The numbers quoted in README/evidence must not silently drift."""
    N = 5_106_385
    G50 = 116_106.0
    assert floor.floor_dti(121_131, G50, N) == pytest.approx(0.1812, abs=0.002)
    assert floor.floor_dti(550_000, G50, N) == pytest.approx(0.2900, abs=0.004)
    assert floor.floor_dti(60_000, G50, N) == pytest.approx(0.1101, abs=0.003)
    lo, hi = floor.floor_band(121_131, 84_294.0, 165_151.0, N)
    assert (lo, hi) == (pytest.approx(0.1683, abs=0.002), pytest.approx(0.1928, abs=0.002))
    n_peak, val = floor.peak_floor(107_000.0, N)
    assert 400_000 < n_peak < 560_000
    assert val == pytest.approx(0.279, abs=0.01)


def test_floor_rises_then_falls_and_is_never_zero() -> None:
    """A random emission is worth more at intermediate budgets than at tiny ones."""
    N = 5_106_385
    G = 116_106.0
    vals = [floor.floor_dti(n, G, N) for n in (10_000, 121_131, 480_000, 2_000_000, N)]
    assert vals[0] < vals[1] < vals[2]
    assert vals[4] < vals[2]
    assert all(v > 0.01 for v in vals)  # even a tiny random emission scores above zero


def test_implied_a_bounds_bracket_clean_and_dirty_emissions() -> None:
    lo, hi = floor.implied_a_bounds(0.1922, 121_131, 116_106.0)
    assert lo == pytest.approx(18_566, abs=40)
    assert hi == pytest.approx(23_409, abs=40)
    assert lo < hi


@pytest.mark.skipif(not (EVIDENCE / "h26_floor_model.json").exists(),
                    reason="floor evidence not built")
def test_evidence_matches_recomputation() -> None:
    ev = json.loads((EVIDENCE / "h26_floor_model.json").read_text())
    row = ev["top_live_anchors_vs_floor"][0]
    assert row["id"].startswith("h19-5")
    lo, hi = floor.floor_band(row["n"], 84_294.0, 165_151.0, ev["live_domain_px"])
    assert row["lift_band"][0] == pytest.approx(row["lb"] - hi, abs=1e-3)
    assert row["lift_band"][1] == pytest.approx(row["lb"] - lo, abs=1e-3)
    # the calibration constant really is the mean of the measured arms
    assert floor.UNION_RATIO_MEASURED == pytest.approx(
        ev["union_mass_ratio_calibrated"]["mean"], abs=5e-4)
    for v in ev["validation_against_measured_random_arms"]:
        assert abs(v["A_rel_err"]) < 0.05
