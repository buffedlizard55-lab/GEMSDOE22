"""Holdout evaluator: pure-logic tests always run; reproduction of the committed H16-1 numbers needs the cached OOF surface."""
from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio

from gems.holdout import Holdout, gate, make_quadrant_folds, thin_components
from gems.paths import DATA_DIR, EVIDENCE_DIR


def test_quadrant_folds_partition_the_footprint(footprint):
    fold, names = make_quadrant_folds(footprint)
    assert names == ["NW", "NE_LidarGapHeavy", "SW", "SE"]
    assert ((fold >= 0) == footprint).all()
    sizes = [int((fold == k).sum()) for k in range(4)]
    assert sum(sizes) == int(footprint.sum()) and min(sizes) > 0.15 * footprint.sum()


def test_thin_components_is_deterministic_and_keeps_whole_components():
    t = np.zeros((40, 40), bool); t[5, 5:15] = True; t[20, 10:30] = True; t[30, 2:8] = True; t[35, 20:25] = True; t[10, 30:38] = True
    a, b = thin_components(t, 0.4, seed=1), thin_components(t, 0.4, seed=1)
    assert np.array_equal(a, b) and 0 < a.sum() < t.sum()


def _summ(d, s):
    return {"mean_dense_dti": float(np.mean(d)), "mean_sparse_dti": float(np.mean(s)), "fold_dense": d, "fold_sparse": s}


def test_gate_rules_behave_as_documented():
    base = _summ([0.2, 0.2, 0.2, 0.2], [0.08, 0.08, 0.08, 0.08])
    assert gate(_summ([0.21] * 4, [0.09] * 4), base)["passed"] is True
    assert gate(_summ([0.21] * 4, [0.07] * 4), base)["passed"] is False                               # sparse must improve
    assert gate(_summ([0.19] * 4, [0.09] * 4), base)["passed"] is False                               # dense must improve
    assert gate(_summ([0.30, 0.30, 0.30, 0.15], [0.09, 0.09, 0.09, 0.07]), base)["passed"] is False    # one fold loses > 0.01 dense
    assert gate(_summ([0.21] * 4, [0.09, 0.09, 0.07, 0.07]), base)["passed"] is False                 # only 2/4 sparse folds win


@pytest.mark.data
def test_evaluator_reproduces_the_committed_h16_1_numbers(real_data):
    cache = DATA_DIR / "cache" / "oof_probs_h16_1.npz"
    if not cache.exists():
        pytest.skip("OOF cache not built (run scripts/run_spatial_holdout_and_build.py)")
    with rasterio.open(real_data / "sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(real_data / "labels.tif") as s:
        lab = (s.read(1) > 0) & fp
    r = Holdout(fp, lab).evaluate(np.load(cache)["h16_1"])
    ref = json.loads((EVIDENCE_DIR / "spatial_holdout_results.json").read_text())["summary"]["H16_1_SeamFree_MultiScale_Synthesis"]
    assert r["fold_dense"] == ref["fold_dense"] and r["fold_sparse"] == ref["fold_sparse"]
