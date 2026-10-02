"""Uniqueness gate and pair metrics on synthetic rasters."""
from __future__ import annotations

import numpy as np

from gems import forensics as F


def _grid(footprint):
    lab = np.zeros(footprint.shape, np.int8)
    lab[1500:1503, 800:1600] = 1
    return F.Grid.from_footprint(footprint, lab)


def test_git_blob_sha1_matches_git_hash_object():
    assert F.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"          # git hash-object of "hello\n"


def test_duplicate_near_duplicate_and_distinct_verdicts(footprint):
    g = _grid(footprint)
    rng = np.random.default_rng(5)
    base = np.where(footprint, (rng.random(footprint.shape) < 0.03), False).astype(np.float32)
    base[~footprint] = np.nan
    hist = [{"id": "h", "array": base, "lb_score": 0.15}]
    assert F.gate_candidate(base.copy(), g, hist)["verdict"] == "DUPLICATE"
    twin = np.nan_to_num(base, nan=0.0)                                   # zero-outside twin: same scored pixels
    assert F.gate_candidate(twin, g, hist)["verdict"] == "DUPLICATE"
    on_cat = base.copy(); on_cat[g.catalogue] = 1.0                        # differs only on known-fault pixels (8GEMSDOE case)
    assert F.gate_candidate(on_cat, g, hist)["verdict"] == "DUPLICATE"
    near = base.copy(); idx = np.flatnonzero(((base == 1) & g.scored).ravel())
    near.ravel()[idx[: len(idx) // 10]] = 0.0                              # drop 10 % of pixels -> Jaccard ~ 0.9
    assert F.gate_candidate(near, g, hist)["verdict"] == "NEAR_DUPLICATE"
    other = np.where(footprint, (np.random.default_rng(99).random(footprint.shape) < 0.03), False).astype(np.float32)
    assert F.gate_candidate(other, g, hist)["verdict"] == "DISTINCT"


def test_pair_metrics_are_symmetric_in_jaccard(footprint):
    g = _grid(footprint)
    a = np.where(footprint, (np.random.default_rng(1).random(footprint.shape) < 0.05), False).astype(np.float32)
    b = np.where(footprint, (np.random.default_rng(2).random(footprint.shape) < 0.05), False).astype(np.float32)
    assert F.pair_metrics(a, b, g)["jaccard_positive"] == F.pair_metrics(b, a, g)["jaccard_positive"]
