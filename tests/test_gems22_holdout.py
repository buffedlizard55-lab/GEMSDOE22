"""Holdout-design invariants. These are the guards that caught two real bugs:
a geographic fold whose evaluation mask deleted its own ground truth, and
catalogue-derived features that leaked the held-out traces (fold DTI 0.977).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import holdout as ho


@pytest.fixture(scope="module")
def grid():
    """Synthetic catalogue: 40 elongated traces on a 200x200 grid."""
    rng = np.random.default_rng(0)
    cat = np.zeros((200, 200), bool)
    for _ in range(40):
        r, c = rng.integers(10, 190, 2)
        L = int(rng.integers(6, 30))
        if rng.random() < 0.5:
            cat[r, c:c + L] = True
        else:
            cat[r:r + L, c] = True
    fp = np.ones((200, 200), bool)
    fp[:5, :] = False
    return cat, fp


@pytest.mark.parametrize("gridname", ["quadrants", "blocks3x3", "checker"])
def test_geographic_folds_partition_and_do_not_leak(grid, gridname):
    cat, fp = grid
    for name, blk in ho.folds(cat.shape, gridname).items():
        em = ho.evaluation_mask(blk, fp, cat, buffer_px=3)
        tg = ho.fold_target(blk, cat, fp, buffer_px=3)
        tm = ho.fold_train_mask(blk, fp, cat, buffer_px=3)
        assert (tg & ~em).sum() == 0, f"{gridname}/{name}: GT outside the scored domain"
        assert (tg & tm).sum() == 0, f"{gridname}/{name}: GT leaked into training"
        assert em.sum() > 0, f"{gridname}/{name}: empty scored domain"
        assert tg.sum() > 0, f"{gridname}/{name}: empty ground truth"


def test_geographic_folds_cover_the_catalogue(grid):
    cat, fp = grid
    seen = np.zeros(cat.shape, bool)
    for blk in ho.folds(cat.shape, "quadrants").values():
        seen |= ho.fold_target(blk, cat, fp, 3)
    # a 3-px collar around every fold boundary necessarily drops traces that
    # straddle the boundary, so full coverage is not expected; 75% is the floor
    assert seen.sum() > 0.75 * (cat & fp).sum()


def test_trace_cluster_folds_invariants(grid):
    cat, fp = grid
    folds = ho.trace_cluster_folds(cat, n_folds=4, n_clusters=12, seed=0)
    assert len(folds) == 4
    union = np.zeros(cat.shape, bool)
    for k, g in folds.items():
        assert g.sum() > 0, f"fold {k} is empty"
        assert (g & ~cat).sum() == 0, f"fold {k} contains non-catalogue pixels"
        m = ho.trace_fold_masks(cat, fp, g, collar=1)
        assert (m["gt"] & m["train_target"]).sum() == 0, "GT overlaps the training target"
        assert (m["gt"] & ~m["scored_domain"]).sum() == 0, "GT outside the scored domain"
        assert (m["gt"] & m["train_domain"]).sum() == 0, "GT inside the training domain"
        union |= g
    assert union.sum() > 0.9 * cat.sum(), "trace folds must cover almost every trace"
    # folds must be disjoint
    tot = sum(int(g.sum()) for g in folds.values())
    assert tot == int(union.sum()), "trace folds overlap"


def test_trace_folds_preserve_map_wide_context(grid):
    """The whole point of trace-cluster folds: the training catalogue must remain
    present across the entire map, unlike a geographic quadrant fold."""
    cat, fp = grid
    folds = ho.trace_cluster_folds(cat, n_folds=4, n_clusters=12, seed=0)
    H, W = cat.shape
    for k, g in folds.items():
        tr = ho.trace_fold_masks(cat, fp, g, 1)["train_target"]
        q = [(slice(0, H // 2), slice(0, W // 2)), (slice(0, H // 2), slice(W // 2, W)),
             (slice(H // 2, H), slice(0, W // 2)), (slice(H // 2, H), slice(W // 2, W))]
        present = sum(1 for s in q if tr[s].any())
        assert present >= 3, f"fold {k}: training catalogue present in only {present}/4 quadrants"


def test_quadrant_folds_destroy_context(grid):
    """Documented failure mode (flag F-09): a geographic fold leaves NO known
    fault in the held-out region, which is why it is not the primary design."""
    cat, fp = grid
    blk = ho.folds(cat.shape, "quadrants")["NW"]
    inner = ho.evaluation_mask(blk, fp, cat, 3)
    tr = ho.fold_train_mask(blk, fp, cat, 3)
    assert (cat & inner & ~ho.fold_target(blk, cat, fp, 3)).sum() == 0
    assert (cat & tr).sum() > 0


def _repo():
    from gems22.spec import REPO
    return REPO


@pytest.mark.skipif(not (_repo() / "data/labels.tif").exists(),
                    reason="official rasters not present")
def test_real_data_folds_reproduce_recorded_evidence():
    """Regression guard on the fold partition that every number in evidence/
    *.json was computed from.  If this fails, the cached OOF maps, the holdout
    sweeps and the budget conclusion are all stale and must be regenerated."""
    import json
    import rasterio
    from gems22.spec import load_labels, REPO
    cat = load_labels()
    with rasterio.open(REPO / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    sg = rasterio.open(REPO / "assets/external/derived_sgmc_faults_100m_u8.tif").read(1) > 0
    gap = sg & ~cat & fp
    tf = ho.trace_cluster_folds(cat | gap, n_folds=4, n_clusters=48, seed=22)
    got = {k: {"A": int((v & cat).sum()), "B": int((v & gap).sum())} for k, v in tf.items()}
    p = REPO / "evidence/holdout_union.json"
    if not p.exists():
        pytest.skip("evidence not generated yet")
    exp = {k: v["gt_px"] for k, v in json.loads(p.read_text())["folds"].items()}
    assert got == exp, f"fold partition drifted:\n got={got}\n exp={exp}"
