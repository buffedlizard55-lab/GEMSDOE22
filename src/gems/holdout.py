"""Spatially blocked holdout used to gate hypotheses *before* any weekly submission slot is spent."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label as ndi_label

from .metric import dti_score_fast, ridge_nms

BUDGET_FRAC = 0.025
FOLD_NAMES = ["NW", "NE_LidarGapHeavy", "SW", "SE"]


def make_quadrant_folds(footprint: np.ndarray) -> tuple[np.ndarray, list[str]]:
    yy, xx = np.nonzero(footprint)
    y_med, x_med = int(np.median(yy)), int(np.median(xx))
    H, W = footprint.shape
    gy, gx = np.ogrid[:H, :W]
    fold = np.full((H, W), -1, dtype=np.int8)
    fold[(gy < y_med) & (gx < x_med) & footprint] = 0
    fold[(gy < y_med) & (gx >= x_med) & footprint] = 1
    fold[(gy >= y_med) & (gx < x_med) & footprint] = 2
    fold[(gy >= y_med) & (gx >= x_med) & footprint] = 3
    return fold, list(FOLD_NAMES)


def thin_components(truth_2d: np.ndarray, keep_frac: float, seed: int) -> np.ndarray:
    comp, n = ndi_label(truth_2d, structure=np.ones((3, 3), dtype=int))
    rng = np.random.default_rng(seed)
    keep = rng.choice(np.arange(1, n + 1), size=max(1, int(round(keep_frac * n))), replace=False)
    return np.isin(comp, keep)


class Holdout:
    def __init__(
        self,
        footprint: np.ndarray,
        labels: np.ndarray,
        budget: float = BUDGET_FRAC,
        sparse_seed_offset: int = 0,
    ):
        self.footprint = footprint
        self.labels = labels
        self.budget = budget
        self.fp_idx = np.flatnonzero(footprint.ravel())
        self.fold_2d, self.fold_names = make_quadrant_folds(footprint)
        self.quads = []
        for f_id, name in enumerate(self.fold_names):
            f_mask = (self.fold_2d == f_id) & footprint
            r0, r1 = np.nonzero(f_mask.any(axis=1))[0][[0, -1]]
            c0, c1 = np.nonzero(f_mask.any(axis=0))[0][[0, -1]]
            sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
            t_dense = labels & f_mask
            t_sparse = thin_components(t_dense, keep_frac=0.20, seed=4242 + f_id + sparse_seed_offset)
            known = t_dense & ~t_sparse
            self.quads.append((f_id, name, f_mask, sl, t_dense[sl], t_sparse[sl], known[sl], f_mask[sl]))

    def to_2d(self, p_fp: np.ndarray) -> np.ndarray:
        a = np.zeros(self.footprint.shape, dtype=np.float32)
        a.ravel()[self.fp_idx] = p_fp
        return a

    def emit(self, p_fp: np.ndarray, *, ridge: bool = True, budget_per_quad: list[int] | None = None) -> np.ndarray:
        """Per-quadrant top-budget binary mask over the full grid (ridge-thinned unless ``ridge=False``)."""
        score = self.to_2d(p_fp)
        if ridge:
            r = ridge_nms(score, self.footprint, sigma=1.0)
            boosted = np.where(r, score + 1.0, score * 0.5)
        else:
            boosted = score
        out = np.zeros(self.footprint.shape, dtype=bool)
        flat = boosted.ravel()
        for f_id, _n, f_mask, _sl, _td, _ts, _kc, fm in self.quads:
            k = int(budget_per_quad[f_id]) if budget_per_quad is not None else int(round(self.budget * fm.sum()))
            idx = np.flatnonzero(f_mask.ravel())
            top = idx[np.argpartition(flat[idx], -k)[-k:]]
            out.ravel()[top] = True
        return out

    def score_mask(self, pred: np.ndarray) -> dict[str, Any]:
        dense, sparse, detail = [], [], {}
        for _f, name, _m, sl, td, ts, kc, fm in self.quads:
            rd = dti_score_fast(pred[sl], td, valid_mask=fm)
            rs = dti_score_fast(pred[sl], ts, valid_mask=fm, catalogue_mask=kc)
            dense.append(rd["dti"])
            sparse.append(rs["dti"])
            detail[name] = {
                "dense_dti": round(rd["dti"], 5),
                "sparse_dti": round(rs["dti"], 5),
                "dense_coverage": round(rd["coverage"], 4),
                "sparse_coverage": round(rs["coverage"], 4),
                "emitted_px": int(pred[sl].sum()),
            }
        return {
            "mean_dense_dti": round(float(np.mean(dense)), 5),
            "mean_sparse_dti": round(float(np.mean(sparse)), 5),
            "fold_dense": [round(x, 5) for x in dense],
            "fold_sparse": [round(x, 5) for x in sparse],
            "folds": detail,
        }

    def evaluate(self, p_fp: np.ndarray, *, ridge: bool = True) -> dict[str, Any]:
        return self.score_mask(self.emit(p_fp, ridge=ridge))

    def emit_quadrant(self, p_fp: np.ndarray, f_id: int, *, ridge: bool = True, k_override: int | None = None) -> np.ndarray:
        score = self.to_2d(p_fp)
        boosted = np.where(ridge_nms(score, self.footprint, sigma=1.0), score + 1.0, score * 0.5) if ridge else score
        _f, _n, f_mask, _sl, _td, _ts, _kc, fm = self.quads[f_id]
        k = int(k_override) if k_override is not None else int(round(self.budget * fm.sum()))
        idx = np.flatnonzero(f_mask.ravel())
        top = idx[np.argpartition(boosted.ravel()[idx], -k)[-k:]]
        out = np.zeros(self.footprint.shape, dtype=bool)
        out.ravel()[top] = True
        return out

    def score_quadrant(self, pred: np.ndarray, f_id: int, mode: str, mask_predictions: bool = False) -> float:
        _f, _n, _m, sl, td, ts, kc, fm = self.quads[f_id]
        if mode == "dense":
            return dti_score_fast(pred[sl], td, valid_mask=fm)["dti"]
        return dti_score_fast(pred[sl], ts, valid_mask=fm, catalogue_mask=kc, mask_predictions=mask_predictions)["dti"]

    def known_for(self, f_id: int, mode: str) -> np.ndarray:
        _f, _n, f_mask, sl, _td, _ts, kc, _fm = self.quads[f_id]
        known = self.labels & ~f_mask
        if mode == "sparse":
            known = known.copy()
            known[sl] |= kc
        return known

    def evaluate_by_quadrant(self, surface_fn) -> dict[str, Any]:
        dense, sparse = [], []
        for f_id in range(len(self.quads)):
            dense.append(self.score_quadrant(self.emit_quadrant(surface_fn(f_id, "dense"), f_id), f_id, "dense"))
            sparse.append(self.score_quadrant(self.emit_quadrant(surface_fn(f_id, "sparse"), f_id), f_id, "sparse"))
        return {
            "mean_dense_dti": round(float(np.mean(dense)), 5),
            "mean_sparse_dti": round(float(np.mean(sparse)), 5),
            "fold_dense": [round(x, 5) for x in dense],
            "fold_sparse": [round(x, 5) for x in sparse],
        }


def gate(candidate: dict[str, Any], baseline: dict[str, Any], *, max_fold_loss: float = 0.01) -> dict[str, Any]:
    """Pre-registered promotion rule.

    PASS requires: higher mean dense AND mean sparse DTI than the baseline; higher sparse DTI in >= 3 of 4 folds;
    and no fold losing more than ``max_fold_loss`` DTI (dense or sparse) relative to the baseline.
    """
    cd, cs = candidate["fold_dense"], candidate["fold_sparse"]
    bd, bs = baseline["fold_dense"], baseline["fold_sparse"]
    sparse_wins = int(sum(c > b for c, b in zip(cs, bs)))
    dense_wins = int(sum(c > b for c, b in zip(cd, bd)))
    worst_dense = round(min(c - b for c, b in zip(cd, bd)), 5)
    worst_sparse = round(min(c - b for c, b in zip(cs, bs)), 5)
    rules = {
        "mean_dense_improves": candidate["mean_dense_dti"] > baseline["mean_dense_dti"],
        "mean_sparse_improves": candidate["mean_sparse_dti"] > baseline["mean_sparse_dti"],
        "sparse_fold_wins_ge_3_of_4": sparse_wins >= 3,
        "no_fold_loses_more_than_0_01": worst_dense >= -max_fold_loss and worst_sparse >= -max_fold_loss,
    }
    return {
        "passed": all(rules.values()),
        "rules": rules,
        "sparse_fold_wins": sparse_wins,
        "dense_fold_wins": dense_wins,
        "worst_fold_delta_dense": worst_dense,
        "worst_fold_delta_sparse": worst_sparse,
        "delta_mean_dense": round(candidate["mean_dense_dti"] - baseline["mean_dense_dti"], 5),
        "delta_mean_sparse": round(candidate["mean_sparse_dti"] - baseline["mean_sparse_dti"], 5),
    }
