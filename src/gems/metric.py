"""Official Distance-Weighted Tversky Index (DTI) and directional ridge-thinning primitives.

Reference:
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/#performance-metric
  alpha = 0.2 (false-positive penalty), beta = 0.8 (false-negative penalty),
  R = 300 m = 3 pixels at 100 m resolution, kernel k(d) = max(1 - d / R, 0).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

ALPHA: float = 0.2
BETA: float = 0.8
RADIUS_PX: float = 3.0
EPS: float = 1e-7


def _kernel_offsets(radius: float = RADIUS_PX) -> list[tuple[int, int, float]]:
    r = int(np.ceil(radius))
    offs: list[tuple[int, int, float]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = float(np.hypot(dy, dx))
            k = max(1.0 - d / radius, 0.0)
            if k > 0.0:
                offs.append((dy, dx, k))
    return offs


KERNEL_OFFSETS = _kernel_offsets(RADIUS_PX)


def dti_components_exact(
    pred: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    catalogue_mask: np.ndarray | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
    mask_predictions: bool = False,
) -> dict[str, float]:
    """Compute exact distance-weighted Tversky index components for arbitrary p(x) in [0, 1]."""
    H, W = truth.shape
    p = np.nan_to_num(pred, nan=0.0).astype(np.float64)
    if valid_mask is not None:
        p = np.where(valid_mask, p, 0.0)
    if mask_predictions and catalogue_mask is not None:
        p = np.where(catalogue_mask, 0.0, p)
    g_mask = (truth > 0) if valid_mask is None else ((truth > 0) & valid_mask)
    yy, xx = np.nonzero(g_mask)
    n_truth = int(len(yy))
    if n_truth == 0:
        fp_only = float(p.sum())
        return {
            "TP_w": 0.0,
            "FP_w": fp_only,
            "FN_w": 0.0,
            "n_truth": 0,
            "n_emitted": float((p > 0).sum()),
            "dti": 0.0,
            "coverage": 0.0,
        }

    credit = np.zeros(n_truth, dtype=np.float64)
    p_flat = p.ravel()
    for dy, dx, k in KERNEL_OFFSETS:
        ny = yy + dy
        nx = xx + dx
        ok = (ny >= 0) & (nx >= 0) & (ny < H) & (nx < W)
        credit[ok] = np.maximum(credit[ok], p_flat[ny[ok] * W + nx[ok]] * k)

    tp_w = float(credit.sum())
    fn_w = float(n_truth) - tp_w

    dist_to_g = distance_transform_edt(~g_mask)
    k_to_g = np.maximum(1.0 - dist_to_g / RADIUS_PX, 0.0)
    fp_weight = 1.0 - k_to_g
    if catalogue_mask is not None:
        fp_weight = np.where(catalogue_mask, 0.0, fp_weight)
    if valid_mask is not None:
        fp_weight = np.where(valid_mask, fp_weight, 0.0)

    fp_w = float(np.sum(p * fp_weight, dtype=np.float64))
    denom = tp_w + alpha * fp_w + beta * fn_w + EPS
    dti = tp_w / denom
    return {
        "TP_w": tp_w,
        "FP_w": fp_w,
        "FN_w": fn_w,
        "n_truth": n_truth,
        "n_emitted": float((p > 0).sum()),
        "mass": float(p.sum()),
        "dti": float(dti),
        "coverage": float(tp_w / max(n_truth, 1)),
    }


def dti_score_fast(
    pred_binary: np.ndarray,
    truth_binary: np.ndarray,
    valid_mask: np.ndarray | None = None,
    catalogue_mask: np.ndarray | None = None,
    mask_predictions: bool = False,
) -> dict[str, float]:
    """Fast exact DTI for binary {0, 1} predictions using Euclidean distance transforms."""
    p = np.nan_to_num(pred_binary, nan=0.0) > 0.5
    if mask_predictions and catalogue_mask is not None:
        p = p & ~catalogue_mask
    g = truth_binary > 0
    if valid_mask is not None:
        p = p & valid_mask
        g = g & valid_mask

    n_truth = int(g.sum())
    if n_truth == 0 or not p.any():
        fp_m = p & (~catalogue_mask) if catalogue_mask is not None else p
        fp_w = float(fp_m.sum())
        fn_w = float(n_truth)
        return {
            "TP_w": 0.0,
            "FP_w": fp_w,
            "FN_w": fn_w,
            "n_truth": n_truth,
            "n_emitted": int(p.sum()),
            "dti": 0.0,
            "coverage": 0.0,
        }

    dist_to_p = distance_transform_edt(~p)
    tp_w = float(np.maximum(1.0 - dist_to_p[g] / RADIUS_PX, 0.0).sum())
    fn_w = float(n_truth) - tp_w

    dist_to_g = distance_transform_edt(~g)
    fp_mask = p & (~catalogue_mask) if catalogue_mask is not None else p
    fp_w = float((1.0 - np.maximum(1.0 - dist_to_g[fp_mask] / RADIUS_PX, 0.0)).sum())

    dti = tp_w / (tp_w + ALPHA * fp_w + BETA * fn_w + EPS)
    return {
        "TP_w": tp_w,
        "FP_w": fp_w,
        "FN_w": fn_w,
        "n_truth": n_truth,
        "n_emitted": int(p.sum()),
        "dti": float(dti),
        "coverage": float(tp_w / n_truth),
    }


def dti_score_masked(
    pred: np.ndarray,
    test_truth: np.ndarray,
    eval_mask: np.ndarray,
    known_catalogue: np.ndarray,
) -> dict[str, float]:
    return dti_components_exact(
        pred=pred,
        truth=test_truth,
        valid_mask=eval_mask,
        catalogue_mask=known_catalogue,
    )


def marginal_inclusion_threshold(current_dti: float, alpha: float = ALPHA) -> float:
    return (alpha * current_dti) / (1.0 - alpha * current_dti)


def verify_organizer_worked_example() -> dict[str, float | bool]:
    tp_w, fp_w, fn_w = 3.00, 1.89, 2.00
    dti_formula = tp_w / (tp_w + ALPHA * fp_w + BETA * fn_w)
    return {
        "TP_w": tp_w,
        "FP_w": fp_w,
        "FN_w": fn_w,
        "DTI_exact": round(dti_formula, 4),
        "DTI_rounded_2dec": round(dti_formula, 2),
        "matches_0_60": bool(round(dti_formula, 2) == 0.60),
    }


def ridge_nms(score: np.ndarray, valid: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """1-pixel Hessian across-strike non-maximum suppression (continuous along fault strike)."""
    s = np.where(valid, np.nan_to_num(score, nan=0.0, neginf=0.0), 0.0).astype(np.float32)
    ss = gaussian_filter(s, sigma) if sigma > 0 else s
    gy, gx = np.gradient(ss)
    hyy, hyx = np.gradient(gy)
    hxy, hxx = np.gradient(gx)
    hxy = 0.5 * (hxy + hyx)
    del gy, gx, hyx

    tmp = np.sqrt(((hxx - hyy) * 0.5) ** 2 + hxy ** 2)
    lam = 0.5 * (hxx + hyy) - tmp
    vx = hxy
    vy = lam - hxx
    small = (np.abs(vx) + np.abs(vy)) < 1e-12
    vx = np.where(small, 1.0, vx)
    vy = np.where(small, 0.0, vy)
    ang = np.mod(np.degrees(np.arctan2(vy, vx)), 180.0)
    del hxx, hyy, hxy, tmp, vx, vy

    q = (np.round(ang / 45.0).astype(np.int8)) % 4
    is_concave_down = lam < -1e-7
    del ang, lam
    pad = np.pad(ss, 1, mode="edge")
    h, w = ss.shape
    c = pad[1:-1, 1:-1]
    offs = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    keep = np.zeros((h, w), dtype=bool)
    for k, (dy, dx) in offs.items():
        a = pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        b = pad[1 - dy : 1 - dy + h, 1 - dx : 1 - dx + w]
        keep |= (q == k) & (c >= a) & (c >= b) & ((c > a) | (c > b))
    return keep & is_concave_down & valid & (s > 0)
