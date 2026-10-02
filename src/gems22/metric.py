"""Distance-weighted Tversky index (DTI) -- the official GEMS Prize metric.

VERBATIM SOURCE OF TRUTH (fetched and read 2026-10-01):
  https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
  section "Performance metric" -> "Mathematical representation"

    TI(a,b) = sum_x p(x)g(x)
              / ( sum_x p(x)g(x) + a*sum_x p(x)(1-g(x)) + b*sum_x (1-p(x))g(x) )

    k(d) = (1 - d/R)_+ = max(1 - d/R, 0),   R = 300 metres

    TP_w = sum_{g in G} max_{x : d(x,g) <= R} p(x) * k(d(x,g))
    FP_w = sum_{x : p(x) > 0} p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w = sum_{g in G} [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))]

    DTI(a,b) = TP_w / (TP_w + a*FP_w + b*FN_w + eps)

  "For this competition, we set alpha = 0.2 and beta = 0.8".

SCORING-DOMAIN MASKING (verified from DrivenData staff, 2026-10-01):
  https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious
  -faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/2
  chrisk-dd (DrivenData Staff), Sep 16:
    "1. Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded
        from evaluation, so they do not count towards penalty terms.
     2. Re-evaluation will also mask/exclude the existing USGS/INGENIOUS faults."
  => known-catalogue pixels are dropped from BOTH the prediction array and the
     ground-truth array before the metric is computed.  This module exposes that
     explicitly through `evaluate_mask` so the same code path is used for the live
     leaderboard proxy and for offline holdout folds.

------------------------------------------------------------------------------
TWO EXACT ALGEBRAIC IDENTITIES USED BY THIS REPO (both derived, both unit-tested)
------------------------------------------------------------------------------
(1) FN_w = |G| - TP_w exactly, because for every g in G the same max term
    m_g = max_{x: d(x,g)<=R} p(x) k(d(x,g)) appears in TP_w as m_g and in FN_w as
    1 - m_g.  Hence

        DTI = A / (0.2*A + 0.2*B + 0.8*|G|)          with A = TP_w, B = FP_w

    i.e. the denominator carries a FIXED floor of 0.8*|G| that no submission can
    remove.  This is why coverage, not precision, dominates the score.

(2) For a FIXED support S and a uniform confidence scale q (p = q on S, 0 off S),

        DTI(q) = q*A1 / (0.2*q*(A1 + B1) + 0.8*|G|)

    with A1 = sum_g kappa_g and B1 = sum_{x in S} (1 - w(x)) both independent of
    q.  d DTI / dq > 0 for all q in (0, 1] whenever A1 > 0, so DTI is strictly
    increasing in q:  **an optimal submission is binary (p in {0, 1})**.
    Intermediate probabilities are never better than saturating the support.

(3) MARGINAL VALUE OF A PIXEL.  Adding a pixel-set with (dA, dB) raises DTI iff

        dA > tau * (dA + dB),      tau = 0.2*DTI / (1 - 0.2*DTI)

    For an isolated candidate pixel with posterior probability pi of being the
    unique best cover of some ground-truth pixel (kappa = 1, else w = 0),
    dA = pi and dB = 1 - pi, so the break-even posterior is

        pi* = tau / (1 + tau)

    At DTI = 0.19 (our 2026-09-30 live anchor) tau = 0.0403 and pi* = 0.0387:
    ANY pixel with a >3.9% chance of sitting on an unmapped fault is worth
    emitting.  `break_even_posterior()` returns this and `optimal_support()`
    applies it.  This is the quantitative replacement for the ad-hoc "top 2.5%"
    budgets used by every prior GEMSDOE session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

ALPHA = 0.2          # false-positive weight  (page 967)
BETA = 0.8           # false-negative weight  (page 967)
RADIUS_M = 300.0     # triangular-kernel support, metres (page 967)
PIXEL_M = 100.0      # official grid resolution (page 967)
RADIUS_PX = RADIUS_M / PIXEL_M   # = 3.0 pixels exactly


def kernel_offsets(radius_px: float = RADIUS_PX):
    """Integer offsets with d <= radius_px and their triangular-kernel weights."""
    r = int(np.ceil(radius_px))
    rows = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = float(np.hypot(dy, dx))
            if d <= radius_px + 1e-12:
                rows.append((dy, dx, d, max(1.0 - d / radius_px, 0.0)))
    rows.sort(key=lambda t: (t[2], t[0], t[1]))
    a = np.asarray(rows, dtype=np.float64)
    return a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2], a[:, 3]


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate `arr` by (dy, dx); vacated border filled with zeros (no wrap)."""
    out = np.zeros_like(arr)
    h, w = arr.shape
    ys, yd = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
    xs, xd = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
    out[yd, xd] = arr[ys, xs]
    return out


def best_weighted_prediction(pred: np.ndarray, radius_px: float = RADIUS_PX) -> np.ndarray:
    """M(z) = max over x within R of p(x)*k(d(x,z)) -- exact max-dilation."""
    pred = np.asarray(pred, dtype=np.float64)
    dy, dx, _d, k = kernel_offsets(radius_px)
    out = np.zeros_like(pred)
    for a, b, kk in zip(dy, dx, k):
        if kk <= 0.0:
            continue
        np.maximum(out, _shift(pred, int(a), int(b)) * kk, out=out)
    return out


def kernel_weight_to_gt(target: np.ndarray, radius_px: float = RADIUS_PX) -> np.ndarray:
    """w(x) = max_g k(d(x,g)) = k(D_g(x)) since k is strictly decreasing in d."""
    g = np.asarray(target).astype(bool)
    if not g.any():
        return np.zeros(g.shape, dtype=np.float64)
    d = ndimage.distance_transform_edt(~g, sampling=1.0)
    return np.maximum(1.0 - d / radius_px, 0.0)


@dataclass(frozen=True)
class Components:
    tp_w: float
    fp_w: float
    fn_w: float
    dti: float
    n_pred_pos: int
    n_gt: int
    alpha: float
    beta: float

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def identity_fn(self) -> float:
        """Residual of identity (1): FN_w == |G| - TP_w. Must be ~0."""
        return abs(self.fn_w - (self.n_gt - self.tp_w))

    @property
    def closed_form(self) -> float:
        """DTI recomputed from identity (1): A / (0.2A + 0.2B + 0.8|G|)."""
        den = (1 - BETA) * self.tp_w + ALPHA * self.fp_w + BETA * self.n_gt
        return float(self.tp_w / den) if den > 0 else 0.0


def components(pred, target, alpha=ALPHA, beta=BETA, radius_px=RADIUS_PX,
               eps=0.0, evaluate_mask=None) -> Components:
    """Exact DTI components.

    evaluate_mask : optional bool array, True on pixels that PARTICIPATE in
        scoring.  Pixels where it is False are dropped from pred (set to 0) and
        from target (set to 0) before anything is computed -- this reproduces the
        organiser's known-fault masking confirmed in forum topic 11516.
    """
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(target).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {g.shape}")
    p = np.nan_to_num(np.clip(p, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    if evaluate_mask is not None:
        m = np.asarray(evaluate_mask).astype(bool)
        p = np.where(m, p, 0.0)
        g = g & m
    n_gt = int(g.sum())
    pos = p > 0.0
    n_pred_pos = int(pos.sum())

    w = kernel_weight_to_gt(g, radius_px)
    fp_w = float((p * (1.0 - w))[pos].sum()) if n_pred_pos else 0.0
    if n_gt == 0:
        tp_w = fn_w = 0.0
    else:
        mg = best_weighted_prediction(p, radius_px)[g]
        tp_w = float(mg.sum())
        fn_w = float((1.0 - mg).sum())
    den = tp_w + alpha * fp_w + beta * fn_w + eps
    dti = float(tp_w / den) if den > 0.0 else 0.0
    return Components(tp_w, fp_w, fn_w, dti, n_pred_pos, n_gt, alpha, beta)


def distance_weighted_tversky(pred, target, **kw) -> float:
    return components(pred, target, **kw).dti


# ---------------------------------------------------------------------------
# Decision theory: identities (2) and (3)
# ---------------------------------------------------------------------------
def tau(dti: float, alpha: float = ALPHA) -> float:
    """Marginal trade-off coefficient tau = alpha*DTI / (1 - alpha*DTI)."""
    return alpha * dti / (1.0 - alpha * dti)


def break_even_posterior(dti: float, alpha: float = ALPHA) -> float:
    """Posterior probability at which emitting one more pixel is DTI-neutral."""
    t = tau(dti, alpha)
    return t / (1.0 + t)


def confidence_scale_monotone(support: np.ndarray, target: np.ndarray,
                              qs=(0.05, 0.2, 0.5, 0.8, 1.0), **kw):
    """Empirical check of identity (2): DTI must increase with the uniform scale."""
    out = []
    for q in qs:
        out.append((float(q), distance_weighted_tversky(support * q, target, **kw)))
    return out


def optimal_support(prob: np.ndarray, dti_at_optimum: float, eligible: np.ndarray,
                    alpha: float = ALPHA) -> np.ndarray:
    """Emit every eligible pixel whose posterior exceeds break_even_posterior().

    `dti_at_optimum` is the operating DTI; because tau depends on DTI the map is
    a fixed point.  `solve_fixed_point()` iterates it to convergence.
    """
    pi_star = break_even_posterior(dti_at_optimum, alpha)
    return eligible & (np.asarray(prob, dtype=np.float64) >= pi_star)


def solve_fixed_point(prob: np.ndarray, target: np.ndarray, eligible: np.ndarray,
                      dti0: float = 0.2, alpha: float = ALPHA, beta: float = BETA,
                      iters: int = 12, **kw) -> dict:
    """Iterate  support(DTI) -> DTI(support)  to a self-consistent operating point.

    Returns the converged DTI, threshold, support size and the trajectory.
    """
    dti = float(dti0)
    traj = []
    sup = np.zeros(prob.shape, dtype=bool)
    for _ in range(iters):
        sup = optimal_support(prob, dti, eligible, alpha)
        c = components(sup.astype(np.float64), target, alpha, beta, evaluate_mask=eligible, **kw)
        new = c.dti if c.n_gt else 0.0
        traj.append({"dti": dti, "pi_star": break_even_posterior(dti, alpha),
                     "n_support": int(sup.sum()), "dti_realised": float(new)})
        if abs(new - dti) < 1e-9:
            dti = float(new)
            break
        dti = float(new)
    return {"dti_fixed_point": dti, "pi_star": break_even_posterior(dti, alpha),
            "n_support": int(sup.sum()), "trajectory": traj, "support": sup}


def analytic_everywhere_score(coverage: float, alpha: float = ALPHA) -> float:
    """DTI of "p = 1 on every scored pixel", to first order: c / (c + a(1-c))."""
    return coverage / (coverage + alpha * (1.0 - coverage))


# ---------------------------------------------------------------------------
# Exact DTI-versus-budget curve in O(|G|*29 + N log N), evaluated once
# ---------------------------------------------------------------------------
def rank_field(score: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer rank (0 = highest score) of every eligible pixel, -1 elsewhere.

    Also returns `order`, the flat indices of eligible pixels sorted by
    descending score.  Ties are broken deterministically by flat index.
    """
    elig_idx = np.flatnonzero(eligible.ravel())
    v = score.ravel()[elig_idx].astype(np.float64)
    o = elig_idx[np.argsort(-v, kind="stable")]
    rank = np.full(score.size, -1, dtype=np.int64)
    rank[o] = np.arange(o.size, dtype=np.int64)
    return rank.reshape(score.shape), o


def budget_curve(score: np.ndarray, target: np.ndarray, eligible: np.ndarray,
                 budgets: list[int] | None = None,
                 alpha: float = ALPHA, beta: float = BETA,
                 radius_px: float = RADIUS_PX) -> list[dict]:
    """EXACT DTI for every top-`n` support of `score`, computed once.

    Equivalent to calling `components()` at each budget, but avoids recomputing
    the distance transform and the max-dilation per budget:

      * B(n) = sum over the top-n pixels of (1 - w(x)), where
        w(x) = max_g k(d(x,g)) depends only on the target -> a cumulative sum
        over the rank order.
      * A(n) = sum_g kappa_g(n) with kappa_g(n) = max{ k(d(x,g)) : rank(x) < n }.
        For each ground-truth pixel g only the <= 29 offsets inside the kernel
        can matter, so kappa_g is a step function of n with at most 29 jumps.
        Accumulating the jumps into a difference array and taking a cumulative
        sum yields A(n) for every n simultaneously.
      * DTI(n) = A / ((1-beta)*A + alpha*B + beta*|G|)   [identity (1)]

    `budgets=None` returns the full curve at every attainable n is too large, so
    by default a geometric grid is used.
    """
    g = np.asarray(target).astype(bool) & eligible
    n_gt = int(g.sum())
    rank, order = rank_field(score, eligible)
    n_elig = int(order.size)
    if budgets is None:
        budgets = sorted({int(x) for x in np.unique(np.concatenate([
            np.geomspace(1000, max(n_elig, 2000), 40),
            np.linspace(1000, max(n_elig, 2000), 20)]))})
        budgets = [b for b in budgets if 1 <= b <= n_elig]

    w = kernel_weight_to_gt(g, radius_px)
    one_minus_w = (1.0 - w).ravel()[order]
    cumB = np.concatenate([[0.0], np.cumsum(one_minus_w)])

    # A(n): difference array over ranks
    dy, dx, _d, kk = kernel_offsets(radius_px)
    keep = kk > 0.0
    dy, dx, kk = dy[keep], dx[keep], kk[keep]
    H, W = rank.shape
    gr, gc = np.nonzero(g)
    delta = np.zeros(n_elig + 2, dtype=np.float64)
    for r0, c0 in zip(gr, gc):
        best = 0.0
        prev_r = -1
        # walk candidate offsets; kappa_g(n) = max{k : rank < n}
        cand = []
        for a, b, k in zip(dy, dx, kk):
            rr, cc = r0 + a, c0 + b
            if 0 <= rr < H and 0 <= cc < W:
                rk = rank[rr, cc]
                if rk >= 0:
                    cand.append((int(rk), float(k)))
        if not cand:
            continue
        cand.sort()
        for rk, k in cand:
            if k > best:
                # kappa jumps from `best` to `k` once n exceeds rk
                delta[rk + 1] += (k - best)
                best = k
    cumA = np.cumsum(delta)[:n_elig + 1]

    out = []
    prev = None
    for n in budgets:
        n = int(n)
        if n < 1 or n > n_elig:
            continue
        A = float(cumA[n]); B = float(cumB[n])
        den = (1.0 - beta) * A + alpha * B + beta * n_gt
        dti = float(A / den) if den > 0 else 0.0
        marg = ((A - prev[0]) / (n - prev[1])) if prev else float("nan")
        t = alpha * dti / (1.0 - alpha * dti)
        out.append({"n": n, "A": round(A, 3), "B": round(B, 3),
                    "avg_eff": round(A / n, 6),
                    "marg_eff": (None if marg != marg else round(marg, 6)),
                    "tau_at_dti": round(t, 6),
                    "marginal_rule_says_add": (None if marg != marg else bool(marg > t)),
                    "dti": round(dti, 7)})
        prev = (A, n)
    return out
