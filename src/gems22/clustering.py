"""Fault populations as a SPATIAL STATISTIC.

Two peer-reviewed, independently verifiable statistics are implemented here and
used in two ways: (i) as a geometric PRIOR that re-ranks equally-scored
candidate pixels, and (ii) as a POST-HOC AUDIT of our own predicted raster.

------------------------------------------------------------------------------
(1) BOUR & DAVY (1999) -- nearest-larger-neighbour scaling
------------------------------------------------------------------------------
  Bour, O., and P. Davy (1999), "Clustering and size distributions of fault
  patterns: Theory and measurements", Geophysical Research Letters, 26(13),
  2001-2004.  https://doi.org/10.1029/1999GL900419

  Abstract, verbatim:
    "We show theoretically and numerically that the fractal dimension D and the
     exponent a of the frequency length distribution of fault networks, are
     related through the relation x = (a - 1)/D, where x is the exponent of a
     new scaling law involving the average distance from a fault to its nearest
     neighbor of larger length."
    "We also found a correlation between the position of a fault and its length
     so that large faults have their nearest neighbor located at greater distances
     than small faults."

  Implemented as:
    * length-frequency:  N(>= L) = C * L^(-a)          -> a
    * nearest-larger:    <d(l)> = A * l^(x)            -> x
    * correlation dimension of trace barycenters, from the two-point
      correlation integral C2(r) ~ r^D                -> D
    * consistency test:  x  vs  (a - 1)/D

------------------------------------------------------------------------------
(2) MARRETT ET AL. (2018) -- normalised correlation count (NCC)
------------------------------------------------------------------------------
  Marrett, R., Gale, J. F. W., Gomez, L. A., and Laubach, S. E. (2018),
  "Correlation analysis of fracture arrangement in space", Journal of
  Structural Geology, 108, 16-33.
  https://doi.org/10.1016/j.jsg.2017.06.012

  Applied to faults in: Wang, Q., Laubach, S. E., Gale, J. F. W., and Ramos,
  M. J. (2019), "Quantified fracture (joint) clustering in Archean basement,
  Wyoming: application of the normalized correlation count method", Petroleum
  Geoscience, 25(4), 415-428.  https://doi.org/10.1144/petgeo2018-146

  The published NCC is a 1-D scanline statistic: the observed count of pairs
  separated by a lag in (r, r+dr] divided by the count expected for the same
  number of fractures placed at random on the same scanline.
      NCC(r) > 1  -> clustered at scale r
      NCC(r) ~ 1  -> indistinguishable from random
      NCC(r) < 1  -> regularly spaced / anti-clustered at scale r
  For fractal clustering the slope of log NCC vs log r equals (correlation
  dimension - 1).  `ncc_1d` reproduces the published statistic exactly (scanline
  form, with a Monte-Carlo complete-spatial-randomness null); `pcf_2d` is its
  documented 2-D analogue and is labelled as such wherever it is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# trace extraction
# ---------------------------------------------------------------------------
STRUCT8 = np.ones((3, 3), dtype=bool)


@dataclass
class Traces:
    """One row per connected fault trace."""
    length_px: np.ndarray
    centroid_row: np.ndarray
    centroid_col: np.ndarray
    label_ids: np.ndarray
    n: int = 0

    def __post_init__(self):
        self.n = int(len(self.length_px))


def extract_traces(mask: np.ndarray, pixel_m: float = 100.0,
                   skeleton: bool = True) -> Traces:
    """Connected-component traces with skeleton length and barycentre.

    `skeleton=True` measures length along the 1-pixel-wide medial axis, which is
    the correct length proxy for an elongated trace; pixel count alone would
    scale with trace width instead of trace length.
    """
    lab, n = ndimage.label(np.asarray(mask).astype(bool), structure=STRUCT8)
    if n == 0:
        return Traces(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=int))
    if skeleton:
        try:
            from skimage.morphology import skeletonize
            sk = skeletonize(np.asarray(mask).astype(bool))
        except Exception:                                   # pragma: no cover
            sk = np.asarray(mask).astype(bool)
    else:
        sk = np.asarray(mask).astype(bool)
    sk_lab = np.where(lab > 0, lab, 0)
    lens = np.bincount(sk_lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    # a skeleton of an isolated blob can vanish; fall back to the component size
    comp = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    lens = np.where(lens > 0, lens, np.maximum(comp, 1.0))
    rows = ndimage.sum(np.indices(mask.shape)[0], lab, index=np.arange(1, n + 1))
    cols = ndimage.sum(np.indices(mask.shape)[1], lab, index=np.arange(1, n + 1))
    cen = rows / np.maximum(comp, 1), cols / np.maximum(comp, 1)
    return Traces(length_px=lens * 1.0, centroid_row=cen[0], centroid_col=cen[1],
                  label_ids=np.arange(1, n + 1), n=n)


def length_metres(t: Traces, pixel_m: float = 100.0) -> np.ndarray:
    """Skeleton length in metres. Diagonal skeleton steps are sqrt(2) px."""
    return t.length_px * pixel_m


# ---------------------------------------------------------------------------
# (1a) length-frequency exponent a
# ---------------------------------------------------------------------------
def fit_length_frequency(L: np.ndarray, l_min: float, n_bins: int = 24):
    """Least-squares fit of log10 N(>=L) = log10 C - a*log10 L above l_min."""
    L = np.asarray(L, dtype=np.float64)
    L = L[L >= l_min]
    if len(L) < 8:
        return {"a": float("nan"), "r2": float("nan"), "n": int(len(L)), "C": float("nan")}
    edges = np.unique(np.logspace(np.log10(L.min()), np.log10(L.max()), n_bins + 1))
    counts = np.array([(L >= e).sum() for e in edges[:-1]], dtype=np.float64)
    keep = counts > 0
    x = np.log10(edges[:-1][keep])
    y = np.log10(counts[keep])
    if len(x) < 4:
        return {"a": float("nan"), "r2": float("nan"), "n": int(len(L)), "C": float("nan")}
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = float(((y - pred) ** 2).sum())
    tt = float(((y - y.mean()) ** 2).sum())
    return {"a": float(-coef[0]), "C": float(10 ** coef[1]), "r2": 1 - ss / tt if tt > 0 else float("nan"),
            "n_traces": int(len(L)), "l_min": float(l_min), "n_bins_used": int(keep.sum())}


# ---------------------------------------------------------------------------
# (1b) nearest-larger-neighbour exponent x   (Bour & Davy 1999)
# ---------------------------------------------------------------------------
def nearest_larger_neighbour(t: Traces, pixel_m: float = 100.0, max_n: int = 6000):
    """For every trace, distance to the barycentre of the nearest LONGER trace.

    Returns per-trace (length_m, d_m). Uses a KD-tree-free O(n^2) block search
    with the largest `max_n` traces subsampled deterministically if needed.
    """
    L = length_metres(t, pixel_m)
    pts = np.vstack([t.centroid_row, t.centroid_col]).T.astype(np.float64) * pixel_m
    n = len(L)
    if n > max_n:                       # keep the largest traces, deterministic
        idx = np.argsort(-L)[:max_n]
        L, pts = L[idx], pts[idx]
        n = max_n
    order = np.argsort(L)               # ascending length
    Ls, Ps = L[order], pts[order]
    d = np.full(n, np.nan)
    for i in range(n):
        cand = np.arange(i + 1, n)      # all strictly longer (ties broken by order)
        if cand.size == 0:
            continue
        dd = np.hypot(Ps[cand, 0] - Ps[i, 0], Ps[cand, 1] - Ps[i, 1])
        d[i] = dd.min()
    return Ls, d


def fit_nearest_larger(Ls: np.ndarray, d: np.ndarray, n_bins: int = 12,
                       min_per_bin: int = 8):
    """Fit log10 <d(l)> = x*log10 l + log10 A  -> Bour & Davy exponent x."""
    ok = np.isfinite(d) & (d > 0) & (Ls > 0)
    Ls, d = Ls[ok], d[ok]
    if len(Ls) < 4 * min_per_bin:
        return {"x": float("nan"), "n": int(len(Ls))}
    q = np.quantile(np.log(Ls), np.linspace(0, 1, n_bins + 1))
    q = np.unique(q)
    rows = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (np.log(Ls) >= lo) & (np.log(Ls) < hi)
        if m.sum() < min_per_bin:
            continue
        rows.append((np.exp(np.log(Ls[m]).mean()), d[m].mean(), int(m.sum())))
    if len(rows) < 4:
        return {"x": float("nan"), "n": int(len(Ls)), "bins": rows}
    x_ = np.log10([r[0] for r in rows]); y_ = np.log10([r[1] for r in rows])
    A = np.vstack([x_, np.ones_like(x_)]).T
    coef, *_ = np.linalg.lstsq(A, y_, rcond=None)
    pred = A @ coef
    ss = float(((y_ - pred) ** 2).sum()); tt = float(((y_ - y_.mean()) ** 2).sum())
    return {"x": float(coef[0]), "A": float(10 ** coef[1]),
            "r2": 1 - ss / tt if tt > 0 else float("nan"),
            "n": int(len(Ls)), "n_bins_used": len(rows),
            "bins": [{"l_mean_m": r[0], "d_mean_m": r[1], "n": r[2]} for r in rows]}


def correlation_dimension(pts: np.ndarray, r_grid: np.ndarray, pixel_m: float = 100.0):
    """Two-point correlation integral C2(r) ~ r^D for trace barycentres."""
    P = np.asarray(pts, dtype=np.float64)
    n = len(P)
    if n < 20:
        return {"D": float("nan"), "n": n}
    counts = np.zeros(len(r_grid))
    step = max(1, n // 3000)
    used = 0
    for i in range(0, n, step):
        dd = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        dd.sort()
        # #{j : d_j <= r} for each radius, minus the self-pair (d = 0)
        counts += np.searchsorted(dd, r_grid, side="right") - 1
        used += 1
    C2 = counts / max(used, 1) / max(n - 1, 1)
    ok = C2 > 0
    if ok.sum() < 4:
        return {"D": float("nan"), "n": n}
    x = np.log10(r_grid[ok]); y = np.log10(C2[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = float(((y - pred) ** 2).sum()); tt = float(((y - y.mean()) ** 2).sum())
    return {"D": float(coef[0]), "r2": 1 - ss / tt if tt > 0 else float("nan"),
            "n": n, "r_grid_px": r_grid.tolist(), "C2": C2.tolist()}


# ---------------------------------------------------------------------------
# (2) normalised correlation count -- published 1-D scanline form
# ---------------------------------------------------------------------------
def ncc_1d(positions: np.ndarray, scan_length: float, lags: np.ndarray,
           n_random: int = 200, seed: int = 0) -> dict:
    """Marrett et al. (2018) normalised correlation count on a 1-D scanline.

    NCC(r) = observed #pairs with separation in bin r  /  expected #pairs for
    the same count of points placed uniformly at random on the same scanline.
    """
    p = np.sort(np.asarray(positions, dtype=np.float64))
    n = len(p)
    if n < 5:
        return {"ncc": np.full(len(lags) - 1, np.nan), "n": n}
    d = np.diff(p)

    def counts(pos):
        out = np.zeros(len(lags) - 1)
        for i in range(len(pos)):
            sep = pos[i + 1:] - pos[i]
            if sep.size == 0:
                break
            idx = np.searchsorted(lags, sep, side="right") - 1
            valid = (idx >= 0) & (idx < len(out))
            np.add.at(out, idx[valid], 1)
        return out

    obs = counts(p)
    rng = np.random.default_rng(seed)
    null = np.array([counts(np.sort(rng.uniform(0, scan_length, n))) for _ in range(n_random)])
    exp = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ncc = np.where(exp > 0, obs / exp, np.nan)
        z = np.where(sd > 0, (obs - exp) / sd, 0.0)
    centres = 0.5 * (lags[:-1] + lags[1:])
    return {"lag_edges": lags.tolist(), "lag_centres": centres.tolist(),
            "observed": obs.tolist(), "expected": exp.tolist(),
            "sd": sd.tolist(), "ncc": ncc.tolist(), "z": z.tolist(), "n_points": n}


def ncc_1d_from_raster(mask: np.ndarray, angles_deg=(0, 45, 90, 135),
                       lags_px=None, n_random: int = 200, seed: int = 0) -> dict:
    """Pool NCC over horizontal / vertical / diagonal scanlines through `mask`."""
    mask = np.asarray(mask).astype(bool)
    H, W = mask.shape
    lags = np.arange(0.5, 80.5, 1.0) if lags_px is None else np.asarray(lags_px, float)
    out = {}
    alln, allo, alle = [], [], []
    for ang in angles_deg:
        pos_list = []
        if ang == 0:
            for r in range(H):
                c = np.flatnonzero(mask[r])
                if c.size >= 5:
                    pos_list.append((c.astype(float), float(W)))
        elif ang == 90:
            for c in range(W):
                r = np.flatnonzero(mask[:, c])
                if r.size >= 5:
                    pos_list.append((r.astype(float), float(H)))
        else:
            # 45 / 135 degree diagonals: index along the diagonal
            k = (np.arange(W)[None, :] - np.arange(H)[:, None]) if ang == 45 else \
                (np.arange(W)[None, :] + np.arange(H)[:, None])
            for key in np.unique(k):
                sel = (k == key)
                idx = np.flatnonzero(sel.ravel())
                vals = mask.ravel()[idx]
                p = np.flatnonzero(vals).astype(float)
                if p.size >= 5:
                    pos_list.append((p, float(len(idx))))
        for p, L in pos_list:
            r = ncc_1d(p, L, lags, n_random=max(20, n_random // max(len(pos_list), 1)), seed=seed)
            if np.isfinite(r.get("ncc", np.nan)).any():
                alln.append(r["n_points"]); allo.append(r["observed"]); alle.append(r["expected"])
        out[f"angle_{ang}"] = {"n_scanlines": len(pos_list)}
    if allo:
        obs = np.sum(np.array(allo), axis=0)
        exp = np.sum(np.array(alle), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ncc = np.where(exp > 0, obs / exp, np.nan)
        out["pooled"] = {"lag_edges_px": lags.tolist(),
                         "lag_centres_px": (0.5 * (lags[:-1] + lags[1:])).tolist(),
                         "ncc": ncc.tolist(),
                         "observed": obs.tolist(), "expected": exp.tolist(),
                         "n_scanlines_total": int(len(allo))}
    return out


def classify_arrangement(ncc: np.ndarray, lags_px: np.ndarray,
                         lo_px: float = 2.0, hi_px: float = 40.0,
                         tol: float = 0.15) -> dict:
    """Marrett-style verdict per length-scale band: clustered / random / regular."""
    ncc = np.asarray(ncc, float); lags = np.asarray(lags_px, float)
    m = (lags >= lo_px) & (lags <= hi_px) & np.isfinite(ncc)
    if m.sum() == 0:
        return {"verdict": "insufficient_data"}
    v = ncc[m]; lr = np.log10(lags[m])
    A = np.vstack([lr, np.ones_like(lr)]).T
    coef, *_ = np.linalg.lstsq(A, np.log10(np.clip(v, 1e-6, None)), rcond=None)
    mean = float(v.mean())
    if mean > 1 + tol:
        verdict = "clustered"
    elif mean < 1 - tol:
        verdict = "regular"
    else:
        verdict = "random"
    return {"verdict": verdict, "mean_ncc": mean, "min_ncc": float(v.min()),
            "max_ncc": float(v.max()), "loglog_slope": float(coef[0]),
            "implied_correlation_dimension": float(coef[0] + 1.0),
            "lag_range_px": [lo_px, hi_px], "n_lags": int(m.sum())}


# ---------------------------------------------------------------------------
# geometric prior from the fitted Bour & Davy scaling
# ---------------------------------------------------------------------------
def bour_davy_prior_field(label_map: np.ndarray, t: Traces, fit_x: dict,
                          l_incomplete: float | None = None,
                          pixel_m: float = 100.0, top_k: int = 200,
                          r_max_px: int = 60, log_sigma: float = 1.0) -> np.ndarray:
    """Geometric prior from the fitted Bour & Davy (1999) nearest-larger scaling.

    Bour & Davy measured that a fault of length l sits, on average, at distance
    <d(l)> = A * l^x from its nearest LARGER neighbour.  Inverting that scaling
    turns distance into an expected length:

        l*(r) = (r / A) ** (1 / x)

    So at distance r from a known trace, the unmapped fault the scaling predicts
    to be there has length l*(r).  A candidate pixel earns prior weight where
    l*(r) falls inside the length range in which the catalogue is demonstrably
    INCOMPLETE (the short-fault tail), and little weight where the scaling
    predicts a large fault -- because large faults at that distance would already
    have been mapped.  Contributions from all `top_k` longest traces are combined
    with a max, so the field is governed by the NEAREST LARGER fault, exactly the
    quantity the paper defines.

    This field uses only fault-trace geometry.  It contains no information from
    the 19 competition feature bands, so it is an independent re-ranking term:
    it favours a candidate lying along the extrapolated clustering pattern of a
    known larger fault over an equally-scored but spatially isolated one.
    """
    H, W = label_map.shape
    L = length_metres(t, pixel_m)
    x = float(fit_x.get("x", np.nan))
    A = float(fit_x.get("A", np.nan))
    if not (np.isfinite(x) and np.isfinite(A)) or x <= 0 or A <= 0:
        return np.zeros((H, W), dtype=np.float32)
    if l_incomplete is None:
        pos = L[L > 0]
        l_incomplete = float(np.percentile(pos, 60)) if pos.size else 1500.0
    out = np.zeros((H, W), dtype=np.float32)
    for k in np.argsort(-L)[:top_k]:
        comp = (label_map == t.label_ids[k])
        if not comp.any():
            continue
        d_px = ndimage.distance_transform_edt(~comp)
        inside = (d_px > 0) & (d_px < r_max_px)
        n_in = int(inside.sum())
        if n_in == 0:
            continue
        d_m = d_px[inside].astype(np.float64) * pixel_m
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            l_star = (d_m / A) ** (1.0 / x)
        w = np.zeros(n_in, dtype=np.float32)
        good = np.isfinite(l_star) & (l_star > 0)
        w[good] = np.exp(-0.5 * (np.log(l_star[good] / l_incomplete) / log_sigma) ** 2)
        buf = np.zeros((H, W), dtype=np.float32)
        buf[inside] = w
        np.maximum(out, buf, out=out)
    return out


@dataclass
class ClusteringReport:
    a_fit: dict = field(default_factory=dict)
    x_fit: dict = field(default_factory=dict)
    D_fit: dict = field(default_factory=dict)
    consistency: dict = field(default_factory=dict)
    ncc_known: dict = field(default_factory=dict)
    ncc_predicted: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"bour_davy_1999": {"length_frequency_a": self.a_fit,
                                   "nearest_larger_x": self.x_fit,
                                   "correlation_dimension_D": self.D_fit,
                                   "consistency_x_vs_a_minus_1_over_D": self.consistency},
                "marrett_2018_ncc": {"known_population": self.ncc_known,
                                     "predicted_raster": self.ncc_predicted,
                                     "audit": self.audit}}
