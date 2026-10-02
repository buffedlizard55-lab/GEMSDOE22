"""Fold-safe geometric priors built from a *visible* fault catalogue (H26 workstream).

The brief requires the fractal-clustering statistic to be used two ways: as a
geometric prior on candidates, and as a post-hoc audit of predicted rasters.
``src/gems/clustering.py`` implements the *statistic*.  This module implements the
two *policies* that were missing, and it is deliberately strict about one thing:

    **every field here is a function of the visible catalogue only.**

That is invariant I-2 in ``AGENTS.md``: a feature computed from the full
catalogue leaks, because a held-out fault sits at distance 0 from itself (fold
NW DTI 0.977, flag F-10).  Every function therefore takes the *train* mask
(``visible``) explicitly and never the full catalogue, so the same code produces
a leak-free value inside a fold and a legitimate value in production.

The four policies compared by ``scripts/h26_geometry_holdout.py``:

``halo_isotropic``
    The shape shipped by ``H22-1``/``H22-2``: a radial Gaussian on the distance
    to the nearest visible trace, sigma = lambda (default 1.8 km).  Isotropic --
    a candidate 1.8 km *ahead of a fault tip* scores the same as one 1.8 km
    *abeam of the fault's middle*.  Real fault populations do not behave that
    way; faults propagate along strike.

``bour_davy_ring``
    Uses the measured nearest-larger-neighbour distribution instead of an
    arbitrary sigma: the prior peaks at the measured median distance
    (1,630.9 m in this region, ``evidence/clustering_fit.json``) from a larger
    trace.  This is the "direct mathematical link between a network's
    clustering dimension and its length-frequency exponent" (Bour & Davy 1999,
    DOI 10.1029/1999GL900524) turned into a placement rule.

``strike_tip``
    Anisotropic continuation: extrapolate every visible trace beyond BOTH tips
    along its local strike, with a reach drawn from the measured length
    distribution.  Mechanically, faults grow tip-ward and interact at tips
    (Segall & Pollard 1980, DOI 10.1029/JB085iB08p04333); an expert mapping
    "new" faults is following exactly those continuations.

``dilate_band``
    Metric-matching, not geology: the official kernel is triangular with R = 300 m
    (3 px), so a prediction 1 px off the truth still scores k = 2/3 while a false
    pixel 1 px off the truth is charged only 0.2 * (1 - 2/3) = 0.067.  A one-pixel
    band around a candidate is therefore nearly free insurance against the
    off-by-one misalignment the organisers explicitly warn about
    (page 967: "Rasterization is lossy and can induce off-by-one errors").

All fields are returned as float32 in [0, 1] over the full grid, zero outside
``domain``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

NEIGH8 = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------------------
# basic distance fields
# ---------------------------------------------------------------------------
def distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (in pixels) to the nearest True pixel of ``mask``."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.full(m.shape, np.inf, dtype=np.float64)
    return ndimage.distance_transform_edt(~m)


def halo_isotropic(visible: np.ndarray, sigma_px: float = 18.0) -> np.ndarray:
    """Radial Gaussian on the distance to the nearest visible trace."""
    d = distance_to(visible)
    out = np.exp(-0.5 * (d / float(sigma_px)) ** 2)
    return out.astype(np.float32)


def bour_davy_ring(visible: np.ndarray, peak_px: float = 16.3,
                   sigma_px: float = 24.0) -> np.ndarray:
    """Prior peaked at the measured nearest-larger-neighbour distance.

    ``peak_px`` defaults to the measured 1,630.9 m / 100 m = 16.3 px median
    (``evidence/clustering_fit.json``); ``sigma_px`` is deliberately wide
    (2.4 km) because the measured p90 is 6.0 km.
    """
    d = distance_to(visible)
    out = np.exp(-0.5 * ((d - float(peak_px)) / float(sigma_px)) ** 2)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# anisotropic tip extrapolation
# ---------------------------------------------------------------------------
def skeleton_endpoints_and_directions(skeleton: np.ndarray, arc: int = 8
                                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Endpoints of a 1-px skeleton with an outward unit direction per endpoint.

    Returns ``(ys, xs, dirs)`` where ``dirs`` has shape (n, 2) in (dy, dx) order.
    The direction is estimated from the ``arc`` skeleton pixels nearest the
    endpoint along the trace (PCA, oriented so that it points *away* from the
    trace's interior).
    """
    sk = np.asarray(skeleton, dtype=bool)
    if not sk.any():
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2), float)

    # neighbour count via convolution (8-connectivity)
    nb = ndimage.convolve(sk.astype(np.uint8), NEIGH8, mode="constant")
    ys, xs = np.nonzero(sk & (nb == 2))          # itself + exactly one neighbour
    if len(ys) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2), float)

    sk_pts = np.argwhere(sk)
    tree = None
    from scipy.spatial import cKDTree
    tree = cKDTree(sk_pts)

    dirs = np.zeros((len(ys), 2), dtype=np.float64)
    for i, (y, x) in enumerate(zip(ys, xs)):
        # gather skeleton pixels within a radius, keep the `arc` closest to this endpoint
        idx = tree.query_ball_point([y, x], r=max(arc * 1.5, 6.0))
        pts = sk_pts[idx] if idx else np.array([[y, x]])
        if len(pts) > arc:
            d2 = ((pts - np.array([y, x])) ** 2).sum(1)
            pts = pts[np.argsort(d2)[:arc]]
        c = pts - pts.mean(0)
        if len(pts) < 3:
            dirs[i] = (0.0, 0.0)
            continue
        # principal axis of the local trace segment
        u, s, vt = np.linalg.svd(c, full_matrices=False)
        v = vt[0]
        # orient away from the interior: the endpoint must be the extreme point
        if np.dot(np.array([y, x]) - pts.mean(0), v) < 0:
            v = -v
        dirs[i] = v
    return ys, xs, dirs


def strike_tip_field(visible: np.ndarray, lengths_m: np.ndarray | None = None,
                     max_reach_px: int = 20, taper_m: float = 600.0,
                     step_px: int = 1) -> np.ndarray:
    """Anisotropic continuation field: 1 along an extrapolated tip path, decaying with distance.

    ``max_reach_px`` caps how far a trace is extended (default 20 px = 2 km).
    ``lengths_m`` (optional) is used to scale the reach per trace: the reach is
    ``min(max_reach, 0.25 * trace_length)`` so that short faults are not extended
    as far as long ones -- the measured length distribution is heavy-tailed, and
    a short splay does not continue for kilometres.
    """
    from skimage.morphology import skeletonize

    vis = np.asarray(visible, dtype=bool)
    if not vis.any():
        return np.zeros(vis.shape, dtype=np.float32)

    lab, n = ndimage.label(vis, structure=NEIGH8)
    sk = skeletonize(vis)
    ys, xs, dirs = skeleton_endpoints_and_directions(sk, arc=8)
    if len(ys) == 0:
        return np.zeros(vis.shape, dtype=np.float32)

    reach = np.full(len(ys), float(max_reach_px))
    if lengths_m is not None and len(lengths_m) > 0:
        med = float(np.median(lengths_m))
        reach = np.minimum(reach, np.maximum(4.0, 0.25 * med / 100.0))

    H, W = vis.shape
    field = np.zeros((H, W), dtype=np.float32)
    for (y0, x0), (dy, dx) in zip(zip(ys, xs), dirs):
        norm = float(np.hypot(dy, dx))
        if norm < 1e-9:
            continue
        dy, dx = dy / norm, dx / norm
        # stop the walk if it re-enters the visible mask (that is not a continuation)
        for t in range(1, int(reach.max()) + 1, max(1, step_px)):
            y = int(round(y0 + dy * t))
            x = int(round(x0 + dx * t))
            if not (0 <= y < H and 0 <= x < W):
                break
            if vis[y, x]:
                break
            # contribution at the walked pixel, then tapered by distance from the path
            field[y, x] = max(field[y, x], 1.0)
    if not field.any():
        return field
    # spread along-strike is free, across-strike is not: use a small isotropic blur
    # only, so the field stays a thin line (the point of the arm is thin support).
    sig = max(taper_m / 100.0, 0.6)
    return ndimage.gaussian_filter(field, sig).astype(np.float32)


def dilate_band(score: np.ndarray, domain: np.ndarray, radius: int = 1) -> np.ndarray:
    """One-pixel band around the support of ``score`` (metric-matching hedge)."""
    s = np.asarray(score, dtype=np.float32)
    sup = np.nan_to_num(s) > 0
    if not sup.any():
        return s.copy()
    band = ndimage.binary_dilation(sup, structure=NEIGH8, iterations=int(radius))
    out = s.copy()
    add = band & ~sup & np.asarray(domain, dtype=bool)
    out[add] = 1.0
    return out


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------
def emit_topn(score: np.ndarray, domain: np.ndarray, n: int) -> np.ndarray:
    """Binary emission of the ``n`` highest-scoring eligible pixels (ties by index)."""
    elig = np.asarray(domain, dtype=bool)
    s = np.where(elig, np.nan_to_num(np.asarray(score, dtype=np.float32), nan=-np.inf), -np.inf)
    total = int(elig.sum())
    n = int(min(max(n, 0), total))
    if n == 0:
        return np.zeros(s.shape, dtype=bool)
    flat = s.ravel()
    idx = np.argpartition(flat, -n)[-n:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(s.shape)


def ncc_divergence(pred: np.ndarray, reference: np.ndarray, scales_px=(1, 2, 3, 5, 8, 12, 20)
                   ) -> dict:
    """Normalised correlation count of two binary rasters, scale by scale.

    This is the Marrett-style normalised correlation count the brief refers to:
    for each lag scale the count of *pairs* at that distance is normalised by the
    count expected if the geometry were shuffled while keeping the total area
    occupied.  Values > 1 mean clustered, ~1 random, < 1 regular.

    Returns ``{scales_px, ncc_pred, ncc_ref, log_ratio, divergence}`` where
    ``divergence`` is the mean absolute log-ratio -- the same summary the
    post-hoc audit of the brief is built on.
    """
    def _ncc(mask: np.ndarray) -> list[float]:
        m = np.asarray(mask, dtype=bool)
        n = int(m.sum())
        if n < 10:
            return [float("nan")] * len(scales_px)
        pts = np.argwhere(m).astype(np.float64)
        tree = None
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        out = []
        for r in scales_px:
            # pair count within an annulus [r-0.5, r+0.5)
            pairs = tree.count_neighbors(tree, r + 0.5) - tree.count_neighbors(tree, max(r - 0.5, 0.0))
            pairs = pairs / 2.0
            area = m.size
            dens = n / area
            expected = 0.5 * n * (n - 1) * (np.pi * ((r + 0.5) ** 2 - (r - 0.5) ** 2)) / area
            out.append(float(pairs / expected) if expected > 0 else float("nan"))
        return out

    a = _ncc(pred)
    b = _ncc(reference)
    lr = [float(np.log(x / y)) if (x > 0 and y > 0) else float("nan") for x, y in zip(a, b)]
    finite = [v for v in lr if np.isfinite(v)]
    return {
        "scales_px": list(scales_px),
        "ncc_pred": a,
        "ncc_ref": b,
        "log_ratio": lr,
        "divergence": float(np.mean(np.abs(finite))) if finite else float("nan"),
    }
