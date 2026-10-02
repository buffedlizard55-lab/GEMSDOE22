"""Fractal fault-population spatial statistics — Bour & Davy (1999) + normalized correlation count.

This module implements the two complementary spatial-statistical descriptors
referenced in the project prompt:

1. **Bour & Davy clustering test** (Geophysical Research Letters, 1999,
   DOI 10.1029/1999GL900524): distance from each fault to its nearest
   *larger* neighbour vs. fault length. The paper establishes a direct
   link between the power-law length exponent ``a`` and the correlation /
   clustering dimension ``D``. Measured over the catalogued INGENIOUS/USGS
   traces inside the GeoDAWN footprint *before* touching any model, the
   fitted ``D`` is then used as a **geometric prior** that favours a
   candidate pixel lying along the extrapolated clustering pattern of a
   known larger fault over an equally-scored but spatially isolated one.

2. **Normalized correlation count / Ripley K** (Ripley 1976; later
   structural-geology applications e.g. Bour et al. 2002, Bonnet et al.
   2001): counts the number of fault-pair separations < r, normalized to
   the expectation for a homogeneous Poisson (random) population. At a
   given length scale r the population is classified as clustered,
   random, or regularly spaced. The same statistic is computed on the
   **predicted raster** as a post-hoc audit — a submission whose
   predicted spatial arrangement diverges sharply from the population
   statistics measured in this region is flagged as a likely detection
   artifact (survey-line aliasing, acquisition-block edges).

Both descriptors treat the fault population as a **spatial statistic**,
not a pile of independent pixels. Nothing in a per-pixel loss checks
whether the output looks like a real fault population; this does.

References (all official, free to verify):
  - Bour, O. & Davy, P., Geophys. Res. Lett. 26, 1999 — clustering
    dimension vs. length exponent via nearest-larger-neighbour distance.
    https://doi.org/10.1029/1999GL900524
  - Bour, O. et al., Water Resour. Res. 38, 2002 — clustering and
    connectivity in field fracture networks.
    https://doi.org/10.1029/2001WR000432
  - Ripley, B.D., J. Roy. Stat. Soc. B 39, 1977 — spatial point pattern
    analysis (K-function).
    https://doi.org/10.1111/j.2517-6161.1977.tb01615.x
  - Bonnet, E. et al., Rev. Geophys. 39, 2001 — review of fracture
    population scaling.
    https://doi.org/10.1029/1999RG000074
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndi_label
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ClusteringFit:
    """Fitted fractal-clustering parameters for the catalogued population."""

    n_traces: int
    alpha_ols: float
    alpha_mle: float
    r2_loglog: float
    l_min_m: float
    D_correlation: float          # correlation dimension from C(r) ~ r^D
    D_bour_davy_predicted: float  # Bour & Davy predicted D from alpha
    D_consistent: bool
    nearest_larger_median_m: float
    nearest_larger_p90_m: float
    ripley_scales_m: list[float]
    C_observed: list[float]
    C_poisson_expected: list[float]
    K_normalized: list[float]     # K(r)/pi r^2 ; >1 clustered, ~1 random, <1 regular
    interpretation: str


def _trace_centroids_and_lengths(binary_mask: np.ndarray, pixel_size_m: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Skeletonize ``binary_mask`` and return centroids (y,x) and lengths (m) per connected trace."""
    mask = binary_mask > 0
    skel = skeletonize(mask)
    comp, n = ndi_label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0)
    # Vectorised per-component reduction (np.bincount), NOT a per-component boolean mask:
    # a prediction raster can contain 10^5+ components and the O(n*H*W) mask loop hangs.
    flat = comp.ravel()
    idx = np.flatnonzero(flat)
    ids = flat[idx]
    area = np.bincount(ids, minlength=n + 1)[1:].astype(np.float64)
    sk_counts = np.bincount(ids, weights=skel.ravel()[idx].astype(np.float64), minlength=n + 1)[1:]
    yy = np.broadcast_to(np.arange(binary_mask.shape[0], dtype=np.float64)[:, None], binary_mask.shape).ravel()[idx]
    xx = np.broadcast_to(np.arange(binary_mask.shape[1], dtype=np.float64)[None, :], binary_mask.shape).ravel()[idx]
    sum_y = np.bincount(ids, weights=yy, minlength=n + 1)[1:]
    sum_x = np.bincount(ids, weights=xx, minlength=n + 1)[1:]
    centroids = np.column_stack([sum_y / np.maximum(area, 1.0), sum_x / np.maximum(area, 1.0)])
    length_m = sk_counts * pixel_size_m
    # sub-resolution components carry no skeleton pixel: fall back to their area
    length_m = np.where(length_m < pixel_size_m, area * pixel_size_m, length_m)
    return centroids, np.maximum(length_m, pixel_size_m)


def _nearest_larger_distances(centroids: np.ndarray, lengths: np.ndarray, pixel_size_m: float = 100.0) -> np.ndarray:
    """For each fault, Euclidean distance (m) to the nearest *larger* fault (centroid distance)."""
    if len(lengths) < 2:
        return np.array([])
    # Sort by length descending so "larger" means earlier in order
    order = np.argsort(-lengths)
    centroids_sorted = centroids[order]
    lengths_sorted = lengths[order]
    dists = np.full(len(lengths), np.nan, dtype=np.float64)
    # For each trace, search among traces that are strictly larger
    # Use cKDTree incrementally — build tree of all larger centroids and query
    for i in range(1, len(lengths)):
        tree = cKDTree(centroids_sorted[:i])
        dist_px, _ = tree.query(centroids_sorted[i], k=1)
        dists[order[i]] = float(dist_px * pixel_size_m)
    # Largest fault has no larger neighbour → NaN
    return dists


def _correlation_integral(centroids: np.ndarray, scales_m: np.ndarray, pixel_size_m: float = 100.0, max_pairs: int | None = 200_000) -> np.ndarray:
    """Normalized correlation count C(r) = 2 * pairs with dist < r / (N*(N-1))."""
    n = len(centroids)
    if n < 2:
        return np.zeros_like(scales_m, dtype=np.float64)
    # If too many faults, subsample centroids to keep O(N^2) tractable
    if n > 800 and max_pairs is not None:
        # Use deterministic subsample (largest faults first to preserve long-range structure)
        # Caller should have sorted by length; we just take first 800 here if needed
        # This path is only hit for very fault-dense quadrants
        idx = np.linspace(0, n - 1, min(n, 800), dtype=int)
        centroids = centroids[idx]
        n = len(centroids)
    # Pairwise distances via pdist-like using broadcasting in chunks to avoid O(N^2) memory blow-up
    # Use cKDTree sparse approach: for each scale, count pairs with dist < r via tree query_ball_point
    C = np.zeros(len(scales_m), dtype=np.float64)
    tree = cKDTree(centroids)
    total_pairs = n * (n - 1) / 2.0
    for j, r_m in enumerate(scales_m):
        r_px = float(r_m / pixel_size_m)
        # count pairs efficiently: sum of neighbours within radius, divided by 2 (each pair counted twice)
        counts = tree.query_ball_point(centroids, r=r_px, return_length=True)
        # counts includes self (distance 0), so subtract 1 per point, then sum/2
        pair_count = (np.asarray(counts, dtype=np.float64).sum() - n) / 2.0
        C[j] = float(pair_count / total_pairs) if total_pairs > 0 else 0.0
    return C


def fit_clustering_dimension(
    binary_fault_mask: np.ndarray,
    footprint: np.ndarray,
    pixel_size_m: float = 100.0,
    l_min_m: float = 1650.0,
) -> ClusteringFit:
    """Fit correlation dimension D and Bour & Davy consistency check on the catalogued population.

    Parameters
    ----------
    binary_fault_mask : 2D bool — catalogued faults (labels.tif >0) intersect footprint.
    footprint : 2D bool — GeoDAWN footprint (finite cells of template).
    """
    valid_mask = binary_fault_mask & footprint
    centroids, lengths = _trace_centroids_and_lengths(valid_mask, pixel_size_m=pixel_size_m)
    n = len(lengths)
    if n < 10:
        raise ValueError(f"Too few traces ({n}) to fit clustering")

    # --- Power-law exponent alpha (OLS on cumulative N(>=L)) ---
    L = np.sort(lengths)
    tail = L[L >= l_min_m]
    if len(tail) < 10:
        # fall back to median-based l_min
        l_min_m = float(np.median(L))
        tail = L[L >= l_min_m]
    # OLS on log-log cumulative
    uniq = np.unique(tail)
    if len(uniq) < 5:
        uniq = np.unique(L)
    n_ge = np.array([(L >= x).sum() for x in uniq], dtype=np.float64)
    # Use middle 80% to avoid roll-off at extremes
    lo, hi = np.percentile(np.log10(uniq), [5, 95])
    m = (np.log10(uniq) >= lo) & (np.log10(uniq) <= hi) & (n_ge > 0)
    if m.sum() < 5:
        m = n_ge > 0
    slope, _ = np.polyfit(np.log10(uniq[m]), np.log10(n_ge[m]), 1)
    alpha_ols = float(-slope)
    alpha_mle = float(len(tail) / np.sum(np.log(tail / l_min_m))) if len(tail) > 0 else alpha_ols
    r2 = float(np.corrcoef(np.log10(uniq[m]), np.log10(n_ge[m]))[0, 1] ** 2) if m.sum() >= 2 else 0.0

    # --- Nearest-larger-neighbour distances (Bour & Davy observable) ---
    nld = _nearest_larger_distances(centroids, lengths, pixel_size_m=pixel_size_m)
    nld_valid = nld[np.isfinite(nld)]
    median_nld = float(np.median(nld_valid)) if len(nld_valid) else float("nan")
    p90_nld = float(np.percentile(nld_valid, 90)) if len(nld_valid) else float("nan")

    # --- Correlation integral C(r) and Ripley K ---
    # Scales: 0.5 km to 20 km in log steps (12 scales)
    scales_m = np.array([500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000], dtype=np.float64)
    # For centroids in pixel units, distances are Euclidean in pixel space * pixel_size
    C_obs = _correlation_integral(centroids, scales_m, pixel_size_m=pixel_size_m)
    # Poisson expectation in a homogeneous distribution: C_poisson(r) ~ pi r^2 / A
    # where A is footprint area. But for normalized K we compare C to pi r^2 density.
    # Simpler: compute expected pair fraction for Poisson with same N and area.
    # Area = footprint pixel count * pixel_size^2
    area_m2 = float(footprint.sum() * pixel_size_m * pixel_size_m)
    # Expected C for Poisson: pi r^2 / A (for r << sqrt(A)), capped at 1
    C_poisson = np.clip(np.pi * scales_m**2 / area_m2, 0, 1)
    # Normalized K = C_obs / C_poisson (≈1 random, >1 clustered, <1 regular)
    # Guard against division by tiny C_poisson at small r
    K_norm = np.where(C_poisson > 1e-12, C_obs / C_poisson, np.nan)

    # --- Correlation dimension D: slope of log C vs log r in the scaling range ---
    # Use scales where 5 < expected neighbor count < 80% of pairs, to stay in power-law regime
    # Choose middle 6 scales (1.5 km to 10 km) where data are most stable
    fit_scales = scales_m[2:9]  # 1000 to 10000 m
    fit_C = C_obs[2:9]
    valid = (fit_C > 1e-6) & (fit_C < 0.8) & np.isfinite(fit_C)
    if valid.sum() >= 3:
        slope_D, _ = np.polyfit(np.log10(fit_scales[valid]), np.log10(fit_C[valid]), 1)
        D_corr = float(slope_D)
    else:
        D_corr = float("nan")

    # --- Bour & Davy predicted D from alpha ---
    # Bour & Davy 1999 show D ≈ 2 - (a-1)/2 for 1 < a < 3 in the synthetic
    # fault-growth model; more generally D decreases as a increases.
    # We implement the published linear approximation D_pred = 2.0 - 0.5*(alpha-1)
    # which gives D≈2 at alpha→1 (space-filling) and D≈1 at alpha→3 (linear clusters).
    # This is stated as an INFERENCE and flagged accordingly; the empirical
    # D_corr is what drives the prior.
    if np.isfinite(alpha_ols):
        D_pred = float(np.clip(2.0 - 0.5 * (alpha_ols - 1.0), 0.5, 2.0))
    else:
        D_pred = float("nan")
    consistent = bool(abs(D_corr - D_pred) < 0.35) if np.isfinite(D_corr) and np.isfinite(D_pred) else False

    # Classification per scale
    # >1.2 clustered, 0.8-1.2 random, <0.8 regular (conservative thresholds)
    interp = "clustered" if np.nanmedian(K_norm[2:7]) > 1.2 else ("regular" if np.nanmedian(K_norm[2:7]) < 0.8 else "random")

    return ClusteringFit(
        n_traces=n,
        alpha_ols=round(alpha_ols, 4) if np.isfinite(alpha_ols) else float("nan"),
        alpha_mle=round(alpha_mle, 4) if np.isfinite(alpha_mle) else float("nan"),
        r2_loglog=round(r2, 4) if np.isfinite(r2) else float("nan"),
        l_min_m=round(float(l_min_m), 1),
        D_correlation=round(float(D_corr), 3) if np.isfinite(D_corr) else float("nan"),
        D_bour_davy_predicted=round(D_pred, 3) if np.isfinite(D_pred) else float("nan"),
        D_consistent=consistent,
        nearest_larger_median_m=round(median_nld, 1) if np.isfinite(median_nld) else float("nan"),
        nearest_larger_p90_m=round(p90_nld, 1) if np.isfinite(p90_nld) else float("nan"),
        ripley_scales_m=[round(float(x), 1) for x in scales_m],
        C_observed=[round(float(x), 6) for x in C_obs],
        C_poisson_expected=[round(float(x), 6) for x in C_poisson],
        K_normalized=[round(float(x), 3) if np.isfinite(x) else float("nan") for x in K_norm],
        interpretation=interp,
    )


def clustering_geometric_prior(
    candidate_score: np.ndarray,
    footprint: np.ndarray,
    binary_known_faults: np.ndarray,
    D_correlation: float,
    pixel_size_m: float = 100.0,
    sigma_km: float = 2.0,
) -> np.ndarray:
    """Modulate ``candidate_score`` with a fractal-clustering geometric prior.

    Favors a candidate pixel that lies along the extrapolated clustering
    pattern of a known larger fault over an equally-scored but spatially
    isolated one. Weight scales as ``exp(-d / lambda)`` where ``lambda``
    is set from the fitted correlation dimension: more strongly clustered
    (low D) ⇒ shorter lambda (tight halos around large faults); near
    space-filling (D≈2) ⇒ broader, more diffuse prior.

    This is intentionally smooth (no hard ridge displacement) so that
    Hessian ``ridge_nms`` centerlines are not perturbed.

    Parameters
    ----------
    candidate_score : 2D float32 in [0, 1] — pre-prior model score.
    footprint : 2D bool
    binary_known_faults : 2D bool — known faults (labels >0)
    D_correlation : float — fitted correlation dimension (≈0.8–2.0)

    Returns
    -------
    2D float32 prior weight in [0.7, 1.3] (multiplicative), 1.0 = neutral.
    """
    if not np.isfinite(D_correlation):
        return np.ones_like(candidate_score, dtype=np.float32)
    # Map D to decay length: D=1.0 → sigma≈1.2 km, D=2.0 → sigma≈3.5 km
    # lambda = sigma_km * (D / 1.5)  — clipped to [1.0, 4.0] km
    lam_km = float(np.clip(sigma_km * (float(D_correlation) / 1.5), 1.0, 4.0))
    lam_px = lam_km * 1000.0 / pixel_size_m

    # Distance to nearest known fault (any size) — Euclidean distance transform
    from scipy.ndimage import distance_transform_edt

    inv = ~binary_known_faults
    # Restrict to footprint for distance transform boundary
    inv = inv | (~footprint)
    dist_px = distance_transform_edt(inv)
    # Prior: 1 + 0.3 * exp(-d/lambda)  — max 1.3 at the fault, 1.0 far away, smooth
    # This gives a 30% boost exactly on known-fault pixels (which are masked at scoring,
    # so the boost matters only for off-fault pixels within ~1–2 lambda)
    w = 1.0 + 0.30 * np.exp(-dist_px / max(lam_px, 1.0))
    # Damp far-field to 0.9 to mildly penalize truly isolated pixels (>3 lambda)
    isolated = dist_px > 3 * lam_px
    w[isolated] *= 0.9
    # Keep within [0.7, 1.3] and neutral outside footprint
    w = np.clip(w, 0.7, 1.3).astype(np.float32)
    w[~footprint] = 1.0
    return w


def audit_predicted_clustering(
    predicted_binary: np.ndarray,
    footprint: np.ndarray,
    expected_fit: ClusteringFit,
    pixel_size_m: float = 100.0,
    scales_m: list[float] | None = None,
) -> dict[str, Any]:
    """Post-hoc audit: compare predicted fault population's spatial arrangement to expected.

    Returns a dict with per-scale K ratios, maximum divergence, and a flag.
    """
    if scales_m is None:
        scales_m = expected_fit.ripley_scales_m
    scales_arr = np.asarray(scales_m, dtype=np.float64)
    # Extract centroids and lengths from predicted binary
    pred_mask = predicted_binary & footprint
    centroids, _ = _trace_centroids_and_lengths(pred_mask, pixel_size_m=pixel_size_m)
    if len(centroids) < 10:
        return {
            "n_pred_traces": int(len(centroids)),
            "scales_m": list(scales_arr),
            "C_pred": [0.0] * len(scales_arr),
            "K_pred_normalized": [float("nan")] * len(scales_arr),
            "K_expected_normalized": list(expected_fit.K_normalized),
            "max_log_K_divergence": float("nan"),
            "flag": "INSUFFICIENT_PREDICTIONS",
            "interpretation": "too few predicted traces to audit clustering",
        }
    C_pred = _correlation_integral(centroids, scales_arr, pixel_size_m=pixel_size_m)
    area_m2 = float(footprint.sum() * pixel_size_m * pixel_size_m)
    C_poisson = np.clip(np.pi * scales_arr**2 / area_m2, 1e-12, 1)
    K_pred = C_pred / C_poisson
    K_exp = np.asarray(expected_fit.K_normalized, dtype=np.float64)
    # Align lengths (expected may have different scales)
    # Our audit uses the expected-fit's scales, so lengths match
    # Compute max |log(K_pred) - log(K_exp)| over middle scales (1–10 km)
    valid = np.isfinite(K_pred) & np.isfinite(K_exp) & (K_pred > 1e-6) & (K_exp > 1e-6)
    # Restrict to 1000–10000 m (indices 2:9)
    valid[0:2] = False
    valid[9:] = False
    if valid.sum() >= 2:
        max_div = float(np.max(np.abs(np.log(K_pred[valid]) - np.log(K_exp[valid]))))
    else:
        max_div = float("nan")
    # Flag thresholds: <0.4 consistent, 0.4–0.8 WARN, >0.8 FLAG artifact
    if not np.isfinite(max_div):
        flag = "UNDETERMINED"
    elif max_div < 0.40:
        flag = "CONSISTENT"
    elif max_div < 0.80:
        flag = "WARN_DIVERGENT"
    else:
        flag = "FLAG_ARTIFACT_LIKELY"

    interpretation_map = {
        "CONSISTENT": "Predicted population's clustering matches measured regional statistics.",
        "WARN_DIVERGENT": "Predicted clustering diverges noticeably — review for acquisition or tiling artifacts.",
        "FLAG_ARTIFACT_LIKELY": "Predicted spatial arrangement diverges SHARPLY from measured population — likely detection artifact (survey-line aliasing, block edges) rather than genuine geology.",
        "INSUFFICIENT_PREDICTIONS": "Too few predicted traces to audit.",
        "UNDETERMINED": "Could not compute divergence.",
    }
    return {
        "n_pred_traces": int(len(centroids)),
        "scales_m": [round(float(x), 1) for x in scales_arr],
        "C_pred": [round(float(x), 6) for x in C_pred],
        "K_pred_normalized": [round(float(x), 3) if np.isfinite(x) else float("nan") for x in K_pred],
        "K_expected_normalized": [round(float(x), 3) if np.isfinite(x) else float("nan") for x in K_exp],
        "max_log_K_divergence": round(max_div, 3) if np.isfinite(max_div) else float("nan"),
        "flag": flag,
        "interpretation": interpretation_map.get(flag, ""),
    }


def fit_clustering_to_labels_file(labels_path: str, footprint_path: str | None = None) -> ClusteringFit:
    """Convenience: fit directly from ``labels.tif`` and ``sample_submission.tif`` files."""
    import rasterio

    with rasterio.open(labels_path) as src:
        labels = src.read(1) > 0
        shape = labels.shape
    if footprint_path is not None:
        with rasterio.open(footprint_path) as src:
            foot = np.isfinite(src.read(1))
    else:
        foot = np.ones(shape, dtype=bool)
    return fit_clustering_dimension(labels, foot)


def grid_alignment_audit(
    predicted_binary: np.ndarray,
    footprint: np.ndarray,
    bin_deg: float = 15.0,
    tol_deg: float = 7.5,
) -> dict[str, object]:
    """Distinguish *acquisition* artifacts from genuine geology in a predicted fault population.

    The prompt asks the post-hoc audit to flag "survey-line aliasing, acquisition-block edges"
    as an alternative explanation to genuine geology. Two signatures separate them:

    1. **Grid alignment.** Airborne survey flight lines and DEM tile seams are straight and
       parallel to the acquisition grid; geological faults are not. We histogram the local
       strike of the predicted ridge field (structure-tensor orientation, 15 deg bins over
       0-180 deg) and report the excess mass in the two raster-axis bins (0 deg and 90 deg)
       relative to the 2/n_bins share expected for an isotropic population.
    2. **Block-edge concentration.** Acquisition-block boundaries form long, perfectly
       straight runs. We report the fraction of predicted pixels lying on runs of >= 40 px
       that are straight to within one pixel.

    Returns a dict with the histogram, the axis-bin excess and a verdict.
    """
    mask = predicted_binary & footprint
    if mask.sum() < 100:
        return {"n_pixels": int(mask.sum()), "verdict": "INSUFFICIENT_PREDICTIONS"}
    f = mask.astype(np.float32)
    theta, coh = _structure_tensor_orientation_cached(f)
    w = coh[mask]
    th = theta[mask]
    n_bins = int(round(180.0 / bin_deg))
    hist, edges = np.histogram(th, bins=n_bins, range=(0.0, 180.0), weights=w)
    hist = hist / max(hist.sum(), 1e-9)
    # the two raster-axis bins: [0, bin_deg) and the bin containing 90 deg
    axis_bins = [0, int(round(90.0 / bin_deg)) % n_bins]
    axis_mass = float(sum(hist[b] for b in axis_bins))
    expected = 2.0 / n_bins
    excess = axis_mass / expected if expected > 0 else float("nan")

    # straight-run statistic along rows and columns
    def longest_straight_fraction(m: np.ndarray, axis: int) -> float:
        a = m if axis == 1 else m.T
        frac = 0.0
        runs = 0
        for row in range(a.shape[0]):
            v = a[row]
            i = 0
            while i < a.shape[1]:
                if v[i]:
                    j = i
                    while j + 1 < a.shape[1] and v[j + 1]:
                        j += 1
                    if j - i + 1 >= 40:
                        runs += 1
                    i = j + 1
                else:
                    i += 1
        total = int(m.sum())
        frac = (40.0 * runs) / max(total, 1)
        return float(min(frac, 1.0))

    row_frac = longest_straight_fraction(mask, axis=1)
    col_frac = longest_straight_fraction(mask, axis=0)
    verdict = "NO_GRID_ALIASING_SIGNATURE" if excess < 1.25 and max(row_frac, col_frac) < 0.02 else (
        "POSSIBLE_GRID_ALIASING" if excess < 1.6 else "STRONG_GRID_ALIGNMENT_SUSPECT_ACQUISITION_ARTIFACT"
    )
    return {
        "n_pixels": int(mask.sum()),
        "bin_deg": bin_deg,
        "orientation_histogram": [round(float(x), 5) for x in hist],
        "bin_edges_deg": [round(float(x), 2) for x in edges],
        "raster_axis_bins": axis_bins,
        "raster_axis_mass": round(axis_mass, 5),
        "expected_axis_mass_if_isotropic": round(expected, 5),
        "axis_excess_ratio": round(float(excess), 4),
        "long_straight_run_fraction_rows": round(row_frac, 5),
        "long_straight_run_fraction_cols": round(col_frac, 5),
        "verdict": verdict,
    }


_ST_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _structure_tensor_orientation_cached(f: np.ndarray, sigma: float = 1.5):
    key = f"{f.shape}:{sigma}"
    if key in _ST_CACHE:
        return _ST_CACHE[key]
    gy, gx = np.gradient(f)
    from scipy.ndimage import gaussian_filter

    jxx = gaussian_filter(gx * gx, sigma)
    jyy = gaussian_filter(gy * gy, sigma)
    jxy = gaussian_filter(gx * gy, sigma)
    theta = np.degrees(0.5 * np.arctan2(2.0 * jxy, (jxx - jyy) + 1e-12)) % 180.0
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    coh = np.clip(np.where(l1 + l2 > 1e-12, (l1 - l2) / (l1 + l2 + 1e-12), 0.0), 0.0, 1.0)
    _ST_CACHE.clear()
    _ST_CACHE[key] = (theta.astype(np.float32), coh.astype(np.float32))
    return _ST_CACHE[key]
