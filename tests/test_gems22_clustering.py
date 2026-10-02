"""Bour & Davy (1999) and Marrett et al. (2018) statistic implementations."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import clustering as cl


def _synthetic_traces(n=200, seed=0, lmin=300.0, lmax=20000.0, a=2.0):
    """Trace population whose CUMULATIVE count follows N(>=L) = C*L^-a.

    `fit_length_frequency` fits the cumulative form, which is the convention used
    by Bour & Davy (1999) ("the exponent a of the frequency length distribution")
    and by the field generally; the corresponding DENSITY is proportional to
    L^-(a+1).  Inverse-CDF sampling of the cumulative law gives
    L = lmin * (1-u)^(-1/a).
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    L = lmin * (1 - u) ** (-1.0 / a)
    L = np.clip(L, lmin, lmax)
    rows = rng.uniform(0, 400, n); cols = rng.uniform(0, 400, n)
    return L, rows, cols


def test_length_frequency_recovers_the_injected_exponent():
    L, r, c = _synthetic_traces(4000, seed=1, a=2.0, lmin=300, lmax=300000)
    f = cl.fit_length_frequency(L, 400.0, n_bins=18)
    assert abs(f["a"] - 2.0) < 0.30, f
    assert f["r2"] > 0.90


def test_nearest_larger_neighbour_is_positive_and_increasing():
    """Bour & Davy: 'large faults have their nearest neighbor located at greater
    distances than small faults' -- so <d(l)> must increase with l."""
    L, r, c = _synthetic_traces(1200, seed=2)
    t = cl.Traces(length_px=L / 100.0, centroid_row=r, centroid_col=c,
                  label_ids=np.arange(1, len(L) + 1), n=len(L))
    Ls, d = cl.nearest_larger_neighbour(t, 100.0)
    ok = np.isfinite(d)
    fit = cl.fit_nearest_larger(Ls, d, n_bins=8, min_per_bin=20)
    assert np.isfinite(fit["x"])
    assert fit["x"] > 0, "x must be positive: bigger faults are further from the next bigger one"
    small = Ls[ok] < np.percentile(Ls[ok], 25)
    large = Ls[ok] > np.percentile(Ls[ok], 75)
    assert np.mean(d[ok][large]) > np.mean(d[ok][small])


def test_extract_traces_counts_and_lengths():
    m = np.zeros((60, 60), bool)
    m[10, 10:30] = True          # 20 px horizontal trace
    m[40:55, 40] = True          # 15 px vertical trace
    t = cl.extract_traces(m, pixel_m=100.0, skeleton=True)
    assert t.n == 2
    L = sorted(cl.length_metres(t).tolist())
    assert L[0] == pytest.approx(1500.0, rel=0.01)
    assert L[1] == pytest.approx(2000.0, rel=0.01)


def test_correlation_dimension_of_a_uniform_set_is_about_two():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 500, (4000, 2))
    D = cl.correlation_dimension(pts, np.array([2, 5, 10, 20, 40, 80.0]))
    assert 1.8 < D["D"] < 2.2, D


def test_correlation_dimension_of_a_line_is_about_one():
    x = np.linspace(0, 500, 3000)
    pts = np.vstack([x, np.zeros_like(x)]).T
    D = cl.correlation_dimension(pts, np.array([2, 5, 10, 20, 40.0]))
    assert 0.85 < D["D"] < 1.2, D


def test_ncc_detects_clustering_and_regularity():
    rng = np.random.default_rng(4)
    lags = np.arange(0.5, 40.5, 1.0)
    # clustered: points drawn from tight Gaussian clumps
    centres = rng.uniform(0, 1000, 25)
    clus = np.concatenate([rng.normal(c, 3.0, 40) for c in centres])
    clus = clus[(clus >= 0) & (clus <= 1000)]
    rc = cl.ncc_1d(clus, 1000.0, lags, n_random=200, seed=5)
    vc = cl.classify_arrangement(np.array(rc["ncc"]), np.array(rc["lag_centres"]), 2, 20)
    assert vc["mean_ncc"] > 1.15, vc
    assert vc["verdict"] == "clustered"
    # regular: evenly spaced
    reg = np.arange(5, 1000, 12.0)
    rr = cl.ncc_1d(reg, 1000.0, lags, n_random=200, seed=5)
    vr = cl.classify_arrangement(np.array(rr["ncc"]), np.array(rr["lag_centres"]), 2, 20)
    assert vr["mean_ncc"] < 0.85, vr
    assert vr["verdict"] == "regular"
    # random: uniform
    rnd = rng.uniform(0, 1000, 500)
    rn = cl.ncc_1d(rnd, 1000.0, lags, n_random=200, seed=5)
    vr2 = cl.classify_arrangement(np.array(rn["ncc"]), np.array(rn["lag_centres"]), 2, 20)
    assert 0.75 < vr2["mean_ncc"] < 1.3, vr2


def test_bour_davy_prior_is_bounded_and_localised():
    m = np.zeros((300, 300), bool)
    m[150, 60:240] = True
    m[40:90, 40] = True
    lab, n = __import__("scipy.ndimage", fromlist=["label"]).label(m, structure=cl.STRUCT8)
    t = cl.extract_traces(m)
    fit = {"x": 0.82, "A": 6.27}
    p = cl.bour_davy_prior_field(lab, t, fit, pixel_m=100.0, top_k=5, r_max_px=40)
    assert p.shape == m.shape
    assert np.isfinite(p).all()
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert (p > 0).sum() > 0, "prior must be non-degenerate"
    # the prior must vanish ON the known traces (they are already mapped)
    assert p[m].max() == 0.0


def test_bour_davy_prior_degrades_gracefully_on_a_bad_fit():
    m = np.zeros((80, 80), bool); m[40, 20:60] = True
    from scipy import ndimage
    lab, n = ndimage.label(m, structure=cl.STRUCT8)
    t = cl.extract_traces(m)
    for bad in ({"x": float("nan"), "A": 1.0}, {"x": -1.0, "A": 1.0}, {"x": 0.8, "A": 0.0}, {}):
        p = cl.bour_davy_prior_field(lab, t, bad)
        assert p.shape == m.shape and (p == 0).all()
