"""Pre-registered geological hypotheses H23-1..H23-5 (22GEMSDOE session, 2026-10-02).

Five NEW candidate geological hypotheses, each naming (a) the specific input layers,
(b) the physical signature / transform, (c) why it should catch a fault that is
*missing* from the USGS/INGENIOUS catalogue rather than one already in it, and
(d) how it differs from anything already implemented in this repository.

Every channel builder here is **label-free** (it never reads ``labels.tif``), so the
channels can be computed once on the full grid and reused by every spatially blocked
holdout fold without leakage.

Ranked by expected DTI gain per unit implementation cost (see
``docs/research/preregistration_h23.md``, committed before any result was measured):

  Rank 1  H23-1  Cross-Scale Orientation Persistence (self-affine fault trace test)
  Rank 2  H23-2  Range-Piedmont Junction Straightness (mountain-front sinuosity, Bull & McFadden 1977)
  Rank 3  H23-5  Aspect-Discontinuity / Hillslope-Fabric Worm (shutter-ridge & beveled-scarp memory)
  Rank 4  H23-4  Radiometric Alteration-Halo Anisotropy (K/Th/U ternary)
  Rank 5  H23-3  Tilt-Depth Edge-Depth Consistency (Salem et al. 2007/2008)

Official, free, verifiable sources for every method used here are listed in
``registry/sources.json`` (S42..S53) and reproduced in ``H23_SOURCES`` below.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import (
    convolve,
    gaussian_filter,
    maximum_filter,
    minimum_filter,
    uniform_filter,
)

H23_SOURCES: dict[str, list[str]] = {
    "H23-1": [
        "Candela, Renard, Klinger, Mair, Schmittbuhl & Brodsky 2012, J. Geophys. Res. 117(B08409) - "
        "anisotropic self-affine fault roughness, H_parallel=0.58+-0.07, H_perp=0.8 (doi 10.1029/2012JB009304)",
        "Renard, Voisin, Marsan & Schmittbuhl 2006, Geophys. Res. Lett. 33(L04301) - fault roughness self-affinity across scales (doi 10.1029/2005GL024948)",
        "Power, Tullis, Brown, Boitnott & Scholz 1987, J. Geophys. Res. 92(B8) - roughness power spectra of natural faults (doi 10.1029/JB092iB08p07897)",
    ],
    "H23-2": [
        "Bull, W.B. & McFadden, L.D. 1977, 'Tectonic geomorphology north and south of the Garlock fault', "
        "Proc. 8th Annual Geomorphology Symposium, SUNY Binghamton, pp. 115-138 - defines Smf = Lmf/Ls",
        "Wells, S.G., Bullard, T.F., Menges, C.M. et al. 1988, Geomorphology 1, 239-265 - regional Smf classes",
        "Silva, Goy, Zazo & Bardaji 2003, Geomorphology 50, 203-225 - Smf/Vf tectonic-activity classes (doi 10.1016/S0169-555X(02)00215-5)",
        "Keller, E.A. & Pinter, N. 1996, Active Tectonics: Earthquakes, Uplift and Landscape, Prentice Hall",
        "USGS 3DEP 1/3-arcsecond seamless DEM (the elevation source): https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt",
    ],
    "H23-5": [
        "Wallace, R.E. 1977, Geol. Soc. Am. Bull. 88, 890-900 - profiles and ages of young fault scarps, "
        "scarp degradation and hillslope-fabric persistence",
        "Hesse, R. 2010, Zeitschrift fur Geomorphologie 54(Suppl.2), 1-25 - Local Relief Model for lineament mapping (doi 10.1127/0372-8854/2010/0054S2-0057)",
        "Yokoyama, Shirasawa & Pike 2002, ISPRS J. Photogramm. Remote Sens. 57, 71-83 - topographic openness (doi 10.1016/S0924-2716(02)00104-9)",
    ],
    "H23-4": [
        "Glen, J.M.G. & Earney, T.E. 2024, 'GeoDAWN: Airborne Magnetic and Radiometric Surveys of the "
        "Northwestern Great Basin, Nevada and California', U.S. Geological Survey (doi 10.5066/P93LGLVQ)",
        "Shives, Charbonneau & Ford 2000, J. Geochem. Explor. 69-70, 119-124 - airborne gamma-ray K/Th/U "
        "alteration mapping for hydrothermal systems (doi 10.1016/S0375-6742(00)00129-0)",
        "Coolbaugh, Zehner, Kreemer, Blackwell, Oppliger, Wagoner & Warren-Smith 2005, Trans. Geothermal "
        "Resources Council 29 - regional-scale hydrothermal alteration in the Great Basin",
    ],
    "H23-3": [
        "Salem, Williams, Fairhead, Ravat & Smith 2007, The Leading Edge 26(12), 1502-1505 - "
        "tilt-depth method (doi 10.1190/1.2821934)",
        "Salem, Williams, Fairhead, Smith & Ravat 2008, Geophysics 73(1), L1-L10 - "
        "interpretation of magnetic data using tilt-angle derivatives (doi 10.1190/1.2799992)",
        "Miller & Singh 1994, J. Appl. Geophys. 32, 213-217 - potential-field tilt filters for edge detection (doi 10.1016/0926-9851(94)90022-1)",
        "Verduzco, Fairhead, Green & MacKenzie 2004, Geophysics 69, 435-439 - total horizontal derivative of the tilt angle (doi 10.1190/1.1707060)",
    ],
}


# --------------------------------------------------------------------------------------
# Shared low-level primitives
# --------------------------------------------------------------------------------------
def _dequant(band_u8: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Invert the 5GEMSDOE aux_bridge uint8 quantisation (q = 1 + round(254*clip((v-lo)/(hi-lo),0,1)))."""
    u = (band_u8.astype(np.float32) - 1.0) / 254.0
    return (lo + u * (hi - lo)).astype(np.float32)


def _grad_mag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a.astype(np.float32))
    return np.hypot(gy, gx).astype(np.float32)


def structure_tensor_orientation(field: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (orientation_deg in [0,180), coherence in [0,1]) of ``field`` at smoothing scale ``sigma``.

    Coherence = (l1 - l2) / (l1 + l2) of the 2x2 gradient structure tensor; 1 = perfectly
    anisotropic (a single dominant direction), 0 = isotropic.
    """
    f = gaussian_filter(field.astype(np.float32), sigma)
    gy, gx = np.gradient(f)
    jxx = gaussian_filter(gx * gx, sigma)
    jyy = gaussian_filter(gy * gy, sigma)
    jxy = gaussian_filter(gx * gy, sigma)
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    coherence = np.where(l1 + l2 > 1e-12, (l1 - l2) / (l1 + l2 + 1e-12), 0.0).astype(np.float32)
    # orientation of the *ridge* (edge) direction = perpendicular to the gradient
    theta = np.degrees(0.5 * np.arctan2(2.0 * jxy, (jxx - jyy) + 1e-12)) % 180.0
    return theta.astype(np.float32), np.clip(coherence, 0.0, 1.0).astype(np.float32)


def orientation_agreement(t1: np.ndarray, c1: np.ndarray, t2: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """cos^2 of the angle between two orientations (mod 180 deg), weighted by both coherences."""
    d = np.radians(np.abs(t1 - t2).astype(np.float32) % 180.0)
    d = np.minimum(d, np.pi - d)
    return (np.cos(d) ** 2 * np.sqrt(np.clip(c1, 0, 1) * np.clip(c2, 0, 1))).astype(np.float32)


def strike_worm(
    edge: np.ndarray,
    valid: np.ndarray,
    half_len: int = 7,
    flank: int = 3,
    std_win: int = 31,
) -> np.ndarray:
    """Multi-orientation strike-coherent line integral minus lateral flanks (H16-2 machinery, reused).

    Integrates a locally standardized ``edge`` field along 6 strike orientations over a
    ``(2*half_len+1)``-px baseline and subtracts the mean of two parallel flanking lines
    displaced ``+-flank`` px perpendicular to strike.
    """
    x = np.where(valid, np.nan_to_num(edge, nan=0.0), 0.0).astype(np.float32)
    loc_mean = uniform_filter(x, size=std_win)
    loc_sq = uniform_filter(x * x, size=std_win)
    loc_std = np.sqrt(np.maximum(loc_sq - loc_mean * loc_mean, 1e-6))
    z = np.clip((x - loc_mean) / (loc_std + 0.25), -3.0, 6.0)
    del loc_mean, loc_sq, loc_std

    best = np.full(x.shape, -1e9, dtype=np.float32)
    L = int(half_len)
    for deg in (0, 30, 60, 90, 120, 150):
        rad = np.radians(deg)
        dy_step, dx_step = np.sin(rad), np.cos(rad)
        py_step, px_step = -dx_step, dy_step
        n = 2 * L + 3
        k_mat = np.zeros((n, n), dtype=np.float32)
        c = L + 1
        for t in range(-L, L + 1):
            ry = int(round(c + t * dy_step))
            rx = int(round(c + t * dx_step))
            if 0 <= ry < n and 0 <= rx < n:
                k_mat[ry, rx] += 1.0 / (2 * L + 1)
            for sgn in (-1, 1):
                fy = int(round(c + t * dy_step + sgn * flank * py_step))
                fx = int(round(c + t * dx_step + sgn * flank * px_step))
                if 0 <= fy < n and 0 <= fx < n:
                    k_mat[fy, fx] -= 0.5 / (2 * L + 1)
        best = np.maximum(best, convolve(z, k_mat, mode="nearest"))
    return np.where(valid, np.maximum(best, 0.0), 0.0).astype(np.float32)


def line_integral_oriented(field: np.ndarray, deg: float, half_len: int, lat_off: int = 0) -> np.ndarray:
    """Mean of ``field`` along a straight segment of length ``2*half_len+1`` at orientation ``deg``,
    displaced laterally by ``lat_off`` px."""
    rad = np.radians(deg)
    dy, dx = np.sin(rad), np.cos(rad)
    py, px = -dx, dy
    n = 2 * half_len + 3
    k = np.zeros((n, n), dtype=np.float32)
    c = half_len + 1
    cnt = 0
    for t in range(-half_len, half_len + 1):
        ry = int(round(c + t * dy + lat_off * py))
        rx = int(round(c + t * dx + lat_off * px))
        if 0 <= ry < n and 0 <= rx < n:
            k[ry, rx] += 1.0
            cnt += 1
    if cnt:
        k /= float(cnt)
    return convolve(field.astype(np.float32), k, mode="nearest").astype(np.float32)


# --------------------------------------------------------------------------------------
# H23-1  Cross-scale orientation persistence (self-affine fault-trace test)
# --------------------------------------------------------------------------------------
def h23_1_cross_scale_orientation_persistence(
    scarp_1m: np.ndarray,
    scarp_10m: np.ndarray,
    geopot_100m: np.ndarray,
    footprint: np.ndarray,
    footprint_valid_1m: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """H23-1: a fault trace is self-affine -> its strike is scale-invariant; artifacts are not.

    Returns
    -------
    ``h23_orient_persist``  product of cos^2(dtheta) over the three cross-scale pairs,
                            weighted by the geometric mean of the two coherences.
    ``h23_orient_persist_12`` / ``_13`` / ``_23``  the individual pairwise terms.
    ``h23_hurst_roughness`` self-affinity proxy: H estimated from the growth of the
                            lateral root-mean-square deviation of the ridge field between
                            a short (2 px) and a long (8 px) baseline; scored with a Gaussian
                            centred on the published H ~ 0.6-0.8 for natural faults.
    """
    edge_fields = {
        "s1": np.where(footprint, np.nan_to_num(scarp_1m, nan=0.0), 0.0).astype(np.float32),
        "s2": np.where(footprint, np.nan_to_num(scarp_10m, nan=0.0), 0.0).astype(np.float32),
        "s3": np.where(footprint, np.nan_to_num(geopot_100m, nan=0.0), 0.0).astype(np.float32),
    }
    # Match the native smoothing of each scale: 1m-derived scarp is already 100 m cell, 10 m DEM
    # channels are cell-level, the 100 m geopotential field is intrinsically coarse.
    sigmas = {"s1": 1.0, "s2": 2.0, "s3": 3.0}
    th, co = {}, {}
    for k, f in edge_fields.items():
        th[k], co[k] = structure_tensor_orientation(f, sigmas[k])

    a12 = orientation_agreement(th["s1"], co["s1"], th["s2"], co["s2"])
    a13 = orientation_agreement(th["s1"], co["s1"], th["s3"], co["s3"])
    a23 = orientation_agreement(th["s2"], co["s2"], th["s3"], co["s3"])
    persist = np.clip(a12 * a13 * a23, 0.0, 1.0).astype(np.float32)

    # --- self-affine roughness proxy -------------------------------------------------
    # Lateral RMS deviation of the ridge field from a straight chord, measured at two baselines.
    base = np.where(footprint, np.nan_to_num(scarp_1m, nan=0.0), 0.0).astype(np.float32)
    if footprint_valid_1m is not None:
        base = base * footprint_valid_1m.astype(np.float32)
    base_n = base / (base.mean() + 1e-6)
    dev_short = _lateral_rms_deviation(base_n, half_len=2)
    dev_long = _lateral_rms_deviation(base_n, half_len=8)
    ratio = (dev_long + 1e-3) / (dev_short + 1e-3)
    # RMS ~ L^H  =>  H = log(dev_long/dev_short) / log(8/2)
    H = np.clip(np.log(np.maximum(ratio, 1e-3)) / np.log(4.0), 0.0, 2.0).astype(np.float32)
    hurst_score = np.exp(-(((H - 0.7) / 0.40) ** 2)).astype(np.float32)  # peak at published H~0.7

    return {
        "h23_orient_persist": persist,
        "h23_orient_persist_1m_10m": a12,
        "h23_orient_persist_1m_geo": a13,
        "h23_orient_persist_10m_geo": a23,
        "h23_hurst_H": H,
        "h23_hurst_roughness": hurst_score,
    }


def _lateral_rms_deviation(field: np.ndarray, half_len: int) -> np.ndarray:
    """RMS over t of the field sampled at lateral offset +-1..2 px around the best straight chord.

    A perfectly straight ridge at the pixel centre has (nearly) zero lateral deviation; a
    wandering/acquisition-artifact ridge has a large one.  Cheap surrogate for the full
    RMS(L) ~ L^H analysis of Candela et al. (2012).
    """
    best = np.zeros(field.shape, dtype=np.float32)
    for deg in (0, 45, 90, 135):
        centre = line_integral_oriented(field, deg, half_len, lat_off=0)
        off = np.maximum(
            line_integral_oriented(field, deg, half_len, lat_off=2),
            line_integral_oriented(field, deg, half_len, lat_off=-2),
        )
        best = np.maximum(best, np.abs(off - centre))
    return best.astype(np.float32)


# --------------------------------------------------------------------------------------
# H23-2  Range-piedmont junction straightness (mountain-front sinuosity)
# --------------------------------------------------------------------------------------
def h23_2_mountain_front_straightness(
    dem_mean: np.ndarray,
    footprint: np.ndarray,
    half_len: int = 10,
    max_lat: int = 3,
    relief_win: int = 21,
) -> dict[str, np.ndarray]:
    """H23-2: Smf-like straightness of the range-piedmont junction.

    ``straightness`` = max over 8 orientations of
        [line integral of |grad E| along the chord at lateral offset 0] /
        [max over lateral offsets +-1..max_lat of the same integral]

    A front held straight by an active range-bounding fault gives straightness -> 1; an
    eroded/inactive front wanders by several pixels and gives a small value.  (Bull &
    McFadden 1977: Smf = Lmf/Ls, with active fronts Smf < 1.4; here straightness ~ 1/Smf
    evaluated in a sliding window instead of along a hand-digitised front.)
    """
    E = np.where(footprint, np.nan_to_num(dem_mean, nan=0.0), 0.0).astype(np.float32)
    G = _grad_mag(E)
    Gn = G / (G.mean() + 1e-6)

    best_straight = np.zeros(G.shape, dtype=np.float32)
    best_chord = np.zeros(G.shape, dtype=np.float32)
    best_corridor = np.zeros(G.shape, dtype=np.float32)
    for deg in (0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
        chord = line_integral_oriented(Gn, deg, half_len, lat_off=0)
        corridor = chord.copy()
        for s in range(1, max_lat + 1):
            corridor = np.maximum(corridor, line_integral_oriented(Gn, deg, half_len, lat_off=s))
            corridor = np.maximum(corridor, line_integral_oriented(Gn, deg, half_len, lat_off=-s))
        straight = chord / (corridor + 1e-6)
        best_straight = np.maximum(best_straight, straight)
        best_chord = np.maximum(best_chord, chord)
        best_corridor = np.maximum(best_corridor, corridor)

    relief = (maximum_filter(E, size=relief_win) - minimum_filter(E, size=relief_win)).astype(np.float32)
    relief_n = relief / (np.percentile(relief[footprint], 95) + 1e-6)

    smf_score = np.clip(best_straight * np.clip(relief_n, 0.0, 1.0), 0.0, 1.0).astype(np.float32)
    return {
        "h23_front_straightness": np.clip(best_straight, 0.0, 1.0).astype(np.float32),
        "h23_front_chord": best_chord,
        "h23_front_relief_n": np.clip(relief_n, 0.0, 1.0).astype(np.float32),
        "h23_smf_score": smf_score,
    }


# --------------------------------------------------------------------------------------
# H23-5  Aspect-discontinuity / hillslope-fabric worm (beveled-scarp memory)
# --------------------------------------------------------------------------------------
def h23_5_aspect_discontinuity_worm(
    aspect_coherence: np.ndarray,
    slope_std: np.ndarray,
    hs_lineament: np.ndarray,
    footprint: np.ndarray,
) -> dict[str, np.ndarray]:
    """H23-5: hillslope-aspect fabric discontinuity is the longest-lived fault signature.

    Where a scarp has been beveled by pedimentation the *height* step is gone but the
    two blocks still dip in opposite directions, so aspect coherence has a local minimum
    along a line while slope variability stays high.  We run the H16-2 strike-worm
    transform over that discontinuity field - something the repo has never done (the worm
    machinery has only ever been applied to tmi_hg, grav_slope, elev_slope and the scarp index).
    """
    coh = np.where(footprint, np.clip(np.nan_to_num(aspect_coherence, nan=0.0), 0.0, 1.0), 0.0).astype(np.float32)
    ss = np.where(footprint, np.nan_to_num(slope_std, nan=0.0), 0.0).astype(np.float32)
    ss_n = ss / (np.percentile(ss[footprint], 95) + 1e-6)
    disc = (1.0 - coh) * np.clip(ss_n, 0.0, 1.0)
    worm = strike_worm(disc, footprint, half_len=7, flank=3)
    hs = np.where(footprint, np.nan_to_num(hs_lineament, nan=0.0), 0.0).astype(np.float32)
    hs_worm = strike_worm(hs, footprint, half_len=7, flank=3)
    return {
        "h23_aspect_disc": disc.astype(np.float32),
        "h23_aspect_disc_worm": worm,
        "h23_hs_lineament_worm": hs_worm,
    }


# --------------------------------------------------------------------------------------
# H23-4  Radiometric alteration-halo anisotropy
# --------------------------------------------------------------------------------------
def h23_4_radiometric_anisotropy(
    rad_bands: dict[str, np.ndarray],
    elev_slope: np.ndarray,
    footprint: np.ndarray,
    trend_win: int = 21,
) -> dict[str, np.ndarray]:
    """H23-4: hydrothermal alteration along a *blind* fault is an *anisotropic* K/U/Th stripe.

    ``rad_bands`` must contain ``rad_k``, ``rad_th``, ``rad_u`` (de-quantised).  We form the
    classic hydrothermal-alteration ratios K/Th, U/Th and U/K (Shives et al. 2000), remove the
    regional trend, and then measure the *anisotropy of the residual anomaly* with a structure
    tensor.  An isotropic lithologic or playa-evaporite potassium anomaly has anisotropy ~ 0;
    a fluid pathway along a fault has anisotropy -> 1 with a stable, straight principal axis.
    """
    non_playa = ((elev_slope > 1.5) & footprint).astype(np.float32)
    k = np.nan_to_num(rad_bands["rad_k"], nan=0.0).astype(np.float32)
    th = np.nan_to_num(rad_bands["rad_th"], nan=0.0).astype(np.float32)
    u = np.nan_to_num(rad_bands["rad_u"], nan=0.0).astype(np.float32)
    eps = 1e-3
    ratios = {
        "kth": (k + eps) / (th + eps),
        "uth": (u + eps) / (th + eps),
        "uk": (u + eps) / (k + eps),
    }
    out: dict[str, np.ndarray] = {}
    aniso_acc = np.zeros(k.shape, dtype=np.float32)
    for name, r in ratios.items():
        anom = (r - uniform_filter(r, size=trend_win)) * non_playa
        std = np.std(anom[footprint]) + 1e-6
        anom_n = anom / std
        mag = np.abs(anom_n)
        _t, coh = structure_tensor_orientation(np.abs(anom_n), 2.0)
        score = np.clip(mag, 0.0, 4.0) * coh
        out[f"h23_rad_{name}_anom"] = anom.astype(np.float32)
        out[f"h23_rad_{name}_aniso"] = (mag * coh).astype(np.float32)
        aniso_acc = np.maximum(aniso_acc, score.astype(np.float32))
    out["h23_rad_anisotropy"] = np.clip(aniso_acc / 2.0, 0.0, 1.0).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# H23-3  Tilt-depth edge-depth consistency
# --------------------------------------------------------------------------------------
def h23_3_tilt_depth_consistency(
    field: np.ndarray,
    vgrad: np.ndarray,
    footprint: np.ndarray,
    smooth_sigma: float = 1.0,
    std_win: int = 15,
) -> dict[str, np.ndarray]:
    """H23-3: tilt-depth edge detection + along-strike *depth consistency* test.

    TDR  = atan2(dT/dz, |grad_h T|)                       (Miller & Singh 1994)
    THDR = |grad_h TDR|                                   (Verduzco et al. 2004)
    Salem et al. (2008) note THDR is the reciprocal of the depth to the contact, so
    z_est proportional to 1/THDR.  A real basement contact yields a *smoothly varying*
    z along strike; flight-line levelling errors, cultural edges and gridding seams
    yield an erratic z.  We therefore score an edge by the local *stability* of log z.
    """
    T = np.where(footprint, np.nan_to_num(field, nan=0.0), 0.0).astype(np.float32)
    Tz = np.where(footprint, np.nan_to_num(vgrad, nan=0.0), 0.0).astype(np.float32)
    if smooth_sigma > 0:
        T = gaussian_filter(T, smooth_sigma)
        Tz = gaussian_filter(Tz, smooth_sigma)
    gy, gx = np.gradient(T)
    gh = np.hypot(gy, gx)
    tdr = np.arctan2(Tz, gh + 1e-9).astype(np.float32)
    thdr = _grad_mag(tdr)

    z_est = 1.0 / (thdr + 1e-3)
    lz = np.log(np.maximum(z_est, 1e-6)).astype(np.float32)
    mean_lz = uniform_filter(lz, size=std_win)
    sq_lz = uniform_filter(lz * lz, size=std_win)
    sd_lz = np.sqrt(np.maximum(sq_lz - mean_lz * mean_lz, 0.0)).astype(np.float32)
    # 1.0 = perfectly depth-consistent edge; ->0 for erratic, artifact-like edges
    coherence = np.exp(-sd_lz / 0.75).astype(np.float32)
    edge_amp = np.exp(-np.abs(tdr) / 0.35).astype(np.float32)  # TDR ~ 0 over a contact
    return {
        "h23_tdr_edge": edge_amp,
        "h23_thdr": thdr.astype(np.float32),
        "h23_depth_sd_logz": sd_lz,
        "h23_depth_coherence": coherence,
        "h23_depth_coherence_edge": (coherence * edge_amp).astype(np.float32),
    }


# --------------------------------------------------------------------------------------
# Emission policy (decision-theoretic; orthogonal to the geological channels)
# --------------------------------------------------------------------------------------
def thinning_allocation(score: np.ndarray, footprint: np.ndarray, pool_frac: float, sigma: float = 1.0) -> np.ndarray:
    """Take the top ``pool_frac`` of the footprint as a candidate pool, then thin to 1-px ridges.

    The GEMSDOE10 blocked-holdout sweep (``reports/budget_density_sweep.json``, arm H20) showed
    that at a fixed emitted-pixel count a *thinned* allocation (``thinXX``) beats a raw
    top-k allocation (``topkXX``) by up to +0.024 DTI, because the DTI true-positive credit is
    a MAX over the 300 m kernel: several pixels packed inside one kernel each pay false-positive
    cost but only the best one is credited.
    """
    from .metric import ridge_nms

    n_fp = int(footprint.sum())
    k = int(round(pool_frac * n_fp))
    flat = np.where(footprint, np.nan_to_num(score, nan=-1e9), -1e9).astype(np.float32).ravel()
    idx = np.flatnonzero(footprint.ravel())
    top = idx[np.argpartition(flat[idx], -k)[-k:]]
    pool = np.zeros(footprint.shape, dtype=bool)
    pool.ravel()[top] = True
    ridge = ridge_nms(np.where(pool, np.nan_to_num(score, nan=0.0), 0.0).astype(np.float32), pool, sigma=sigma)
    out = pool & ridge
    # Never return fewer than the ridge themselves: if the ridge is empty (degenerate) fall back to the pool.
    if out.sum() == 0:
        return pool
    return out


def marginal_inclusion_rule(expected_hit: np.ndarray, dti: float, alpha: float = 0.2) -> np.ndarray:
    """Bang-bang emission rule derived from the official DTI.

    Adding a pixel with expected true-positive credit ``g`` and false-positive weight ``f``
    improves DTI iff  g > alpha * DTI * (g + f).  With ``f ~ 1`` for a pixel farther than the
    300 m kernel radius this reduces to  E[credit] > alpha * DTI (see
    ``docs/research/preregistration_h23.md`` section E0).
    """
    return (expected_hit > alpha * dti).astype(np.float32)


HYPOTHESIS_H23_ORDER = ["H23-1", "H23-2", "H23-5", "H23-4", "H23-3"]


def spec_table() -> list[dict[str, Any]]:
    """Machine-readable hypothesis register (mirrors ``docs/research/hypothesis_register.md``)."""
    return [
        {
            "id": "H23-1",
            "rank": 1,
            "name": "Cross-Scale Orientation Persistence & Self-Affine Roughness Test",
            "layers": [
                "1m USGS 3DEP lidar scarp stack (lidar_scarp_features_u8, 12 ch) resampled to the 100 m grid",
                "10m USGS 3DEP seamless DEM scarp channels (dem10_*, 13 ch)",
                "100m GeoDAWN tilt/curvature band 'tc' (training_features.tif band 6)",
            ],
            "physical_signature": (
                "Structure-tensor principal orientation of each of the three edge fields, then the product of "
                "cos^2(dtheta) over the three cross-scale pairs weighted by both coherences; plus a self-affinity "
                "proxy H fitted from the growth of the lateral RMS deviation of the ridge between a 2 px and an "
                "8 px baseline, scored against the published H = 0.6-0.8 for natural fault surfaces."
            ),
            "why_unmapped_not_catalogued": (
                "Compilers drew the faults they could see as continuous scarps on 1:100k imagery. A short "
                "(<2 km) fault has exactly the same self-affine trace geometry, only less length, so it fails a "
                "length-based compilation rule while passing a scale-invariance one. Acquisition artifacts "
                "(flight-line seams, DEM tile edges, hillshade striping) are NOT self-affine: they appear at one "
                "scale with a fixed grid-aligned strike and vanish at the others."
            ),
            "differs_from_repo": (
                "seamfree_scarp_index fuses amplitudes across scales but never tests orientation agreement or "
                "roughness scaling; ridge_nms enforces only local ridge continuity; compute_strike_worm uses fixed "
                "6 orientations on a single field. No prior GEMSDOE module computes a Hurst/self-affinity statistic "
                "or cross-scale orientation coherence."
            ),
            "expected_dti_gain": "medium (multiplicative prior on the dominant topographic arm)",
            "implementation_cost": "medium (3 structure tensors + 4 line-integral orientations per baseline)",
            "sources": H23_SOURCES["H23-1"],
        },
        {
            "id": "H23-2",
            "rank": 2,
            "name": "Range-Piedmont Junction Straightness (Smf-like, Bull & McFadden 1977)",
            "layers": [
                "topo_u8 band 9 'dem_mean' (100 m elevation, de-quantised from the USGS 3DEP 1/3-arcsec seamless DEM VRT)",
                "topo_u8 band 5 'relief_local'",
                "10m DEM relief channels (dem10_slope_std, dem10_hgm200_mean)",
            ],
            "physical_signature": (
                "Windowed mountain-front straightness: max over 8 orientations of "
                "[chord line-integral of |grad E| over +-10 px] / [max over lateral offsets +-1..3 px of the same "
                "integral] x normalised local relief. Straightness -> 1 for a front held straight by active "
                "displacement (Bull & McFadden Smf < 1.4); small for an eroded, inactive front."
            ),
            "why_unmapped_not_catalogued": (
                "In the Great Basin many range fronts are straight but unmapped: the scarp is buried by late "
                "Quaternary alluvium or the front is partly erosional, so air-photo compilers drew nothing. "
                "Smf was designed precisely to flag a fault-generated mountain front BEFORE the fault is mapped. "
                "Where the scarp is obvious the fault is already in the catalogue and Smf adds nothing, so Smf's "
                "incremental information is concentrated in the unmapped set."
            ),
            "differs_from_repo": (
                "No prior GEMSDOE arm computes any morphotectonic index (Smf, Vf, SL, AF). Every topographic arm "
                "is a local scarp/curvature detector (lid1m_*, dem10_*, openness/LRM) that responds to the scarp "
                "face itself; Smf responds to the 2-8 km plan-view geometry of the range front - an orthogonal, "
                "much longer-wavelength signal. It is also the first arm to use the topo_u8 'dem_mean' elevation."
            ),
            "expected_dti_gain": "unknown / holdout-limited (see limitations)",
            "implementation_cost": "medium (8 orientations x 7 lateral offsets of line integrals)",
            "sources": H23_SOURCES["H23-2"],
        },
        {
            "id": "H23-5",
            "rank": 3,
            "name": "Hillslope-Aspect Fabric Discontinuity Worm (beveled-scarp memory)",
            "layers": [
                "topo_u8 band 7 'aspect_coherence' (new to this repo)",
                "dem10 'slope_std' / topo_u8 band 4 'slope_std'",
                "topo_u8 band 8 'hs_lineament'",
            ],
            "physical_signature": (
                "discontinuity = (1 - aspect_coherence) x normalised slope_std, passed through the H16-2 "
                "multi-orientation strike-coherent line integral minus lateral flanks (half_len=7, flank=3). "
                "A fault produces a line along which hillslope aspect flips while slope variability stays high."
            ),
            "why_unmapped_not_catalogued": (
                "Where a scarp has been beveled by pedimentation the height step is gone but the two blocks still "
                "dip oppositely; the aspect-fabric discontinuity persists for 10^5-10^6 yr, far longer than the "
                "scarp itself (Wallace 1977 degradation). So this signature survives exactly where the scarp - "
                "the thing compilers mapped - has disappeared, and it is invisible to amplitude-only scarp detectors."
            ),
            "differs_from_repo": (
                "aspect_coherence and hs_lineament (topo_u8) are not used anywhere in the repository. The strike-worm "
                "machinery has only ever been applied to tmi_hg, grav_slope, elev_slope and the seam-free scarp index - "
                "never to an aspect/hillslope-fabric field."
            ),
            "expected_dti_gain": "low-medium (new independent topographic channel)",
            "implementation_cost": "low (reuses the existing worm transform on two new channels)",
            "sources": H23_SOURCES["H23-5"],
        },
        {
            "id": "H23-4",
            "rank": 4,
            "name": "Radiometric Alteration-Halo Anisotropy (K/Th/U ternary)",
            "layers": [
                "radiometric_u8 7-band stack (rad_k, rad_th, rad_u, rad_tc, rad_thk, rad_uk, rad_uth) - new to this repo",
                "training_features.tif band 19 'det_elev_slope' (playa gate)",
            ],
            "physical_signature": (
                "Hydrothermal ratios K/Th, U/Th, U/K (Shives et al. 2000) after removing the 2.1 km regional trend, "
                "gated to non-playa ground; then the structure-tensor ANISOTROPY (l1-l2)/(l1+l2) of the residual "
                "magnitude. Score = |anomaly| x anisotropy."
            ),
            "why_unmapped_not_catalogued": (
                "A blind fault with active fluid flow leaves an alteration stripe in the airborne radiometrics while "
                "showing no scarp at all, so compilers mapping from imagery see nothing. The anisotropy test is what "
                "separates a structurally controlled stripe from the ISOTROPIC potassium anomaly of a playa or of a "
                "lithologic contact, which follows geology rather than a straight line."
            ),
            "differs_from_repo": (
                "H16-4/H19-2 use a scalar 'hydro_k_th_anom' with isotropic 2.1 km trend removal and a non_playa gate; "
                "no prior arm measures the ANISOTROPY/orientation of the radiometric anomaly, and none uses the U/Th "
                "or U/K ratios or the 7-band stack (the repo only has the 4-band geodawn_rad_u8)."
            ),
            "expected_dti_gain": "low (geochemical family has historically been weak: H16-4 = 0.15996 dense)",
            "implementation_cost": "low (7-band stack already downloaded)",
            "sources": H23_SOURCES["H23-4"],
        },
        {
            "id": "H23-3",
            "rank": 5,
            "name": "Tilt-Depth Edge-Depth Consistency Filter on GeoDAWN Potential Fields",
            "layers": [
                "training_features.tif: iso_grav_anom (13), iso_grav_anom_hg (18), iso_grav_anom_vg (11); "
                "tmi (14), tmi_hg (3), tmi_vg (9)",
                "geodawn_extensions_u8 upward-continued magnetic band (persistence check)",
            ],
            "physical_signature": (
                "TDR = atan2(dT/dz, |grad_h T|); THDR = |grad_h TDR|; depth proxy z ~ 1/THDR (Salem et al. 2007, 2008). "
                "Score = exp(-sd_local(log z) / 0.75) x exp(-|TDR| / 0.35): an edge is only credited if the estimated "
                "source depth varies SMOOTHLY along strike over a 1.5 km window."
            ),
            "why_unmapped_not_catalogued": (
                "A buried basement fault with no surface scarp still produces a coherent geopotential contact at a "
                "stable estimated depth, but precisely because it lacks a scarp it was never drawn. Flight-line "
                "levelling artifacts, cultural edges and gridding seams also produce edges - but their estimated depth "
                "is erratic. The consistency filter therefore removes exactly the false positives that make the "
                "geopotential-only arm (H16-2, dense DTI 0.16266) fail."
            ),
            "differs_from_repo": (
                "H16-2 uses a fixed 6-orientation strike-coherent line integral on tmi_hg/grav_slope plus an "
                "upward-continuation persistence difference; it never computes the tilt angle, THDR, or an estimated "
                "source depth, and never applies a depth-variance consistency test. Band 6 'tc' is a supplied tilt band "
                "but is only used as a raw feature."
            ),
            "expected_dti_gain": "low (geophysical family historically weakest on the holdout)",
            "implementation_cost": "low (finite-difference derivatives on bands already loaded)",
            "sources": H23_SOURCES["H23-3"],
        },
    ]
