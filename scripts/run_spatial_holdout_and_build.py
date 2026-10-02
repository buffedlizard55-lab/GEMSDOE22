"""Run the four-quadrant spatial holdout for the existing H16 hypotheses.

This script evaluates transfer to held-out portions of the known catalogue and a
synthetic sparse-component proxy. It does not evaluate hidden leaderboard labels,
calibrate a private score, or spend a DrivenData submission slot.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import rasterio
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    label as ndi_label,
    maximum_filter,
    uniform_filter,
)
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.paths import AUDIT_CLONES, DATA_DIR, EVIDENCE_DIR  # noqa: E402
from gems.metric import dti_score_fast, ridge_nms  # noqa: E402

SEED = 20260929


def dequant_lidar(band_name: str, q_arr: np.ndarray, q_meta: dict) -> np.ndarray:
    xmax, mode = q_meta[band_name]
    u = np.clip((q_arr.astype(np.float32) - 1.0) / 254.0, 0.0, 1.0)
    if mode == "sqrt":
        val = (u ** 2) * float(xmax)
    else:
        val = u * float(xmax)
    return np.where(q_arr > 0, val, 0.0).astype(np.float32)


def compute_strike_worm_feature(
    edge_2d: np.ndarray, valid_2d: np.ndarray, half_len: int = 7, flank: int = 3
) -> np.ndarray:
    """H16-2: Multi-orientation strike-coherent line integral minus lateral flanks.

    Integrates normalized geopotential/scarp horizontal gradient along 6 strike orientations
    (0, 30, 60, 90, 120, 150 deg) over a (2*half_len+1)-px baseline and subtracts the mean
    of two parallel flanking lines displaced +-flank px perpendicular to strike.
    """
    from scipy.ndimage import convolve

    x = np.where(valid_2d, np.nan_to_num(edge_2d, nan=0.0), 0.0).astype(np.float32)
    # Local standardization (3 km window) so regional amplitude differences don't dominate
    loc_mean = uniform_filter(x, size=31)
    loc_sq = uniform_filter(x * x, size=31)
    loc_std = np.sqrt(np.maximum(loc_sq - loc_mean * loc_mean, 1e-6))
    z = np.clip((x - loc_mean) / (loc_std + 0.25), -3.0, 6.0)
    del loc_mean, loc_sq, loc_std

    best = np.full(x.shape, -1e9, dtype=np.float32)
    L = int(half_len)
    flank = int(flank)
    for deg in (0, 30, 60, 90, 120, 150):
        rad = np.radians(deg)
        dy_step = np.sin(rad)
        dx_step = np.cos(rad)
        py_step = -dx_step
        px_step = dy_step
        k_mat = np.zeros((2 * L + 3, 2 * L + 3), dtype=np.float32)
        c = L + 1
        for t in range(-L, L + 1):
            ry = int(round(c + t * dy_step))
            rx = int(round(c + t * dx_step))
            if 0 <= ry < k_mat.shape[0] and 0 <= rx < k_mat.shape[1]:
                k_mat[ry, rx] += 1.0 / (2 * L + 1)
            for sgn in (-1, 1):
                fy = int(round(c + t * dy_step + sgn * flank * py_step))
                fx = int(round(c + t * dx_step + sgn * flank * px_step))
                if 0 <= fy < k_mat.shape[0] and 0 <= fx < k_mat.shape[1]:
                    k_mat[fy, fx] -= 0.5 / (2 * L + 1)
        resp = convolve(z, k_mat, mode="nearest")
        best = np.maximum(best, resp)
    return np.where(valid_2d, np.maximum(best, 0.0), 0.0).astype(np.float32)


def build_feature_matrix_on_footprint(footprint: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build footprint-indexed (N=5,167,373) feature vectors for all 5 hypotheses."""
    t0 = time.time()
    H, W = footprint.shape
    fp_idx = np.flatnonzero(footprint.ravel())
    cache_path = DATA_DIR / "cache" / "features_fp.npz"
    if cache_path.exists():
        print(f"  [Cache] Loading precomputed footprint features from {cache_path}...")
        loaded = np.load(cache_path)
        return {k: loaded[k] for k in loaded.files}, fp_idx
    feats: dict[str, np.ndarray] = {}

    def to_fp(arr2d: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr2d.ravel()[fp_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    print("  [Features 1/4] Loading & de-regionalizing 19 GeoDAWN bands + Strike Worms (H16-2, H16-5)...")
    # Band indices (1-based in training_features.tif):
    # 1:mag_anom, 2:rtp, 3:tmi_hg, 4:geod_2ndinv, 5:iso_grav_anom_slope, 6:tc,
    # 7:geod_shearrate, 8:geod_dilaterate, 9:tmi_vg, 10:deq_n100a15, 11:iso_grav_anom_vg,
    # 12:det_elev, 13:iso_grav_anom, 14:tmi, 15:depth_to_base_surf, 16:ieq_n100a15,
    # 17:cond_surf, 18:iso_grav_anom_hg, 19:det_elev_slope
    with rasterio.open(DATA_DIR / "training_features.tif") as src:
        b_rtp = src.read(2).astype(np.float32)
        b_tmi_hg = src.read(3).astype(np.float32)
        b_geod_2nd = src.read(4).astype(np.float32)
        b_grav_slope = src.read(5).astype(np.float32)
        b_tc = src.read(6).astype(np.float32)
        b_geod_shear = src.read(7).astype(np.float32)
        b_geod_dil = src.read(8).astype(np.float32)
        b_tmi_vg = src.read(9).astype(np.float32)
        b_grav_vg = src.read(11).astype(np.float32)
        b_det_elev = src.read(12).astype(np.float32)
        b_grav = src.read(13).astype(np.float32)
        b_tmi = src.read(14).astype(np.float32)
        b_depth_base = src.read(15).astype(np.float32)
        b_ieq = src.read(16).astype(np.float32)
        b_cond = src.read(17).astype(np.float32)
        b_grav_hg = src.read(18).astype(np.float32)
        b_elev_slope = src.read(19).astype(np.float32)

    for arr in (
        b_rtp, b_tmi_hg, b_geod_2nd, b_grav_slope, b_tc, b_geod_shear, b_geod_dil,
        b_tmi_vg, b_grav_vg, b_det_elev, b_grav, b_tmi, b_depth_base, b_ieq,
        b_cond, b_grav_hg, b_elev_slope,
    ):
        arr[(arr < -1e20) | ~np.isfinite(arr)] = 0.0

    # Local derivative / de-regionalized baseline bands (cannot memorize absolute basin ID!)
    feats["tmi_hg"] = np.log1p(np.maximum(to_fp(b_tmi_hg), 0.0))
    feats["tmi_vg"] = to_fp(b_tmi_vg)
    feats["tc"] = to_fp(b_tc)
    feats["grav_slope"] = to_fp(b_grav_slope)
    feats["grav_vg"] = to_fp(b_grav_vg)
    feats["grav_hg_abs"] = np.abs(to_fp(b_grav_hg))
    feats["elev_slope"] = to_fp(b_elev_slope)
    feats["elev_slope_anom15"] = to_fp(b_elev_slope - uniform_filter(b_elev_slope, size=15))
    feats["det_elev_local15"] = to_fp(b_det_elev - uniform_filter(b_det_elev, size=15))
    feats["rtp_local15"] = to_fp(b_rtp - uniform_filter(b_rtp, size=15))
    feats["grav_local15"] = to_fp(b_grav - uniform_filter(b_grav, size=15))
    feats["cond_local15"] = to_fp(b_cond - uniform_filter(b_cond, size=15))
    feats["depth_base_grad"] = to_fp(np.hypot(*np.gradient(b_depth_base)))

    # H16-5: Strain & seismic local tectonic intensity
    feats["strain_2ndinv"] = to_fp(b_geod_2nd)
    feats["strain_shear"] = to_fp(b_geod_shear)
    feats["strain_dilate_abs"] = np.abs(to_fp(b_geod_dil))
    feats["seismic_ieq_log"] = np.log1p(np.maximum(to_fp(b_ieq), 0.0))

    # H16-2: Geopotential Strike-Integrated Worm & Upward-Continuation Persistence
    with rasterio.open(DATA_DIR / "external" / "geodawn_extensions_u8.tif") as src:
        thk_u8 = src.read(1).astype(np.float32)
        uk_u8 = src.read(2).astype(np.float32)
        uth_u8 = src.read(3).astype(np.float32)
        up150_u8 = src.read(4).astype(np.float32)

    worm_mag = compute_strike_worm_feature(np.log1p(np.maximum(b_tmi_hg, 0.0)), footprint)
    worm_grav = compute_strike_worm_feature(b_grav_slope, footprint)
    worm_elev = compute_strike_worm_feature(b_elev_slope, footprint)
    # Upward continuation high-frequency residual (local shallow vs deep persistent magnetic edge)
    up150_grad = np.hypot(*np.gradient(up150_u8))
    feats["worm_mag_1500m"] = to_fp(worm_mag)
    feats["worm_grav_1500m"] = to_fp(worm_grav)
    feats["worm_elev_1500m"] = to_fp(worm_elev)
    feats["worm_joint_mag_grav"] = to_fp(np.sqrt(worm_mag * worm_grav))
    feats["up150_deep_grad"] = to_fp(up150_grad)
    del worm_mag, worm_grav, worm_elev, up150_grad

    print("  [Features 2/4] Computing H16-4 Hydrothermal Potassic + Demagnetization Conduit features...")
    with rasterio.open(DATA_DIR / "external" / "geodawn_rad_u8.tif") as src:
        k_u8 = src.read(1).astype(np.float32)
        th_u8 = src.read(2).astype(np.float32)
        u_u8 = src.read(3).astype(np.float32)
        tc_rad_u8 = src.read(4).astype(np.float32)

    # Potassic alteration index: high K relative to Th (low ThK), gated away from flat playa bottoms
    non_playa = (b_elev_slope > 1.5).astype(np.float32)
    k_over_th = (k_u8 + 1.0) / (th_u8 + 10.0)
    k_th_anom = (k_over_th - uniform_filter(k_over_th, size=21)) * non_playa
    # Demagnetization trough along magnetic gradient: negative local RTP anomaly * tmi_hg
    demag_conduit = np.maximum(-(b_rtp - uniform_filter(b_rtp, size=15)), 0.0) * np.log1p(np.maximum(b_tmi_hg, 0.0)) * non_playa
    # Clay cap conductivity anomaly * structural gradient
    clay_conduit = np.maximum(b_cond - uniform_filter(b_cond, size=21), 0.0) * (b_elev_slope + 10.0 * b_grav_slope) * non_playa
    rad_grad = np.hypot(*np.gradient(tc_rad_u8)) * non_playa

    feats["hydro_k_th_anom"] = to_fp(k_th_anom)
    feats["hydro_demag_conduit"] = to_fp(demag_conduit)
    feats["hydro_clay_conduit"] = to_fp(clay_conduit)
    feats["hydro_rad_edge"] = to_fp(rad_grad)
    feats["hydro_uk_anom"] = to_fp((uk_u8 - uniform_filter(uk_u8, size=21)) * non_playa)
    del k_u8, th_u8, u_u8, tc_rad_u8, thk_u8, uk_u8, uth_u8, up150_u8
    del b_rtp, b_tmi_hg, b_geod_2nd, b_grav_slope, b_tc, b_geod_shear, b_geod_dil
    del b_tmi_vg, b_grav_vg, b_det_elev, b_grav, b_tmi, b_depth_base, b_ieq, b_cond, b_grav_hg, b_elev_slope
    gc.collect()

    print("  [Features 3/4] Loading H16-3 1m USGS 3DEP Lidar Scarp + 10m DEM Asymmetry Gap-Fill...")
    lid_meta = json.loads((DATA_DIR / "external" / "lidar_scarp_features.json").read_text())["quantisation"]
    with rasterio.open(DATA_DIR / "external" / "lidar_scarp_features_u8.tif") as src:
        bnames = list(src.descriptions)
        lid = {n: dequant_lidar(n, src.read(i + 1), lid_meta) for i, n in enumerate(bnames)}

    lid_valid = (lid["valid"] > 0.5) & footprint
    # Tectonic-to-fluvial scarp physics:
    # Tectonic scarps have strong downface or upface (antislope graben) + crest*toe dipole + high coh100,
    # whereas fluvial channels have cross_max >> max(upface, downface) and low coh100.
    dipole_1m = np.sqrt(np.maximum(lid["lapneg_max"] * lid["lappos_max"], 0.0))
    tect_vs_fluv = (np.maximum(lid["downface_max"], 1.35 * lid["upface_max"]) * (0.3 + lid["coh100"])) / (
        lid["cross_max"] + 0.08
    )
    # Relief-normalized piedmont scarp step (prevents high-relief range crests from drowning piedmont scarps)
    piedmont_step_1m = lid["step_max"] / (np.sqrt(lid["relief"] + 9.0))
    antislope_graben_1m = lid["upface_max"] * (0.25 + lid["coh100"]) / (lid["cross_max"] + 0.10)

    feats["lid1m_valid"] = to_fp(lid_valid.astype(np.float32))
    feats["lid1m_ex_max"] = to_fp(lid["ex_max"])
    feats["lid1m_step_max"] = to_fp(lid["step_max"])
    feats["lid1m_dipole"] = to_fp(dipole_1m)
    feats["lid1m_tect_vs_fluv"] = to_fp(np.where(lid_valid, tect_vs_fluv, 0.0))
    feats["lid1m_piedmont_step"] = to_fp(np.where(lid_valid, piedmont_step_1m, 0.0))
    feats["lid1m_antislope"] = to_fp(np.where(lid_valid, antislope_graben_1m, 0.0))
    feats["lid1m_coh100"] = to_fp(lid["coh100"])
    feats["lid1m_step_max5"] = to_fp(maximum_filter(lid["step_max"], size=5))
    feats["lid1m_step_anom15"] = to_fp(lid["step_max"] - uniform_filter(lid["step_max"], size=15))
    del lid, dipole_1m, tect_vs_fluv, piedmont_step_1m, antislope_graben_1m
    gc.collect()

    # Load 10m USGS 3DEP DEM scarp channels (100% footprint coverage, bridging the 24.6% 1m-lidar gap!)
    dem10_names = [
        "slope_max", "slope_mean", "slope_std", "hgm20_max", "hgm50_max", "hgm200_mean",
        "steep_ratio_max", "resid_std", "resid_range", "curv_absmax", "onesided", "onesided3",
    ]
    for ch in dem10_names:
        v = np.load(DATA_DIR / "dem10" / f"dem10_{ch}.f32.npy").astype(np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        feats[f"dem10_{ch}"] = v

    # Construct a unified seam-free cross-scale scarp index (z-standardized on overlap, seamlessly bridging gap)
    lid_ok_fp = feats["lid1m_valid"] > 0.5
    dem10_scarp_raw = (
        0.35 * (feats["dem10_onesided3"] / (np.std(feats["dem10_onesided3"]) + 1e-6))
        + 0.25 * (feats["dem10_onesided"] / (np.std(feats["dem10_onesided"]) + 1e-6))
        + 0.25 * (feats["dem10_steep_ratio_max"] / (np.std(feats["dem10_steep_ratio_max"]) + 1e-6))
        + 0.15 * (feats["dem10_curv_absmax"] / (np.std(feats["dem10_curv_absmax"]) + 1e-6))
    )
    lid1m_scarp_raw = (
        0.35 * (feats["lid1m_tect_vs_fluv"] / (np.std(feats["lid1m_tect_vs_fluv"][lid_ok_fp]) + 1e-6))
        + 0.30 * (feats["lid1m_piedmont_step"] / (np.std(feats["lid1m_piedmont_step"][lid_ok_fp]) + 1e-6))
        + 0.20 * (feats["lid1m_dipole"] / (np.std(feats["lid1m_dipole"][lid_ok_fp]) + 1e-6))
        + 0.15 * (feats["lid1m_antislope"] / (np.std(feats["lid1m_antislope"][lid_ok_fp]) + 1e-6))
    )
    feats["seamfree_scarp_index"] = np.where(
        lid_ok_fp, 0.70 * lid1m_scarp_raw + 0.30 * dem10_scarp_raw, dem10_scarp_raw
    ).astype(np.float32)

    print("  [Features 4/4] Loading Out-of-Fold Spatial Context Probability fields...")
    with rasterio.open(DATA_DIR / "derived" / "context_detector_prob_4fold_base.tif") as src:
        feats["ctx_oof_base"] = to_fp(src.read(1).astype(np.float32) / 255.0)
    with rasterio.open(DATA_DIR / "derived" / "context_detector_prob_topo.tif") as src:
        feats["ctx_oof_topo"] = to_fp(src.read(1).astype(np.float32) / 255.0)
    with rasterio.open(DATA_DIR / "derived" / "context_detector_prob_topo_rad.tif") as src:
        feats["ctx_oof_topo_rad"] = to_fp(src.read(1).astype(np.float32) / 255.0)

    print(f"  Built {len(feats)} footprint-indexed channels in {time.time() - t0:.1f}s.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **feats)
    return feats, fp_idx


def make_quadrant_folds(footprint: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Partition footprint into 4 contiguous geographic quadrants (NW=0, NE=1, SW=2, SE=3)."""
    yy, xx = np.nonzero(footprint)
    y_med = int(np.median(yy))
    x_med = int(np.median(xx))
    H, W = footprint.shape
    grid_y, grid_x = np.ogrid[:H, :W]
    fold_2d = np.full((H, W), -1, dtype=np.int8)
    fold_2d[(grid_y < y_med) & (grid_x < x_med) & footprint] = 0   # NW
    fold_2d[(grid_y < y_med) & (grid_x >= x_med) & footprint] = 1  # NE (47.4% lidar gap!)
    fold_2d[(grid_y >= y_med) & (grid_x < x_med) & footprint] = 2  # SW
    fold_2d[(grid_y >= y_med) & (grid_x >= x_med) & footprint] = 3 # SE
    return fold_2d, ["NW", "NE_LidarGapHeavy", "SW", "SE"]


def thin_components(truth_2d: np.ndarray, keep_frac: float, seed: int) -> np.ndarray:
    """Thin connected fault components to simulate sparse unmapped new-fault truth."""
    comp, n_comp = ndi_label(truth_2d, structure=np.ones((3, 3), dtype=int))
    rng = np.random.default_rng(seed)
    keep_ids = rng.choice(np.arange(1, n_comp + 1), size=max(1, int(round(keep_frac * n_comp))), replace=False)
    return np.isin(comp, keep_ids)


if __name__ == "__main__":
    with rasterio.open(DATA_DIR / "sample_submission.tif") as src:
        footprint = np.isfinite(src.read(1))
    with rasterio.open(DATA_DIR / "labels.tif") as src:
        labels = (src.read(1) > 0) & footprint

    feats, fp_idx = build_feature_matrix_on_footprint(footprint)
    fold_2d, fold_names = make_quadrant_folds(footprint)
    fold_fp = fold_2d.ravel()[fp_idx]
    y_fp = labels.ravel()[fp_idx]

    # PU collar: exclude pixels within 300m (3 px) of known catalogue traces from negative sampling
    near_cat_2d = binary_dilation(labels, iterations=3)
    near_cat_fp = near_cat_2d.ravel()[fp_idx]

    # Define feature sets for our 5 hypotheses + baseline
    BASE_COLS = [
        "tmi_hg", "tmi_vg", "tc", "grav_slope", "grav_vg", "grav_hg_abs",
        "elev_slope", "elev_slope_anom15", "det_elev_local15",
        "rtp_local15", "grav_local15", "cond_local15", "depth_base_grad",
    ]
    H16_5_COLS = BASE_COLS + ["strain_2ndinv", "strain_shear", "strain_dilate_abs", "seismic_ieq_log"]
    H16_4_COLS = BASE_COLS + [
        "hydro_k_th_anom", "hydro_demag_conduit", "hydro_clay_conduit", "hydro_rad_edge", "hydro_uk_anom",
    ]
    H16_2_COLS = BASE_COLS + [
        "worm_mag_1500m", "worm_grav_1500m", "worm_elev_1500m", "worm_joint_mag_grav", "up150_deep_grad",
    ]
    # Compute 100% unsupervised multi-scale physical spatial context features once on the 2D grid
    # (zero label usage -> zero cross-fold leakage, bridging fragmented scarp/geopotential segments)
    ctx_cache = DATA_DIR / "cache" / "unsupervised_scarp_ctx_fp.npz"
    if ctx_cache.exists():
        print(f"  [Cache] Loading unsupervised multi-scale scarp context from {ctx_cache}...")
        loaded_ctx = np.load(ctx_cache)
        for k in loaded_ctx.files:
            feats[k] = loaded_ctx[k]
    else:
        print("  Computing unsupervised multi-scale scarp & worm spatial context channels...")
        scarp_2d = np.zeros(footprint.shape, dtype=np.float32)
        scarp_2d.ravel()[fp_idx] = feats["seamfree_scarp_index"]
        dem10_s_2d = np.zeros(footprint.shape, dtype=np.float32)
        dem10_s_2d.ravel()[fp_idx] = feats["dem10_onesided3"]
        lid_s_2d = np.zeros(footprint.shape, dtype=np.float32)
        lid_s_2d.ravel()[fp_idx] = feats["lid1m_step_max"]

        ctx_dict = {
            "scarp_worm_1100m": compute_strike_worm_feature(scarp_2d, footprint, half_len=5, flank=3).ravel()[fp_idx],
            "dem10_worm_1100m": compute_strike_worm_feature(dem10_s_2d, footprint, half_len=5, flank=3).ravel()[fp_idx],
            "lid1m_worm_1100m": compute_strike_worm_feature(lid_s_2d, footprint, half_len=5, flank=3).ravel()[fp_idx],
            "scarp_gauss_300m": gaussian_filter(scarp_2d, sigma=3.0).ravel()[fp_idx].astype(np.float32),
            "scarp_gauss_800m": gaussian_filter(scarp_2d, sigma=8.0).ravel()[fp_idx].astype(np.float32),
            "scarp_max_500m": maximum_filter(scarp_2d, size=5).ravel()[fp_idx].astype(np.float32),
        }
        np.savez(ctx_cache, **ctx_dict)
        feats.update(ctx_dict)
        del scarp_2d, dem10_s_2d, lid_s_2d, ctx_dict

    SCARP_PURE_COLS = [
        "lid1m_valid", "lid1m_ex_max", "lid1m_step_max", "lid1m_dipole",
        "lid1m_tect_vs_fluv", "lid1m_piedmont_step", "lid1m_antislope",
        "lid1m_coh100", "lid1m_step_max5", "lid1m_step_anom15",
        "dem10_slope_max", "dem10_slope_mean", "dem10_slope_std",
        "dem10_hgm20_max", "dem10_hgm50_max", "dem10_hgm200_mean",
        "dem10_steep_ratio_max", "dem10_resid_std", "dem10_resid_range",
        "dem10_curv_absmax", "dem10_onesided", "dem10_onesided3",
        "seamfree_scarp_index",
    ]
    H16_3_COLS = BASE_COLS + SCARP_PURE_COLS

    ARMS = {
        "Baseline_Bands19_DeReg": BASE_COLS,
        "H16_5_Strain_Seismic_Completeness": H16_5_COLS,
        "H16_4_Hydrothermal_Conduit": H16_4_COLS,
        "H16_2_Geopotential_Strike_Worm": H16_2_COLS,
        "H16_3_ScarpPure_1m_10m": SCARP_PURE_COLS,
        "H16_3_Antislope_Piedmont_Scarp_1m_10m": H16_3_COLS,
    }

    oof_all_cache = DATA_DIR / "cache" / "oof_probs_all_arms.npz"
    if oof_all_cache.exists():
        print(f"  [Cache] Loading 4-quadrant OOF probabilities from {oof_all_cache}...")
        loaded_oof = np.load(oof_all_cache)
        oof_probs = {k: loaded_oof[k] for k in loaded_oof.files}
    else:
        oof_probs = {arm: np.zeros(len(fp_idx), dtype=np.float32) for arm in ARMS}
        rng = np.random.default_rng(SEED)
        for f_id in range(4):
            test_2d = (fold_2d == f_id) & footprint
            buf_2d = binary_dilation(test_2d, iterations=15)
            train_fp_mask = (fold_fp != f_id) & (~buf_2d.ravel()[fp_idx])
            test_fp_mask = (fold_fp == f_id)

            pos_idx = np.flatnonzero(train_fp_mask & y_fp)
            neg_idx = np.flatnonzero(train_fp_mask & (~near_cat_fp))
            sel_pos = rng.choice(pos_idx, size=min(40000, len(pos_idx)), replace=False)
            sel_neg = rng.choice(neg_idx, size=min(120000, len(neg_idx)), replace=False)
            tr_idx = np.concatenate([sel_pos, sel_neg])
            y_tr = np.r_[np.ones(len(sel_pos), dtype=np.int8), np.zeros(len(sel_neg), dtype=np.int8)]
            w_tr = np.where(y_tr == 1, 0.5 / len(sel_pos), 0.5 / len(sel_neg)) * len(tr_idx)

            te_idx = np.flatnonzero(test_fp_mask)
            for arm, cols in ARMS.items():
                X_tr = np.column_stack([feats[c][tr_idx] for c in cols])
                clf = HistGradientBoostingClassifier(
                    max_iter=150,
                    max_leaf_nodes=31,
                    min_samples_leaf=80,
                    learning_rate=0.06,
                    l2_regularization=2.0,
                    early_stopping=False,
                    random_state=SEED + f_id,
                )
                clf.fit(X_tr, y_tr, sample_weight=w_tr)
                X_te = np.column_stack([feats[c][te_idx] for c in cols])
                oof_probs[arm][te_idx] = clf.predict_proba(X_te)[:, 1].astype(np.float32)
                del X_tr, X_te, clf
            print(f"  Completed fold {f_id} ({fold_names[f_id]}) training & OOF inference.")
        np.savez(oof_all_cache, **oof_probs)

    # H16-1 Seam-Free Multi-Scale Synthesis:
    # Regime-conditional modular synthesis of pure scarp expert + GeoDAWN-scarp expert +
    # 1.5 km geopotential strike worm + hydrothermal conduit, with cross-regime quantile
    # calibration across the 24.6% 1m-lidar gap seam.
    lid_ok = feats["lid1m_valid"] > 0.5
    p_scarp_pure = oof_probs["H16_3_ScarpPure_1m_10m"]
    p3 = oof_probs["H16_3_Antislope_Piedmont_Scarp_1m_10m"]
    p2 = oof_probs["H16_2_Geopotential_Strike_Worm"]
    p4 = oof_probs["H16_4_Hydrothermal_Conduit"]

    p_regime = np.where(
        lid_ok,
        0.55 * p_scarp_pure + 0.35 * p3 + 0.06 * p2 + 0.04 * p4,
        0.40 * p_scarp_pure + 0.35 * p3 + 0.15 * p2 + 0.10 * p4,
    ).astype(np.float32)
    p_h16_1 = p_regime.copy()
    q_grid = np.linspace(0.0, 1.0, 1001)
    for f_id in range(4):
        m_lid = (fold_fp == f_id) & lid_ok
        m_gap = (fold_fp == f_id) & (~lid_ok)
        p_h16_1[m_gap] = np.interp(
            p_regime[m_gap], np.quantile(p_regime[m_gap], q_grid), np.quantile(p_regime[m_lid], q_grid)
        )
    oof_probs["H16_1_SeamFree_MultiScale_Synthesis"] = p_h16_1

    # Also evaluate sibling comparators on the exact same 4 geographic folds:
    with rasterio.open(AUDIT_CLONES / "7GEMSDOE/downloads/gems7-lidarscarp-ridge-top2pct-36c3a3f341c8.tif") as src:
        sib_g7_2d = (np.nan_to_num(src.read(1), nan=0.0) > 0.5) & footprint

    BUDGET_FRAC = 0.025
    data_ver = json.loads((EVIDENCE_DIR / "data_verification.json").read_text())
    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_scope": (
            "Spatial-transfer proxy on held-out portions of the known fault catalogue; "
            "sparse scores use a deterministic 20%-of-components simulation. Not the hidden competition test set."
        ),
        "methodology": {
            "split": "four contiguous NW/NE/SW/SE geographic quadrants from footprint row/column medians",
            "buffer_pixels": 15,
            "training_sampling": "up to 40,000 known-catalogue positives and 120,000 negatives outside a 3-pixel known-catalogue collar per fold",
            "classifier": "HistGradientBoostingClassifier; max_iter=150, max_leaf_nodes=31, min_samples_leaf=80, learning_rate=0.06, l2_regularization=2.0",
            "prediction_budget_per_fold": BUDGET_FRAC,
            "sparse_proxy": "deterministic 20% subset of connected held-out catalogue components; remaining held-out components treated as neutral for sparse false-positive calculation",
            "random_seed": SEED,
            "note": "The holdout does not reproduce private test geography, hidden faults, or leaderboard scoring conditions. Scores are not predictions of competition DTI.",
        },
        "input_sha256": {
            key: value["sha256"]
            for key, value in data_ver["rasters"].items()
            if key in ("training_features.tif", "labels.tif", "sample_submission.tif")
        },
        "submission_slot_spent": False,
        "folds": {},
        "summary": {},
    }

    # Precompute quadrant bounding-box slices & held-out truth masks once
    quads = []
    for f_id, fname in enumerate(fold_names):
        f_mask = (fold_2d == f_id) & footprint
        r0, r1 = np.nonzero(f_mask.any(axis=1))[0][[0, -1]]
        c0, c1 = np.nonzero(f_mask.any(axis=0))[0][[0, -1]]
        sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        t_dense = labels & f_mask
        t_sparse = thin_components(t_dense, keep_frac=0.20, seed=4242 + f_id)
        known_cat = t_dense & (~t_sparse)
        quads.append((f_id, fname, f_mask, sl, t_dense[sl], t_sparse[sl], known_cat[sl], f_mask[sl]))

    eval_order = [
        "Baseline_Bands19_DeReg",
        "H16_5_Strain_Seismic_Completeness",
        "H16_4_Hydrothermal_Conduit",
        "H16_2_Geopotential_Strike_Worm",
        "H16_3_ScarpPure_1m_10m",
        "H16_3_Antislope_Piedmont_Scarp_1m_10m",
        "H16_1_SeamFree_MultiScale_Synthesis",
    ]

    for arm in eval_order:
        p_fp = oof_probs[arm]
        score_2d = np.zeros(footprint.shape, dtype=np.float32)
        score_2d.ravel()[fp_idx] = p_fp
        ridge_2d = ridge_nms(score_2d, footprint, sigma=1.0)
        boosted_2d = np.where(ridge_2d, score_2d + 1.0, score_2d * 0.5)

        dense_dtis, sparse_dtis = [], []
        fold_detail = {}
        for f_id, fname, f_mask, sl, td, ts, kc, fm in quads:
            k_emit = int(round(BUDGET_FRAC * fm.sum()))
            f_idx = np.flatnonzero(f_mask.ravel())
            vals = boosted_2d.ravel()[f_idx]
            top_sel = f_idx[np.argpartition(vals, -k_emit)[-k_emit:]]
            pred_f = np.zeros(footprint.shape, dtype=bool)
            pred_f.ravel()[top_sel] = True

            r_dense = dti_score_fast(pred_f[sl], td, valid_mask=fm)
            r_sparse = dti_score_fast(pred_f[sl], ts, valid_mask=fm, catalogue_mask=kc)
            dense_dtis.append(r_dense["dti"])
            sparse_dtis.append(r_sparse["dti"])
            fold_detail[fname] = {
                "dense_dti": round(r_dense["dti"], 5),
                "sparse_dti": round(r_sparse["dti"], 5),
                "dense_coverage": round(r_dense["coverage"], 4),
                "sparse_coverage": round(r_sparse["coverage"], 4),
                "emitted_px": int(pred_f[sl].sum()),
            }
        results["folds"][arm] = fold_detail
        results["summary"][arm] = {
            "mean_dense_dti": round(float(np.mean(dense_dtis)), 5),
            "mean_sparse_dti": round(float(np.mean(sparse_dtis)), 5),
            "fold_dense": [round(x, 5) for x in dense_dtis],
            "fold_sparse": [round(x, 5) for x in sparse_dtis],
        }
        print(
            f"  {arm:40s} | Mean Dense DTI: {np.mean(dense_dtis):.5f} | Mean Sparse DTI: {np.mean(sparse_dtis):.5f}"
        )

    g7_dense, g7_sparse = [], []
    g7_detail = {}
    for f_id, fname, f_mask, sl, td, ts, kc, fm in quads:
        pred_f = sib_g7_2d & f_mask
        r_d = dti_score_fast(pred_f[sl], td, valid_mask=fm)
        r_s = dti_score_fast(pred_f[sl], ts, valid_mask=fm, catalogue_mask=kc)
        g7_dense.append(r_d["dti"])
        g7_sparse.append(r_s["dti"])
        g7_detail[fname] = {
            "dense_dti": round(r_d["dti"], 5),
            "sparse_dti": round(r_s["dti"], 5),
            "dense_coverage": round(r_d["coverage"], 4),
            "sparse_coverage": round(r_s["coverage"], 4),
            "emitted_px": int(pred_f[sl].sum()),
        }
    results["folds"]["Sibling_7GEMSDOE_LidarOnly_36c3a3f3"] = g7_detail
    results["summary"]["Sibling_7GEMSDOE_LidarOnly_36c3a3f3"] = {
        "mean_dense_dti": round(float(np.mean(g7_dense)), 5),
        "mean_sparse_dti": round(float(np.mean(g7_sparse)), 5),
        "fold_dense": [round(x, 5) for x in g7_dense],
        "fold_sparse": [round(x, 5) for x in g7_sparse],
    }
    print(
        f"  {'Sibling_7GEMSDOE_LidarOnly_36c3a3f3':40s} | Mean Dense DTI: {np.mean(g7_dense):.5f} | Mean Sparse DTI: {np.mean(g7_sparse):.5f}"
    )

    (EVIDENCE_DIR / "spatial_holdout_results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez(
        DATA_DIR / "cache" / "oof_probs_h16_1.npz",
        h16_1=oof_probs["H16_1_SeamFree_MultiScale_Synthesis"],
        h16_3_pure=oof_probs["H16_3_ScarpPure_1m_10m"],
        h16_3=oof_probs["H16_3_Antislope_Piedmont_Scarp_1m_10m"],
        h16_2=oof_probs["H16_2_Geopotential_Strike_Worm"],
        h16_4=oof_probs["H16_4_Hydrothermal_Conduit"],
    )
    print("  Saved evidence/spatial_holdout_results.json and data/cache/oof_probs_h16_1.npz.")
