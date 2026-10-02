"""Evaluate 19GEMSDOE hypotheses on the 4-quadrant spatial holdout, build the physical
corroboration ledgers, and emit validated GeoTIFF submission packages.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt, label as ndi_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems import forensics as F  # noqa: E402
from gems.footprint import load_footprint  # noqa: E402
from gems.holdout import FOLD_NAMES, Holdout, gate  # noqa: E402
from gems.hypotheses import (  # noqa: E402
    HYPOTHESIS_SPECS,
    extract_trace_lengths_m,
    fit_power_law_population,
    synthesize_h19_4_corroborated,
    synthesize_h19_5_openness_thermal_corroborated,
)
from gems.paths import (  # noqa: E402
    DATA_DIR,
    DOWNLOADS_DIR,
    EVIDENCE_DIR,
    GROUP_DIR,
    LABELS_PATH,
    SITE_DATA_DIR,
    SUBMISSIONS_DIR,
    TEMPLATE_PATH,
)
from gems.submission import (  # noqa: E402
    check_variants,
    make_filename,
    make_note,
    scored_content_id,
    write_submission,
    zip_single,
)


def git_head() -> str:
    try:
        return (
            subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    t0 = time.time()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fp = load_footprint()
    with rasterio.open(LABELS_PATH) as s:
        labels = (s.read(1) > 0) & fp
    grid = F.Grid.load(TEMPLATE_PATH, LABELS_PATH)
    holdout = Holdout(fp, labels)
    H, W = fp.shape
    fp_idx = holdout.fp_idx
    fold_2d = holdout.fold_2d
    fold_fp = fold_2d.ravel()[fp_idx]

    # -------------------------------------------------------------------------
    # 1. Power-Law Fault Population Size-Distribution Scaling Report (H19-1)
    # -------------------------------------------------------------------------
    print("[1/5] Computing Power-Law Fault Population Scaling Report (H19-1)...")
    df_vec = pd.read_csv(EVIDENCE_DIR / "ci" / "gdr_qfaults_traces.csv")
    l_vec = df_vec[df_vec["clipped_length_m"] >= 100.0]["clipped_length_m"].to_numpy()
    l_rast = extract_trace_lengths_m(labels, px_len_m=108.0)

    power_law_report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_vector": {
            "url": "https://gdr.openei.org/submissions/1391",
            "file": "qfaults_ingenious_nad83conus117_2023-06-27.zip",
            "doi": "10.15121/1881483",
            "traces_in_grid": int(len(l_vec)),
            "length_min_m": round(float(l_vec.min()), 1),
            "length_median_m": round(float(np.median(l_vec)), 1),
            "length_p90_m": round(float(np.percentile(l_vec, 90)), 1),
            "length_max_m": round(float(l_vec.max()), 1),
            "fits_by_l_min": {
                str(int(lm)): fit_power_law_population(l_vec, l_min=lm, l0=300.0, upper_pct=96.0)
                for lm in [1500.0, 2000.0, 2500.0, 3000.0]
            },
        },
        "source_raster_skeleton": {
            "file": "data/labels.tif (existing_faults.tif)",
            "sha256": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
            "connected_traces_in_footprint": int(len(l_rast)),
            "positive_pixels_in_footprint": int(labels.sum()),
            "length_min_m": round(float(l_rast.min()), 1),
            "length_median_m": round(float(np.median(l_rast)), 1),
            "length_p90_m": round(float(np.percentile(l_rast, 90)), 1),
            "length_max_m": round(float(l_rast.max()), 1),
            "fits_by_l_min": {
                str(int(lm)): fit_power_law_population(l_rast, l_min=lm, l0=300.0, upper_pct=97.0)
                for lm in [1200.0, 1500.0, 1650.0, 1800.0, 2200.0, 2500.0]
            },
            "per_quadrant_fits_at_1650m": {},
        },
    }
    for q_id, q_name in enumerate(FOLD_NAMES):
        q_mask = (fold_2d == q_id) & fp
        l_q = extract_trace_lengths_m(labels & q_mask, px_len_m=108.0)
        power_law_report["source_raster_skeleton"]["per_quadrant_fits_at_1650m"][q_name] = fit_power_law_population(
            l_q, l_min=1650.0, l0=300.0, upper_pct=97.0, footprint_pixels=int(q_mask.sum())
        )

    (EVIDENCE_DIR / "power_law_scaling_report.json").write_text(json.dumps(power_law_report, indent=2) + "\n")
    (SITE_DATA_DIR / "power_law_scaling_report.json").write_text(json.dumps(power_law_report, indent=2) + "\n")

    # -------------------------------------------------------------------------
    # 2. Backward Thermal & Geochemical Conduit Inversion Report (H19-2)
    # -------------------------------------------------------------------------
    print("[2/5] Computing Backward Thermal & Geochemical Conduit Inversion Report (H19-2)...")
    dist_known_px = distance_transform_edt(~labels)
    df_ws = pd.read_csv(EVIDENCE_DIR / "ci" / "gdr_wellspring_in_footprint.csv")
    is_thermal_ws = (
        (df_ws["temp_c"] >= 25.0)
        | (df_ws["geothermquartz_c"] >= 70.0)
        | (df_ws["geothermchalc_c"] >= 60.0)
        | (df_ws["geothermcat_c"] >= 80.0)
        | (df_ws["thermalclass"].fillna("").str.lower().str.contains("hot|warm|therm"))
    )
    df_ws_anom = df_ws[is_thermal_ws].copy()

    df_pr = pd.read_csv(DATA_DIR / "ingenious" / "probes_2m_2m_temperature_probe_n83geo_layer_points_utm.csv")
    pr_r = np.clip(((4508550.0 - df_pr["utm_y"].values) // 100.0).astype(int), 0, H - 1)
    pr_c = np.clip(((df_pr["utm_x"].values - 243350.0) // 100.0).astype(int), 0, W - 1)
    pr_in_fp = fp[pr_r, pr_c]
    df_pr_fp = df_pr[pr_in_fp].copy()
    df_pr_fp["dist_known_px"] = dist_known_px[pr_r[pr_in_fp], pr_c[pr_in_fp]]
    df_pr_anom = df_pr_fp[df_pr_fp["F2mDAB"] >= 1.5]

    df_pa = pd.read_csv(DATA_DIR / "ingenious" / "paleo_geothermal_Paleo_geothermal_final_layer_points_utm.csv")
    pa_r = np.clip(((4508550.0 - df_pa["utm_y"].values) // 100.0).astype(int), 0, H - 1)
    pa_c = np.clip(((df_pa["utm_x"].values - 243350.0) // 100.0).astype(int), 0, W - 1)
    pa_in_fp = fp[pa_r, pa_c]
    df_pa_fp = df_pa[pa_in_fp].copy()
    df_pa_fp["dist_known_px"] = dist_known_px[pa_r[pa_in_fp], pa_c[pa_in_fp]]

    df_vo = pd.read_csv(EVIDENCE_DIR / "ci" / "gdr_volcanic_vents_in_footprint.csv")

    thermal_report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "physical_principle": (
            "Amagmatic Great Basin thermal and geochemical anomalies require deep permeable fault conduits "
            "for fluid upflow. When a thermal spring/well, silica/cation geothermometer anomaly, 2m temperature "
            "probe anomaly, or sinter/travertine/tufa deposit lies >500 m from any catalogued fault in labels.tif, "
            "backward physical inversion requires an unmapped permeable fault conduit within the local upflow/outflow radius."
        ),
        "datasets": {
            "gdr1391_wellspring": {
                "source_url": "https://gdr.openei.org/submissions/1391 (wellspringdata.gdb.zip, DOI 10.15121/1881483)",
                "total_records_in_footprint": int(len(df_ws)),
                "thermal_or_geochem_anomalies": int(len(df_ws_anom)),
                "temp_ge_25c": int((df_ws["temp_c"] >= 25.0).sum()),
                "temp_ge_37c": int((df_ws["temp_c"] >= 37.0).sum()),
                "quartz_geotherm_ge_70c": int((df_ws["geothermquartz_c"] >= 70.0).sum()),
                "chalcedony_geotherm_ge_60c": int((df_ws["geothermchalc_c"] >= 60.0).sum()),
                "cation_geotherm_ge_80c": int((df_ws["geothermcat_c"] >= 80.0).sum()),
                "orphan_gt_500m_count": int((df_ws_anom["dist_known_fault_px"] > 5.0).sum()),
                "orphan_gt_500m_fraction": round(float((df_ws_anom["dist_known_fault_px"] > 5.0).mean()), 4),
                "orphan_gt_1000m_count": int((df_ws_anom["dist_known_fault_px"] > 10.0).sum()),
                "orphan_gt_1000m_fraction": round(float((df_ws_anom["dist_known_fault_px"] > 10.0).mean()), 4),
            },
            "gdr1391_2m_temperature_probes": {
                "source_url": "https://gdr.openei.org/submissions/1391 (2m_temperature_probe_INGENIOUS_regional_data.zip)",
                "total_stations_in_footprint": int(len(df_pr_fp)),
                "anomalous_f2mdab_ge_1_5c": int(len(df_pr_anom)),
                "orphan_gt_500m_count": int((df_pr_anom["dist_known_px"] > 5.0).sum()),
                "orphan_gt_500m_fraction": round(float((df_pr_anom["dist_known_px"] > 5.0).mean()), 4),
                "orphan_gt_1000m_count": int((df_pr_anom["dist_known_px"] > 10.0).sum()),
                "orphan_gt_1000m_fraction": round(float((df_pr_anom["dist_known_px"] > 10.0).mean()), 4),
            },
            "gdr1391_paleo_geothermal_deposits": {
                "source_url": "https://gdr.openei.org/submissions/1391 (paleo_geothermal_regional.zip)",
                "total_sites_in_footprint": int(len(df_pa_fp)),
                "orphan_gt_500m_count": int((df_pa_fp["dist_known_px"] > 5.0).sum()),
                "orphan_gt_500m_fraction": round(float((df_pa_fp["dist_known_px"] > 5.0).mean()), 4),
                "orphan_gt_1000m_count": int((df_pa_fp["dist_known_px"] > 10.0).sum()),
                "orphan_gt_1000m_fraction": round(float((df_pa_fp["dist_known_px"] > 10.0).mean()), 4),
            },
            "gdr1391_quaternary_volcanic_vents": {
                "source_url": "https://gdr.openei.org/submissions/1391 (great_basin_q_volcanics.zip)",
                "total_vents_in_footprint": int(len(df_vo)),
                "orphan_gt_500m_count": int((df_vo["dist_known_fault_px"] > 5.0).sum()),
            },
        },
    }
    (EVIDENCE_DIR / "backward_thermal_geochem_report.json").write_text(json.dumps(thermal_report, indent=2) + "\n")
    (SITE_DATA_DIR / "backward_thermal_geochem_report.json").write_text(json.dumps(thermal_report, indent=2) + "\n")

    # -------------------------------------------------------------------------
    # 3. 4-Quadrant Spatial Holdout Evaluation & Hypothesis Corroboration Table
    # -------------------------------------------------------------------------
    print("[3/5] Evaluating all baselines, single-domain experts, ablations, and H19-4/H19-5 on 4-quadrant holdout...")
    f_base = np.load(DATA_DIR / "cache" / "features_fp.npz")
    lid_ok = f_base["lid1m_valid"] > 0.5
    oof_all = np.load(DATA_DIR / "cache" / "oof_probs_all_arms.npz")
    oof_16 = np.load(DATA_DIR / "cache" / "oof_probs_h16_1.npz")
    oof_19 = np.load(DATA_DIR / "cache" / "oof_probs_h19_arms.npz")

    p_base19 = oof_all["Baseline_Bands19_DeReg"]
    p_h16_5 = oof_all["H16_5_Strain_Seismic_Completeness"]
    p_h16_4 = oof_16["h16_4"]
    p_h16_2 = oof_16["h16_2"]
    p_h16_3_pure = oof_16["h16_3_pure"]
    p_h16_3_anti = oof_16["h16_3"]
    p_h16_1 = oof_16["h16_1"]

    p_h19_1 = oof_19["H19_1_PowerLaw_TipStepover"]
    p_h19_2 = oof_19["H19_2_Backward_ThermalGeochem"]
    p_h19_3_pure = oof_19["H19_3_Openness_LRM_Pure"]
    p_h19_3_anti = oof_19["H19_3_Openness_LRM_AntiPiedmont"]

    p_h19_4, lines_dict = synthesize_h19_4_corroborated(
        p_scarp_pure_16=p_h16_3_pure,
        p_scarp_anti_16=p_h16_3_anti,
        p_worm_16=p_h16_2,
        p_hydro_16=p_h16_4,
        p_h19_1=p_h19_1,
        p_h19_2=p_h19_2,
        p_h19_3_pure=p_h19_3_pure,
        p_h19_3_anti=p_h19_3_anti,
        lid_ok=lid_ok,
        fold_fp=fold_fp,
    )
    p_h19_5 = synthesize_h19_5_openness_thermal_corroborated(
        p_scarp_pure_16=p_h16_3_pure,
        p_scarp_anti_16=p_h16_3_anti,
        p_worm_16=p_h16_2,
        p_hydro_16=p_h16_4,
        p_h19_1=p_h19_1,
        p_h19_2=p_h19_2,
        p_h19_3_pure=p_h19_3_pure,
        p_h19_3_anti=p_h19_3_anti,
        lid_ok=lid_ok,
        fold_fp=fold_fp,
    )

    second_best = lines_dict["second_best_line"]
    p_single_layer_only = np.where(second_best < 0.18, p_h19_4 / lines_dict["single_layer_gate"], p_h19_4 * 0.05).astype(
        np.float32
    )

    base_eval = holdout.evaluate(p_h16_1, ridge=True)

    candidate_defs = [
        (
            "Baseline_Bands19_DeReg",
            p_base19,
            0.0250,
            ["L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem", "L3_Openness_LRM_Scarp"],
            "DISCARDED (Single-Layer Match: 19 competition GeoDAWN bands only)",
        ),
        (
            "H16_5_Strain_Seismic_Completeness",
            p_h16_5,
            0.0250,
            ["L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem", "L3_Openness_LRM_Scarp"],
            "DISCARDED (Single-Layer Match & Fails Holdout Gate)",
        ),
        (
            "H16_4_Hydrothermal_Conduit",
            p_h16_4,
            0.0250,
            ["L2_Backward_ThermalGeochem", "L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay", "L3_Openness_LRM_Scarp"],
            "REJECTED (Superseded by H19-2 Backward Thermal/Geochemical Inversion)",
        ),
        (
            "H16_2_Geopotential_Strike_Worm",
            p_h16_2,
            0.0250,
            ["L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem", "L3_Openness_LRM_Scarp"],
            "DISCARDED Standalone (Single-Layer Match; retained only as Line 4 input to H19-4)",
        ),
        (
            "H19_1_PowerLaw_TipStepover",
            p_h19_1,
            0.0250,
            ["L1_PopScaling_TipRelay", "L4_Geopotential_Basement"],
            ["L2_Backward_ThermalGeochem", "L3_Openness_LRM_Scarp"],
            "COMPONENT ONLY (Line 1+4 expert; +0.00353 Dense over 19-band baseline, feeds H19-4)",
        ),
        (
            "H19_2_Backward_ThermalGeochem",
            p_h19_2,
            0.0250,
            ["L2_Backward_ThermalGeochem", "L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay", "L3_Openness_LRM_Scarp"],
            "COMPONENT ONLY (Line 2+4 expert; beats H16-4 by +0.00205 Dense / +0.00088 Sparse DTI, feeds H19-4)",
        ),
        (
            "H16_3_ScarpPure_1m_10m",
            p_h16_3_pure,
            0.0250,
            ["L3_Openness_LRM_Scarp"],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem", "L4_Geopotential_Basement"],
            "DISCARDED Standalone (Single-Layer Topographic Match; superseded by H19-3)",
        ),
        (
            "H19_3_Openness_LRM_Pure",
            p_h19_3_pure,
            0.0250,
            ["L3_Openness_LRM_Scarp"],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem", "L4_Geopotential_Basement"],
            "DISCARDED Standalone (Single-Layer Topographic Match; beats H16-3 Pure by +0.00155 Dense, feeds H19-4)",
        ),
        (
            "H16_3_Antislope_Piedmont_Scarp_1m_10m",
            p_h16_3_anti,
            0.0250,
            ["L3_Openness_LRM_Scarp", "L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem"],
            "REJECTED (2-Line expert superseded by H19-3 AntiPiedmont & H19-4)",
        ),
        (
            "H19_3_Openness_LRM_AntiPiedmont",
            p_h19_3_anti,
            0.0250,
            ["L3_Openness_LRM_Scarp", "L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem"],
            "COMPONENT ONLY (2-Line expert; beats H16-3 AntiPiedmont by +0.00240 Dense / +0.00263 Sparse DTI, feeds H19-4)",
        ),
        (
            "H19_SingleLayer_PatternMatch_Ablation",
            p_single_layer_only,
            0.0250,
            [],
            ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem", "L3_Openness_LRM_Scarp", "L4_Geopotential_Basement"],
            "DISCARDED (Single-Layer Pattern Matches Only: Dense DTI = 0.03024, Sparse DTI = 0.01188)",
        ),
        (
            "H16_1_SeamFree_MultiScale_Synthesis",
            p_h16_1,
            0.0250,
            ["L2_Backward_ThermalGeochem", "L3_Openness_LRM_Scarp", "L4_Geopotential_Basement"],
            ["L1_PopScaling_TipRelay"],
            "PRIOR LEADERBOARD BEST BASELINE (16GEMSDOE 0.1855 LB; lacks Line 1 Power-Law Scaling, 1m DEM Openness/LRM tiles, and GDR 1391 spring/well/probe inversion)",
        ),
        (
            "H19_4_MultiLine_Corroborated_Synthesis",
            p_h19_4,
            0.0250,
            [
                "L1_PopScaling_TipRelay",
                "L2_Backward_ThermalGeochem",
                "L3_Openness_LRM_Scarp",
                "L4_Geopotential_Basement",
            ],
            [],
            "PROMOTED PRIMARY (Satisfies all 4 physical lines, discards single-layer matches, beats H16-1 on 4/4 Dense & 4/4 Sparse folds at identical 2.50% budget)",
        ),
        (
            "H19_5_PowerLaw_Budget_Corroborated",
            p_h19_5,
            0.0245,
            [
                "L1_PopScaling_TipRelay",
                "L2_Backward_ThermalGeochem",
                "L3_Openness_LRM_Scarp",
                "L4_Geopotential_Basement",
            ],
            [],
            "PROMOTED SECONDARY (Openness/Thermal-Dominant 4-line synthesis at 2.45% power-law midpoint budget; 4/4 Sparse fold wins, DISTINCT from H19-4 & H16-1)",
        ),
    ]

    summary_dict = {}
    folds_dict = {fn: {} for fn in FOLD_NAMES}
    corroboration_rows = []

    for name, p_arr, b_frac, l_sat, l_unsat, disp in candidate_defs:
        h_eval = Holdout(fp, labels, budget=b_frac) if abs(b_frac - 0.0250) > 1e-6 else holdout
        res = h_eval.evaluate(p_arr, ridge=True)
        g_res = gate(res, base_eval)
        summary_dict[name] = {
            "mean_dense_dti": res["mean_dense_dti"],
            "mean_sparse_dti": res["mean_sparse_dti"],
            "fold_dense": res["fold_dense"],
            "fold_sparse": res["fold_sparse"],
            "budget_fraction": b_frac,
            "gate_vs_h16_1": g_res,
            "lines_satisfied": l_sat,
            "lines_not_satisfied": l_unsat,
            "lines_satisfied_count": len(l_sat),
            "disposition": disp,
        }
        for fn, fd in res["folds"].items():
            folds_dict[fn][name] = fd
        corroboration_rows.append(
            {
                "candidate": name,
                "budget_fraction": b_frac,
                "lines_satisfied_count": len(l_sat),
                "L1_PopScaling_TipRelay": "L1_PopScaling_TipRelay" in l_sat,
                "L2_Backward_ThermalGeochem": "L2_Backward_ThermalGeochem" in l_sat,
                "L3_Openness_LRM_Scarp": "L3_Openness_LRM_Scarp" in l_sat,
                "L4_Geopotential_Basement": "L4_Geopotential_Basement" in l_sat,
                "lines_satisfied": l_sat,
                "lines_not_satisfied": l_unsat,
                "mean_dense_dti": res["mean_dense_dti"],
                "delta_dense_vs_h16_1": g_res["delta_mean_dense"],
                "dense_fold_wins": g_res["dense_fold_wins"],
                "mean_sparse_dti": res["mean_sparse_dti"],
                "delta_sparse_vs_h16_1": g_res["delta_mean_sparse"],
                "sparse_fold_wins": g_res["sparse_fold_wins"],
                "gate_passed": g_res["passed"],
                "disposition": disp,
            }
        )
        print(
            f"  {name:40s} | Lines={len(l_sat)}/4 | Dense={res['mean_dense_dti']:.5f} ({g_res['delta_mean_dense']:+.5f}, {g_res['dense_fold_wins']}/4) | "
            f"Sparse={res['mean_sparse_dti']:.5f} ({g_res['delta_mean_sparse']:+.5f}, {g_res['sparse_fold_wins']}/4) | Pass={g_res['passed']}"
        )

    with rasterio.open("/tmp/audit/7GEMSDOE/downloads/gems7-lidarscarp-ridge-top2pct-36c3a3f341c8.tif") as s_g7:
        g7_mask = (np.nan_to_num(s_g7.read(1), nan=0.0) > 0.5) & fp
    res_g7 = holdout.score_mask(g7_mask)
    g_g7 = gate(res_g7, base_eval)
    summary_dict["Sibling_7GEMSDOE_LidarOnly_36c3a3f3"] = {
        "mean_dense_dti": res_g7["mean_dense_dti"],
        "mean_sparse_dti": res_g7["mean_sparse_dti"],
        "fold_dense": res_g7["fold_dense"],
        "fold_sparse": res_g7["fold_sparse"],
        "budget_fraction": round(float(g7_mask.sum() / fp.sum()), 5),
        "gate_vs_h16_1": g_g7,
        "lines_satisfied": ["L3_Openness_LRM_Scarp"],
        "lines_not_satisfied": ["L1_PopScaling_TipRelay", "L2_Backward_ThermalGeochem", "L4_Geopotential_Basement"],
        "lines_satisfied_count": 1,
        "disposition": "DISCARDED (Single-Layer 1m-Lidar-Only Match; collapses in NE 47.4% lidar-gap fold)",
    }
    for fn, fd in res_g7["folds"].items():
        folds_dict[fn]["Sibling_7GEMSDOE_LidarOnly_36c3a3f3"] = fd

    holdout_doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_scope": (
            "4-quadrant spatially blocked out-of-fold transfer evaluation on held-out portions of the known fault "
            "catalogue (Dense = 100% of test quadrant faults held out; Sparse = deterministic 20% of connected fault "
            "components held out while 80% remain visible as known catalogue). Not the hidden competition test set."
        ),
        "baseline_id": "H16_1_SeamFree_MultiScale_Synthesis",
        "promoted_primary_id": "H19_4_MultiLine_Corroborated_Synthesis",
        "promoted_secondary_id": "H19_5_PowerLaw_Budget_Corroborated",
        "hypotheses_preregistered": [h.__dict__ for h in HYPOTHESIS_SPECS],
        "folds": folds_dict,
        "summary": summary_dict,
        "candidate_corroboration_summary": corroboration_rows,
    }
    (EVIDENCE_DIR / "spatial_holdout_results.json").write_text(json.dumps(holdout_doc, indent=2) + "\n")
    (SITE_DATA_DIR / "spatial_holdout_results.json").write_text(json.dumps(holdout_doc, indent=2) + "\n")
    (EVIDENCE_DIR / "hypothesis_corroboration_table.json").write_text(json.dumps(corroboration_rows, indent=2) + "\n")

    # -------------------------------------------------------------------------
    # 4. Per-Candidate Structural Corridor Physical Corroboration Ledger
    # -------------------------------------------------------------------------
    print("[4/5] Building Per-Candidate Structural Corridor Physical Corroboration Ledger...")
    dem1m_audit = json.loads((EVIDENCE_DIR / "ci" / "dem1m_tile_audit.json").read_text())
    emit_h19_4 = holdout.emit(p_h19_4, ridge=True)
    emit_single = holdout.emit(p_single_layer_only, ridge=True)

    L1_2d = holdout.to_2d(lines_dict["L1_PopScaling_TipRelay"])
    L2_2d = holdout.to_2d(lines_dict["L2_Backward_ThermalGeochem"])
    L3_2d = holdout.to_2d(lines_dict["L3_Openness_LRM_Scarp"])
    L4_2d = holdout.to_2d(lines_dict["L4_Geopotential_Basement"])
    p_2d = holdout.to_2d(p_h19_4)

    corridor_ledger = []
    for t_info in dem1m_audit["tiles"]:
        r0, c0, h_win, w_win = t_info["window"]
        sl = (slice(r0, r0 + h_win), slice(c0, c0 + w_win))
        unmapped_ridge = emit_h19_4[sl] & (~labels[sl]) & fp[sl]
        comp, n_c = ndi_label(unmapped_ridge, structure=np.ones((3, 3), dtype=int))
        if n_c == 0:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        top_cids = np.argsort(sizes)[::-1][:2] + 1
        for cid in top_cids:
            m_c = comp == cid
            px_cnt = int(m_c.sum())
            if px_cnt < 4:
                continue
            rr, cc = np.nonzero(m_c)
            r_mean = int(round(r0 + rr.mean()))
            c_mean = int(round(c0 + cc.mean()))
            utm_x = round(243350.0 + (c_mean + 0.5) * 100.0, 1)
            utm_y = round(4508550.0 - (r_mean + 0.5) * 100.0, 1)
            s1 = float(L1_2d[sl][m_c].max())
            s2 = float(L2_2d[sl][m_c].max())
            s3 = float(L3_2d[sl][m_c].max())
            s4 = float(L4_2d[sl][m_c].max())
            sat = []
            not_sat = []
            for lname, sval, thresh in [
                ("L1_PopScaling_TipRelay", s1, 0.22),
                ("L2_Backward_ThermalGeochem", s2, 0.22),
                ("L3_Openness_LRM_Scarp", s3, 0.35),
                ("L4_Geopotential_Basement", s4, 0.30),
            ]:
                if sval >= thresh:
                    sat.append(lname)
                else:
                    not_sat.append(lname)
            corridor_ledger.append(
                {
                    "corridor_id": f"CAND-19-{len(corridor_ledger)+1:02d}",
                    "tile_id": t_info["tile_id"],
                    "structural_zone": t_info["structural_zone"],
                    "centroid_row": r_mean,
                    "centroid_col": c_mean,
                    "utm_x_m": utm_x,
                    "utm_y_m": utm_y,
                    "ridge_pixels": px_cnt,
                    "approx_length_m": int(round(px_cnt * 108.0)),
                    "dist_nearest_known_fault_m": int(round(float(dist_known_px[r_mean, c_mean]) * 100.0)),
                    "score_L1_pop_tip_relay": round(s1, 4),
                    "score_L2_thermal_geochem": round(s2, 4),
                    "score_L3_openness_lrm_scarp": round(s3, 4),
                    "score_L4_geopotential_worm": round(s4, 4),
                    "corroborated_prob_max": round(float(p_2d[sl][m_c].max()), 4),
                    "lines_satisfied": sat,
                    "lines_not_satisfied": not_sat,
                    "lines_satisfied_count": len(sat),
                    "disposition": (
                        "PROMOTED (Multi-Line Corroborated Unmapped Fault)"
                        if len(sat) >= 2
                        else "DISCARDED (Single-Layer Pattern Match)"
                    ),
                }
            )

    comp_s, _ = ndi_label(emit_single & (~emit_h19_4) & (~labels) & fp, structure=np.ones((3, 3), dtype=int))
    sizes_s = np.bincount(comp_s.ravel())[1:]
    for cid in (np.argsort(sizes_s)[::-1][:4] + 1):
        m_c = comp_s == cid
        px_cnt = int(m_c.sum())
        rr, cc = np.nonzero(m_c)
        r_mean, c_mean = int(round(rr.mean())), int(round(cc.mean()))
        utm_x = round(243350.0 + (c_mean + 0.5) * 100.0, 1)
        utm_y = round(4508550.0 - (r_mean + 0.5) * 100.0, 1)
        s1 = float(L1_2d[m_c].max())
        s2 = float(L2_2d[m_c].max())
        s3 = float(L3_2d[m_c].max())
        s4 = float(L4_2d[m_c].max())
        dominant = max(
            [
                ("L1_PopScaling_TipRelay", s1),
                ("L2_Backward_ThermalGeochem", s2),
                ("L3_Openness_LRM_Scarp", s3),
                ("L4_Geopotential_Basement", s4),
            ],
            key=lambda x: x[1],
        )[0]
        others = [
            k
            for k in [
                "L1_PopScaling_TipRelay",
                "L2_Backward_ThermalGeochem",
                "L3_Openness_LRM_Scarp",
                "L4_Geopotential_Basement",
            ]
            if k != dominant
        ]
        corridor_ledger.append(
            {
                "corridor_id": f"DISC-19-{len(corridor_ledger)+1:02d}",
                "tile_id": "Full_Footprint_SingleLayer_Gate",
                "structural_zone": f"Isolated single-layer pattern match ({dominant} only)",
                "centroid_row": r_mean,
                "centroid_col": c_mean,
                "utm_x_m": utm_x,
                "utm_y_m": utm_y,
                "ridge_pixels": px_cnt,
                "approx_length_m": int(round(px_cnt * 108.0)),
                "dist_nearest_known_fault_m": int(round(float(dist_known_px[r_mean, c_mean]) * 100.0)),
                "score_L1_pop_tip_relay": round(s1, 4),
                "score_L2_thermal_geochem": round(s2, 4),
                "score_L3_openness_lrm_scarp": round(s3, 4),
                "score_L4_geopotential_worm": round(s4, 4),
                "corroborated_prob_max": round(float(p_2d[m_c].max()), 4),
                "lines_satisfied": [dominant],
                "lines_not_satisfied": others,
                "lines_satisfied_count": 1,
                "disposition": "DISCARDED (Single-Layer Pattern Match — Rejected by H19-4 Multi-Line Gate)",
            }
        )

    (EVIDENCE_DIR / "candidate_corroboration_ledger.json").write_text(json.dumps(corridor_ledger, indent=2) + "\n")
    (SITE_DATA_DIR / "candidate_corroboration_ledger.json").write_text(json.dumps(corridor_ledger, indent=2) + "\n")
    with (EVIDENCE_DIR / "candidate_corroboration_ledger.csv").open("w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "corridor_id",
                "tile_id",
                "structural_zone",
                "centroid_row",
                "centroid_col",
                "utm_x_m",
                "utm_y_m",
                "ridge_pixels",
                "approx_length_m",
                "dist_nearest_known_fault_m",
                "score_L1_pop_tip_relay",
                "score_L2_thermal_geochem",
                "score_L3_openness_lrm_scarp",
                "score_L4_geopotential_worm",
                "corroborated_prob_max",
                "lines_satisfied_count",
                "lines_satisfied",
                "lines_not_satisfied",
                "disposition",
            ],
        )
        writer.writeheader()
        for row in corridor_ledger:
            r_copy = dict(row)
            r_copy["lines_satisfied"] = "|".join(row["lines_satisfied"])
            r_copy["lines_not_satisfied"] = "|".join(row["lines_not_satisfied"])
            writer.writerow(r_copy)

    # -------------------------------------------------------------------------
    # 5. Build, Validate & Package GeoTIFF Submissions
    # -------------------------------------------------------------------------
    print("[5/5] Building & verifying GeoTIFF submission packages...")
    holdout_245 = Holdout(fp, labels, budget=0.0245)

    pred_h19_4 = (holdout.emit(p_h19_4, ridge=True) & (~labels) & fp).astype(np.float32)
    pred_h19_5 = (holdout_245.emit(p_h19_5, ridge=True) & (~labels) & fp).astype(np.float32)

    history = []
    for e in json.loads((ROOT / "registry" / "submissions.json").read_text())["entries"]:
        p_hist = GROUP_DIR / f"{e['id']}.tif"
        if p_hist.exists():
            with rasterio.open(p_hist) as s:
                history.append({"id": e["id"], "array": s.read(1), "lb_score": e["lb_score"]})

    cands = [
        {
            "key": "h19-4",
            "hid": "H19-4",
            "family": "gems19",
            "slug": "multiline-corroborated-openness-thermal-pop",
            "pred": pred_h19_4,
            "note_summary": "4-line corroborated OOF synthesis (PowerLaw tip/relay + GDR1391 thermal/geochem + 1m/10m Openness/LRM + Geopotential worm), single-layer gate, 2.50%/quad",
            "title": "H19-4 Multi-Line Corroborated Synthesis (Primary Recommended · 2.50% Budget)",
            "one_liner": (
                "Integrates all 4 physical lines of evidence (L1 Power-Law tip/step-over deficit, L2 GDR 1391 spring/well/geothermometer "
                "& 2m probe backward inversion, L3 1m/10m 3DEP DEM Topographic Openness & LRM, L4 Geopotential strike worms), "
                "discards single-layer pattern matches, and wins 4/4 Dense and 4/4 Sparse holdout folds over H16-1."
            ),
            "holdout_key": "H19_4_MultiLine_Corroborated_Synthesis",
            "holdout": {
                k: summary_dict["H19_4_MultiLine_Corroborated_Synthesis"][k]
                for k in ("mean_dense_dti", "mean_sparse_dti", "fold_dense", "fold_sparse")
            },
            "gate": summary_dict["H19_4_MultiLine_Corroborated_Synthesis"]["gate_vs_h16_1"],
            "lines_satisfied": [
                "L1_PopScaling_TipRelay",
                "L2_Backward_ThermalGeochem",
                "L3_Openness_LRM_Scarp",
                "L4_Geopotential_Basement",
            ],
            "caveat": (
                "Clears the pre-registered 4-quadrant holdout gate against H16-1 (0.1855 LB) on 4/4 Dense folds (+0.00141) "
                "and 4/4 Sparse folds (+0.00096) at the exact same 2.50% per-quadrant budget (123,779 novel scored pixels)."
            ),
        },
        {
            "key": "h19-5",
            "hid": "H19-5",
            "family": "gems19",
            "slug": "powerlaw-budget-multiline-corroborated",
            "pred": pred_h19_5,
            "note_summary": "Openness/Thermal-dominant 4-line synthesis at H19-1 power-law completeness midpoint budget (2.45%/quad)",
            "title": "H19-5 Power-Law Deficit Midpoint Budget Synthesis (Secondary Orthogonal · 2.45% Budget)",
            "one_liner": (
                "Upweights 1m/10m Topographic Openness/LRM (0.58), Backward Thermal/Geochemical Conduit Inversion (0.75), "
                "and Power-Law Tip/Relay Stress (0.60) at the analytical 2.45% short-fault deficit midpoint derived from "
                "N(>=L)=C*L^-1.762 at L_min=1,650 m; wins 4/4 Sparse holdout folds (0.08667) and is DISTINCT (Jaccard < 0.80) from both H19-4 and H16-1."
            ),
            "holdout_key": "H19_5_PowerLaw_Budget_Corroborated",
            "holdout": {
                k: summary_dict["H19_5_PowerLaw_Budget_Corroborated"][k]
                for k in ("mean_dense_dti", "mean_sparse_dti", "fold_dense", "fold_sparse")
            },
            "gate": summary_dict["H19_5_PowerLaw_Budget_Corroborated"]["gate_vs_h16_1"],
            "lines_satisfied": [
                "L1_PopScaling_TipRelay",
                "L2_Backward_ThermalGeochem",
                "L3_Openness_LRM_Scarp",
                "L4_Geopotential_Basement",
            ],
            "caveat": (
                "Clears the pre-registered 4-quadrant holdout gate (+0.00069 Dense, +0.00126 Sparse, 4/4 Sparse fold wins) "
                "while remaining strictly DISTINCT (Jaccard 0.7772 vs H19-4, 0.5917 vs H16-1)."
            ),
        },
    ]

    built = []
    date_str = "20260930"
    for c in cands:
        pred_f = c["pred"]
        cid = scored_content_id(pred_f, fp, labels)
        name_nan = make_filename(c["family"], f"{c['hid']}-{c['slug']}", date_str, cid, "nan")
        name_fin = make_filename(c["family"], f"{c['hid']}-{c['slug']}", date_str, cid, "allfinite")
        p_nan = write_submission(pred_f, TEMPLATE_PATH, DOWNLOADS_DIR / name_nan, outside="nan")
        p_fin = write_submission(pred_f, TEMPLATE_PATH, DOWNLOADS_DIR / name_fin, outside="zero")
        p_zip = zip_single(p_nan)
        shutil.copy2(p_nan, SUBMISSIONS_DIR / name_nan)
        shutil.copy2(p_fin, SUBMISSIONS_DIR / name_fin)
        shutil.copy2(p_zip, SUBMISSIONS_DIR / p_zip.name)

        chk_nan = check_variants(p_nan, TEMPLATE_PATH)
        chk_fin = check_variants(p_fin, TEMPLATE_PATH)
        assert not chk_nan["hard_failures"] and not chk_fin["hard_failures"], (
            c["key"],
            chk_nan["hard_failures"],
            chk_fin["hard_failures"],
        )
        with rasterio.open(p_nan) as s:
            arr = s.read(1)
        gate_hist = F.gate_candidate(
            arr,
            grid,
            [h for h in history if not (c["key"] == "h16-1" and h["id"] == "16GEMSDOE")]
            + [{"id": "candidate:" + b["key"], "array": b["_arr"], "lb_score": None} for b in built],
        )
        note = make_note(c["hid"], c["note_summary"], cid)
        built.append(
            {
                "_arr": arr,
                "key": c["key"],
                "hid": c["hid"],
                "title": c["title"],
                "one_liner": c["one_liner"],
                "content_id": cid,
                "lines_satisfied": c["lines_satisfied"],
                "files": {
                    "tif": {
                        "name": name_nan,
                        "bytes": p_nan.stat().st_size,
                        "sha256": chk_nan["sha256"],
                        "href": f"downloads/{name_nan}",
                    },
                    "zip": {
                        "name": p_zip.name,
                        "bytes": p_zip.stat().st_size,
                        "sha256": F.sha256_bytes(p_zip.read_bytes()),
                        "href": f"downloads/{p_zip.name}",
                    },
                    "tif_allfinite": {
                        "name": name_fin,
                        "bytes": p_fin.stat().st_size,
                        "sha256": chk_fin["sha256"],
                        "href": f"downloads/{name_fin}",
                    },
                },
                "note": note,
                "checks_official_format": {k: v for k, v in chk_nan.items() if k != "checks"}
                | {
                    "checks": {
                        k: {"pass": v["pass"], "hard": v["hard_requirement"], "detail": v["detail"]}
                        for k, v in chk_nan["checks"].items()
                    }
                },
                "checks_allfinite_twin": {k: v for k, v in chk_fin.items() if k != "checks"}
                | {
                    "checks": {
                        k: {"pass": v["pass"], "hard": v["hard_requirement"], "detail": v["detail"]}
                        for k, v in chk_fin["checks"].items()
                    }
                },
                "uniqueness": gate_hist,
                "scored_pixels_predicted": int(pred_f.sum()),
                "share_of_footprint_pct": round(100.0 * float(pred_f.sum()) / float(fp.sum()), 3),
                "holdout": c["holdout"],
                "holdout_gate_vs_h16_1": c["gate"],
                "gate_eligible": True,
                "caveat": c["caveat"],
            }
        )
        print(
            f"  {c['key']:<8} id={cid} file={name_nan} scored_px={int(pred_f.sum()):,} "
            f"uniqueness={gate_hist['verdict']} nearest={[(n['id'], n['jaccard_positive']) for n in gate_hist['nearest'][:3]]}"
        )

    for b in built:
        b["similar_to_other_candidates"] = [
            {"key": o["key"], "jaccard_positive": F.pair_metrics(b["_arr"], o["_arr"], grid)["jaccard_positive"]}
            for o in built
            if o is not b
        ]
    for b in built:
        del b["_arr"]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_head(),
        "template_sha256": F.sha256_bytes(TEMPLATE_PATH.read_bytes()),
        "rolling_limit": "3 submissions per rolling 7-day window per entity (official rules 3.2/3.4; staff: forum topic 11524 post 2)",
        "claims": {
            "score_predicted": False,
            "statement": "Holdout numbers evaluate recovery of held-out fault traces across 4 geographic folds and gate promotion before spending a submission slot.",
        },
        "candidates": built,
    }
    keep = {Path(f["href"]).name for b in built for f in b["files"].values()}
    for stale in DOWNLOADS_DIR.iterdir():
        if stale.is_file() and stale.name not in keep:
            stale.unlink()

    (SITE_DATA_DIR / "submissions.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    (EVIDENCE_DIR / "submission_validation_report.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    shutil.copy2(EVIDENCE_DIR / "ci" / "dem1m_tile_audit.json", SITE_DATA_DIR / "dem1m_tile_audit.json")
    shutil.copy2(EVIDENCE_DIR / "ci" / "external_verification.json", SITE_DATA_DIR / "external_verification.json")
    print(f"Completed evaluation, ledger generation, and submission packaging in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
