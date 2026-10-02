"""Independent verification and data extraction against OFFICIAL open-data sources on GitHub-hosted runners.

Steps:
  A  Official provenance of ``labels.tif`` AND vector fault trace length extraction from GDR INGENIOUS Quaternary
     faults v1 and v2 (DOI 10.15121/1881483) -> writes ``evidence/ci/gdr_qfaults_traces.csv``.
  B  GDR INGENIOUS thermal & geochemical compilation (DOI 10.15121/1881483):
     extracts all footprint records from ``wellspringdata.gdb.zip`` (spring & well temperature and chemistry,
     including quartz, chalcedony, and cation geothermometers), ``2m_temperature_probe``, ``paleo_geothermal``,
     and ``great_basin_q_volcanics`` -> writes ``evidence/ci/gdr_wellspring_in_footprint.csv`` and
     ``evidence/ci/gdr_volcanic_vents_in_footprint.csv``.
  C  USGS SGMC geologic-map faults (NV, CA) -> writes ``evidence/ci/derived_sgmc_faults_100m_u8.tif``.
  D  Targeted 1 m USGS 3DEP DEM Openness (Yokoyama et al. 2002) and Local Relief Model (Hesse 2010) on the
     handful of high-prior 10 km x 10 km tiles flagged jointly by power-law short-fault tip/step-over deficit
     and backward thermal/geochemical anomaly inversion -> writes ``evidence/ci/dem1m_high_prior_openness_lrm.npz``
     and ``evidence/ci/dem1m_tile_audit.json``.

No DrivenData host is contacted (DrivenData Terms of Use prohibit automated access).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "ci"
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path(tempfile.mkdtemp(prefix="gems19_ci_"))

RAW = "https://raw.githubusercontent.com/buffedlizard55-lab"
GDR = "https://gdr.openei.org/files/1391"
S3_1M = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects"

SRC = {
    "labels": f"{RAW}/GEMSDOE/main/data/bridge/existing_faults.tif",
    "template": f"{RAW}/GEMSDOE/main/data/bridge/example_submission.tif",
    "ens12": f"{RAW}/5GEMSDOE/main/data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif",
    "lidar7": f"{RAW}/7GEMSDOE/main/downloads/gems7-lidarscarp-ridge-top2pct-36c3a3f341c8.tif",
    "h16_1": f"{RAW}/16GEMSDOE/main/docs/downloads/gems16-h16-1-topo-geophys-baseline-ridges-20260930-df20f65e-nan.tif",
    "gdr_qfaults_v2": f"{GDR}/qfaults_ingenious_nad83conus117_2023-06-27.zip",
    "gdr_qfaults_v1": f"{GDR}/faults_quaternary_INGENIOUS_regional_data.zip",
    "gdr_wellspring": f"{GDR}/wellspringdata.gdb.zip",
    "gdr_paleo": f"{GDR}/paleo_geothermal_regional.zip",
    "gdr_volcanics": f"{GDR}/great_basin_q_volcanics.zip",
    "gdr_2m_probes": f"{GDR}/2m_temperature_probe_INGENIOUS_regional_data.zip",
    "sgmc_nv": "https://mrdata.usgs.gov/geology/state/shp/NV.zip",
    "sgmc_ca": "https://mrdata.usgs.gov/geology/state/shp/CA.zip",
}

# Top 8 high-prior 10 km x 10 km 1m DEM tiles flagged jointly by:
#   (1) Power-law short-fault deficit near mapped fault tips & step-overs (L >= L_min)
#   (2) Backward thermal & geochemical orphan anomalies (>500 m from mapped faults)
HIGH_PRIOR_1M_TILES = [
    {
        "tile_id": "USGS_1M_11_x34y441_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x34y441_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Desert Queen / Hot Springs Mountains step-over & fault tips",
        "tips_in_tile": 18,
        "orphan_thermal_anomalies_1500m": 40,
        "joint_prior_score": 0.036471,
    },
    {
        "tile_id": "USGS_1M_11_x36y436_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x36y436_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Stillwater / Salt Wells / Carson Sink accommodation & step-over zone",
        "tips_in_tile": 13,
        "orphan_thermal_anomalies_1500m": 38,
        "joint_prior_score": 0.024895,
    },
    {
        "tile_id": "USGS_1M_11_x32y441_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x32y441_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Bradys / Patua / Hazen displacement transfer & relay ramp",
        "tips_in_tile": 18,
        "orphan_thermal_anomalies_1500m": 28,
        "joint_prior_score": 0.021425,
    },
    {
        "tile_id": "USGS_1M_11_x26y446_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x26y446_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Astor Pass / Pyramid Lake tufa & fault termination corridor",
        "tips_in_tile": 6,
        "orphan_thermal_anomalies_1500m": 94,
        "joint_prior_score": 0.021152,
    },
    {
        "tile_id": "USGS_1M_11_x33y441_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x33y441_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Bradys / Desert Peak step-over & blind geothermal corridor",
        "tips_in_tile": 9,
        "orphan_thermal_anomalies_1500m": 53,
        "joint_prior_score": 0.020921,
    },
    {
        "tile_id": "USGS_1M_11_x38y430_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x38y430_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Southern Dixie Valley / Pirouette Mountain piedmont step-over",
        "tips_in_tile": 5,
        "orphan_thermal_anomalies_1500m": 49,
        "joint_prior_score": 0.013850,
    },
    {
        "tile_id": "USGS_1M_11_x39y430_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x39y430_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Dixie Valley / Clan Alpine range-front step-over",
        "tips_in_tile": 8,
        "orphan_thermal_anomalies_1500m": 46,
        "joint_prior_score": 0.013034,
    },
    {
        "tile_id": "USGS_1M_11_x32y439_NV_WestCentral_EarthMRI_2020_D20",
        "project": "NV_WestCentral_EarthMRI_2020_D20",
        "url": f"{S3_1M}/NV_WestCentral_EarthMRI_2020_D20/TIFF/USGS_1M_11_x32y439_NV_WestCentral_EarthMRI_2020_D20.tif",
        "structural_zone": "Soda Lake / Upsal Hogback blind quaternary basin-floor fault zone",
        "tips_in_tile": 16,
        "orphan_thermal_anomalies_1500m": 13,
        "joint_prior_score": 0.013011,
    },
]

GRID_CRS = "EPSG:32611"
GRID_TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
GRID_SHAPE = (3730, 3292)

report: dict = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "downloads": {},
    "steps": {},
}


def fetch(key: str) -> Path | None:
    url = SRC[key]
    dest = WORK / Path(url).name
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "19GEMSDOE-source-verification/1.0 (research; contact via repo)"},
        )
        with urllib.request.urlopen(req, timeout=240) as r:
            data = r.read()
        dest.write_bytes(data)
        report["downloads"][key] = {
            "url": url,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "ok": True,
        }
        return dest
    except Exception as exc:  # noqa: BLE001
        report["downloads"][key] = {"url": url, "ok": False, "error": repr(exc)[:300]}
        return None


def step(name: str):
    def deco(fn):
        def run(*a, **k):
            try:
                report["steps"][name] = fn(*a, **k)
            except Exception as exc:  # noqa: BLE001
                report["steps"][name] = {
                    "ok": False,
                    "error": repr(exc)[:400],
                    "trace": traceback.format_exc()[-1500:],
                }
            print(f"[{name}] done: {str(report['steps'][name])[:300]}", flush=True)
        return run
    return deco


def compute_openness_and_lrm_2m(z: np.ndarray, valid: np.ndarray, res: float = 2.0) -> dict[str, np.ndarray]:
    """Compute Topographic Openness (Yokoyama et al. 2002) and Local Relief Model (Hesse 2010) on a 2m DEM grid.

    Returns:
      - lrm_abs: |LRM(x)| purged local relief step (m) at 20m-50m wavelength
      - lrm_grad: |grad(LRM)| sharp breakline gradient across subtle buried scarps (m/m)
      - openness_diff: (Phi_neg - Phi_pos) openness scarp asymmetry (radians)
      - openness_dipole: local crest-toe openness dipole (max Phi_pos - min Phi_pos in 12m window)
      - valid: valid sample mask
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, maximum_filter, minimum_filter

    if valid.all():
        zf = z.astype(np.float32)
    else:
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        zf = z[tuple(idx)].astype(np.float32)

    px = lambda m: m / res  # noqa: E731
    pad = int(np.ceil(px(80.0))) + 2
    zp = np.pad(zf, pad, mode="reflect", reflect_type="odd")
    crop = (slice(pad, -pad), slice(pad, -pad))

    # 1. Local Relief Model (Hesse 2010):
    # Step 1: Smooth DEM with a 25 m Gaussian low-pass filter
    z_lp1 = gaussian_filter(zp, px(25.0), mode="nearest")
    d0 = zp - z_lp1
    # Step 2: Purge local relief mounds/scarps (|d0| > 0.35 m) and re-interpolate smooth trend surface
    breakline_mask = np.abs(d0) <= 0.35
    if breakline_mask.any() and (~breakline_mask).any():
        b_idx = distance_transform_edt(~breakline_mask, return_distances=False, return_indices=True)
        z_purged = zp[tuple(b_idx)]
    else:
        z_purged = z_lp1
    z_trend = gaussian_filter(z_purged, px(12.0), mode="nearest")
    lrm = gaussian_filter(zp, px(2.0), mode="nearest") - z_trend
    gy_lrm, gx_lrm = np.gradient(lrm, res)
    lrm_grad = np.hypot(gx_lrm, gy_lrm)

    # 2. Topographic Openness (Yokoyama et al. 2002):
    # Evaluate elevation angles along 8 azimuths at radii R = 6m, 12m, 24m
    z_s = gaussian_filter(zp, px(2.0), mode="nearest")
    dirs = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
    ]
    radii_px = [max(1, int(round(r_m / res))) for r_m in (6.0, 12.0, 24.0)]
    pos_sum = np.zeros_like(z_s, dtype=np.float32)
    neg_sum = np.zeros_like(z_s, dtype=np.float32)

    H, W = z_s.shape
    for dy, dx in dirs:
        step_scale = np.hypot(dy, dx) * res
        max_ang = np.full_like(z_s, -np.pi / 2, dtype=np.float32)
        min_ang = np.full_like(z_s, np.pi / 2, dtype=np.float32)
        for rp in radii_px:
            dist_m = rp * step_scale
            shifted = np.roll(np.roll(z_s, -dy * rp, axis=0), -dx * rp, axis=1)
            ang = np.arctan2(shifted - z_s, dist_m).astype(np.float32)
            max_ang = np.maximum(max_ang, ang)
            min_ang = np.minimum(min_ang, ang)
        # Zenith / nadir angles in radians:
        pos_sum += (np.pi / 2.0 - max_ang)
        neg_sum += (np.pi / 2.0 + min_ang)

    phi_pos = pos_sum / 8.0
    phi_neg = neg_sum / 8.0
    openness_asymm = np.abs(phi_pos - phi_neg)
    openness_dipole = maximum_filter(phi_pos, size=5) + maximum_filter(phi_neg, size=5) - np.pi

    safe = binary_erosion(valid, iterations=int(np.ceil(px(40.0))), border_value=1)
    out = {}
    for k, arr in (
        ("lrm_abs_max", np.abs(lrm[crop])),
        ("lrm_grad_max", lrm_grad[crop]),
        ("openness_asymm_max", openness_asymm[crop]),
        ("openness_dipole_max", np.maximum(openness_dipole[crop], 0.0)),
    ):
        a = arr.astype(np.float32)
        a[~safe] = np.nan
        out[k] = a
    out["valid"] = valid.astype(np.float32)
    return out


def main() -> None:
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    import rasterio
    from rasterio import features
    from rasterio.enums import Resampling
    from rasterio.transform import Affine, array_bounds
    from rasterio.warp import reproject, transform_bounds
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import box

    p_lab, p_tmp = fetch("labels"), fetch("template")
    if not (p_lab and p_tmp):
        report["fatal"] = "could not fetch competition-grid copy from GitHub"
        (OUT / "external_verification.json").write_text(json.dumps(report, indent=2) + "\n")
        return

    with rasterio.open(p_tmp) as s:
        footprint = np.isfinite(s.read(1))
        transform, crs, shape = s.transform, s.crs, s.shape
    with rasterio.open(p_lab) as s:
        labels = (s.read(1) > 0) & footprint
    dist_cat = distance_transform_edt(~labels).astype(np.float32)
    n_lab = int(labels.sum())
    grid_box = box(243350, 4135550, 572550, 4508550)

    def read_any(zip_path: Path):
        ex = WORK / (zip_path.stem + "_x")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ex)
        out = {}
        cands = [p for p in ex.rglob("*") if p.suffix.lower() == ".shp"] + [
            p for p in ex.rglob("*.gdb") if p.is_dir()
        ]
        for c in cands:
            try:
                if c.suffix.lower() == ".gdb":
                    for lname, _g in pyogrio.list_layers(c):
                        out[f"{c.name}:{lname}"] = gpd.read_file(c, layer=lname)
                else:
                    out[c.name] = gpd.read_file(c)
            except Exception as exc:  # noqa: BLE001
                out[f"{c.name}:ERROR"] = repr(exc)[:200]
        return out

    def to_utm(gdf):
        if gdf.crs is None:
            gdf = gdf.set_crs(4269)
        return gdf.to_crs(32611)

    def raster_lines(gdf, all_touched=False):
        shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
        if not shapes:
            return np.zeros(shape, dtype=bool)
        return (
            features.rasterize(
                shapes, out_shape=shape, transform=transform, all_touched=all_touched, dtype="uint8"
            ).astype(bool)
            & footprint
        )

    def compare_to_labels(R):
        nR = int(R.sum())
        dR = distance_transform_edt(~R)
        out = {"raster_pixels": nR, "labels_pixels": n_lab, "exact_overlap_px": int((R & labels).sum())}
        for k in (0, 1, 2, 3):
            out[f"labels_within_{k}px_of_raster_frac"] = round(float((dR[labels] <= k).mean()), 4)
            out[f"raster_within_{k}px_of_labels_frac"] = round(float((dist_cat[R] <= k).mean()), 4) if nR else None
        return out

    # ---------------- A: labels provenance + exact fault trace lengths ---------------------------------
    @step("A_labels_provenance_vs_GDR_qfaults")
    def step_a():
        res = {
            "ok": True,
            "official_source": "https://gdr.openei.org/submissions/1391 (DOI 10.15121/1881483)",
            "versions": {},
        }
        for key in ("gdr_qfaults_v2", "gdr_qfaults_v1"):
            zp = fetch(key)
            if not zp:
                res["versions"][key] = {"ok": False, "error": report["downloads"][key].get("error")}
                continue
            layers = read_any(zp)
            v = {"ok": True, "layers": {}}
            merged = None
            for lname, gdf in layers.items():
                if isinstance(gdf, str):
                    v["layers"][lname] = gdf
                    continue
                v["layers"][lname] = {
                    "rows": int(len(gdf)),
                    "crs": str(gdf.crs),
                    "geom_types": sorted(gdf.geom_type.dropna().unique().tolist()),
                    "columns": [c for c in gdf.columns][:25],
                }
                if gdf.geom_type.astype(str).str.contains("Line").any():
                    g = to_utm(gdf[gdf.geom_type.astype(str).str.contains("Line")])
                    g = g[g.intersects(grid_box)]
                    merged = g if merged is None else gpd.GeoDataFrame(pd.concat([merged, g]), crs=32611)
            if merged is not None and len(merged):
                v["lines_in_grid_bounds"] = int(len(merged))
                v["all_touched_false"] = compare_to_labels(raster_lines(merged, False))
                v["all_touched_true"] = compare_to_labels(raster_lines(merged, True))
                if key == "gdr_qfaults_v2":
                    # Save exact vector trace lengths & attributes for power-law scaling analysis
                    rows_out = []
                    for idx, row in merged.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        clipped = geom.intersection(grid_box)
                        if clipped.is_empty:
                            continue
                        c_pt = clipped.centroid
                        r_px = int((4508550.0 - c_pt.y) / 100.0)
                        c_px = int((c_pt.x - 243350.0) / 100.0)
                        in_fp = bool(
                            0 <= r_px < shape[0] and 0 <= c_px < shape[1] and footprint[r_px, c_px]
                        )
                        rows_out.append({
                            "trace_id": int(idx),
                            "name": str(row.get("NAME", "")),
                            "slip_rate": str(row.get("SLIPRT2023", "")),
                            "recency": str(row.get("REC2023", "")),
                            "dip_direct": str(row.get("DIPDIRECT", "")),
                            "slip_sense": str(row.get("SLIPSENSE", "")),
                            "map_scale": str(row.get("MAPSCALE", "")),
                            "full_length_m": round(float(geom.length), 2),
                            "clipped_length_m": round(float(clipped.length), 2),
                            "centroid_utm_x": round(float(c_pt.x), 1),
                            "centroid_utm_y": round(float(c_pt.y), 1),
                            "centroid_row": r_px,
                            "centroid_col": c_px,
                            "centroid_in_footprint": int(in_fp),
                        })
                    df_tr = pd.DataFrame(rows_out)
                    df_tr.to_csv(OUT / "gdr_qfaults_traces.csv", index=False)
                    v["saved_traces_csv"] = "evidence/ci/gdr_qfaults_traces.csv"
                    v["saved_traces_count"] = len(df_tr)
            res["versions"][key] = v
        return res

    step_a()

    # ---------------- B: thermal & geochemical features + CSV extraction -------------------------------
    @step("B_thermal_features")
    def step_b():
        res = {"ok": True, "datasets": {}}
        rng = np.random.default_rng(7)
        fp_rc = np.column_stack(np.nonzero(footprint))
        rand = fp_rc[rng.choice(len(fp_rc), 50000, replace=False)]
        rand_d = dist_cat[rand[:, 0], rand[:, 1]]
        res["random_footprint_pixels_distance_to_known_fault_px"] = {
            "p25": float(np.percentile(rand_d, 25)),
            "median": float(np.median(rand_d)),
            "p75": float(np.percentile(rand_d, 75)),
            "frac_within_10px_1km": float((rand_d <= 10).mean()),
        }
        wellspring_rows = []
        volcanic_rows = []

        for key in ("gdr_wellspring", "gdr_paleo", "gdr_volcanics", "gdr_2m_probes"):
            zp = fetch(key)
            if not zp:
                res["datasets"][key] = {"ok": False, "error": report["downloads"][key].get("error")}
                continue
            layers = read_any(zp)
            dd = {"ok": True, "layers": {}}
            for lname, gdf in layers.items():
                if isinstance(gdf, str):
                    dd["layers"][lname] = gdf
                    continue
                info = {
                    "rows": int(len(gdf)),
                    "crs": str(gdf.crs),
                    "geom_types": sorted(gdf.geom_type.dropna().unique().tolist()),
                    "columns": list(gdf.columns)[:30],
                }
                try:
                    g = to_utm(gdf)
                    pts = g.geometry.representative_point()
                    r = ((4508550.0 - pts.y.values) / 100.0).astype(int)
                    c = ((pts.x.values - 243350.0) / 100.0).astype(int)
                    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
                    inside = np.zeros(len(g), bool)
                    inside[ok] = footprint[r[ok], c[ok]]
                    info["in_footprint"] = int(inside.sum())
                    if inside.any():
                        d = dist_cat[r[inside], c[inside]]
                        info["distance_to_known_fault_px"] = {
                            "p25": float(np.percentile(d, 25)),
                            "median": float(np.median(d)),
                            "p75": float(np.percentile(d, 75)),
                            "frac_within_10px_1km": round(float((d <= 10).mean()), 4),
                            "frac_beyond_30px_3km": round(float((d > 30).mean()), 4),
                        }
                        # Extract thermal/geochemical records inside footprint
                        if key == "gdr_wellspring":
                            sub_g = g.iloc[np.where(inside)[0]].copy()
                            sub_r = r[inside]
                            sub_c = c[inside]
                            sub_x = pts.x.values[inside]
                            sub_y = pts.y.values[inside]
                            sub_d = d
                            short_layer = lname.split(":")[-1]
                            for i_row, (_, rec) in enumerate(sub_g.iterrows()):
                                def _num(col_candidates):
                                    for cc in col_candidates:
                                        if cc in rec and pd.notna(rec[cc]):
                                            try:
                                                v_f = float(rec[cc])
                                                if np.isfinite(v_f) and -50 < v_f < 500:
                                                    return round(v_f, 2)
                                            except Exception:  # noqa: BLE001
                                                pass
                                    return np.nan

                                t_meas = _num(["temp_c", "maxmeasuredtemp_c", "measuredtemp_c", "bottommeasuredtemp_c", "correctedbottomtemp_c", "surfacemeasuredtemp_c"])
                                t_qtz = _num(["geothermquartz_c"])
                                t_chalc = _num(["geothermchalc_c"])
                                t_cat = _num(["geothermcat_c"])
                                t_class = str(rec.get("thermalclass", "") or "")
                                # Keep spring records and any well record with temperature or geothermometer >= 25 C or thermalclass
                                is_spring = "spring" in short_layer
                                max_t = np.nanmax([t_meas, t_qtz, t_chalc, t_cat, -999.0])
                                if is_spring or max_t >= 25.0 or (t_class and t_class.lower() not in ("none", "nan", "cold", "non-thermal")):
                                    wellspring_rows.append({
                                        "layer": short_layer,
                                        "name": str(rec.get("springname", rec.get("wellname", "")) or "")[:60],
                                        "thermalclass": t_class[:30],
                                        "row": int(sub_r[i_row]),
                                        "col": int(sub_c[i_row]),
                                        "utm_x": round(float(sub_x[i_row]), 1),
                                        "utm_y": round(float(sub_y[i_row]), 1),
                                        "dist_known_fault_px": round(float(sub_d[i_row]), 2),
                                        "temp_c": t_meas,
                                        "geothermquartz_c": t_qtz,
                                        "geothermchalc_c": t_chalc,
                                        "geothermcat_c": t_cat,
                                    })
                        elif key == "gdr_volcanics" and "vents" in lname:
                            sub_g = g.iloc[np.where(inside)[0]].copy()
                            sub_r = r[inside]
                            sub_c = c[inside]
                            sub_d = d
                            for i_row, (_, rec) in enumerate(sub_g.iterrows()):
                                volcanic_rows.append({
                                    "name": str(rec.get("Name", "") or ""),
                                    "rock_type": str(rec.get("Rock_Type", "") or ""),
                                    "age": str(rec.get("Geo_Age", "") or ""),
                                    "row": int(sub_r[i_row]),
                                    "col": int(sub_c[i_row]),
                                    "dist_known_fault_px": round(float(sub_d[i_row]), 2),
                                })
                except Exception as exc:  # noqa: BLE001
                    info["reproject_error"] = repr(exc)[:200]
                dd["layers"][lname] = info
            res["datasets"][key] = dd

        if wellspring_rows:
            df_ws = pd.DataFrame(wellspring_rows)
            df_ws.to_csv(OUT / "gdr_wellspring_in_footprint.csv", index=False)
            res["saved_wellspring_csv"] = {
                "path": "evidence/ci/gdr_wellspring_in_footprint.csv",
                "rows": len(df_ws),
            }
        if volcanic_rows:
            df_vv = pd.DataFrame(volcanic_rows)
            df_vv.to_csv(OUT / "gdr_volcanic_vents_in_footprint.csv", index=False)
            res["saved_volcanics_csv"] = {
                "path": "evidence/ci/gdr_volcanic_vents_in_footprint.csv",
                "rows": len(df_vv),
            }
        return res

    step_b()

    # ---------------- C: SGMC geologic-map faults ------------------------------------------------------
    @step("C_sgmc_faults")
    def step_c():
        res = {
            "ok": True,
            "source": "https://mrdata.usgs.gov/geology/state/ (SGMC; nominal scale 1:1,000,000 per ScienceBase item 5888bf4fe4b05ccb964bab9d)",
            "states": {},
        }
        fault_gdfs = []
        for key in ("sgmc_nv", "sgmc_ca"):
            zp = fetch(key)
            if not zp:
                res["states"][key] = {"ok": False, "error": report["downloads"][key].get("error")}
                continue
            layers = read_any(zp)
            st = {"ok": True, "layers": {}}
            for lname, gdf in layers.items():
                if isinstance(gdf, str):
                    st["layers"][lname] = gdf
                    continue
                if not gdf.geom_type.astype(str).str.contains("Line").any():
                    continue
                cols = list(gdf.columns)
                text_cols = [c for c in cols if c != "geometry" and gdf[c].dtype == object]
                mask = np.zeros(len(gdf), bool)
                for tc in text_cols:
                    mask |= gdf[tc].astype(str).str.contains("fault|thrust|fissure|scarp|lineament", case=False, na=False).values
                fg = to_utm(gdf[mask if mask.any() else slice(None)])
                fg = fg[fg.intersects(grid_box)]
                st["layers"][lname] = {
                    "rows": int(len(gdf)),
                    "rows_mentioning_fault_or_thrust": int(mask.sum()),
                    "in_grid_bounds": int(len(fg)),
                }
                if len(fg):
                    fault_gdfs.append(fg[["geometry"]])
            res["states"][key] = st
        if fault_gdfs:
            all_f = gpd.GeoDataFrame(pd.concat(fault_gdfs, ignore_index=True), crs=32611)
            R = raster_lines(all_f, all_touched=False)
            off = R & (dist_cat > 3)
            res["fault_lines_in_grid"] = int(len(all_f))
            res["sgmc_fault_pixels_in_footprint"] = int(R.sum())
            res["sgmc_off_catalogue_pixels_gt_3px"] = int(off.sum())
            out_tif = OUT / "derived_sgmc_faults_100m_u8.tif"
            with rasterio.open(
                out_tif,
                "w",
                driver="GTiff",
                height=shape[0],
                width=shape[1],
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(R.astype(np.uint8), 1)
            res["derived_raster"] = {"path": "evidence/ci/derived_sgmc_faults_100m_u8.tif", "bytes": out_tif.stat().st_size}
        return res

    step_c()

    # ---------------- D: Targeted 1 m USGS 3DEP DEM Openness & Local Relief Model ----------------------
    @step("D_targeted_1m_dem_openness_lrm")
    def step_d():
        tile_logs = []
        channel_names = ["lrm_abs_max", "lrm_grad_max", "openness_asymm_max", "openness_dipole_max", "valid"]
        mosaic = {ch: np.zeros(GRID_SHAPE, dtype=np.float32) for ch in channel_names}
        work_res = 2.0  # 2m native sub-grid resolution for 1m DEM openness/LRM

        for spec in HIGH_PRIOR_1M_TILES:
            t0 = time.time()
            tid = spec["tile_id"]
            url = spec["url"]
            local_tif = WORK / f"{tid}.tif"
            entry = dict(spec)
            try:
                subprocess.run(
                    ["curl", "-fsSL", "--retry", "3", "--connect-timeout", "30", "-o", str(local_tif), url],
                    check=True,
                )
                entry["bytes"] = local_tif.stat().st_size
                h = hashlib.sha256()
                with local_tif.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(chunk)
                entry["sha256"] = h.hexdigest()

                with rasterio.open(local_tif) as src:
                    fac = work_res / abs(src.res[0])
                    oh, ow = int(round(src.height / fac)), int(round(src.width / fac))
                    z = src.read(1, out_shape=(oh, ow), resampling=Resampling.average, masked=True)
                    valid = ~np.ma.getmaskarray(z) & np.isfinite(z.data) & (z.data > -1000) & (z.data < 9000)
                    src_t = src.transform * src.transform.scale(src.width / ow, src.height / oh)
                    src_crs = src.crs
                    ch_2m = compute_openness_and_lrm_2m(z.data.astype(np.float32), valid, res=work_res)

                # Aggregate onto the 100m competition grid window
                bounds = array_bounds(oh, ow, src_t)
                gb = transform_bounds(src_crs, GRID_CRS, *bounds, densify_pts=21)
                a, b, c, d, e, f = GRID_TRANSFORM
                c0 = max(0, int(np.floor((gb[0] - c) / a)) - 1)
                c1 = min(GRID_SHAPE[1], int(np.ceil((gb[2] - c) / a)) + 1)
                r0 = max(0, int(np.floor((f - gb[3]) / -e)) - 1)
                r1 = min(GRID_SHAPE[0], int(np.ceil((f - gb[1]) / -e)) + 1)
                hh, ww = r1 - r0, c1 - c0
                dst_t = Affine(a, b, c + c0 * a, d, e, f + r0 * e)

                for ch_name in channel_names:
                    dst = np.full((hh, ww), np.nan, dtype=np.float32)
                    src_arr = ch_2m[ch_name]
                    if ch_name == "valid":
                        src_arr = np.where(np.isfinite(src_arr), src_arr, 0.0).astype(np.float32)
                        resamp = Resampling.average
                    else:
                        resamp = Resampling.max
                    reproject(
                        src_arr,
                        dst,
                        src_transform=src_t,
                        src_crs=src_crs,
                        src_nodata=np.nan,
                        dst_transform=dst_t,
                        dst_crs=GRID_CRS,
                        dst_nodata=np.nan,
                        resampling=resamp,
                    )
                    cur = mosaic[ch_name][r0:r1, c0:c1]
                    mosaic[ch_name][r0:r1, c0:c1] = np.fmax(cur, np.nan_to_num(dst, nan=0.0))

                valid_win = (mosaic["valid"][r0:r1, c0:c1] > 0.5) & footprint[r0:r1, c0:c1]
                entry.update(
                    status="ok",
                    window=[int(r0), int(c0), int(hh), int(ww)],
                    valid_footprint_cells_100m=int(valid_win.sum()),
                    mean_lrm_abs_m=round(float(mosaic["lrm_abs_max"][r0:r1, c0:c1][valid_win].mean()), 4) if valid_win.any() else 0.0,
                    mean_lrm_grad=round(float(mosaic["lrm_grad_max"][r0:r1, c0:c1][valid_win].mean()), 4) if valid_win.any() else 0.0,
                    mean_openness_asymm_rad=round(float(mosaic["openness_asymm_max"][r0:r1, c0:c1][valid_win].mean()), 4) if valid_win.any() else 0.0,
                    seconds=round(time.time() - t0, 2),
                )
            except Exception as exc:  # noqa: BLE001
                entry.update(status="failed", error=repr(exc)[:300], seconds=round(time.time() - t0, 2))
            finally:
                local_tif.unlink(missing_ok=True)
            tile_logs.append(entry)
            print(f"  [1m Tile {tid}] status={entry['status']} sec={entry['seconds']}", flush=True)

        # Save sparse footprint-indexed arrays for the covered cells so the artifact is compact (<1 MB)
        fp_idx = np.flatnonzero(footprint.ravel())
        cov_mask_fp = mosaic["valid"].ravel()[fp_idx] > 0.5
        cov_fp_indices = np.flatnonzero(cov_mask_fp).astype(np.int32)
        npz_path = OUT / "dem1m_high_prior_openness_lrm.npz"
        np.savez_compressed(
            npz_path,
            cov_fp_indices=cov_fp_indices,
            lrm_abs_max=mosaic["lrm_abs_max"].ravel()[fp_idx][cov_fp_indices].astype(np.float16),
            lrm_grad_max=mosaic["lrm_grad_max"].ravel()[fp_idx][cov_fp_indices].astype(np.float16),
            openness_asymm_max=mosaic["openness_asymm_max"].ravel()[fp_idx][cov_fp_indices].astype(np.float16),
            openness_dipole_max=mosaic["openness_dipole_max"].ravel()[fp_idx][cov_fp_indices].astype(np.float16),
        )
        audit_summary = {
            "ok": True,
            "method": "Topographic Openness (Yokoyama et al. 2002) + Local Relief Model (Hesse 2010) on 1m USGS 3DEP DEM tiles",
            "tiles_requested": len(HIGH_PRIOR_1M_TILES),
            "tiles_succeeded": sum(1 for t in tile_logs if t.get("status") == "ok"),
            "total_covered_footprint_cells_100m": int(len(cov_fp_indices)),
            "npz_artifact": "evidence/ci/dem1m_high_prior_openness_lrm.npz",
            "npz_bytes": npz_path.stat().st_size,
            "tiles": tile_logs,
        }
        (OUT / "dem1m_tile_audit.json").write_text(json.dumps(audit_summary, indent=2) + "\n")
        return audit_summary

    step_d()

    (OUT / "external_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Wrote evidence/ci/external_verification.json", flush=True)


if __name__ == "__main__":
    main()
