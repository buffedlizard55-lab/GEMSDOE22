"""Build, verify and label DrivenData GEMS submission GeoTIFFs."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from .validator import sha256_file

EXPECTED = {
    "crs": "EPSG:32611",
    "shape": (3730, 3292),
    "transform": (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0),
    "dtype": "float32",
    "bounds": (243350.0, 4135550.0, 572550.0, 4508550.0),
    "footprint_pixels": 5167373,
    "outside_pixels": 7111787,
}


def sanitize(pred: np.ndarray) -> np.ndarray:
    """NaN/Inf-safe float32 in [0, 1]. NaN -> 0 (no evidence), +Inf -> 1, -Inf -> 0, then clip."""
    a = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def write_submission(
    pred_2d: np.ndarray,
    template_path: Path | str,
    out_path: Path | str,
    *,
    outside: str = "nan",
) -> Path:
    """Write ``pred_2d`` using the *template's own raster profile*."""
    if outside not in ("nan", "zero"):
        raise ValueError("outside must be 'nan' or 'zero'")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_path) as t:
        profile = t.profile.copy()
        footprint = np.isfinite(t.read(1))
    if pred_2d.shape != footprint.shape:
        raise ValueError(f"prediction shape {pred_2d.shape} != template {footprint.shape}")
    arr = sanitize(pred_2d)
    arr = np.where(footprint, arr, np.float32(np.nan) if outside == "nan" else np.float32(0.0)).astype(np.float32)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=(np.nan if outside == "nan" else None))
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(AREA_OR_POINT="Area")
    return out_path


def zip_single(tif_path: Path | str, zip_path: Path | str | None = None) -> Path:
    tif_path = Path(tif_path)
    zip_path = Path(zip_path) if zip_path else tif_path.with_suffix(".zip")
    info = zipfile.ZipInfo(tif_path.name, date_time=(2026, 9, 30, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(info, tif_path.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert names == [tif_path.name], names
    return zip_path


def check_variants(path: Path | str, template_path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with rasterio.open(template_path) as t:
        t_arr = t.read(1)
        t_prof = t.profile
        footprint = np.isfinite(t_arr)
        t_crs, t_tr, t_shape = t.crs, t.transform, t.shape
    with rasterio.open(path) as s:
        arr = s.read(1)
        masked = s.read(1, masked=True)
        prof = s.profile
        crs, tr, shape, count, dtypes, nodata = s.crs, s.transform, s.shape, s.count, s.dtypes, s.nodata
    shape_ok = arr.shape == footprint.shape
    if shape_ok:
        inside, outside = arr[footprint], arr[~footprint]
    else:
        inside = outside = np.array([], dtype=np.float32)
    fin_in = np.isfinite(inside)

    def rng_ok(v: np.ndarray) -> bool:
        return bool(v.size == 0 or (np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0))

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, ok: bool, detail: str, *, hard: bool = True) -> None:
        checks[name] = {"pass": bool(ok), "detail": detail, "hard_requirement": hard}

    add("single_band", count == 1, f"count={count}")
    add("dtype_float32", dtypes == ("float32",), f"dtypes={dtypes}")
    add("crs_epsg_32611", crs == t_crs, f"{crs}")
    add("shape_matches_template", shape == t_shape, f"{shape} vs {t_shape}")
    add("geotransform_matches_template", tr == t_tr, f"{tuple(tr)[:6]}")
    if shape_ok:
        add(
            "footprint_all_finite",
            bool(fin_in.all()),
            f"{int((~fin_in).sum())} NaN/Inf inside the {int(footprint.sum()):,}-pixel footprint",
        )
        add(
            "footprint_range_0_1",
            rng_ok(inside),
            f"min={float(np.nanmin(inside)):.6g} max={float(np.nanmax(inside)):.6g}",
        )
        add(
            "outside_is_nan_official_text",
            bool(np.isnan(outside).all()),
            f"{int(np.isnan(outside).sum()):,} of {outside.size:,} outside pixels are NaN",
            hard=False,
        )
    else:
        for name in ("footprint_all_finite", "footprint_range_0_1"):
            add(name, False, "not evaluated: the raster does not have the template's shape")
        add("outside_is_nan_official_text", False, "not evaluated: the raster does not have the template's shape", hard=False)
    add("nodata_tag_is_nan", nodata is not None and np.isnan(nodata), f"nodata={nodata}", hard=False)
    add("variant_nan_aware_whole_array", rng_ok(arr), "np.nanmin/np.nanmax over the whole array")
    add("variant_masked_read", rng_ok(masked.compressed()), "rasterio read(masked=True).compressed()")
    add(
        "variant_strict_whole_array_no_nan_allowed",
        bool(((arr >= 0) & (arr <= 1)).all()),
        "((a>=0)&(a<=1)).all() over the whole array - fails for ANY NaN, including the official sample's NaN outside",
        hard=False,
    )

    def _same(a, b) -> bool:
        if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
            return True
        return a == b

    same_profile = all(
        _same(prof.get(k), t_prof.get(k))
        for k in ("driver", "dtype", "nodata", "width", "height", "crs", "transform")
    )
    add(
        "profile_matches_official_sample",
        bool(same_profile),
        f"driver/dtype/nodata/size/crs/transform equal to template; compress={prof.get('compress')} (template {t_prof.get('compress')})",
        hard=False,
    )
    hard_fail = [k for k, v in checks.items() if v["hard_requirement"] and not v["pass"]]
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "ok_to_upload": not hard_fail,
        "hard_failures": hard_fail,
        "official_format_compliant": checks["outside_is_nan_official_text"]["pass"] and not hard_fail,
        "in_footprint_positive_pixels": int((inside > 0).sum()),
        "in_footprint_mass": round(float(np.nansum(inside, dtype=np.float64)), 2) if inside.size else 0.0,
        "checks": checks,
    }


def scored_content_id(pred_2d: np.ndarray, footprint: np.ndarray, catalogue: np.ndarray) -> str:
    scored = footprint & ~catalogue
    v = (np.nan_to_num(pred_2d, nan=0.0)[scored] > 0.5).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:8]


def make_filename(family: str, hypothesis: str, date: str, content_id: str, outside: str = "nan") -> str:
    slug = "".join(c if c.isalnum() else "-" for c in hypothesis.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"{family}-{slug}-{date}-{content_id}-{outside}.tif"


def make_note(hypothesis_id: str, summary: str, content_id: str) -> str:
    prefix = f"19GEMSDOE {hypothesis_id} | "
    suffix = f" | id {content_id} | not yet live-scored"
    max_sum = 200 - len(prefix) - len(suffix)
    return f"{prefix}{summary[:max_sum]}{suffix}"


def dump_json(obj: Any, path: Path | str) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float) + "\n")
