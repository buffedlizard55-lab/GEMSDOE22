"""GeoTIFF submission writing and the pre-flight checks that prevent the
"Predicted values must be in range [0, 1]" rejection.

ROOT CAUSE OF THAT ERROR (measured in this repo, see evidence/data_provenance.json):
`training_features.tif` carries the float32 sentinel -3.4028234663852886e+38 on
3,073 pixels that are INSIDE the official footprint (5,167,373 finite pixels in
`sample_submission.tif` versus 5,164,300 pixels where all 19 bands are finite;
band 6 `tc` alone is sentinel on 12 extra pixels).  Any submission that lets a
derived value propagate from those pixels into the output is instantly outside
[0, 1] and is rejected before scoring.

Every writer here therefore:
  1. works only inside the official footprint mask;
  2. replaces every non-finite value with 0.0 BEFORE clipping;
  3. clips to [0, 1] and re-reads the written file to confirm;
  4. writes the exact official profile (float32, single band, EPSG:32611,
     3292 x 3730, transform (100, 0, 243350, 0, -100, 4508550)).

TWO OUTSIDE-FOOTPRINT CONVENTIONS ARE EMITTED, because the official text and the
observed platform behaviour differ:
  * `-nan`       : NaN on all 7,111,787 outside pixels, nodata = NaN.  This
                   matches `sample_submission.tif` byte-for-byte in convention
                   and the problem-description sentence "data outside the bounds
                   is null or nan".
  * `-allfinite` : 0.0 on all outside pixels, no NaN anywhere.  Immune to any
                   scorer path that does not use NaN-aware reductions.
EMPIRICAL EVIDENCE THEY SCORE IDENTICALLY: the group registry contains
`r7-nms3-dem10-scarp_0c9199f14e62` = 0.1294 and
`r7-nms3-dem10-scarp_0c9199f14e62_allfinite` = 0.1294 -- the same prediction
under both conventions, same public score.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from .spec import HEIGHT, N_CATALOGUE, N_FOOTPRINT, N_OUTSIDE, N_TOTAL, TRANSFORM, WIDTH

PROFILE_BASE = {
    "driver": "GTiff",
    "dtype": "float32",
    "width": WIDTH,
    "height": HEIGHT,
    "count": 1,
    "crs": "EPSG:32611",
    "transform": rasterio.Affine(*TRANSFORM),
    "compress": "lzw",
    "tiled": False,
    "interleave": "band",
}


@dataclass(frozen=True)
class Written:
    path: Path
    sha256: str
    bytes: int
    n_positive_footprint: int
    n_positive_scored: int
    variant: str
    checks: dict


def sanitise(values: np.ndarray, footprint: np.ndarray,
             catalogue: np.ndarray | None = None) -> np.ndarray:
    """Coerce a raw score field into a legal in-footprint prediction.

    * non-finite -> 0.0 (this is what kills the sentinel-pixel rejection)
    * clip to [0, 1]
    * outside the footprint -> 0.0
    * optionally zero the known-catalogue pixels.  Per DrivenData staff
      (forum topic 11516, post 2) those pixels are excluded from evaluation
      entirely, so emitting there is score-neutral; zeroing them spends no
      emitted-pixel budget on pixels that cannot earn credit.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.shape != footprint.shape:
        raise ValueError(f"shape {v.shape} != footprint {footprint.shape}")
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
    v = np.clip(v, 0.0, 1.0)
    v = np.where(footprint, v, 0.0)
    if catalogue is not None:
        v = np.where(catalogue, 0.0, v)
    return v.astype(np.float32)


def write_submission(values: np.ndarray, out_path: Path, footprint: np.ndarray,
                     variant: str = "allfinite",
                     catalogue: np.ndarray | None = None,
                     strict_shape: bool = True) -> Written:
    """Write and immediately re-verify one submission GeoTIFF.

    `strict_shape=True` (the default, and what every real submission uses) refuses
    to write anything that is not exactly the official 3730 x 3292 grid.  Tests set
    it False so they can exercise the sanitisation logic on small synthetic grids.
    """
    if variant not in ("allfinite", "nan"):
        raise ValueError("variant must be 'allfinite' or 'nan'")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = sanitise(values, footprint, catalogue)
    if strict_shape and v.shape != (HEIGHT, WIDTH):
        raise ValueError(f"submission must be {(HEIGHT, WIDTH)}, got {v.shape}")
    arr = v.copy()
    prof = dict(PROFILE_BASE)
    prof["height"], prof["width"] = v.shape
    if variant == "nan":
        arr = np.where(footprint, arr, np.nan).astype(np.float32)
        prof["nodata"] = float("nan")
    else:
        prof["nodata"] = None
    with rasterio.open(out_path, "w", **prof) as ds:
        ds.write(arr, 1)
        ds.update_tags(AREA_OR_POINT="Area")
    chk = verify_file(out_path, footprint, catalogue, strict_shape=strict_shape)
    assert chk["hard_all_pass"], json.dumps(chk, indent=1, default=str)
    return Written(path=out_path, sha256=chk["sha256"], bytes=chk["bytes"],
                   n_positive_footprint=chk["n_positive_footprint"],
                   n_positive_scored=chk["n_positive_scored"], variant=variant,
                   checks=chk)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, footprint: np.ndarray,
                catalogue: np.ndarray | None = None,
                strict_shape: bool = True) -> dict:
    """Re-read a written file from disk and run every hard + advisory check.

    This is deliberately a *fresh read of the file on disk*, not a check of the
    in-memory array, so it catches driver-level surprises (compression, nodata
    tagging, dtype promotion).
    """
    path = Path(path)
    with rasterio.open(path) as ds:
        a = ds.read(1)
        prof = ds.profile.copy()
        crs = str(ds.crs)
        nodata = ds.nodata
        tr = tuple(ds.transform)[:6]
        dt = ds.dtypes[0]
        nb = ds.count
        shp = (ds.height, ds.width)
    fin = np.isfinite(a)
    inside = a[footprint]
    outside = a[~footprint]
    scored = inside if catalogue is None else a[footprint & ~catalogue]
    checks = {
        "single_band": nb == 1,
        "dtype_float32": dt == "float32",
        "crs_epsg_32611": crs.upper().endswith("32611"),
        "shape_3730x3292": (shp == (HEIGHT, WIDTH)) if strict_shape else (shp == footprint.shape),
        "geotransform_matches_official": tr == TRANSFORM,
        "footprint_all_finite": bool(fin[footprint].all()),
        "footprint_min_ge_0": bool(np.min(inside) >= 0.0) if inside.size else False,
        "footprint_max_le_1": bool(np.max(inside) <= 1.0) if inside.size else False,
        "scored_all_finite": bool(np.isfinite(scored).all()),
        "scored_min_ge_0": bool(np.min(scored) >= 0.0) if scored.size else False,
        "scored_max_le_1": bool(np.max(scored) <= 1.0) if scored.size else False,
        # three independent range probes a scorer might use
        "probe_nanaware_whole_array": bool(
            np.nanmin(a) >= 0.0 and np.nanmax(a) <= 1.0),
        "probe_masked_read": bool(
            np.min(rasterio.open(path).read(1, masked=True).compressed()) >= 0.0),
        "probe_strict_no_nan_anywhere": bool(np.isfinite(a).all()),
        "n_footprint_pixels": int(footprint.sum()),
        "n_outside_pixels": int((~footprint).sum()),
    }
    hard = ["single_band", "dtype_float32", "crs_epsg_32611", "shape_3730x3292"] + (
            ["geotransform_matches_official"] if strict_shape else []) + [
            "footprint_all_finite",
            "footprint_min_ge_0", "footprint_max_le_1", "scored_all_finite",
            "scored_min_ge_0", "scored_max_le_1", "probe_nanaware_whole_array"]
    return {
        "path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size,
        "profile": {k: str(v) for k, v in prof.items()},
        "nodata": (None if nodata is None else ("nan" if np.isnan(nodata) else float(nodata))),
        "checks": checks,
        "hard_all_pass": all(checks[k] for k in hard),
        "hard_keys": hard,
        "n_positive_footprint": int((inside > 0).sum()),
        "n_positive_scored": int((scored > 0).sum()),
        "positive_on_catalogue": (int((a[catalogue] > 0).sum()) if catalogue is not None else None),
        "in_footprint_nan": int((~fin[footprint]).sum()),
        "outside_all_nan": bool(np.isnan(outside).all()) if outside.size else False,
        "outside_all_zero": bool(np.all(outside == 0.0)) if outside.size else False,
        "unique_values": int(np.unique(a[np.isfinite(a)]).size),
        "is_binary_0_1": bool(set(np.unique(a[np.isfinite(a)]).tolist()) <= {0.0, 1.0}),
        "footprint_frac_positive": float((inside > 0).mean()),
    }


def zip_submission(tif: Path) -> Path:
    """DrivenData also accepts '.zip containing a single GeoTIFF'."""
    tif = Path(tif)
    zp = tif.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tif, arcname=tif.name)
    return zp


def make_submission_name(tag: str, content_id: str, variant: str,
                         when: str | None = None) -> str:
    """Unique, self-describing filename.

    Pattern: gems22-<tag>-<UTC stamp>-<content id>-<variant>.tif
    The content id is the first 8 hex of the SHA-256 of the scored pixel set, so
    two files with the same id are provably the same prediction.
    """
    when = when or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"gems22-{tag}-{when}-{content_id}-{variant}.tif"


def content_id(values: np.ndarray, footprint: np.ndarray,
               catalogue: np.ndarray | None = None) -> str:
    """Hash of the scored pixel set only -- ignores the outside convention."""
    v = np.asarray(values)
    m = footprint if catalogue is None else (footprint & ~catalogue)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(v[m]).astype(np.float32).tobytes())
    h.update(str(int(m.sum())).encode())
    return h.hexdigest()[:8]


def registry_record(w: Written, note: str, extra: dict | None = None) -> dict:
    rec = {"file": w.path.name, "variant": w.variant, "sha256": w.sha256,
           "bytes": w.bytes, "n_positive_footprint": w.n_positive_footprint,
           "n_positive_scored": w.n_positive_scored,
           "drivendata_note": note,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "hard_all_pass": w.checks["hard_all_pass"],
           # the full per-check verdicts, so the published site can render them
           # instead of guessing at a schema (and so a reviewer can audit them)
           "checks": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                      for k, v in w.checks.items()}}
    if extra:
        rec.update(extra)
    return rec
