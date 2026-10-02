#!/usr/bin/env python3
"""Step 00 -- re-assert every published constant against the bytes on disk.

Nothing downstream is allowed to run on an assumption.  This script re-hashes the
three official rasters, re-measures the grid, footprint, catalogue and nodata
structure, and writes evidence/data_provenance.json.  Any drift fails loudly.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22.spec import (DATA, N_CATALOGUE, N_FOOTPRINT, N_OUTSIDE, N_TOTAL,
                         SHA256_PINS, Spec, load_features, load_labels)

BAND_CATEGORIES_EXPECTED = {
    "magnetic_data", "gravity_data", "geodetic_strain", "topographic",
    "subsurface", "seismic",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "hashes": {}, "checks": {}, "measurements": {}, "flags": []}
    ok = True
    for name, pin in SHA256_PINS.items():
        p = DATA / name
        if not p.exists():
            out["hashes"][name] = {"exists": False, "pin": pin}
            out["flags"].append(f"F-00 MISSING {p} -- run scripts/fetch_data.sh")
            ok = False
            continue
        got = sha256(p)
        match = got == pin
        ok &= match
        out["hashes"][name] = {"exists": True, "bytes": p.stat().st_size,
                               "sha256": got, "pin": pin, "match": match}
        print(f"{'PASS' if match else 'FAIL'} sha256 {name} = {got}")

    spec = Spec.load()
    checks = spec.verify()
    out["checks"]["spec"] = checks
    ok &= all(checks.values())
    out["measurements"]["spec"] = spec.summary()
    print(f"{'PASS' if all(checks.values()) else 'FAIL'} spec.verify() {checks}")

    cat = load_labels()
    A, names = load_features()
    with rasterio.open(DATA / "training_features.tif") as ds:
        cats = sorted({ds.tags(i).get("data_category", "?") for i in range(1, ds.count + 1)})
        band_names_tag = [ds.tags(i).get("band_name", "") for i in range(1, ds.count + 1)]
    out["measurements"]["band_names"] = names
    out["measurements"]["band_names_from_tags"] = band_names_tag
    out["measurements"]["band_categories"] = cats
    out["measurements"]["n_catalogue"] = int(cat.sum())
    out["measurements"]["n_footprint"] = int(spec.footprint.sum())
    out["measurements"]["n_outside"] = int((~spec.footprint).sum())
    out["measurements"]["n_total"] = int(spec.footprint.size)
    out["measurements"]["n_feature_all_finite_in_footprint"] = int(spec.feature_valid.sum())
    out["measurements"]["n_sentinel_pixels_inside_footprint"] = \
        int((spec.footprint & ~spec.feature_valid).sum())
    per_band = []
    for i, nm in enumerate(names):
        bad = int((~np.isfinite(A[i]) & spec.footprint).sum())
        per_band.append({"band": i + 1, "name": nm,
                         "non_finite_inside_footprint": bad})
    out["measurements"]["per_band_nonfinite_inside_footprint"] = per_band

    # ---- flags -------------------------------------------------------------
    n_sent = out["measurements"]["n_sentinel_pixels_inside_footprint"]
    if n_sent > 0:
        out["flags"].append(
            f"F-03 {n_sent} pixels INSIDE the official footprint carry the float32 "
            f"sentinel -3.4028235e+38 in at least one of the 19 bands "
            f"({', '.join(p['name'] for p in per_band if p['non_finite_inside_footprint'])}). "
            "Any submission that lets a derived value propagate from these pixels is "
            "rejected with 'Predicted values must be in range [0, 1]'. "
            "gems22.submission.sanitise() forces them to 0.0 before clipping.")
    with rasterio.open(DATA / "sample_submission.tif") as ds:
        s = ds.read(1)
    n_one = int((s == 1).sum())
    if n_one > 0:
        out["flags"].append(
            f"F-04 IRREGULARITY: the mirrored 'example_submission.tif' "
            f"(= data/sample_submission.tif, sha256 {SHA256_PINS['sample_submission.tif'][:16]}) "
            f"contains {n_one} pixels equal to 1.0, exactly the {int(cat.sum())} "
            "positive pixels of labels.tif.  The problem description says the sample "
            "'predicts total fault absence'.  The file is still a valid FORMAT template "
            "(CRS/shape/transform/dtype/nodata all match), so it is used here only for "
            "the footprint mask and the write profile -- never as a prediction.")
    if "radiometric" not in " ".join(cats).lower():
        out["flags"].append(
            "F-06 The 19 competition bands carry data_category values "
            f"{cats} -- there is NO radiometric family, although GeoDAWN is an "
            "'airborne magnetic AND RADIOMETRIC' survey (USGS doi:10.5066/P93LGLVQ). "
            "The K/Th/U/TC grids from that same release are therefore genuinely "
            "orthogonal information and are used as external features.")

    out["all_checks_pass"] = bool(ok)
    (Path(__file__).resolve().parents[1] / "evidence").mkdir(exist_ok=True)
    p = Path(__file__).resolve().parents[1] / "evidence/data_provenance.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nmeasurements: footprint={out['measurements']['n_footprint']:,} "
          f"outside={out['measurements']['n_outside']:,} "
          f"catalogue={out['measurements']['n_catalogue']:,} "
          f"sentinel_inside_footprint={n_sent:,}")
    for f in out["flags"]:
        print("FLAG:", f[:190], "...")
    print(f"\nwrote {p}   all_checks_pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
