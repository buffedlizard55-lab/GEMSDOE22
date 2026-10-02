"""Profile the 19-band competition feature raster and record verified facts (evidence/feature_profile.json).

Facts recorded: per-band name/category/description (from the file's own tags), observed footprint min/max/mean,
invalid (-3.4e38) pixels inside the footprint, valid pixels outside it, and the identity check that band 6 (`tc`) is
radiometric total count (correlation with the external USGS GeoDAWN TC channel vs a computed magnetic tilt angle).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.paths import DATA_DIR, EVIDENCE_DIR, FEATURES_PATH, LABELS_PATH, TEMPLATE_PATH  # noqa: E402


def main() -> None:
    with rasterio.open(TEMPLATE_PATH) as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(LABELS_PATH) as s:
        lab = s.read(1)
    assert np.array_equal(lab >= 0, fp), "labels footprint differs from template footprint"
    bands = []
    with rasterio.open(FEATURES_PATH) as s:
        nodata = s.nodata
        arrs = {}
        for i in range(1, s.count + 1):
            a = s.read(i)
            t = s.tags(i)
            ok = np.isfinite(a) & (a > -1e30)
            v = a[ok & fp]
            bands.append({
                "band": i, "name": t.get("band_name"), "category": t.get("data_category"), "description": t.get("description"),
                "footprint_min": float(v.min()), "footprint_max": float(v.max()), "footprint_mean": round(float(v.mean()), 4),
                "invalid_inside_footprint": int((fp & ~ok).sum()), "valid_outside_footprint": int((~fp & ok).sum()),
            })
            if i in (3, 6, 9):
                arrs[i] = (a, ok)
    tc, ok_tc = arrs[6]
    hg, ok_hg = arrs[3]
    vg, ok_vg = arrs[9]
    identity = {}
    rad = DATA_DIR / "external" / "geodawn_rad_u8.tif"
    if rad.exists():
        with rasterio.open(rad) as s:
            tc_ext = s.read(4).astype(np.float32)
        m = fp & ok_tc & (tc_ext > 0)
        identity["corr_band6_vs_external_geodawn_TC"] = round(float(np.corrcoef(tc[m], tc_ext[m])[0, 1]), 4)
    m = fp & ok_hg & ok_vg & ok_tc      # all three bands must be valid or the -3.4e38 sentinel contaminates the correlation
    tilt = np.degrees(np.arctan2(vg[m], hg[m]))
    identity["corr_band6_vs_computed_magnetic_tilt"] = round(float(np.corrcoef(tilt, tc[m])[0, 1]), 4)
    identity["band6_min"] = float(tc[fp & ok_tc].min())
    identity["band6_negative_values"] = int((tc[fp & ok_tc] < 0).sum())
    identity["computed_tilt_range_deg"] = [round(float(tilt.min()), 1), round(float(tilt.max()), 1)]
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nodata_sentinel": nodata, "footprint_pixels": int(fp.sum()),
        "labels_footprint_equals_template_footprint": True,
        "bands": bands, "tc_identity_check": identity,
    }
    (EVIDENCE_DIR / "feature_profile.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(identity, indent=1))
    print("invalid inside footprint per band:", sorted({b["invalid_inside_footprint"] for b in bands}))


if __name__ == "__main__":
    main()
