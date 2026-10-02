"""Verify competition rasters and prepare derived multi-resolution tectonic feature caches.

Usage:
    bash scripts/download_competition_data.sh
    python scripts/prepare_data.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.paths import DATA_DIR, EVIDENCE_DIR  # noqa: E402

EXPECTED = {
    "training_features.tif": {
        "sha256": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
        "bytes": 418912844,
        "count": 19,
        "dtype": "float32",
    },
    "labels.tif": {
        "sha256": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
        "bytes": 425830,
        "count": 1,
        "dtype": "int8",
    },
    "sample_submission.tif": {
        "sha256": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
        "bytes": 1599597,
        "count": 1,
        "dtype": "float32",
    },
    "external/lidar_scarp_features_u8.tif": {
        "sha256": "d580bb8bdcdb941e32fefb8b38044bc5bf04e199bf2e83498c3576e6fc465568",
        "bytes": 36943606,
        "count": 12,
        "dtype": "uint8",
    },
    "external/geodawn_rad_u8.tif": {
        "sha256": "c22420f75999030d7cc65c9e31e50d232ea6158423bca051613a18a8b20ba682",
        "bytes": 26612970,
        "count": 4,
        "dtype": "uint8",
    },
    "external/geodawn_extensions_u8.tif": {
        "sha256": "a35a9c6d2a14786f4dab85481ee59769213072f5dab5b2535ea82ae4d9bb7d9b",
        "bytes": 27132925,
        "count": 4,
        "dtype": "uint8",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rasters": {},
        "dem10_channels": {},
        "irregularities_verified": [],
    }

    for rel, spec in EXPECTED.items():
        p = DATA_DIR / rel
        if not p.exists():
            sys.exit(f"Missing required file {p}. Run `bash scripts/download_competition_data.sh` first.")
        actual_sha = sha256_file(p)
        actual_bytes = p.stat().st_size
        if actual_sha != spec["sha256"]:
            sys.exit(f"SHA-256 mismatch for {rel}: {actual_sha} != {spec['sha256']}")
        if actual_bytes != spec["bytes"]:
            sys.exit(f"File-size mismatch for {rel}: {actual_bytes} != {spec['bytes']}")
        with rasterio.open(p) as src:
            assert src.shape == (3730, 3292), f"Unexpected shape for {rel}: {src.shape}"
            assert str(src.crs) == "EPSG:32611", f"Unexpected CRS for {rel}: {src.crs}"
            assert src.count == spec["count"], f"Unexpected band count for {rel}: {src.count}"
            assert src.dtypes[0] == spec["dtype"], f"Unexpected dtype for {rel}: {src.dtypes[0]}"
            report["rasters"][rel] = {
                "sha256": actual_sha,
                "bytes": actual_bytes,
                "shape": list(src.shape),
                "crs": str(src.crs),
                "transform": list(src.transform)[:6],
                "count": src.count,
                "dtype": src.dtypes[0],
                "descriptions": list(src.descriptions),
                "status": "VERIFIED",
            }

    with rasterio.open(DATA_DIR / "sample_submission.tif") as s_sub, rasterio.open(
        DATA_DIR / "labels.tif"
    ) as s_lab:
        sub_arr = s_sub.read(1)
        lab_arr = s_lab.read(1)
        footprint = np.isfinite(sub_arr)
        sub_pos = (sub_arr > 0) & footprint
        lab_pos = (lab_arr > 0) & footprint
        identical_to_catalogue = bool(np.array_equal(sub_pos, lab_pos))
        report["footprint_pixels"] = int(footprint.sum())
        report["outside_pixels"] = int((~footprint).sum())
        report["catalogue_positive_pixels"] = int(lab_pos.sum())
        report["sample_submission_equals_catalogue"] = identical_to_catalogue
        report["irregularities_verified"].append(
            {
                "id": "FLAG-01",
                "finding": (
                    "The locally bridged sample_submission.tif positive mask equals (labels.tif > 0) "
                    "inside the 5,167,373-pixel footprint (60,988 positive pixels). This conflicts "
                    "with the official problem page describing an all-absence sample. The private "
                    "DrivenData download was not independently retrieved here, so provenance/cause remain unresolved."
                ),
                "measured": {
                    "sample_submission_positives": int(sub_pos.sum()),
                    "labels_positives": int(lab_pos.sum()),
                    "differing_pixels_in_footprint": int((sub_pos ^ lab_pos).sum()),
                },
            }
        )

    with rasterio.open(DATA_DIR / "external" / "lidar_scarp_features_u8.tif") as s_lid:
        lid_valid = (s_lid.read(12) > 0) & footprint
        lid_px = int(lid_valid.sum())
        gap_px = int((footprint & ~lid_valid).sum())
        cat_in_gap = int((lab_pos & ~lid_valid).sum())
        report["lidar_1m_coverage"] = {
            "covered_pixels": lid_px,
            "covered_fraction": round(lid_px / float(footprint.sum()), 6),
            "gap_pixels": gap_px,
            "gap_fraction": round(gap_px / float(footprint.sum()), 6),
            "catalogue_pixels_in_gap": cat_in_gap,
            "catalogue_fraction_in_gap": round(cat_in_gap / float(lab_pos.sum()), 6),
        }

    dem10_man = json.loads((DATA_DIR / "dem10" / "manifest.json").read_text())
    for ch in dem10_man["channels"]:
        p = DATA_DIR / "dem10" / f"{ch}.f32.npy"
        arr = np.load(p, mmap_mode="r")
        assert arr.shape == (5167373,), f"Unexpected shape for dem10/{ch}: {arr.shape}"
        report["dem10_channels"][ch] = {
            "sha256": dem10_man["channel_stats"][ch]["sha256"],
            "shape": [5167373],
            "finite_fraction": float(np.isfinite(arr).mean()),
        }

    report["elapsed_seconds"] = round(time.time() - t0, 2)
    out_json = EVIDENCE_DIR / "data_verification.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    (DATA_DIR / "data_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Data preparation & verification complete in {report['elapsed_seconds']}s -> {out_json}")


if __name__ == "__main__":
    main()
