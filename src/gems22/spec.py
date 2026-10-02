"""Raster specification, footprint and scoring-domain construction.

Every constant in this module is MEASURED from the byte-verified official rasters
in ``data/`` (see ``scripts/00_verify_data.py`` for the SHA-256 pins) and is
asserted at import time by ``Spec.verify()``.

Official format requirements, quoted from
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
section "Submission format":

  * "same projected coordinate reference system as the training data
     (projected coordinate system for UTM zone 11N, EPSG 32611)"
  * "same resolution as the training data (100m)"
  * "same bounds as the training data, and data outside the bounds is null or nan"
  * "a single layer with datatype of 32-bit float (float32) with values between
     0 and 1 indicating the confidence or probability of fault presence"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

FEATURES_TIF = DATA / "training_features.tif"
LABELS_TIF = DATA / "labels.tif"
SAMPLE_TIF = DATA / "sample_submission.tif"

# SHA-256 pins carried by src/6GEMSDOE/data/bridge/manifest.json (generated
# 2026-09-17T01:02:28+00:00 by scripts/make_data_bridge.py from the official
# DrivenData data tab mirrors).  Re-verified byte-for-byte in this repo on
# 2026-10-01 by scripts/00_verify_data.py.
SHA256_PINS = {
    "training_features.tif":
        "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
    "labels.tif":
        "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
    "sample_submission.tif":
        "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
}

EPSG = 32611
WIDTH = 3292
HEIGHT = 3730
TRANSFORM = (100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)
N_FOOTPRINT = 5_167_373        # finite pixels in sample_submission.tif
N_OUTSIDE = 7_111_787          # NaN pixels in sample_submission.tif
N_CATALOGUE = 60_988           # labels.tif == 1
N_TOTAL = WIDTH * HEIGHT       # 12,279,160
FEATURE_NODATA = np.float32(-3.4028234663852886e38)


@dataclass(frozen=True)
class Spec:
    """Immutable description of the competition grid + scoring domain."""

    footprint: np.ndarray      # bool (H, W) -- scored region ("inside the bounds")
    catalogue: np.ndarray      # bool (H, W) -- labels.tif == 1 (known faults)
    evaluate_mask: np.ndarray  # bool (H, W) -- footprint & ~catalogue  (forum 11516)
    feature_valid: np.ndarray  # bool (H, W) -- all 19 bands finite inside footprint
    profile: dict

    # ---- sizes ------------------------------------------------------------
    @property
    def shape(self):
        return self.footprint.shape

    @property
    def n_footprint(self) -> int:
        return int(self.footprint.sum())

    @property
    def n_catalogue(self) -> int:
        return int(self.catalogue.sum())

    @property
    def n_evaluate(self) -> int:
        return int(self.evaluate_mask.sum())

    def summary(self) -> dict:
        return {
            "shape": list(self.shape),
            "n_total": int(self.footprint.size),
            "n_footprint": self.n_footprint,
            "n_outside": int(self.footprint.size - self.n_footprint),
            "n_catalogue": self.n_catalogue,
            "n_evaluate_mask": self.n_evaluate,
            "n_feature_valid": int(self.feature_valid.sum()),
            "n_footprint_feature_nodata": int((self.footprint & ~self.feature_valid).sum()),
            "profile": self.profile,
        }

    # ---- construction -----------------------------------------------------
    @classmethod
    def load(cls, data_dir: Path | str = DATA) -> "Spec":
        data_dir = Path(data_dir)
        with rasterio.open(data_dir / "sample_submission.tif") as ds:
            sample = ds.read(1)
            profile = ds.profile.copy()
        with rasterio.open(data_dir / "labels.tif") as ds:
            labels = ds.read(1)
        with rasterio.open(data_dir / "training_features.tif") as ds:
            feat = ds.read()
            fnodata = ds.nodata

        footprint = np.isfinite(sample)
        catalogue = (labels == 1) & footprint
        valid = np.all(np.isfinite(feat) & (feat != fnodata), axis=0) & footprint
        return cls(
            footprint=footprint,
            catalogue=catalogue,
            evaluate_mask=footprint & ~catalogue,
            feature_valid=valid,
            profile=profile,
        )

    # ---- assertions -------------------------------------------------------
    def verify(self) -> dict:
        """Hard-check every published constant. Raises AssertionError on drift."""
        p = self.profile
        checks = {
            "shape_is_3730x3292": self.footprint.shape == (HEIGHT, WIDTH),
            "crs_is_epsg_32611": str(p["crs"]).upper().endswith("32611"),
            "transform_matches_official": tuple(p["transform"])[:6] == TRANSFORM,
            "dtype_float32": p["dtype"] == "float32",
            "count_single_band": int(p["count"]) == 1,
            "nodata_is_nan": bool(np.isnan(p["nodata"])) if p["nodata"] is not None else False,
            "n_footprint_5167373": self.n_footprint == N_FOOTPRINT,
            "n_outside_7111787": (self.footprint.size - self.n_footprint) == N_OUTSIDE,
            "n_catalogue_60988": self.n_catalogue == N_CATALOGUE,
            "n_total_12279160": int(self.footprint.size) == N_TOTAL,
            "catalogue_inside_footprint": bool((self.catalogue & ~self.footprint).sum() == 0),
        }
        bad = [k for k, v in checks.items() if not v]
        assert not bad, f"spec verification failed: {bad}"
        return checks


def load_features(data_dir: Path | str = DATA, dtype: str = "float32") -> tuple[np.ndarray, list[str]]:
    """Return (bands (19,H,W) with nodata -> NaN, band_names).

    float32 by default: the full stack is 19 x 3730 x 3292, i.e. 0.93 GB in
    float32 versus 1.87 GB in float64.  Every downstream transform is a
    z-scored or filtered derivative whose useful precision is far below float32,
    and the sandbox budget is 3.8 GB RAM.
    """
    npdt = np.dtype(dtype)
    with rasterio.open(Path(data_dir) / "training_features.tif") as ds:
        a = ds.read().astype(npdt)
        nod = ds.nodata
        names = []
        for i in range(1, ds.count + 1):
            t = ds.tags(i)
            names.append(t.get("band_name") or (t.get("description", f"band{i}").split(" - ")[0]))
    if nod is not None:
        a[a == npdt.type(nod)] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a, names


def load_labels(data_dir: Path | str = DATA) -> np.ndarray:
    with rasterio.open(Path(data_dir) / "labels.tif") as ds:
        lab = ds.read(1)
    return (lab == 1)
