"""Streaming multi-scale feature bank for the GEMS grid.

Memory-safe by construction: features are *yielded* one at a time and written
straight into a float16 memmap, so peak RAM is ~2 full-grid layers regardless of
how many features the bank contains.  float16 is adequate because the downstream
learner (HistGradientBoosting) bins every feature into 256 quantile bins anyway,
and every layer is either z-scored into [-8, 8] or a uint8 rank in [0, 1].

Layer families
--------------
1. The 19 official GeoDAWN competition bands (robust-standardised).
2. Multi-scale structure on the 12 most fault-relevant bands: high-pass residual
   (band - Gaussian), gradient magnitude, Frangi-style Hessian *linearity* and
   *ridge strength* at sigma = 2 px (200 m), local standard deviation at
   sigma = 4 px (400 m).  Sigma brackets the 300 m DTI kernel radius.
3. Band-group edge composites (magnetic / gravity / strain / topographic /
   subsurface / seismic).
4. EXTERNAL, independently-sourced layers that are NOT in the 19 bands:
     * GeoDAWN airborne radiometrics K, Th, U, TC  (DOI 10.5066/P93LGLVQ,
       ScienceBase item 657e1d85d34e23d3533209f7) -- a data family entirely
       absent from the competition stack (which has magnetics, gravity, geodetic
       strain, topography, seismicity, subsurface but NO radiometrics).
     * GeoDAWN contractor ratio grids Th/K, U/K, U/Th and TMI upward-continued
       to 150 m (same DOI).
     * USGS 3DEP 1 m LiDAR-derived scarp descriptors aggregated to 100 m
       (slope excess, step height, positive/negative Laplacian, up/down-facing
       slope, cross-slope, relief, strike coherence).
     * USGS 1 m DEM topographic openness / local-relief-model fields for the
       eight high-prior 10 km tiles.
5. Structural priors: distance to the known catalogue, and the Bour & Davy
   (1999) nearest-larger-neighbour geometric prior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from scipy import ndimage

BAND_GROUPS = {
    "magnetic": [0, 1, 2, 5, 8, 13],
    "gravity": [4, 10, 12, 17],
    "strain": [3, 6, 7],
    "topographic": [11, 18],
    "subsurface": [14, 16],
    "seismic": [9, 15],
}
MULTISCALE_BANDS = (11, 18, 12, 17, 10, 13, 2, 1, 14, 16, 3, 5)
RAD_BANDS = ("K", "Th", "U", "TC")
EXT_BANDS = ("ThK", "UK", "UTh", "TMI_up150")
LIDAR_BANDS = ("ex_max", "ex_mean", "step_max", "lapneg_max", "lappos_max",
               "downface_max", "upface_max", "cross_max", "relief", "coh100",
               "strike", "valid")


def z(a: np.ndarray) -> np.ndarray:
    """Robust (median/IQR) standardisation, clipped to +-8, NaN -> 0, float32."""
    a = np.asarray(a)
    ok = np.isfinite(a)
    out = np.zeros(a.shape, np.float32)
    if not ok.any():
        return out
    v = a[ok]
    med = float(np.median(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    s = iqr / 1.349 if iqr > 0 else float(v.std())
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    out[ok] = np.clip((a[ok] - med) / s, -8.0, 8.0)
    return out


def hessian(a: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Frangi-style (linearity, ridge_strength) of a Gaussian-smoothed field."""
    s = ndimage.gaussian_filter(a, sigma)
    gy, gx = np.gradient(s)
    gyy = np.gradient(gy, axis=0)
    gxx = np.gradient(gx, axis=1)
    gxy = np.gradient(gy, axis=1)
    tmp = np.sqrt(((gyy - gxx) / 2.0) ** 2 + gxy ** 2)
    l1 = (gyy + gxx) / 2.0 + tmp
    l2 = (gyy + gxx) / 2.0 - tmp
    mx = np.maximum(np.abs(l1), np.abs(l2))
    mn = np.minimum(np.abs(l1), np.abs(l2))
    lin = np.zeros_like(mx)
    nz = mx > 0
    lin[nz] = (mx[nz] - mn[nz]) / mx[nz]
    return lin.astype(np.float32), np.maximum(0.0, -(l1 + l2)).astype(np.float32)


class U8Stack:
    """Lazy per-band reader for a uint8 multi-band external raster.

    Reading all 12 LiDAR bands at once would cost 0.74 GB; reading one band on
    demand costs 0.05 GB.  Values are the original uint8 ranks (0 = nodata).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.ok = self.path.exists()
        self._ds = rasterio.open(self.path) if self.ok else None
        self.count = self._ds.count if self.ok else 0

    def band(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (value/255 as float32, valid mask) for 0-based band index j."""
        a = self._ds.read(j + 1)
        return (a.astype(np.float32) / 255.0), (a > 0)

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None


def build_stream(A: np.ndarray, band_names: list[str],
                 external_dir: Path | None = None,
                 structural: dict[str, np.ndarray] | None = None,
                 verbose: bool = True,
                 include_thermal_inversion: bool = False) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (feature_name, float32 full-grid layer) one at a time."""
    nB = A.shape[0]

    for i in range(nB):
        yield band_names[i], z(A[i])
    yield "valid_all_bands", np.isfinite(A).all(axis=0).astype(np.float32)
    yield "n_invalid_bands", (~np.isfinite(A)).sum(axis=0).astype(np.float32)

    for i in MULTISCALE_BANDS:
        a = np.nan_to_num(A[i], nan=0.0)
        bn = band_names[i]
        sm2 = ndimage.gaussian_filter(a, 2.0)
        yield f"{bn}__hp_s2", z(a - sm2)
        yield f"{bn}__grad_s2", z(np.hypot(*np.gradient(sm2)))
        del sm2
        lin, ridge = hessian(a, 2.0)
        yield f"{bn}__linearity_s2", lin
        yield f"{bn}__ridge_s2", z(ridge)
        del lin, ridge
        sm4 = ndimage.gaussian_filter(a, 4.0)
        yield f"{bn}__hp_s4", z(a - sm4)
        yield f"{bn}__grad_s4", z(np.hypot(*np.gradient(sm4)))
        del sm4
        m = ndimage.uniform_filter(a, 9)
        m2 = ndimage.uniform_filter(a * a, 9)
        yield f"{bn}__locstd_s4", z(np.sqrt(np.maximum(m2 - m * m, 0.0)))
        del a, m, m2
        if verbose and i == MULTISCALE_BANDS[0]:
            print(f"    ...multiscale bank over {len(MULTISCALE_BANDS)} bands", flush=True)

    for gname, idx in BAND_GROUPS.items():
        comp = np.nanmean(np.stack([z(A[i]) for i in idx]), axis=0)
        yield f"group_{gname}_grad_s2", z(np.hypot(*np.gradient(ndimage.gaussian_filter(comp, 2.0))))

    # ---- external, independently sourced layers ---------------------------
    if external_dir is not None:
        external_dir = Path(external_dir)

        rad = U8Stack(external_dir / "geodawn_rad_u8.tif")
        if rad.ok:
            grads, valid0 = [], None
            for j, nm in enumerate(RAD_BANDS):
                v, msk = rad.band(j)
                if valid0 is None:
                    valid0 = msk
                vv = np.where(msk, v, np.nan)
                yield f"rad_{nm}", z(vv)
                g1 = ndimage.gaussian_filter(np.nan_to_num(vv), 1.0)
                e1 = np.hypot(*np.gradient(g1))
                yield f"rad_{nm}_grad_s1", z(e1)
                g3 = ndimage.gaussian_filter(np.nan_to_num(vv), 3.0)
                yield f"rad_{nm}_grad_s3", z(np.hypot(*np.gradient(g3)))
                grads.append(z(e1))
                del v, vv, g1, g3, e1
            # Radiometric edge coherence: does the SAME lineament step in K, Th,
            # U and total count simultaneously?  A fault cutting hydrothermally
            # altered ground should, whereas a survey-line artefact will not.
            E = np.stack(grads)
            yield "rad_edge_rss_s1", np.sqrt((E ** 2).sum(axis=0)).astype(np.float32)
            yield "rad_edge_max_s1", E.max(axis=0).astype(np.float32)
            yield "rad_edge_coherence_s1", z(
                np.nanmean(E / np.maximum(E.max(axis=0, keepdims=True), 1e-6), axis=0))
            yield "rad_valid", valid0.astype(np.float32)
            del E, grads
            rad.close()

        ext = U8Stack(external_dir / "geodawn_extensions_u8.tif")
        if ext.ok:
            for j, nm in enumerate(EXT_BANDS):
                v, msk = ext.band(j)
                yield f"ext_{nm}", z(np.where(msk, v, np.nan))
                del v
            ext.close()

        lid = U8Stack(external_dir / "lidar_scarp_features_u8.tif")
        if lid.ok:
            cache = {}
            for j, nm in enumerate(LIDAR_BANDS):
                v, msk = lid.band(j)
                if nm == "valid":
                    yield "lidar_valid", msk.astype(np.float32)
                else:
                    yield f"lidar_{nm}", z(np.where(msk, v, np.nan))
                if nm in ("ex_max", "step_max", "upface_max", "downface_max"):
                    cache[nm] = (v, msk)
                del v
            # Tectonic vs fluvial discriminant: a fault scarp is a COHERENT,
            # ONE-SIDED step (up-facing on one limb, down-facing on the other),
            # whereas a channel or arroyo is a symmetric negative step.
            if {"upface_max", "downface_max"} <= cache.keys():
                (up, mu), (dn, md) = cache["upface_max"], cache["downface_max"]
                yield "lidar_onesided", z(np.where(mu | md, np.abs(up - dn), np.nan))
            if {"step_max", "ex_max"} <= cache.keys():
                (st, ms), (ex, me) = cache["step_max"], cache["ex_max"]
                yield "lidar_step_over_excess", z(np.where(ms & me, st - ex, np.nan))
            cache.clear()
            lid.close()

        npz = external_dir / "dem1m_high_prior_openness_lrm.npz"
        if npz.exists():
            zz = np.load(npz)
            idx = zz["cov_fp_indices"].astype(np.int64)
            HH, WW = A.shape[1], A.shape[2]
            cov = np.zeros(HH * WW, bool)
            cov[idx] = True
            for key in ("openness_dipole_max", "openness_asymm_max",
                        "lrm_grad_max", "lrm_abs_max"):
                if key not in zz:
                    continue
                g = np.zeros(HH * WW, np.float32)
                g[idx] = zz[key].astype(np.float32)
                yield f"dem1m_{key}", z(np.where(g > 0, g, np.nan).reshape(HH, WW))
                del g
            yield "dem1m_coverage", cov.reshape(HH, WW).astype(np.float32)
            del cov
            zz.close()

        if include_thermal_inversion:
            from gems22.hypotheses import build_thermal_conduit_layers
            fp_mask = np.isfinite(A).any(axis=0)
            th_layers, _ = build_thermal_conduit_layers((A.shape[1], A.shape[2]), fp_mask)
            for k_th, v_th in th_layers.items():
                yield k_th, v_th

    if structural:
        for k, v in structural.items():
            if v is not None:
                yield k, (v.astype(np.float32) if v.dtype == np.float32 else z(v))


def build_cache(A: np.ndarray, band_names: list[str], out_path: Path,
                external_dir: Path | None = None,
                structural: dict[str, np.ndarray] | None = None,
                dtype: str = "float16", verbose: bool = True) -> tuple[Path, list[str]]:
    """Materialise the bank to a (n_features, H, W) memmap; returns (path, names).

    Single-pass and memory-safe: each yielded layer is streamed straight to disk
    as raw C-order bytes, so peak RAM is one layer, not the whole bank.  The file
    is then re-opened as a memmap once the layer count is known.
    """
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = int(A.shape[1]), int(A.shape[2])
    if out_path.exists():
        out_path.unlink()
    names: list[str] = []
    itemsize = np.dtype(dtype).itemsize
    with open(out_path, "wb") as fh:
        for nm, arr in build_stream(A, band_names, external_dir, structural, verbose):
            buf = np.asarray(arr, dtype=dtype)
            assert buf.shape == (H, W), f"{nm}: {buf.shape} != {(H, W)}"
            buf.tofile(fh)
            names.append(nm)
            if verbose and len(names) % 25 == 0:
                print(f"    cached {len(names)} layers ({nm}) "
                      f"{out_path.stat().st_size/1e9:.2f} GB", flush=True)
    n = len(names)
    assert out_path.stat().st_size == n * H * W * itemsize, "cache size mismatch"
    mm = np.memmap(out_path, dtype=dtype, mode="r", shape=(n, H, W))
    del mm
    out_path.with_suffix(".names.json").write_text(
        json.dumps({"names": names, "dtype": dtype, "shape": [n, H, W],
                    "bytes": out_path.stat().st_size}, indent=1))
    if verbose:
        print(f"  feature cache: {n} layers, {out_path.stat().st_size/1e9:.2f} GB")
    return out_path, names
