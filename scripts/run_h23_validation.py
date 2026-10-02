"""Validate the pre-registered H23-1..H23-5 geological channels and the E0 emission-policy sweep.

Pre-registration: ``docs/research/preregistration_h23.md`` (committed before any result).
Protocol (no tuning after the fact):

  * Primary gate   : off-catalogue USGS SGMC fault pixels (79,615 px), 4 spatially blocked quadrants.
  * Secondary gate : inherited 4-quadrant catalogue holdout (dense + 20%-of-components sparse).
  * E0 sweep       : pool_frac in {2.5, 3.5, 5, 6.5, 8, 10, 12.5} % x policy in {topk, thinned}.
  * Channels       : standalone arm + fused arm with a FIXED beta = 0.6 modulation.

Usage:
    export GEMS_DATA_DIR=/path/to/data
    python scripts/run_h23_validation.py [--skip-channels]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy.ndimage import binary_dilation, label as ndi_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.metric import dti_score_fast, ridge_nms  # noqa: E402
from gems.paths import DATA_DIR, EVIDENCE_DIR  # noqa: E402
from gems import hypotheses_h23 as h23  # noqa: E402

SEED = 20260929
BETA_FUSE = 0.6  # FIXED by pre-registration; never tuned per channel
POOL_FRACS = [0.025, 0.035, 0.05, 0.065, 0.08, 0.10, 0.125]
FOLD_NAMES = ["NW", "NE_LidarGapHeavy", "SW", "SE"]


# ----------------------------------------------------------------------------------
# Grids, folds, truths
# ----------------------------------------------------------------------------------
def load_grids() -> dict:
    with rasterio.open(DATA_DIR / "sample_submission.tif") as src:
        sub = src.read(1)
    footprint = np.isfinite(sub)
    with rasterio.open(DATA_DIR / "labels.tif") as src:
        labels = (src.read(1) > 0) & footprint
    sgmc_path = ROOT / "evidence" / "ci" / "derived_sgmc_faults_100m_u8.tif"
    with rasterio.open(sgmc_path) as src:
        sgmc = (src.read(1) > 0) & footprint
    sgmc_offcat = sgmc & ~labels

    yy, xx = np.nonzero(footprint)
    y_med, x_med = int(np.median(yy)), int(np.median(xx))
    H, W = footprint.shape
    gy, gx = np.ogrid[:H, :W]
    fold_2d = np.full((H, W), -1, dtype=np.int8)
    fold_2d[(gy < y_med) & (gx < x_med) & footprint] = 0
    fold_2d[(gy < y_med) & (gx >= x_med) & footprint] = 1
    fold_2d[(gy >= y_med) & (gx < x_med) & footprint] = 2
    fold_2d[(gy >= y_med) & (gx >= x_med) & footprint] = 3
    return {
        "footprint": footprint,
        "labels": labels,
        "sgmc": sgmc,
        "sgmc_offcat": sgmc_offcat,
        "fold_2d": fold_2d,
    }


def make_quads(g: dict):
    """Four (fold_id, name, fold_mask, slice, truths..., fold_mask_slice) tuples."""
    footprint, fold_2d = g["footprint"], g["fold_2d"]
    labels, sgmc_offcat = g["labels"], g["sgmc_offcat"]
    quads = []
    for f_id, fname in enumerate(FOLD_NAMES):
        f_mask = (fold_2d == f_id) & footprint
        r0, r1 = np.nonzero(f_mask.any(axis=1))[0][[0, -1]]
        c0, c1 = np.nonzero(f_mask.any(axis=0))[0][[0, -1]]
        sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        t_dense = labels & f_mask
        t_sparse = thin_components(t_dense, keep_frac=0.20, seed=4242 + f_id)
        known_cat = t_dense & ~t_sparse
        t_sgmc = sgmc_offcat & f_mask
        quads.append((f_id, fname, f_mask, sl, t_dense[sl], t_sparse[sl], known_cat[sl], t_sgmc[sl], f_mask[sl]))
    return quads


def thin_components(truth_2d: np.ndarray, keep_frac: float, seed: int) -> np.ndarray:
    comp, n = ndi_label(truth_2d, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.zeros_like(truth_2d, dtype=bool)
    rng = np.random.default_rng(seed)
    keep = rng.choice(np.arange(1, n + 1), size=max(1, int(round(keep_frac * n))), replace=False)
    return np.isin(comp, keep)


# ----------------------------------------------------------------------------------
# Emission policies
# ----------------------------------------------------------------------------------
def emit_topk(score_2d: np.ndarray, footprint: np.ndarray, fold_2d: np.ndarray, pool_frac: float) -> np.ndarray:
    out = np.zeros(footprint.shape, dtype=bool)
    for f_id in range(4):
        f_mask = (fold_2d == f_id) & footprint
        k = int(round(pool_frac * int(f_mask.sum())))
        idx = np.flatnonzero(f_mask.ravel())
        vals = score_2d.ravel()[idx]
        out.ravel()[idx[np.argpartition(vals, -k)[-k:]]] = True
    return out


def emit_ridge_topk(score_2d: np.ndarray, footprint: np.ndarray, fold_2d: np.ndarray, pool_frac: float) -> np.ndarray:
    """Per-quadrant: ridge-NMS thinning -> rank by ridge-boosted score -> take the top (pool_frac * |quad|)."""
    ridge = ridge_nms(score_2d, footprint, sigma=1.0)
    boosted = np.where(ridge, score_2d + 1.0, score_2d * 0.5).astype(np.float32)
    out = np.zeros(footprint.shape, dtype=bool)
    for f_id in range(4):
        f_mask = (fold_2d == f_id) & footprint
        k = int(round(pool_frac * int(f_mask.sum())))
        idx = np.flatnonzero(f_mask.ravel())
        vals = boosted.ravel()[idx]
        out.ravel()[idx[np.argpartition(vals, -k)[-k:]]] = True
    return out


def emit_thinned_pool(score_2d: np.ndarray, footprint: np.ndarray, fold_2d: np.ndarray, pool_frac: float) -> np.ndarray:
    """Take the top pool_frac as a POOL, skeleton-thin it to 1-px ridge centre-lines, emit the thinned set.

    This is the allocation that GEMSDOE10's blocked sweep showed to beat raw top-k at equal pixel
    count (thin12 = 0.1664 vs topk04 = 0.1422 at ~37.5k px).
    """
    out = np.zeros(footprint.shape, dtype=bool)
    for f_id in range(4):
        f_mask = (fold_2d == f_id) & footprint
        k = int(round(pool_frac * int(f_mask.sum())))
        idx = np.flatnonzero(f_mask.ravel())
        vals = score_2d.ravel()[idx]
        pool = np.zeros(footprint.shape, dtype=bool)
        pool.ravel()[idx[np.argpartition(vals, -k)[-k:]]] = True
        ridge = ridge_nms(np.where(pool, score_2d, 0.0).astype(np.float32), pool, sigma=1.0)
        thin = pool & ridge
        out |= thin
    return out


POLICIES = {
    "topk": emit_topk,
    "ridge_topk": emit_ridge_topk,
    "thinned_pool": emit_thinned_pool,
}


def score_pred(pred: np.ndarray, quads) -> dict:
    dense, sparse, sgmc = [], [], []
    detail = {}
    for _f, name, _m, sl, td, ts, kc, tsg, fm in quads:
        rd = dti_score_fast(pred[sl], td, valid_mask=fm)
        rs = dti_score_fast(pred[sl], ts, valid_mask=fm, catalogue_mask=kc)
        rg = dti_score_fast(pred[sl], tsg, valid_mask=fm, catalogue_mask=(td | kc), mask_predictions=True)
        dense.append(rd["dti"])
        sparse.append(rs["dti"])
        sgmc.append(rg["dti"])
        detail[name] = {
            "dense_dti": round(rd["dti"], 5),
            "sparse_dti": round(rs["dti"], 5),
            "sgmc_offcat_dti": round(rg["dti"], 5),
            "emitted_px": int(pred[sl].sum()),
        }
    return {
        "mean_dense_dti": round(float(np.mean(dense)), 5),
        "mean_sparse_dti": round(float(np.mean(sparse)), 5),
        "mean_sgmc_offcat_dti": round(float(np.mean(sgmc)), 5),
        "fold_dense": [round(x, 5) for x in dense],
        "fold_sparse": [round(x, 5) for x in sparse],
        "fold_sgmc_offcat": [round(x, 5) for x in sgmc],
        "emitted_px_total": int(pred.sum()),
        "emitted_fraction": round(float(pred.sum()) / float(sum(int(q[8].sum()) for q in quads)), 5),
        "folds": detail,
    }


# ----------------------------------------------------------------------------------
# Feature loading helpers
# ----------------------------------------------------------------------------------
def to_2d(fp_vec: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    a = np.zeros(footprint.shape, dtype=np.float32)
    a.ravel()[np.flatnonzero(footprint.ravel())] = fp_vec
    return a


def rank_in_footprint(a2d: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1] of each in-footprint pixel (ties averaged)."""
    v = a2d[footprint]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.shape, dtype=np.float32)
    ranks[order] = np.arange(v.shape[0], dtype=np.float32)
    # average ties
    sv = v[order]
    i = 0
    n = sv.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    out = np.zeros(a2d.shape, dtype=np.float32)
    out[footprint] = ranks / max(n - 1, 1)
    return out


def dequant_topo(path: Path) -> dict[str, np.ndarray]:
    """Read the 5GEMSDOE aux_bridge 9-band topo_u8 stack (100 m) and de-quantise it."""
    spec = {
        1: ("slope_p90", 0.0, 41.82450717926024),
        2: ("slope_max", 0.0, 48.14020595550532),
        3: ("steep_frac", 0.0, 1.0),
        4: ("slope_std", 0.0, 9.837751884460445),
        5: ("relief_local", 0.0, 80.53856750488279),
        6: ("curv_prof_absmax", 0.0, 40.50098724365233),
        7: ("aspect_coherence", 0.09517142064869404, 0.9983641988039017),
        8: ("hs_lineament", 0.0, 0.5624071541428564),
        9: ("dem_mean", 1099.7082177734376, 2481.5011486816393),
    }
    with rasterio.open(path) as src:
        return {name: h23._dequant(src.read(idx), lo, hi) for idx, (name, lo, hi) in spec.items()}


def dequant_radiometric(path: Path) -> dict[str, np.ndarray]:
    spec = {
        1: ("rad_k", 1.6337181997299195, 2.9305647492408773),
        2: ("rad_th", 0.5500761246681214, 2.1914254903793355),
        3: ("rad_u", 1.333544532060623, 2.3317787337303164),
        4: ("rad_tc", 5.467920293807984, 30.272063579559337),
        5: ("rad_thk", 0.335282980799675, 0.8956930857896809),
        6: ("rad_uk", 0.6093248844146724, 1.1472516965866115),
        7: ("rad_uth", 0.8803293263912201, 2.8154791545867965),
    }
    with rasterio.open(path) as src:
        return {name: h23._dequant(src.read(idx), lo, hi) for idx, (name, lo, hi) in spec.items()}


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-channels", action="store_true")
    ap.add_argument("--policies", default="topk,ridge_topk,thinned_pool")
    args = ap.parse_args()

    t0 = time.time()
    g = load_grids()
    footprint, fold_2d = g["footprint"], g["fold_2d"]
    fp_idx = np.flatnonzero(footprint.ravel())
    quads = make_quads(g)
    n_fp = int(footprint.sum())
    print(f"footprint={n_fp}  catalogue={int(g['labels'].sum())}  sgmc={int(g['sgmc'].sum())}  "
          f"sgmc_off_catalogue={int(g['sgmc_offcat'].sum())}")

    cache = DATA_DIR / "cache" / "oof_probs_h16_1.npz"
    with np.load(cache) as z:
        h16_1_fp = z["h16_1"].astype(np.float32)
    base_2d = to_2d(h16_1_fp, footprint)

    report: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": {
            "file": "docs/research/preregistration_h23.md",
            "git_commit_before_results": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
        },
        "evaluation_scope": (
            "PRIMARY gate = off-catalogue USGS SGMC faults (79,615 px) in 4 spatially blocked quadrants; "
            "SECONDARY gate = inherited 4-quadrant catalogue holdout (dense + 20%-of-components sparse). "
            "Neither is the hidden competition test set."
        ),
        "truth_pixels": {
            "footprint": n_fp,
            "catalogue": int(g["labels"].sum()),
            "sgmc_total": int(g["sgmc"].sum()),
            "sgmc_off_catalogue": int(g["sgmc_offcat"].sum()),
        },
        "submission_slot_spent": False,
        "e0_emission_policy_sweep": {},
        "channels": {},
    }

    # ---------------- E0: emission-policy sweep on the baseline surface ----------------
    print("\n=== E0 emission-policy sweep (baseline H16-1 out-of-fold surface) ===")
    best = None
    for pname in args.policies.split(","):
        fn = POLICIES[pname]
        for pf in POOL_FRACS:
            pred = fn(base_2d, footprint, fold_2d, pf)
            s = score_pred(pred, quads)
            key = f"{pname}@{pf:.3f}"
            report["e0_emission_policy_sweep"][key] = s
            print(f"  {key:28s} px={s['emitted_px_total']:7d} ({s['emitted_fraction']*100:5.2f}%)  "
                  f"SGMC={s['mean_sgmc_offcat_dti']:.5f}  dense={s['mean_dense_dti']:.5f}  sparse={s['mean_sparse_dti']:.5f}")
            if best is None or s["mean_sgmc_offcat_dti"] > best[1]["mean_sgmc_offcat_dti"]:
                best = (key, s, pname, pf)
            del pred
            gc.collect()
    report["e0_best_by_primary_gate"] = {"key": best[0], "policy": best[2], "pool_frac": best[3], **best[1]}
    print(f"  --> E0 best by PRIMARY gate: {best[0]}  SGMC={best[1]['mean_sgmc_offcat_dti']:.5f}")

    # Establish the baseline at the *inherited* policy for comparison (ridge_topk @ 2.5%)
    base_key = "ridge_topk@0.025"
    baseline = report["e0_emission_policy_sweep"][base_key]

    if args.skip_channels:
        out = EVIDENCE_DIR / "hypothesis_h23_validation.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {out} in {time.time()-t0:.0f}s")
        return

    # ---------------- H23 channels ----------------
    chans_fp: dict[str, np.ndarray] = {}
    chan_cache = DATA_DIR / "cache" / "h23_channels_fp.npz"
    if chan_cache.exists():
        print(f"\n[Cache] loading H23 channels from {chan_cache}")
        with np.load(chan_cache) as z:
            def _fixkey(k: str) -> str:
                if k.startswith("h23_"):
                    return k
                if k.startswith("mag_"):
                    return "h23_mag_" + k[4:]
                if k.startswith("grav_"):
                    return "h23_grav_" + k[5:]
                return k
            chans_fp = {_fixkey(k): z[k] for k in z.files}
    else:
        # mmap the cache: it is ~1.2 GB and loading it all would risk OOM on a 4 GB runner
        _fz = np.load(DATA_DIR / "cache" / "features_fp.npz", mmap_mode="r")
        _need = ["lid1m_valid", "lid1m_step_max", "dem10_onesided3", "elev_slope", "seamfree_scarp_index"]
        feats = {k: np.asarray(_fz[k]) for k in _need}
        del _fz
        gc.collect()
        lid_ok = feats["lid1m_valid"] > 0.5
        scarp_1m = to_2d(feats["lid1m_step_max"], footprint) * to_2d(lid_ok.astype(np.float32), footprint)
        scarp_10m = to_2d(feats["dem10_onesided3"], footprint)
        elev_slope = to_2d(feats["elev_slope"], footprint)

        with rasterio.open(DATA_DIR / "training_features.tif") as src:
            tc = src.read(6).astype(np.float32)
            tmi = src.read(14).astype(np.float32)
            tmi_vg = src.read(9).astype(np.float32)
            grav = src.read(13).astype(np.float32)
            grav_vg = src.read(11).astype(np.float32)
        for a in (tc, tmi, tmi_vg, grav, grav_vg):
            a[(a < -1e20) | ~np.isfinite(a)] = 0.0

        print("\n[H23-1] cross-scale orientation persistence + self-affine roughness ...")
        r = h23.h23_1_cross_scale_orientation_persistence(scarp_1m, scarp_10m, tc, footprint,
                                                           footprint_valid_1m=to_2d(lid_ok.astype(np.float32), footprint) > 0.5)
        for k, v in r.items():
            chans_fp[k] = v.ravel()[fp_idx].astype(np.float32)
        del r, scarp_1m, scarp_10m, tc
        gc.collect()
        np.savez(chan_cache, **chans_fp); print(f"  [checkpoint] {len(chans_fp)} chans")

        print("[H23-2] range-piedmont junction straightness (Smf-like) ...")
        topo = dequant_topo(DATA_DIR / "external" / "topo_u8.tif")
        r = h23.h23_2_mountain_front_straightness(topo["dem_mean"], footprint)
        for k, v in r.items():
            chans_fp[k] = v.ravel()[fp_idx].astype(np.float32)
        # complementarity variant: straight front WITHOUT a local scarp
        scarp_rank = rank_in_footprint(to_2d(feats["seamfree_scarp_index"], footprint), footprint)
        front_rank = rank_in_footprint(to_2d(chans_fp["h23_smf_score"], footprint), footprint)
        comp = np.clip(front_rank * (1.0 - scarp_rank), 0.0, 1.0)
        chans_fp["h23_smf_complement"] = comp.ravel()[fp_idx].astype(np.float32)
        del r, topo["dem_mean"], scarp_rank, front_rank, comp
        gc.collect()
        np.savez(chan_cache, **chans_fp); print(f"  [checkpoint] {len(chans_fp)} chans")

        print("[H23-5] hillslope-aspect fabric discontinuity worm ...")
        r = h23.h23_5_aspect_discontinuity_worm(topo["aspect_coherence"], topo["slope_std"], topo["hs_lineament"], footprint)
        for k, v in r.items():
            chans_fp[k] = v.ravel()[fp_idx].astype(np.float32)
        del r, topo
        gc.collect()
        np.savez(chan_cache, **chans_fp); print(f"  [checkpoint] {len(chans_fp)} chans")

        print("[H23-4] radiometric alteration-halo anisotropy ...")
        rad = dequant_radiometric(DATA_DIR / "external" / "radiometric_u8.tif")
        r = h23.h23_4_radiometric_anisotropy(rad, elev_slope, footprint)
        for k, v in r.items():
            chans_fp[k] = v.ravel()[fp_idx].astype(np.float32)
        del r, rad, elev_slope
        gc.collect()
        np.savez(chan_cache, **chans_fp); print(f"  [checkpoint] {len(chans_fp)} chans")

        print("[H23-3] tilt-depth edge-depth consistency ...")
        r = h23.h23_3_tilt_depth_consistency(tmi, tmi_vg, footprint)
        for k, v in r.items():
            chans_fp["h23_mag_" + k] = v.ravel()[fp_idx].astype(np.float32)
        del r, tmi, tmi_vg
        gc.collect()
        r = h23.h23_3_tilt_depth_consistency(grav, grav_vg, footprint)
        for k, v in r.items():
            chans_fp["h23_grav_" + k] = v.ravel()[fp_idx].astype(np.float32)
        del r, grav, grav_vg
        gc.collect()

        np.savez(chan_cache, **chans_fp)
        print(f"  saved {len(chans_fp)} H23 channels -> {chan_cache}")

    # ---------------- Evaluate each channel ----------------
    e0_policy = POLICIES[best[2]]
    e0_pf = best[3]
    print(f"\n=== H23 channel evaluation at E0-optimal policy {best[2]}@{e0_pf} and at the inherited 2.5% ===")
    keys = [k for k in chans_fp if k.startswith("h23_")]
    for ck in sorted(keys):
        q = rank_in_footprint(to_2d(chans_fp[ck], footprint), footprint)
        fused = np.clip(base_2d * (1.0 + BETA_FUSE * (q - 0.5)), 0.0, 1.0).astype(np.float32)
        entry: dict = {}
        # (a) standalone at the E0 policy
        entry["standalone_e0"] = score_pred(e0_policy(to_2d(chans_fp[ck], footprint).astype(np.float32), footprint, fold_2d, e0_pf), quads)
        # (b) fused at the E0 policy
        entry["fused_e0"] = score_pred(e0_policy(fused, footprint, fold_2d, e0_pf), quads)
        # (c) fused at the inherited policy (ridge_topk @ 2.5%)
        entry["fused_inherited"] = score_pred(POLICIES["ridge_topk"](fused, footprint, fold_2d, 0.025), quads)
        entry["delta_vs_baseline_sgmc"] = round(
            entry["fused_e0"]["mean_sgmc_offcat_dti"] - best[1]["mean_sgmc_offcat_dti"], 5
        )
        entry["delta_vs_baseline_dense"] = round(
            entry["fused_e0"]["mean_dense_dti"] - best[1]["mean_dense_dti"], 5
        )
        entry["delta_vs_baseline_sparse"] = round(
            entry["fused_e0"]["mean_sparse_dti"] - best[1]["mean_sparse_dti"], 5
        )
        entry["sgmc_fold_wins"] = int(
            sum(a > b for a, b in zip(entry["fused_e0"]["fold_sgmc_offcat"], best[1]["fold_sgmc_offcat"]))
        )
        entry["gate_passed"] = bool(
            entry["sgmc_fold_wins"] >= 3
            and entry["delta_vs_baseline_sgmc"] > 0
            and entry["delta_vs_baseline_dense"] >= -0.005
        )
        report["channels"][ck] = entry
        print(f"  {ck:34s} stand(SGMC)={entry['standalone_e0']['mean_sgmc_offcat_dti']:.5f}  "
              f"fused(SGMC)={entry['fused_e0']['mean_sgmc_offcat_dti']:.5f} "
              f"({entry['delta_vs_baseline_sgmc']:+.5f}, {entry['sgmc_fold_wins']}/4)  "
              f"dense={entry['fused_e0']['mean_dense_dti']:.5f} ({entry['delta_vs_baseline_dense']:+.5f})  "
              f"{'PASS' if entry['gate_passed'] else 'REJECTED'}")
        del q, fused
        gc.collect()

    report["baseline_at_inherited_policy"] = baseline
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    out = EVIDENCE_DIR / "hypothesis_h23_validation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out} in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
