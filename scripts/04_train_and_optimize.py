#!/usr/bin/env python3
"""Step 04 -- spatially-blocked training, DTI-budget optimisation, and gating.

Two independently-trained heads.  Each head has its OWN target for training AND
for evaluation, because the two targets are structurally different objects:

  HEAD A "signature"
      target = labels.tif catalogue faults (USGS QFD + INGENIOUS).
      Learns what a fault looks like in the 19 GeoDAWN bands plus the external
      radiometric / LiDAR layers.  This is the localisation engine.

  HEAD B "gap"
      target = SGMC-gap = faults in the USGS State Geologic Map Compilation
      (NV + CA, mrdata.usgs.gov/geology/state) that are NOT pixels of the
      competition catalogue.  A real, public, independent sample of exactly the
      object the private test set contains: a genuine fault the catalogue
      missed.  Head B therefore learns the catalogue's blind spots.

  HEAD C "blend"  weighted rank blend of A and B, evaluated against BOTH targets.

LEAKAGE CONTROL (critical, and the reason the structural layers are NOT in the
shared feature cache): `distance-to-catalogue`, `catalogue-dilated` and the
Bour & Davy (1999) prior field are all functions of the known-fault map.  A
held-out catalogue fault sits AT ZERO DISTANCE from itself, so a globally-built
distance-to-catalogue feature predicts the held-out truth perfectly (measured
DTI 0.977 on fold NW before this fix).  They are therefore rebuilt per fold from
`catalogue & train_mask` only.  This is also the honest analogue of the live
configuration, where the private ground truth is disjoint from the catalogue and
so lies at a strictly positive distance from it.

Emission support is chosen by the EXACT DTI marginal rule from `gems22.metric`:
emit ranked pixels while the marginal efficiency exceeds
tau = 0.2*DTI / (1 - 0.2*DTI).  Rank-based selection means the decision is
invariant to the model's probability calibration.

Outputs
  data/derived/oof_{A,B,C}_{fold}.f32   per-fold full-grid probability maps
  data/derived/oof_{A,B,C}.npy          out-of-fold stacks (NaN off-fold)
  evidence/holdout_results.json         per-fold DTI, budget sweep, gate decision
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import clustering as cl
from gems22 import holdout as ho
from gems22.metric import budget_curve, components, tau
from gems22.spec import REPO, load_labels

FEAT = REPO / "data/derived/features.f16"
DERIV = REPO / "data/derived"
BLOCK = 64
BUDGETS = [10_000, 25_000, 50_000, 75_000, 100_000, 120_000, 150_000, 200_000,
           250_000, 300_000, 400_000, 500_000, 650_000, 800_000, 1_000_000,
           1_250_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000, 4_000_000]

# Feature-cache layers that are functions of the known-fault map. They are
# EXCLUDED from the shared cache read and rebuilt per fold instead.
LEAKY_PREFIXES = ("struct_dist_cat", "struct_cat_dilate", "struct_bour_davy")


def open_features():
    meta = json.loads((DERIV / "features.names.json").read_text())
    n, H, W = meta["shape"]
    mm = np.memmap(FEAT, dtype=meta["dtype"], mode="r", shape=(n, H, W))
    names = meta["names"]
    keep = [i for i, nm in enumerate(names) if not nm.startswith(LEAKY_PREFIXES)]
    dropped = [names[i] for i in range(n) if i not in set(keep)]
    return mm, names, keep, dropped


def gather_layers(layers: np.ndarray, keep: list[int], sel: np.ndarray,
                  extra: np.ndarray | None = None) -> np.ndarray:
    """Rows where `sel` is True -> float32 (n_sel, len(keep) + n_extra)."""
    F, H, W = layers.shape
    n_sel = int(sel.sum())
    n_ex = 0 if extra is None else extra.shape[0]
    out = np.empty((n_sel, len(keep) + n_ex), dtype=np.float32)
    k = 0
    for r0 in range(0, H, BLOCK):
        r1 = min(r0 + BLOCK, H)
        sub = sel[r0:r1].ravel()
        if not sub.any():
            continue
        m = sub.sum()
        blk = np.stack([np.asarray(layers[i, r0:r1, :], dtype=np.float32)
                        for i in keep]).reshape(len(keep), -1)
        out[k:k + m, :len(keep)] = blk[:, sub].T
        if n_ex:
            exb = np.stack([extra[j, r0:r1, :] for j in range(n_ex)]).reshape(n_ex, -1)
            out[k:k + m, len(keep):] = exb[:, sub].T
        k += m
    assert k == n_sel, (k, n_sel)
    return out


def predict_map(layers: np.ndarray, keep: list[int], model, H: int, W: int,
                extra: np.ndarray | None = None) -> np.ndarray:
    n_ex = 0 if extra is None else extra.shape[0]
    out = np.zeros((H, W), dtype=np.float32)
    for r0 in range(0, H, BLOCK):
        r1 = min(r0 + BLOCK, H)
        blk = np.stack([np.asarray(layers[i, r0:r1, :], dtype=np.float32)
                        for i in keep]).reshape(len(keep), -1).T
        if n_ex:
            exb = np.stack([extra[j, r0:r1, :] for j in range(n_ex)]).reshape(n_ex, -1).T
            blk = np.hstack([blk, exb])
        out[r0:r1] = model.predict_proba(blk)[:, 1].reshape(r1 - r0, W).astype(np.float32)
    return out


def fold_structural(cat_train: np.ndarray, cfit: dict, top_k: int = 80) -> np.ndarray:
    """Catalogue-derived layers rebuilt from the TRAINING catalogue only."""
    d = ndimage.distance_transform_edt(~cat_train)
    layers = [np.log1p(d).astype(np.float32),
              (1.0 / (1.0 + d / 3.0)).astype(np.float32),
              ndimage.binary_dilation(cat_train, np.ones((3, 3), bool),
                                      iterations=3).astype(np.float32)]
    pop = next((p for p in cfit.get("populations", []) if p["population"].startswith("A_")), None)
    if pop is not None and cat_train.any():
        lab, n = ndimage.label(cat_train, structure=cl.STRUCT8)
        t = cl.extract_traces(cat_train, skeleton=True)
        prior = cl.bour_davy_prior_field(lab, t, pop.get("nearest_larger", {}),
                                         pixel_m=100.0, top_k=top_k, r_max_px=60)
        layers.append(prior)
        layers.append((prior > 0).astype(np.float32))
    return np.stack(layers)


def sample_plan(target: np.ndarray, train: np.ndarray, rng: np.random.Generator,
                n_pos_max: int, n_neg: int, ring: int = 2) -> np.ndarray:
    """Positives (thinned if needed) + labelled halo ring + random negatives."""
    pos = target & train
    sel = np.zeros(target.shape, bool)
    flat = sel.ravel()
    idx = np.flatnonzero(pos.ravel())
    if idx.size > n_pos_max:
        idx = rng.choice(idx, n_pos_max, replace=False)
    flat[idx] = True
    if ring > 0 and pos.any():
        halo = ndimage.binary_dilation(pos, np.ones((3, 3), bool), iterations=ring) & ~pos & train
        hidx = np.flatnonzero(halo.ravel())
        if hidx.size > n_pos_max:
            hidx = rng.choice(hidx, n_pos_max, replace=False)
        flat[hidx] = True
    negpool = np.flatnonzero((train & ~target).ravel())
    flat[rng.choice(negpool, min(n_neg, negpool.size), replace=False)] = True
    return sel


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss", max_iter=160, learning_rate=0.09, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, max_bins=127,
        early_stopping=False, random_state=seed)


def budget_sweep(prob: np.ndarray, target: np.ndarray, elig: np.ndarray,
                 budgets: list[int]) -> list[dict]:
    """Exact DTI-vs-budget curve via `metric.budget_curve` (verified identical to
    calling `components()` at every budget, 200x faster)."""
    if int((target & elig).sum()) == 0 or int(elig.sum()) == 0:
        return []
    return budget_curve(prob, target, elig, budgets)


def rank_blend(pa: np.ndarray, pb: np.ndarray, elig: np.ndarray, wa: float = 0.5) -> np.ndarray:
    out = np.zeros(pa.shape, np.float32)
    va, vb = pa[elig].astype(np.float64), pb[elig].astype(np.float64)
    if va.size < 2:
        return out
    ra = np.argsort(np.argsort(va, kind="stable"), kind="stable") / (va.size - 1)
    rb = np.argsort(np.argsort(vb, kind="stable"), kind="stable") / (vb.size - 1)
    out[elig] = (wa * ra + (1.0 - wa) * rb).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="union_traces",
                    choices=["union_traces", "quadrants", "blocks3x3", "traces_cat"])
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--heads", default="A,B")
    ap.add_argument("--n-pos", type=int, default=140_000)
    ap.add_argument("--n-neg", type=int, default=420_000)
    ap.add_argument("--blend-w", type=float, default=0.5)
    ap.add_argument("--buffer-px", type=int, default=12)
    ap.add_argument("--bd-topk", type=int, default=80)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--out", default="holdout_results.json")
    args = ap.parse_args()

    t0 = time.time()
    mm, all_names, keep, dropped = open_features()
    F, H, W = mm.shape
    print(f"cache: {F} layers; using {len(keep)}; EXCLUDED as catalogue-derived: {dropped}",
          flush=True)

    cat = load_labels()
    with rasterio.open(REPO / "data/sample_submission.tif") as ds:
        fp = np.isfinite(ds.read(1))
    sg = (rasterio.open(REPO / "assets/external/derived_sgmc_faults_100m_u8.tif").read(1) > 0)
    gap = sg & ~cat & fp                       # pixel-disjoint from the catalogue
    cfit = json.loads((REPO / "evidence/clustering_fit.json").read_text()) \
        if (REPO / "evidence/clustering_fit.json").exists() else {}
    print(f"catalogue={int(cat.sum()):,} sgmc_gap_strict={int(gap.sum()):,} "
          f"footprint={int(fp.sum()):,}", flush=True)

    targets = {"A": cat, "B": gap}

    # ---- fold construction -------------------------------------------------
    # union_traces: cluster the CONNECTED TRACES of (catalogue U SGMC-gap) into
    # spatially coherent groups and hold whole groups out.  Both heads therefore
    # get a genuine object-level holdout, while the rest of the map keeps its
    # full catalogue + gap context exactly as the live task does.
    if args.grid in ("union_traces", "traces_cat"):
        pop = (cat | gap) if args.grid == "union_traces" else cat
        tf = ho.trace_cluster_folds(pop, n_folds=4, n_clusters=48, seed=args.seed)
        blocks = {k: v for k, v in tf.items()}
        trace_mode = True
    else:
        blocks = ho.folds((H, W), args.grid)
        trace_mode = False
    keys = list(blocks)[:args.folds] if args.folds else list(blocks)
    heads = [h.strip().upper() for h in args.heads.split(",") if h.strip()]
    rng = np.random.default_rng(args.seed)

    oof = {h: np.full((H, W), np.nan, np.float32) for h in heads + ["C"]}
    results = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "grid": args.grid, "buffer_px": args.buffer_px, "seed": args.seed,
               "n_cache_layers": F, "n_used_layers": len(keep),
               "excluded_leaky_layers": dropped, "heads": heads,
               "blend_weight_A": args.blend_w,
               "targets": {"A": "labels.tif catalogue faults",
                           "B": "SGMC-gap (USGS state geologic map compilation faults "
                                "pixel-disjoint from the competition catalogue)"},
               "leakage_control": "catalogue-derived structural layers rebuilt per fold "
                                  "from catalogue & train_mask only",
               "folds": {}}

    for bk in keys:
        blk = blocks[bk]
        if trace_mode:
            # per-head ground truth and per-head scored domain, from ONE shared
            # object-level partition of (catalogue U SGMC-gap)
            fold_masks = {}
            for h in heads:
                gt_h = blk & targets[h]
                fold_masks[h] = ho.trace_fold_masks(targets[h], fp, gt_h, collar=1)
            tm = np.logical_and.reduce([fold_masks[h]["train_domain"] for h in heads])
            cat_train = cat & tm
            share = 1.0
            fb = list(BUDGETS)
        else:
            tm = ho.fold_train_mask(blk, fp, cat, args.buffer_px)
            cat_train = cat & tm
            fold_masks = {h: {"scored_domain": ho.evaluation_mask(blk, fp, cat, args.buffer_px),
                              "gt": ho.fold_target(blk, cat, fp, args.buffer_px),
                              "train_target": cat & tm} for h in heads}
            share = float(fold_masks[heads[0]]["scored_domain"].sum()) / max(float(fp.sum()), 1.0)
            fb = [b for b in BUDGETS]
        extra = fold_structural(cat_train, cfit, args.bd_topk)
        ems = {h: fold_masks[h]["scored_domain"] for h in heads}
        gts = {h: fold_masks[h]["gt"] for h in heads}
        rec = {"mode": "trace_cluster" if trace_mode else "geographic_block",
               "train_px": int(tm.sum()), "grid_share": round(share, 4),
               "budgets": fb, "n_extra_layers": int(extra.shape[0]),
               "eval_px": {h: int(ems[h].sum()) for h in heads},
               "gt_px": {h: int(gts[h].sum()) for h in heads}}
        print(f"\n=== fold {bk}: train={rec['train_px']:,} eval={rec['eval_px']} "
              f"gt={rec['gt_px']} t={time.time()-t0:.0f}s", flush=True)

        probs: dict[str, np.ndarray] = {}
        for h in heads:
            tgt = fold_masks[h]["train_target"]
            sel = sample_plan(tgt, fold_masks[h]["train_domain"], rng,
                              args.n_pos, args.n_neg)
            X = gather_layers(mm, keep, sel, extra)
            y = tgt[sel].astype(np.int8)
            print(f"   head {h}: rows={X.shape[0]:,} cols={X.shape[1]} pos={int(y.sum()):,} "
                  f"({100*y.mean():.2f}%)  t={time.time()-t0:.0f}s", flush=True)
            mdl = make_model(args.seed + (0 if h == "A" else 1000))
            mdl.fit(X, y)
            del X, y
            P = predict_map(mm, keep, mdl, H, W, extra)
            del mdl
            probs[h] = P
            oof[h][blk] = P[blk]
            (DERIV / f"oof_{h}_{bk}.f32").write_bytes(P.tobytes())
            print(f"   head {h}: p99(eval)={np.percentile(P[ems[h]],99):.4f} max={P.max():.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        if "A" in probs and "B" in probs:
            em = ems["A"] & ems["B"]
            probs["C"] = rank_blend(probs["A"], probs["B"], em, args.blend_w)
            oof["C"][blk] = probs["C"][blk]
            (DERIV / f"oof_C_{bk}.f32").write_bytes(probs["C"].tobytes())

        # ---- random-null control: how much DTI is bought by budget alone ----
        null = rng.random((H, W)).astype(np.float32)
        rec["null_random"] = {}
        for tn in ("A", "B"):
            if int(gts[tn].sum()) == 0:
                continue
            e_domain = ems["A"] & ems["B"]
            sw = budget_sweep(null, gts[tn], e_domain, fb)
            if sw:
                best = max(sw, key=lambda r: r["dti"])
                at120 = min(sw, key=lambda r: abs(r["n"] - 120_000))
                rec["null_random"][tn] = {"best": best, "at_120k": at120}
                print(f"   NULL random vs {tn}: best n={best['n']:,} DTI={best['dti']:.5f} | "
                      f"n=120k DTI={at120['dti']:.5f}", flush=True)
        del null

        for h, P in probs.items():
            rec[f"head_{h}"] = {}
            for tn in ("A", "B"):
                if int(gts[tn].sum()) == 0:
                    continue
                e_domain = ems["A"] & ems["B"]
                sw = budget_sweep(P, gts[tn], e_domain, fb)
                if not sw:
                    continue
                best = max(sw, key=lambda r: r["dti"])
                anchor_n = int(120_000 * share)
                near = min(sw, key=lambda r: abs(r["n"] - anchor_n))
                # DTI-optimal budget implied by the marginal rule alone
                rule_stop = next((r["n"] for r in sw
                                  if r["marginal_rule_says_add"] is False), None)
                rec[f"head_{h}"][f"vs_target_{tn}"] = {
                    "sweep": sw, "best": best, "at_anchor_budget": near,
                    "gain_vs_anchor": round(best["dti"] - near["dti"], 6),
                    "marginal_rule_stop_n": rule_stop,
                    "p99_eval": float(np.percentile(P[e_domain], 99))}
                print(f"   head {h} vs {tn}: BEST n={best['n']:,} DTI={best['dti']:.5f} | "
                      f"anchor-equiv n={near['n']:,} DTI={near['dti']:.5f} | "
                      f"gain={best['dti']-near['dti']:+.5f} | rule_stop={rule_stop}", flush=True)
        results["folds"][bk] = rec
        del extra
        (REPO / "evidence").mkdir(exist_ok=True)
        (REPO / "evidence" / args.out).write_text(json.dumps(results, indent=1, default=str))

    for h, arr in oof.items():
        if np.isfinite(arr).any():
            np.save(DERIV / f"oof_{h}.npy", arr)
    results["seconds"] = round(time.time() - t0, 1)
    (REPO / "evidence" / args.out).write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote evidence/{args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
