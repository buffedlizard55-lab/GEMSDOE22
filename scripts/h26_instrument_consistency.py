"""H26 — do the published anchor files beat a random control on the *valid* instrument?

Why this script exists
----------------------
The repository's headline holdout numbers for the live anchors (``h19-5`` 0.21341,
``h19-4`` 0.21413, ``h16-1`` 0.16953 ...) were measured on the four-quadrant
Dense/Sparse holdout, which the repository itself flags as structurally invalid
(flag F-09: a quadrant fold deletes *every* known fault from the held-out region,
while the live task keeps the catalogue visible everywhere).  The trace-cluster
fold is the instrument the repository chose as valid.

This script measures the *published* GeoTIFFs on the trace-cluster instrument with
one consistent domain convention -- ``domain = footprint & ~visible``, exactly the
live masking rule (known-fault pixels are excluded from scoring) -- and compares
each file with a random binary control of the *same* size on the *same* fold:

    lift = DTI(file) - floor_dti(n_file, |G|_fold, N_domain)

Run:
    python3 scripts/h26_instrument_consistency.py
    -> evidence/h26_instrument_consistency.json
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

from gems import floor as FL                      # noqa: E402
from gems.metric import dti_components_exact      # noqa: E402
from gems.paths import DATA_DIR, DOWNLOADS_DIR    # noqa: E402
from gems22.holdout import trace_cluster_folds    # noqa: E402

FILES = [
    "gems19-h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan.tif",
    "gems19-h19-4-multiline-corroborated-openness-thermal-pop-20260930-691e4dfa-nan.tif",
    "gems22-h22-1-fractal-clustering-prior-multiline-20261002-7fd2f28b-nan.tif",
    "gems22-h22-2-fractal-243pct-budget-corroborated-20261002-00a4a807-nan.tif",
    "gems22-h23-a-dti-optimal-emission-6pct-20261002-e2ec4b49-nan.tif",
    "gems22-h23-b-dti-optimal-emission-10pct-20261002-86176698-nan.tif",
]


def main() -> int:
    with rasterio.open(DATA_DIR / "sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(DATA_DIR / "labels.tif") as s:
        cat = s.read(1) > 0
    with rasterio.open(ROOT / "assets/external/derived_sgmc_faults_100m_u8.tif") as s:
        gap = s.read(1) > 0
    visible_all = cat | gap
    folds = trace_cluster_folds(visible_all, 4, 48, seed=22)

    rng = np.random.default_rng(20261002)
    rows = []
    for fname in FILES:
        with rasterio.open(DOWNLOADS_DIR / fname) as s:
            pred_full = np.nan_to_num(s.read(1), nan=0.0) > 0.5
        for fold_name, fold_mask in folds.items():
            visible = visible_all & ~fold_mask
            domain = fp & ~visible
            for head, head_mask in (("A", cat), ("B", gap)):
                truth = head_mask & fold_mask
                c = dti_components_exact(pred_full.astype(np.float32),
                                         truth.astype(np.uint8), valid_mask=domain)
                n, G, ndom = int(c["n_emitted"]), int(c["n_truth"]), int(domain.sum())
                fl = FL.floor_dti(n, float(G), float(ndom))
                # random control of the same size, same fold, same domain
                idx = np.flatnonzero(domain.ravel())
                pick = rng.choice(idx.size, size=min(n, idx.size), replace=False)
                ctrl = np.zeros(domain.size, dtype=np.float32)
                ctrl[idx[pick]] = 1.0
                cc = dti_components_exact(ctrl.reshape(domain.shape),
                                          truth.astype(np.uint8), valid_mask=domain)
                rows.append({
                    "file": fname.split("-")[2] + "-" + fname.split("-")[4][:8],
                    "fold": fold_name, "head": head, "n": n, "G": G, "N_domain": ndom,
                    "dti": round(float(c["dti"]), 5), "A": round(float(c["TP_w"]), 1),
                    "B": round(float(c["FP_w"]), 1), "coverage": round(float(c["coverage"]), 4),
                    "floor_closed_form": round(float(fl), 5),
                    "random_control_dti": round(float(cc["dti"]), 5),
                    "lift_vs_floor": round(float(c["dti"]) - fl, 5),
                    "lift_vs_random_control": round(float(c["dti"]) - float(cc["dti"]), 5),
                })
        print(f"  {rows[-1]['file']}: done")

    # per-file summary over the 8 fold x head cells
    summary = {}
    for fname in {r["file"] for r in rows}:
        sel = [r for r in rows if r["file"] == fname]
        lift = [r["lift_vs_floor"] for r in sel]
        lift_rc = [r["lift_vs_random_control"] for r in sel]
        summary[fname] = {
            "n_fold_head_cells": len(sel),
            "n_emitted_median": int(np.median([r["n"] for r in sel])),
            "mean_dti": round(float(np.mean([r["dti"] for r in sel])), 5),
            "mean_floor": round(float(np.mean([r["floor_closed_form"] for r in sel])), 5),
            "mean_lift_vs_floor": round(float(np.mean(lift)), 5),
            "mean_lift_vs_random_control": round(float(np.mean(lift_rc)), 5),
            "cells_positive_vs_random_control": int(sum(1 for x in lift_rc if x > 0)),
            "fold_head_cells": len(sel),
        }
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/h26_instrument_consistency.py",
        "status": "COMPUTED",
        "instrument": "trace_cluster_folds(cat|gap, 4, 48, seed=22); domain = footprint & ~visible",
        "note": ("This is the repository's chosen *valid* instrument. It was built to "
                 "test whether held-out traces can be recovered; because each fold holds "
                 "out a spatially coherent cluster, proximity priors are penalised harder "
                 "here than in the live task, where new faults are often continuations of "
                 "mapped systems. Rank results accordingly."),
        "rows": rows, "summary": summary,
    }
    (ROOT / "evidence/h26_instrument_consistency.json").write_text(json.dumps(out, indent=1))
    print(f"{'file':>16} {'n':>8} {'meanDTI':>8} {'meanFloor':>9} {'lift_floor':>10} "
          f"{'lift_random':>11} {'cells>0':>8}")
    for k, v in summary.items():
        print(f"{k:>16} {v['n_emitted_median']:>8,} {v['mean_dti']:>8.5f} {v['mean_floor']:>9.5f} "
              f"{v['mean_lift_vs_floor']:>+10.5f} {v['mean_lift_vs_random_control']:>+11.5f} "
              f"{v['cells_positive_vs_random_control']:>5}/{v['n_fold_head_cells']}")
    print("wrote evidence/h26_instrument_consistency.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
