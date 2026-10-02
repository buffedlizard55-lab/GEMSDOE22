#!/usr/bin/env bash
# Full pipeline. Safe to re-run; every step is idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
echo "### 0/9 measure network egress (records what can and cannot be fetched here)"
$PY scripts/07_probe_network.py
echo "### 1/9 fetch + hash-verify the official data"; bash scripts/fetch_data.sh
echo "### 2/9 calibrate offline proxies against the 3 live-scored anchors"; $PY scripts/01_calibrate_anchors.py
echo "### 3/9 fit Bour&Davy(1999) + Marrett(2018) NCC BEFORE any model"; $PY scripts/02_fit_clustering.py
echo "### 4/9 build the 156-layer feature cache (~195 s, 3.8 GB)"; $PY scripts/03_build_features.py
echo "### 5/9 blocked CV + exact DTI budget sweep (~22 min)"; $PY scripts/04_train_and_optimize.py --grid union_traces --heads A,B
echo "### 6/9 infer |G| from the live scores and rescale the budget curve"; $PY scripts/04b_infer_G_and_rescale.py
echo "### 7/9 head-to-head gate: our maps vs the live-scored anchors"; $PY scripts/04c_head_to_head.py
echo "### 8/9 build + audit + write the submission"; $PY scripts/05_build_submission.py
echo "### 9/9 re-verify the SERVED bytes, then build the site"; $PY scripts/05b_reverify_published.py && $PY scripts/06_build_site.py
echo "### tests"; $PY -m pytest tests -q
