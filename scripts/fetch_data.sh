#!/usr/bin/env bash
# Reconstruct data/ entirely from PUBLIC sources. No manual input.
#
# Two routes, tried in order. Route A works from any machine with network access
# to the official mirrors; Route B works from a sandbox where only GitHub is
# reachable (which is the situation this repo was built in -- see
# evidence/network_reachability.json).
#
# Official data tab (login required):
#   https://www.drivendata.org/competitions/306/competition-doe-gems/data/
# Public mirrors recorded in buffedlizard55-lab/6GEMSDOE data/bridge/manifest.json:
#   training_features.tif <- gems-geodawn-numerical-features.tif
#   labels.tif            <- existing_faults.tif
#   sample_submission.tif <- example_submission.tif
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data data/derived

PY=${PYTHON:-python3}

echo "== Route A: official Dropbox mirrors =="
declare -A M=(
 [training_features.tif]="https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&dl=1"
 [labels.tif]="https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&dl=1"
 [sample_submission.tif]="https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&dl=1"
)
for f in "${!M[@]}"; do
  if [ -s "data/$f" ]; then echo "  have data/$f"; continue; fi
  echo "  GET $f"; curl -sSL --max-time 1800 -o "data/$f" "${M[$f]}" || echo "  FAILED $f (will try Route B)"
done

echo "== Route B: GitHub-transported byte-identical copies =="
# 6GEMSDOE committed the official rasters split under GitHub's 100 MB blob limit.
BR="https://codeload.github.com/buffedlizard55-lab/6GEMSDOE/tar.gz/refs/heads/main"
if [ ! -s data/training_features.tif ] || [ ! -s data/labels.tif ] || [ ! -s data/sample_submission.tif ]; then
  tmp=$(mktemp -d)
  echo "  fetching 6GEMSDOE tarball (~397 MB)"
  curl -sSL --max-time 1800 -o "$tmp/6.tar.gz" "$BR"
  mkdir -p "$tmp/x" && tar xzf "$tmp/6.tar.gz" -C "$tmp/x" --strip-components=1
  cp -n "$tmp/x/data/labels.tif" data/ 2>/dev/null || true
  cp -n "$tmp/x/data/sample_submission.tif" data/ 2>/dev/null || true
  if [ ! -s data/training_features.tif ]; then
    $PY - "$tmp/x/data/bridge" <<'PY'
import hashlib, sys
from pathlib import Path
src = Path(sys.argv[1]); out = Path("data/training_features.tif")
h = hashlib.sha256(); n = 0
with open(out, "wb") as fo:
    for i in range(5):
        p = src / f"gems-geodawn-numerical-features.tif.part-{i:03d}"
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                fo.write(c); h.update(c); n += len(c)
print("reassembled", n, "bytes sha256", h.hexdigest())
PY
  fi
  rm -rf "$tmp"
fi

echo "== verify every byte against the SHA-256 pins =="
$PY scripts/00_verify_data.py
