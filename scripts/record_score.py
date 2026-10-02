"""Record the public score DrivenData shows for one of our candidates (the single manual step).

DrivenData's Terms of Use forbid robots/spiders, and scores sit behind login, so a person copies the number here.
Usage:  python scripts/record_score.py <candidate-key> <score> [--account NAME] [--comment TEXT]
Keys: see docs/data/submissions.json (h18-3a, h16-1, sgmc-gap).  Appends to registry/ledger.json and rebuilds the site.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "registry" / "ledger.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key")
    ap.add_argument("score", type=float)
    ap.add_argument("--account", default=None, help="DrivenData account that uploaded it (optional)")
    ap.add_argument("--comment", default="")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if not 0.0 <= a.score <= 1.0:
        print("score must be in [0, 1]")
        return 2
    subs = json.loads((ROOT / "docs" / "data" / "submissions.json").read_text())
    cand = next((c for c in subs["candidates"] if c["key"] == a.key), None)
    if cand is None:
        print(f"unknown key {a.key!r}; choose from {[c['key'] for c in subs['candidates']]}")
        return 2
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"schema_version": 1, "entries": []}
    entry = {"candidate_key": a.key, "content_id": cand["content_id"], "file_sha256": cand["files"]["tif"]["sha256"],
             "score": a.score, "account": a.account, "comment": a.comment,
             "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ledger["entries"].append(entry)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"recorded {a.key} ({cand['content_id']}) = {a.score:.4f}")
    by = {}
    for e in ledger["entries"]:
        by[e["candidate_key"]] = e["score"]
    if "h18-3a" in by and "h16-1" in by:
        d = by["h18-3a"] - by["h16-1"]
        print(f"A/B (H18-3a minus H16-1) = {d:+.4f}. Holdout proxy predicted H18-3a ahead. "
              + ("Direction agrees with the proxy." if d > 0 else "Direction DISAGREES with the proxy: down-weight the holdout gate."))
    print("historical best public score in the registry: 0.1563 (ens12 cluster)")
    if not a.no_build:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py")], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
