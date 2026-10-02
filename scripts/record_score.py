#!/usr/bin/env python3
"""Append an observed DrivenData public score to registry/submissions.json.

The sandbox cannot log in to DrivenData, so the live score of a new file is not
observable here. This is the single manual-input escape hatch in the repo: run it
from a machine that can see the submission list, or paste the number by hand.

    python3 scripts/record_score.py --id gems22-... --score 0.2031
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22.spec import REPO

ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True, help="content_id, filename, or tag")
ap.add_argument("--score", required=True, type=float)
ap.add_argument("--account", default=None)
ap.add_argument("--url", default=None)
a = ap.parse_args()
reg = REPO / "registry/submissions.json"
d = json.loads(reg.read_text()) if reg.exists() else {"schema_version": 1, "entries": []}
hit = None
for e in d["entries"]:
    if a.id in (e.get("content_id"), e.get("tag")) or any(
            a.id in (f.get("file") or "") for f in e.get("files", [])):
        hit = e
if hit is None:
    hit = {"content_id": a.id, "tag": a.id, "files": []}
    d["entries"].append(hit)
hit["observed_public_score"] = a.score
hit["score_observed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if a.account: hit["account"] = a.account
if a.url: hit["submission_page"] = a.url
reg.write_text(json.dumps(d, indent=1))
print(f"recorded {a.id} -> {a.score}")
print("now re-run: python3 scripts/04b_infer_G_and_rescale.py   (re-fits |G| with the new point)")
