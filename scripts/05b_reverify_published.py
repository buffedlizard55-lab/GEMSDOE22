#!/usr/bin/env python3
"""Re-verify the GeoTIFFs that are ACTUALLY SERVED, against the recorded build.

`05_build_submission.py` verifies each file the instant it writes it.  That is
necessary but not sufficient: between the write and the upload the artefact is
copied into `docs/downloads/gems22/`, zipped, committed to git, and served over GitHub
Pages.  Any of those steps could in principle substitute, truncate or re-encode
the file.  This script closes that gap by treating the *served* bytes as the
object under test:

  1. SHA-256 of every file in `docs/downloads/gems22/` must equal the value recorded in
     `evidence/submission_build.json` (and in `submissions/`, if present).
  2. Every served `.tif` is re-parsed from disk and re-run through
     `submission.verify_file`, i.e. the same 12 hard checks the writer ran.
  3. Each `.zip` must contain exactly one `.tif` whose bytes equal the loose one.
  4. The full per-check verdicts are merged back into the evidence JSON so the
     site renders measured values rather than `pending`.

Exit status is non-zero if any check fails, so this can gate a release.

Usage:  python3 scripts/05b_reverify_published.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import submission as sub                     # noqa: E402
from gems22.spec import REPO                              # noqa: E402

DOCS = REPO / "docs" / "downloads" / "gems22"
ALT = REPO / "submissions"
EV = REPO / "evidence" / "submission_build.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not EV.exists():
        print(f"MISSING {EV} — run scripts/05_build_submission.py first")
        return 2
    ev = json.loads(EV.read_text())
    recs = {r["file"]: r for r in (ev.get("files") or [])}
    if not recs:
        print("evidence/submission_build.json has no file records")
        return 2

    # Pixel-level verification needs the two official rasters, which are
    # gitignored (419 MB) and absent on a fresh clone or a CI runner.  When they
    # are missing this script degrades to SHA-256 + size + zip-integrity checking
    # of the committed artefacts -- still a real gate, because it proves the bytes
    # in git are the bytes the build recorded -- and says so loudly instead of
    # crashing or, worse, passing silently.
    sample = REPO / "data/sample_submission.tif"
    labels = REPO / "data/labels.tif"
    pixel_level = sample.exists() and labels.exists()
    if pixel_level:
        with rasterio.open(sample) as ds:
            footprint = np.isfinite(ds.read(1))
        with rasterio.open(labels) as ds:
            catalogue = ds.read(1) == 1
    else:
        footprint = catalogue = None
        print("  NOTE: official rasters absent -> pixel-level checks SKIPPED; "
              "running SHA-256 / size / zip integrity only\n")

    failures: list[str] = []
    checked: list[dict] = []
    print(f"re-verifying served artefacts in {DOCS.relative_to(REPO)}/ "
          f"and {ALT.relative_to(REPO)}/\n")

    for d in (DOCS, ALT):
        if not d.is_dir():
            print(f"  (skip {d.relative_to(REPO)}/ — not present)")
            continue
        for p in sorted(d.glob("*.tif")):
            rec = recs.get(p.name)
            digest = sha256(p)
            row = {"dir": str(d.relative_to(REPO)), "file": p.name, "sha256": digest,
                   "bytes": p.stat().st_size}
            if rec is None:
                failures.append(f"{p.relative_to(REPO)}: not in evidence/submission_build.json")
                row["recorded"] = False
                print(f"  {p.name}: NO RECORD")
            else:
                row["recorded"] = True
                row["sha256_matches"] = (digest == rec["sha256"])
                row["bytes_match"] = (p.stat().st_size == rec["bytes"])
                if not row["sha256_matches"]:
                    failures.append(f"{p.relative_to(REPO)}: SHA-256 {digest[:16]} != recorded "
                                    f"{rec['sha256'][:16]}")
                if not row["bytes_match"]:
                    failures.append(f"{p.relative_to(REPO)}: size changed")
            if not pixel_level:
                checked.append(row)
                print(f"  {p.relative_to(REPO)}")
                print(f"      sha256 {digest[:16]}…  "
                      f"{'MATCH' if row.get('sha256_matches') else 'MISMATCH'}"
                      f"   bytes {p.stat().st_size:,}   (pixel checks skipped)")
                continue
            c = sub.verify_file(p, footprint, catalogue)
            # merge the SAME structure registry_record() emits: the outer
            # verify_file dict, whose "checks" member holds the 16 boolean
            # verdicts.  The site reads checks["hard_all_pass"] and
            # checks["checks"][k], so both must be present.
            row["checks"] = {k: v for k, v in c.items() if k != "path"}
            row["hard_all_pass"] = bool(c["hard_all_pass"])
            if not row["hard_all_pass"]:
                bad = [k for k in c["hard_keys"] if not c["checks"].get(k)]
                failures.append(f"{p.relative_to(REPO)}: hard checks failed {bad}")
            row["n_positive_scored"] = int(c["n_positive_scored"])
            row["positive_on_catalogue"] = int(c["positive_on_catalogue"])
            row["is_binary_0_1"] = bool(c["is_binary_0_1"])
            checked.append(row)
            nhard = len(c["hard_keys"])
            npass = sum(1 for k in c["hard_keys"] if c["checks"].get(k))
            print(f"  {p.relative_to(REPO)}")
            print(f"      sha256 {digest[:16]}…  {'MATCH' if row.get('sha256_matches') else 'MISMATCH'}"
                  f"   bytes {p.stat().st_size:,}")
            print(f"      hard checks {npass}/{nhard} pass  "
                  f"pos_scored={row['n_positive_scored']:,}  "
                  f"pos_on_catalogue={row['positive_on_catalogue']}  "
                  f"binary_0_1={row['is_binary_0_1']}")

        for z in sorted(d.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                row = {"dir": str(d.relative_to(REPO)), "file": z.name,
                       "zip_members": names}
                ok = len(names) == 1 and names[0].endswith(".tif")
                if not ok:
                    failures.append(f"{z.relative_to(REPO)}: must contain exactly one .tif, got {names}")
                else:
                    inner = zf.read(names[0])
                    loose = d / names[0]
                    same = loose.exists() and hashlib.sha256(inner).hexdigest() == sha256(loose)
                    row["zip_matches_loose_tif"] = bool(same)
                    if not same:
                        failures.append(f"{z.relative_to(REPO)}: zipped .tif differs from the loose .tif")
                    ok = ok and same
                row["zip_ok"] = bool(ok)
                checked.append(row)
                print(f"  {z.relative_to(REPO)}: {len(names)} member(s) "
                      f"{'MATCH loose .tif' if row.get('zip_matches_loose_tif') else ''} "
                      f"-> {'OK' if ok else 'FAIL'}")

    # merge the per-check verdicts back into the evidence so the site can render them
    for r in ev["files"]:
        for row in checked:
            if row["file"] == r["file"] and "checks" in row:
                r["checks"] = row["checks"]
                r["hard_all_pass"] = row["hard_all_pass"]
                r["sha256_matches_served"] = row.get("sha256_matches")
    ev["reverify"] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/05b_reverify_published.py",
        "artefacts_checked": len(checked),
        "pixel_level_checks_run": pixel_level,
        "all_pass": not failures,
        "failures": failures,
        "detail": checked,
    }
    EV.write_text(json.dumps(ev, indent=1, default=str))
    print(f"\nartefacts checked: {len(checked)}   failures: {len(failures)}")
    for f in failures:
        print(f"  FAIL {f}")
    print(f"wrote {EV.relative_to(REPO)} (merged per-check verdicts + reverify block)")
    print("ALL SERVED ARTEFACTS PASS" if not failures else "REVERIFICATION FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
