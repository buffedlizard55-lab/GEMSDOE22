"""Scan the group's public GEMS repositories for NEW GeoTIFFs and compare each against everything already uploaded.

Answers "are we about to submit the same thing again?" automatically: any published .tif that is not in
registry/submissions.json is fetched (via the authenticated ``gh`` CLI), format-checked, and compared on the scored pixels
(footprint minus known-fault mask) with every registered file.  It also cross-checks this hub's own ready-to-upload
candidates (docs/data/submissions.json) against every newly found file, because a candidate can become a duplicate of a file
somebody else in the group published after the registry was built.  Output: docs/data/group_scan.json.
Needs only the footprint payload in docs/data and two small public files (labels + registered submissions).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.io import MemoryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems import forensics as F  # noqa: E402
from gems.footprint import load_footprint, write_template  # noqa: E402
from gems.paths import GROUP_DIR, SITE_DATA_DIR  # noqa: E402
from gems.submission import check_variants  # noqa: E402

OWNER = "buffedlizard55-lab"
REPO_RE = re.compile(r"^(\d+)?GEMSDOE\d*$")
IGNORE = ("/bridge/", "fixture", "labels.tif", "sample_submission", "example_submission", "existing_faults", "proxy_", "qfaults_catalogue",
          "/external/", "/derived/", "/cache/", "/dem/", "geodawn_", "prior_u8", "/evidence/runs/", "/evidence/baseline", "/evidence/proxy", "/evidence/xcat")
LABELS_SHA256 = "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093"
MAX_BYTES = 8_000_000
MAX_FILES = 150                 # never truncate silently: the report records found_total and cap_reached
SELF_REPO = "16GEMSDOE"         # this hub: its own downloads are gated by scripts/build_candidates.py, not listed as "new"
OVERLAP_JACCARD = 0.50          # below the 0.80 duplicate threshold, but the "same idea tried differently" zone


def decode_pages(raw: bytes | str) -> list:
    """Decode ``gh api --paginate`` output: one JSON document per page, written back to back.

    ``json.loads`` fails on that ("Extra data") as soon as a listing has more than one page, and older ``gh`` releases
    have no ``--slurp``; so decode the documents one by one.
    """
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    decoder, pos, docs = json.JSONDecoder(), 0, []
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return docs
        doc, pos = decoder.raw_decode(text, pos)
        docs.append(doc)


def gh_json(path: str):
    docs = decode_pages(subprocess.run(["gh", "api", path, "--paginate"], check=True, capture_output=True).stdout)
    if docs and all(isinstance(d, list) for d in docs):          # list endpoint: flatten the pages
        return [item for page in docs for item in page]
    return docs[0] if len(docs) == 1 else docs


# --- GitHub access layer (module-level so tests can substitute an offline fake) ----------------------------------------
def list_repos(owner: str) -> list[str]:
    return [r["name"] for r in gh_json(f"users/{owner}/repos?per_page=100") if REPO_RE.match(r["name"])]


WARNINGS: list[str] = []        # blind spots found while listing; written to the report so a gap is never silent


def list_tree(owner: str, repo: str) -> list[dict]:
    try:
        tree = json.loads(subprocess.run(["gh", "api", f"repos/{owner}/{repo}/git/trees/main?recursive=1"], check=True, capture_output=True).stdout)
    except subprocess.CalledProcessError:
        WARNINGS.append(f"{repo}: tree listing failed (no 'main' branch or an API error); its files were NOT scanned")
        return []
    if tree.get("truncated"):
        WARNINGS.append(f"{repo}: GitHub truncated the recursive tree listing; some of its files may not have been scanned")
    return tree.get("tree", [])


def fetch_file(owner: str, repo: str, path: str, dest: Path) -> bytes:
    return F.gh_fetch_raw(owner, repo, path, dest)


def load_candidates(grid) -> list[dict]:
    """This hub's ready-to-upload candidates (docs/data/submissions.json) as arrays; [] when the manifest is absent."""
    manifest = SITE_DATA_DIR / "submissions.json"
    if not manifest.exists():
        return []
    cands = []
    for c in json.loads(manifest.read_text()).get("candidates", []):
        path = ROOT / "docs" / "downloads" / c["files"]["tif"]["name"]
        if path.exists():
            with rasterio.open(path) as s:
                arr = s.read(1)
            cands.append({"key": c["key"], "file": path.name, "array": arr, "pos": F.scored_positive(arr, grid), "best": None})
    return cands


def overlap_verdict(m: dict) -> str:
    if m["identical_on_scored_pixels"]:
        return "DUPLICATE"
    if m["jaccard_positive"] >= F.NEAR_DUP_JACCARD:
        return "NEAR_DUPLICATE"
    return "OVERLAP" if m["jaccard_positive"] >= OVERLAP_JACCARD else "DISTINCT"


def main() -> int:
    WARNINGS.clear()
    reg = json.loads((ROOT / "registry" / "submissions.json").read_text())["entries"]
    reg_blobs = {e["git_blob_sha1"] for e in reg}
    fp = load_footprint()
    labels_path = GROUP_DIR / "_labels.tif"
    data = fetch_file(OWNER, "GEMSDOE", "data/bridge/existing_faults.tif", labels_path)
    if F.sha256_bytes(data) != LABELS_SHA256:
        print("labels file hash mismatch; aborting")
        return 1
    with rasterio.open(labels_path) as s:
        lab = s.read(1)
    grid = F.Grid.from_footprint(fp, lab)
    template = write_template(Path(tempfile.mkdtemp()) / "template.tif")
    history = []
    for e in reg:
        dest = GROUP_DIR / f"{e['id']}.tif"
        if not dest.exists() or F.git_blob_sha1(dest.read_bytes()) != e["git_blob_sha1"]:
            fetch_file(OWNER, e["github_repo"], e["repo_path"], dest)
        with rasterio.open(dest) as s:
            history.append({"id": e["id"], "array": s.read(1), "lb_score": e["lb_score"]})

    cands = load_candidates(grid)
    repos = [r for r in list_repos(OWNER) if r != SELF_REPO]
    seen_blobs, found = set(), []
    tifs_seen = 0
    for repo in sorted(repos):
        for ent in list_tree(OWNER, repo):
            path = ent["path"]
            if ent["type"] != "blob" or not path.lower().endswith((".tif", ".tiff")):
                continue
            tifs_seen += 1
            if any(tok in path for tok in IGNORE) or ent["sha"] in reg_blobs or ent["sha"] in seen_blobs:
                continue
            if not (path.startswith(("docs/", "downloads/")) or "/downloads/" in path):
                continue
            if (ent.get("size") or 0) > MAX_BYTES:
                continue
            seen_blobs.add(ent["sha"])
            found.append((repo, path, ent["sha"], ent.get("size")))
    found_total = len(found)
    found = found[:MAX_FILES]
    unreg = []
    for repo, path, sha, size in found:
        dest = GROUP_DIR / "scan" / f"{repo}__{path.replace('/', '__')}"
        try:
            blob = fetch_file(OWNER, repo, path, dest)
            with MemoryFile(blob) as m, m.open() as s:
                arr = s.read(1)
                ok_shape = arr.shape == fp.shape
            if not ok_shape:
                unreg.append({"repo": repo, "path": path, "bytes": size, "git_blob_sha1": sha, "verdict": "NOT_A_SUBMISSION_GRID"})
                continue
            ph = F.scored_positive(arr, grid)
            for cand in cands:
                m = F.pair_metrics(cand["array"], arr, grid, pa=cand["pos"], pb=ph)
                if cand["best"] is None or m["jaccard_positive"] > cand["best"]["m"]["jaccard_positive"]:
                    cand["best"] = {"repo": repo, "path": path, "sha": sha, "m": m}
            chk = check_variants(dest, template)
            gate = F.gate_candidate(arr, grid, history)
            nn = gate["nearest"][0] if gate["nearest"] else None
            unreg.append({"repo": repo, "path": path, "bytes": size, "git_blob_sha1": sha, "format_ok": chk["ok_to_upload"],
                          "hard_failures": chk["hard_failures"], "verdict": gate["verdict"],
                          "nearest_id": nn["id"] if nn else None, "nearest_jaccard": nn["jaccard_positive"] if nn else None,
                          "nearest_lb_score": nn["lb_score"] if nn else None, "scored_positive_pixels": gate["candidate_positive_scored_pixels"]})
        except Exception as exc:  # noqa: BLE001
            unreg.append({"repo": repo, "path": path, "bytes": size, "git_blob_sha1": sha, "verdict": "ERROR", "error": repr(exc)[:160]})
    cand_rows = []
    for cand in cands:
        b = cand["best"]
        cand_rows.append({"key": cand["key"], "file": cand["file"],
                          "verdict": overlap_verdict(b["m"]) if b else "DISTINCT",
                          "nearest_repo": b["repo"] if b else None, "nearest_path": b["path"] if b else None,
                          "nearest_git_blob_sha1": b["sha"] if b else None,
                          "jaccard_positive": b["m"]["jaccard_positive"] if b else 0.0,
                          "shared_positive": b["m"]["shared_positive"] if b else 0,
                          "candidate_positive": int(cand["pos"].sum()), "found_positive": b["m"]["b_positive"] if b else 0})
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repos_scanned": len(repos), "tif_blobs_seen": tifs_seen,
           "registered_entries": len(reg), "found_total": found_total, "cap_reached": found_total > MAX_FILES,
           "unregistered": unreg, "candidates_vs_found": cand_rows, "warnings": list(WARNINGS),
           "note": (f"Only published downloads are scanned (docs/, downloads/); the hub repository {SELF_REPO} is skipped because its downloads are gated "
                    f"by scripts/build_candidates.py. A DISTINCT verdict means Jaccard < {F.NEAR_DUP_JACCARD} with every registered file on scored pixels. "
                    f"candidates_vs_found compares this hub's ready-to-upload files with every newly found file (OVERLAP = Jaccard >= {OVERLAP_JACCARD}).")}
    (SITE_DATA_DIR / "group_scan.json").write_text(json.dumps(out, indent=2) + "\n")
    dups = [u for u in unreg if u["verdict"] in ("DUPLICATE", "NEAR_DUPLICATE")]
    print(f"repos={len(repos)} tif blobs={tifs_seen} unregistered scanned={len(unreg)} of {found_total} duplicates/near={len(dups)}")
    for w in WARNINGS:
        print("  WARNING:", w)
    for r in cand_rows:
        print(f"  candidate {r['key']}: {r['verdict']} (nearest J={r['jaccard_positive']} {r['nearest_repo']}/{(r['nearest_path'] or '')[-50:]})")
    for u in unreg[:12]:
        print(" ", u["repo"], u["path"][-50:], u["verdict"], u.get("nearest_id"), u.get("nearest_jaccard"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
