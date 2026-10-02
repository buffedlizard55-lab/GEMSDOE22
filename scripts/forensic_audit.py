"""Reproducible forensic audit of every group submission GeoTIFF.

What it does (all from public GitHub content; nothing is sent to DrivenData):
  1. fetches each registered file through the authenticated ``gh`` CLI and verifies its Git blob SHA-1
     against both the registry and the live repository tree;
  2. computes SHA-256 and label-free descriptive statistics;
  3. compares every pair of files on the *scored* pixels (footprint minus known-fault mask);
  4. maps byte-identical files across every GEMS repository (lineage: how the same artifact propagates);
  5. reports descriptive, deduplicated rank correlations between reported public scores and geometry.

Outputs: ``evidence/submission_similarity.json`` and ``docs/data/forensics.json``.
The analysis is descriptive: a public score cannot be inverted to hidden truth.

Usage:  GEMS_DATA_DIR=/path/to/data python scripts/forensic_audit.py [--offline]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems import forensics as F  # noqa: E402
from gems.paths import EVIDENCE_DIR, GROUP_DIR, LABELS_PATH, SITE_DATA_DIR, TEMPLATE_PATH  # noqa: E402

REGISTRY = ROOT / "registry" / "submissions.json"
OWNER = "buffedlizard55-lab"
EXTRA_REPOS = ["16GEMSDOE", "LEARNGEMSDOE"]


def live_tree(repo: str) -> dict[str, tuple[str, int | None]]:
    out = subprocess.run(
        ["gh", "api", f"repos/{OWNER}/{repo}/git/trees/main?recursive=1"], check=True, capture_output=True
    ).stdout
    d = json.loads(out)
    return {e["path"]: (e["sha"], e.get("size")) for e in d.get("tree", []) if e["type"] == "blob"}


def _frac_within(points: np.ndarray, tree) -> float:
    """Fraction of ``points`` closer than 3 px (300 m) to any point in ``tree`` (DTI kernel support)."""
    if tree is None or len(points) == 0:
        return 0.0
    d, _ = tree.query(points, k=1, distance_upper_bound=F.NEAR_PX + 1e-6)
    return float((d < F.NEAR_PX).mean())  # strictly inside the kernel support (k(d) > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use cached files only; skip GitHub calls")
    args = ap.parse_args()

    reg = F.load_json(REGISTRY)
    entries = reg["entries"]
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    grid = F.Grid.load(TEMPLATE_PATH, LABELS_PATH)

    trees: dict[str, dict] = {}
    repos = sorted({e["github_repo"] for e in entries} | set(EXTRA_REPOS))
    if not args.offline:
        for r in repos:
            trees[r] = live_tree(r)

    rows, arrays_idx, trees_kd = [], {}, {}
    integrity_problems = []
    for e in entries:
        dest = GROUP_DIR / f"{e['id']}.tif"
        if not dest.exists() or F.git_blob_sha1(dest.read_bytes()) != e["git_blob_sha1"]:
            if args.offline:
                raise SystemExit(f"missing/invalid cache for {e['id']} (offline)")
            F.gh_fetch_raw(OWNER, e["github_repo"], e["repo_path"], dest)
        data = dest.read_bytes()
        blob = F.git_blob_sha1(data)
        live = trees.get(e["github_repo"], {}).get(e["repo_path"], (None, None))[0] if trees else None
        ok_registry = blob == e["git_blob_sha1"]
        ok_live = (live == blob) if live else None
        if not ok_registry or ok_live is False:
            integrity_problems.append({"id": e["id"], "registry": e["git_blob_sha1"], "downloaded": blob, "live": live})
        arr = F.read_prediction(dest)
        desc = F.describe(arr, grid)
        pos = F.scored_positive(arr, grid)
        arrays_idx[e["id"]] = np.flatnonzero(pos.ravel()).astype(np.int32)
        rc = np.column_stack(np.nonzero(pos)).astype(np.float32)
        trees_kd[e["id"]] = (cKDTree(rc) if len(rc) else None, rc)
        scored_vals = np.nan_to_num(arr[grid.scored], nan=0.0)
        rows.append({
            "id": e["id"], "display_name": e["display_name"], "github_repo": e["github_repo"],
            "repo_path": e["repo_path"], "site_url": e["site_url"], "label": e["label"],
            "lb_score": e["lb_score"], "account_named_by_owner": e.get("account_named_by_owner"),
            "bytes": len(data), "sha256": F.sha256_bytes(data), "git_blob_sha1": blob,
            "blob_matches_registry": ok_registry, "blob_matches_live_repo": ok_live,
            "scored_content_sha256": F.sha256_bytes(np.ascontiguousarray((scored_vals > 0.5).astype(np.uint8)).tobytes()),
            **desc,
        })
        del arr
        print(f"  {e['id']:<22} n_scored_pos={desc['positive_scored_pixels']:>7,} outside={desc['outside_mode']:<5} "
              f"in_fp_nan={desc['in_footprint_nan']} range=[{desc['in_footprint_min']},{desc['in_footprint_max']}] "
              f"blob_ok={ok_registry}")

    # ---- pairwise on scored pixels --------------------------------------------------------------
    pairs = []
    ids = [r["id"] for r in rows]
    by_id = {r["id"]: r for r in rows}
    for a, b in combinations(ids, 2):
        ia, ib = arrays_idx[a], arrays_idx[b]
        inter = int(np.intersect1d(ia, ib, assume_unique=True).size)
        union = int(ia.size + ib.size - inter)
        ident = by_id[a]["scored_content_sha256"] == by_id[b]["scored_content_sha256"]
        within_ab = _frac_within(trees_kd[a][1], trees_kd[b][0])
        within_ba = _frac_within(trees_kd[b][1], trees_kd[a][0])
        pairs.append({
            "a": a, "b": b, "identical_on_scored_pixels": bool(ident),
            "jaccard_positive": round(inter / union, 5) if union else 1.0,
            "shared_positive": inter, "a_positive": int(ia.size), "b_positive": int(ib.size),
            "frac_a_within_300m_of_b": round(within_ab, 4), "frac_b_within_300m_of_a": round(within_ba, 4),
        })
    # identical-on-scored groups (union-find)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        if p["identical_on_scored_pixels"]:
            parent[find(p["a"])] = find(p["b"])
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    identical_groups = [sorted(v) for v in groups.values() if len(v) > 1]
    near_dups = sorted(
        [p for p in pairs if not p["identical_on_scored_pixels"] and p["jaccard_positive"] >= F.NEAR_DUP_JACCARD],
        key=lambda p: -p["jaccard_positive"],
    )

    # ---- lineage: identical bytes across repos ---------------------------------------------------
    lineage = []
    if not trees and (EVIDENCE_DIR / "submission_similarity.json").exists():
        prev = json.loads((EVIDENCE_DIR / "submission_similarity.json").read_text())
        lineage = prev.get("lineage_identical_bytes_across_repos", [])
    if not lineage and Path("/tmp/audit/16GEMSDOE/evidence/submission_similarity.json").exists():
        prev = json.loads(Path("/tmp/audit/16GEMSDOE/evidence/submission_similarity.json").read_text())
        lineage = prev.get("lineage_identical_bytes_across_repos", [])
    if trees:
        by_blob: dict[str, list] = defaultdict(list)
        for repo, tree in trees.items():
            for path, (sha, size) in tree.items():
                if path.lower().endswith((".tif", ".tiff")) and "fixture" not in path and "/bridge/" not in path \
                        and not path.endswith(("labels.tif", "sample_submission.tif", "example_submission.tif", "existing_faults.tif")) \
                        and "proxy_" not in path and "qfaults_catalogue" not in path:
                    by_blob[sha].append({"repo": repo, "path": path, "bytes": size})
        reg_blob_to_ids = defaultdict(list)
        for e in entries:
            reg_blob_to_ids[e["git_blob_sha1"]].append(e["id"])
        for sha, locs in by_blob.items():
            repos_in = sorted({l["repo"] for l in locs})
            if len(locs) > 1:
                lineage.append({
                    "git_blob_sha1": sha, "copies": len(locs), "repos": repos_in, "paths": locs,
                    "registry_entries": reg_blob_to_ids.get(sha, []),
                    "lb_scores": sorted({by_id[i]["lb_score"] for i in reg_blob_to_ids.get(sha, []) if by_id[i]["lb_score"] is not None}),
                })
        lineage.sort(key=lambda d: (-d["copies"], d["git_blob_sha1"]))

    # ---- descriptive correlation (deduplicated by identical scored content) -----------------------
    rep = {}
    for g in [sorted(v) for v in groups.values()]:
        scored_members = [i for i in g if by_id[i]["lb_score"] is not None]
        if scored_members:
            rep[g[0]] = by_id[scored_members[0]]
    xs = list(rep.values())
    corr = {}
    if len(xs) >= 6:
        y = [r["lb_score"] for r in xs]
        for key in ("positive_scored_pixels", "frac_near_le_300m", "frac_far_gt_1500m", "dist_gt_1500m", "dist_le_300m"):
            vals = [r[key] for r in xs]
            rho, p = spearmanr(vals, y)
            corr[key] = {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4)}
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scored_pixel_definition": "footprint (finite cells of the official template) minus pixel-exact known-fault mask; "
                                   "DrivenData forum 11516 posts 2 and 4",
        "n_entries": len(rows),
        "integrity_problems": integrity_problems,
        "entries": rows,
        "identical_on_scored_pixel_groups": identical_groups,
        "near_duplicate_pairs_jaccard_ge_0_80": near_dups,
        "pairs": sorted(pairs, key=lambda p: -p["jaccard_positive"]),
        "lineage_identical_bytes_across_repos": lineage,
        "score_vs_geometry_descriptive": {
            "n_unique_scored_contents": len(xs),
            "spearman": corr,
            "caveat": "Descriptive only. Small n, confounded (pixel count, method, threshold and geography all vary); "
                      "public scores are rounded, time-stamped observations, not evidence of causation.",
        },
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "submission_similarity.json").write_text(json.dumps(report, indent=2) + "\n")
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    site = {k: report[k] for k in ("generated_utc", "scored_pixel_definition", "n_entries", "integrity_problems",
                                   "identical_on_scored_pixel_groups", "score_vs_geometry_descriptive")}
    site["near_duplicate_pairs"] = near_dups[:20]
    site["lineage"] = [l for l in lineage if l["copies"] >= 2][:25]
    site["entries"] = [{k: r[k] for k in ("id", "display_name", "github_repo", "repo_path", "site_url", "label",
                                           "lb_score", "account_named_by_owner", "sha256", "positive_scored_pixels",
                                           "frac_near_le_300m", "frac_far_gt_1500m", "outside_mode",
                                           "in_footprint_nan", "is_binary_0_1")} for r in rows]
    (SITE_DATA_DIR / "forensics.json").write_text(json.dumps(site, indent=2) + "\n")
    print(f"\nidentical-on-scored groups: {identical_groups}")
    print(f"near-duplicate pairs (J>=0.8): {[(p['a'], p['b'], p['jaccard_positive']) for p in near_dups]}")
    print(f"integrity problems: {integrity_problems}")
    print(f"spearman (dedup n={len(xs)}): {corr}")
    print("wrote evidence/submission_similarity.json and docs/data/forensics.json")


if __name__ == "__main__":
    main()
