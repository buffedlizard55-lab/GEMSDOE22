"""End-to-end test of scripts/scan_group_repos.py against an OFFLINE fake GitHub (no network, no `gh`)."""
from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np
import pytest
import rasterio

from gems.paths import ROOT
from gems.submission import write_submission

spec = importlib.util.spec_from_file_location("scan_group_repos", ROOT / "scripts" / "scan_group_repos.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def _pred(footprint, seed):
    rng = np.random.default_rng(seed)
    a = np.where(footprint, (rng.random(footprint.shape) < 0.02), False).astype(np.float32)
    a[~footprint] = np.nan
    return a


def test_scan_flags_duplicates_ignores_noise_and_writes_report(tmp_path, monkeypatch, footprint, template_tif):
    reg_dir = tmp_path / "root" / "registry"
    reg_dir.mkdir(parents=True)
    entries = [{"id": f"R{i}", "github_repo": "1GEMSDOE", "repo_path": f"docs/downloads/r{i}.tif", "git_blob_sha1": f"{i:040x}", "lb_score": 0.1 * i} for i in (1, 2, 3)]
    (reg_dir / "submissions.json").write_text(json.dumps({"entries": entries}))

    files: dict[tuple[str, str], bytes] = {}
    for i, e in enumerate(entries):                                             # registered files
        p = write_submission(_pred(footprint, 100 + i), template_tif, tmp_path / f"r{i}.tif")
        files[(e["github_repo"], e["repo_path"])] = p.read_bytes()
    dup = write_submission(_pred(footprint, 100), template_tif, tmp_path / "dup.tif", outside="zero")   # same scored pixels as R1 (first entry), zero-outside container
    new = write_submission(_pred(footprint, 999), template_tif, tmp_path / "new.tif")
    files[("77GEMSDOE", "docs/downloads/copy_of_r1.tif")] = dup.read_bytes()
    files[("77GEMSDOE", "docs/downloads/brand_new.tif")] = new.read_bytes()
    files[("77GEMSDOE", "docs/downloads/labels.tif")] = b"ignored"
    files[("77GEMSDOE", "data/bridge/existing_faults.tif")] = b"ignored"

    lab = np.zeros(footprint.shape, dtype=np.int8)
    lab[1500:1503, 800:1600] = 1
    lab_path = tmp_path / "lab.tif"
    with rasterio.open(template_tif) as t:
        prof = t.profile.copy()
    prof.update(dtype="int8", nodata=-1)
    with rasterio.open(lab_path, "w", **prof) as d:
        d.write(np.where(footprint, lab, -1).astype(np.int8), 1)
    files[("GEMSDOE", "data/bridge/existing_faults.tif")] = lab_path.read_bytes()

    def fake_fetch(owner, repo, path, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(files[(repo, path)])
        return files[(repo, path)]

    def fake_tree(owner, repo):
        return [{"type": "blob", "path": p, "sha": hashlib.sha1(p.encode()).hexdigest(), "size": len(b)} for (r, p), b in files.items() if r == repo]

    monkeypatch.setattr(scan, "ROOT", tmp_path / "root")
    monkeypatch.setattr(scan, "GROUP_DIR", tmp_path / "group")
    monkeypatch.setattr(scan, "SITE_DATA_DIR", tmp_path / "out")
    monkeypatch.setattr(scan, "LABELS_SHA256", hashlib.sha256(lab_path.read_bytes()).hexdigest())
    monkeypatch.setattr(scan, "list_repos", lambda owner: ["77GEMSDOE"])
    monkeypatch.setattr(scan, "list_tree", fake_tree)
    monkeypatch.setattr(scan, "fetch_file", fake_fetch)
    (tmp_path / "out").mkdir()

    assert scan.main() == 0
    out = json.loads((tmp_path / "out" / "group_scan.json").read_text())
    by_path = {u["path"]: u for u in out["unregistered"]}
    assert set(by_path) == {"docs/downloads/copy_of_r1.tif", "docs/downloads/brand_new.tif"}        # labels.tif and bridge/ ignored
    assert by_path["docs/downloads/copy_of_r1.tif"]["verdict"] == "DUPLICATE"
    assert by_path["docs/downloads/copy_of_r1.tif"]["nearest_id"] == "R1"
    assert by_path["docs/downloads/brand_new.tif"]["verdict"] == "DISTINCT"
    assert all(u["format_ok"] for u in by_path.values())
    assert out["candidates_vs_found"] == [] and out["found_total"] == 2 and out["cap_reached"] is False   # no candidate manifest in this fake world
    assert out["warnings"] == []


def test_scan_aborts_when_the_labels_file_hash_is_wrong(tmp_path, monkeypatch):
    (tmp_path / "root" / "registry").mkdir(parents=True)
    (tmp_path / "root" / "registry" / "submissions.json").write_text(json.dumps({"entries": []}))
    monkeypatch.setattr(scan, "ROOT", tmp_path / "root")
    monkeypatch.setattr(scan, "GROUP_DIR", tmp_path / "group")
    monkeypatch.setattr(scan, "fetch_file", lambda o, r, p, d: (d.parent.mkdir(parents=True, exist_ok=True), d.write_bytes(b"not the labels"), b"not the labels")[2])
    assert scan.main() == 1


class _Completed:                               # minimal stand-in for subprocess.CompletedProcess
    def __init__(self, stdout: bytes):
        self.stdout = stdout


@pytest.mark.parametrize("raw, expected", [
    (b"", []),
    (b"[]", []),
    (b'[{"a": 1}]', [{"a": 1}]),
    (b'[{"a": 1}][{"a": 2}]', [{"a": 1}, {"a": 2}]),                     # two pages back to back, as `gh api --paginate` prints them
    (b'[{"a": 1}]\n\n [{"a": 2}]\n[]', [{"a": 1}, {"a": 2}]),             # whitespace between pages, empty last page
    ('[{"n": "\u00e9"}][{"n": "\u00fc"}]'.encode("utf-8"), [{"n": "\u00e9"}, {"n": "\u00fc"}]),
    (b'{"tree": []}', {"tree": []}),                                      # a single object is returned as is
])
def test_gh_json_decodes_paginated_output(monkeypatch, raw, expected):
    """Regression: the first real run failed with json 'Extra data' because an account with >100 repos yields several pages."""
    monkeypatch.setattr(scan.subprocess, "run", lambda *a, **k: _Completed(raw))
    assert scan.gh_json("users/x/repos?per_page=100") == expected


def test_gh_json_asks_gh_to_paginate(monkeypatch):
    seen = {}
    monkeypatch.setattr(scan.subprocess, "run", lambda cmd, **k: seen.setdefault("cmd", cmd) and _Completed(b"[]"))
    scan.gh_json("users/x/repos?per_page=100")
    assert seen["cmd"][:2] == ["gh", "api"] and "--paginate" in seen["cmd"]


def _labels_tif(tmp_path, footprint, template_tif):
    lab = np.zeros(footprint.shape, dtype=np.int8)
    lab[1500:1503, 800:1600] = 1
    path = tmp_path / "lab.tif"
    with rasterio.open(template_tif) as t:
        prof = t.profile.copy()
    prof.update(dtype="int8", nodata=-1)
    with rasterio.open(path, "w", **prof) as d:
        d.write(np.where(footprint, lab, -1).astype(np.int8), 1)
    return path


def _world(tmp_path, monkeypatch, footprint, template_tif, found, candidates, repos):
    """Offline GitHub: one registered file, `found` = {(repo, path): array} of published files, `candidates` = {key: array} of ready-to-upload files."""
    root, out = tmp_path / "root", tmp_path / "out"
    (root / "registry").mkdir(parents=True)
    (root / "docs" / "downloads").mkdir(parents=True)
    out.mkdir()
    entries = [{"id": "R1", "github_repo": "1GEMSDOE", "repo_path": "docs/downloads/r1.tif", "git_blob_sha1": "0" * 40, "lb_score": 0.15}]
    (root / "registry" / "submissions.json").write_text(json.dumps({"entries": entries}))
    lab_path = _labels_tif(tmp_path, footprint, template_tif)
    files = {("1GEMSDOE", "docs/downloads/r1.tif"): write_submission(_pred(footprint, 100), template_tif, tmp_path / "r1.tif").read_bytes(),
             ("GEMSDOE", "data/bridge/existing_faults.tif"): lab_path.read_bytes()}
    for i, ((repo, path), arr) in enumerate(found.items()):
        files[(repo, path)] = write_submission(arr, template_tif, tmp_path / f"found{i}.tif").read_bytes()
    manifest = {"candidates": []}
    for key, arr in candidates.items():
        write_submission(arr, template_tif, root / "docs" / "downloads" / f"{key}.tif")
        manifest["candidates"].append({"key": key, "files": {"tif": {"name": f"{key}.tif"}}})
    (out / "submissions.json").write_text(json.dumps(manifest))

    def fake_fetch(owner, repo, path, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(files[(repo, path)])
        return files[(repo, path)]

    def fake_tree(owner, repo):
        return [{"type": "blob", "path": p, "sha": hashlib.sha1(f"{r}/{p}".encode()).hexdigest(), "size": len(b)} for (r, p), b in files.items() if r == repo]

    monkeypatch.setattr(scan, "ROOT", root)
    monkeypatch.setattr(scan, "GROUP_DIR", tmp_path / "group")
    monkeypatch.setattr(scan, "SITE_DATA_DIR", out)
    monkeypatch.setattr(scan, "LABELS_SHA256", hashlib.sha256(lab_path.read_bytes()).hexdigest())
    monkeypatch.setattr(scan, "list_repos", lambda owner: repos)
    monkeypatch.setattr(scan, "list_tree", fake_tree)
    monkeypatch.setattr(scan, "fetch_file", fake_fetch)
    return out


def test_scan_skips_the_hub_repo_and_cross_checks_our_candidates(tmp_path, monkeypatch, footprint, template_tif):
    base = _pred(footprint, 555)
    partial = base.copy()                                     # keeps ~70 % of base's positives: an overlap, not a duplicate
    pos = np.flatnonzero(base.ravel() == 1)
    partial.ravel()[np.random.default_rng(3).choice(pos, size=int(0.3 * len(pos)), replace=False)] = 0
    found = {("77GEMSDOE", "docs/downloads/published_copy.tif"): base,
             ("16GEMSDOE", "docs/downloads/our_own_file.tif"): _pred(footprint, 4242)}         # the hub's own file must not be listed as new
    cands = {"cand-dup": base, "cand-overlap": partial, "cand-distinct": _pred(footprint, 9001)}
    out = _world(tmp_path, monkeypatch, footprint, template_tif, found, cands, repos=["77GEMSDOE", "16GEMSDOE"])

    assert scan.main() == 0
    rep = json.loads((out / "group_scan.json").read_text())
    assert rep["repos_scanned"] == 1 and [u["repo"] for u in rep["unregistered"]] == ["77GEMSDOE"]
    by_key = {r["key"]: r for r in rep["candidates_vs_found"]}
    assert by_key["cand-dup"]["verdict"] == "DUPLICATE" and by_key["cand-dup"]["nearest_path"] == "docs/downloads/published_copy.tif"
    assert by_key["cand-overlap"]["verdict"] == "OVERLAP"
    assert scan.OVERLAP_JACCARD <= by_key["cand-overlap"]["jaccard_positive"] < scan.F.NEAR_DUP_JACCARD
    assert by_key["cand-distinct"]["verdict"] == "DISTINCT" and by_key["cand-distinct"]["jaccard_positive"] < 0.05
    assert rep["found_total"] == 1 and rep["cap_reached"] is False


def test_scan_reports_when_the_file_cap_truncates(tmp_path, monkeypatch, footprint, template_tif):
    found = {("77GEMSDOE", f"docs/downloads/f{i}.tif"): _pred(footprint, 70 + i) for i in range(3)}
    out = _world(tmp_path, monkeypatch, footprint, template_tif, found, {}, repos=["77GEMSDOE"])
    monkeypatch.setattr(scan, "MAX_FILES", 2)
    assert scan.main() == 0
    rep = json.loads((out / "group_scan.json").read_text())
    assert rep["found_total"] == 3 and rep["cap_reached"] is True and len(rep["unregistered"]) == 2
    assert rep["candidates_vs_found"] == []


def test_list_tree_reports_failures_and_truncation_instead_of_hiding_them(monkeypatch):
    scan.WARNINGS.clear()

    def boom(*a, **k):
        raise scan.subprocess.CalledProcessError(1, "gh")

    monkeypatch.setattr(scan.subprocess, "run", boom)
    assert scan.list_tree("o", "r1") == [] and "r1" in scan.WARNINGS[0] and "NOT scanned" in scan.WARNINGS[0]
    scan.WARNINGS.clear()
    monkeypatch.setattr(scan.subprocess, "run", lambda *a, **k: _Completed(b'{"truncated": true, "tree": [{"path": "a.tif"}]}'))
    assert scan.list_tree("o", "r2") == [{"path": "a.tif"}] and "truncated" in scan.WARNINGS[0]
    scan.WARNINGS.clear()
