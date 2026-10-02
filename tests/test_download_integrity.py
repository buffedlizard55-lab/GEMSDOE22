"""Integrity guards for everything published under ``docs/downloads/``.

These tests exist because two defects were found by manual audit on 2026-10-02 (session H23):

* **F24** — the two H22 submission GeoTIFFs carried content-IDs in their filenames
  (``16dbe573``, ``4131bb57``) that did not equal the SHA-256 of their own scored-pixel
  content (``7fd2f28b``, ``00a4a807``).  The manifest builder had been *reading* the id out of
  the filename instead of recomputing it, so the mismatch was invisible.  Files were renamed
  to the recomputed ids; this test stops it recurring.
* **F25** — ``README.md`` quoted a SHA-256 for the H22-1 file (``45e30d01…``) that is not the
  file's hash (``b6ba77cf…``), and a visibly synthetic placeholder for H22-2
  (``ee58482a48f85c2f12a4c4d3b2e3e3c3e3e3c3e3c3e3c3e3``).  Both were corrected; this test
  re-derives every hash in the README from the files on disk.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.paths import DATA_DIR, DOWNLOADS_DIR, SITE_DATA_DIR  # noqa: E402
from gems.validator import sha256_file  # noqa: E402


def _grid():
    tpl = DATA_DIR / "sample_submission.tif"
    lab = DATA_DIR / "labels.tif"
    if not (tpl.exists() and lab.exists()):
        pytest.skip("competition rasters not placed (set GEMS_DATA_DIR)")
    with rasterio.open(tpl) as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(lab) as s:
        cat = (s.read(1) > 0) & fp
    return fp, cat


def _content_id(path: Path, fp: np.ndarray, cat: np.ndarray) -> str:
    import hashlib

    with rasterio.open(path) as s:
        a = s.read(1)
    scored = fp & ~cat
    v = (np.nan_to_num(a, nan=0.0)[scored] > 0.5).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:8]


def test_filename_content_id_matches_raster() -> None:
    """Every published `.tif` must embed the SHA-256[:8] of its own scored-pixel content (flag F24)."""
    fp, cat = _grid()
    files = sorted(DOWNLOADS_DIR.glob("*.tif"))
    assert files, "no published submissions found"
    bad = []
    for p in files:
        parts = p.stem.split("-")
        embedded = parts[-2] if len(parts) >= 2 else ""
        actual = _content_id(p, fp, cat)
        if embedded != actual:
            bad.append((p.name, embedded, actual))
    assert not bad, f"filename content-id mismatch (F24): {bad}"


def test_manifest_hashes_and_names_match_disk() -> None:
    """`docs/data/submissions.json` must not drift from the files it advertises."""
    import json

    man_p = SITE_DATA_DIR / "submissions.json"
    if not man_p.exists():
        pytest.skip("site manifest not built")
    man = json.loads(man_p.read_text())
    problems = []
    for c in man["candidates"]:
        for kind in ("tif", "tif_allfinite", "zip"):
            f = c["files"][kind]
            p = DOWNLOADS_DIR / f["name"]
            if not p.exists():
                problems.append(f"{c['key']}/{kind}: missing file {f['name']}")
                continue
            if f["sha256"] != sha256_file(p):
                problems.append(f"{c['key']}/{kind}: sha256 mismatch for {f['name']}")
            if f["bytes"] != p.stat().st_size:
                problems.append(f"{c['key']}/{kind}: byte-size mismatch for {f['name']}")
    assert not problems, problems


def test_readme_sha256_values_are_real_file_hashes() -> None:
    """Every 64-hex string in README.md that sits next to a published filename must be a real hash (flag F25)."""
    readme = (ROOT / "README.md").read_text()
    real = {}
    for p in sorted(DOWNLOADS_DIR.glob("*")):
        real[p.name] = sha256_file(p)
    # collect 64-hex tokens
    tokens = set(re.findall(r"\b[0-9a-f]{64}\b", readme))
    unknown = []
    for t in tokens:
        if t in real.values():
            continue
        # otherwise it must be a documented non-file hash (e.g. the competition rasters)
        unknown.append(t)
    # The only allowed non-file hashes are the three competition rasters, the template and
    # the geodawn/lidar stacks pinned in scripts/prepare_data.py.
    allowed = {
        "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",  # training_features.tif
        "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",  # labels.tif
        "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",  # sample_submission.tif
        "d580bb8bdcdb941e32fefb8b38044bc5bf04e199bf2e83498c357e6fc465568",  # lidar_scarp_features_u8.tif
        "c22420f75999030d7cc65c9e31e50d232ea6158423bca051613a18a8b20ba682",  # geodawn_rad_u8.tif
        "a35a9c6d2a14786f4dab85481ee59769213072f5dab5b2535ea82ae4d9bb7d9b",  # geodawn_extensions_u8.tif
        "a6398d9950965dec6aae6ccecdaa6ced48645d133eab222cbdd11def9bdabfa4",  # topo_u8.tif
        "6cb051f70f941fd78028fe66a9f71e87204fcd8d9a85903df0b94993bad1ec4d",  # radiometric_u8.tif
        "055309694ed499ca3d87e76b3f5e81c292aef6e39c418d33ff5a817aa1309bea",  # 16GEMSDOE h16-1 (external repo)
    }
    leftover = [t for t in unknown if t not in allowed]
    assert not leftover, f"README quotes hash(es) that match no published file (F25): {leftover}"


def test_no_placeholder_style_hashes_anywhere() -> None:
    """Guard against obviously synthetic hex (e.g. repeating `e3e3c3e3…`) being published."""
    for name in ("README.md",):
        text = (ROOT / name).read_text()
        for tok in set(re.findall(r"\b[0-9a-f]{16,}\b", text)):
            # a run of >=6 identical 2-char repeats is a hand-made placeholder
            if re.search(r"([0-9a-f]{2})\1{5,}", tok):
                pytest.fail(f"{name} contains a synthetic-looking hex token: {tok}")
