"""The content id must have exactly one definition (flag F27).

`74cb4afe` was the id of the published gems22 value-emit file computed by
hashing the raw float32 values; the same file hashes to `f6777492` when the
binary scored-pixel set is hashed, which is the definition the repository's
integrity test and the trunk's own files use.  Two definitions of one name is a
defect: a reader cannot verify a filename.  These tests pin the single canonical
definition so it cannot drift again.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.paths import DATA_DIR  # noqa: E402
from gems22.submission import content_id  # noqa: E402


def test_content_id_is_the_binary_support_hash() -> None:
    """`content_id` must equal the plain-SHA-256 reference of the >0.5 support."""
    rng = np.random.default_rng(0)
    values = rng.random((40, 60), dtype=np.float32)
    values[values < 0.5] = 0.0
    fp = np.ones_like(values, dtype=bool)
    cat = np.zeros_like(values, dtype=bool)
    cat[0, :5] = True
    m = fp & ~cat
    want = hashlib.sha256(
        np.ascontiguousarray((values[m] > 0.5).astype(np.uint8)).tobytes()
    ).hexdigest()[:8]
    assert content_id(values, fp, cat) == want


def test_content_id_ignores_value_scale_above_threshold() -> None:
    """Scaling a binary submission's values cannot change its identity."""
    values = np.zeros((20, 20), dtype=np.float32)
    values[5:15, 5] = 1.0
    fp = np.ones_like(values, dtype=bool)
    a = content_id(values, fp)
    b = content_id((values * 0.75).astype(np.float32), fp)
    assert a == b


@pytest.mark.skipif(not (DATA_DIR / "sample_submission.tif").exists(),
                    reason="competition rasters not placed (set GEMS_DATA_DIR)")
def test_published_files_satisfy_the_canonical_definition() -> None:
    """Every published .tif whose name embeds an id must satisfy the definition."""
    from gems.paths import DOWNLOADS_DIR
    with rasterio.open(DATA_DIR / "sample_submission.tif") as s:
        fp = np.isfinite(s.read(1))
    with rasterio.open(DATA_DIR / "labels.tif") as s:
        cat = (s.read(1) > 0) & fp
    bad = []
    for p in sorted(DOWNLOADS_DIR.rglob("*.tif")):
        embedded = p.stem.split("-")[-2]
        if len(embedded) != 8:
            continue
        with rasterio.open(p) as s:
            a = s.read(1)
        if content_id(a, fp, cat) != embedded:
            bad.append((p.name, embedded, content_id(a, fp, cat)))
    assert not bad, f"filename/definition mismatch: {bad}"
