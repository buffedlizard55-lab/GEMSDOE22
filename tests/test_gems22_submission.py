"""The [0,1] guarantee. This is the check that prevents the DrivenData rejection
"Predicted values must be in range [0, 1]".
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22 import submission as sub
from gems22.spec import REPO

H, W = 60, 50


@pytest.fixture(scope="module")
def masks():
    rng = np.random.default_rng(0)
    fp = np.zeros((H, W), bool); fp[5:55, 5:45] = True
    cat = np.zeros((H, W), bool); cat[20, 10:30] = True
    return fp, cat, rng


def _write(tmp_path, values, fp, cat, variant):
    p = tmp_path / f"s_{variant}.tif"
    # strict_shape=False: these tests exercise the sanitisation logic on a small
    # synthetic grid. Every real submission uses strict_shape=True (the default),
    # which refuses anything that is not the official 3730 x 3292 grid.
    return sub.write_submission(values, p, fp, variant, cat, strict_shape=False), p


@pytest.mark.parametrize("variant", ["allfinite", "nan"])
def test_sentinel_pixels_never_reach_the_output(tmp_path, masks, variant):
    """Flag F-03: 3,073 real footprint pixels carry -3.4028235e+38 in the source."""
    fp, cat, rng = masks
    v = rng.random((H, W)).astype(np.float32)
    v[7, 7] = np.float32(-3.4028234663852886e38)      # the actual sentinel
    v[8, 8] = np.nan
    v[9, 9] = np.inf
    v[10, 10] = -np.inf
    v[11, 11] = 5.0
    v[12, 12] = -5.0
    w, p = _write(tmp_path, v, fp, cat, variant)
    assert w.checks["hard_all_pass"]
    a = rasterio.open(p).read(1)
    ins = a[fp]
    assert np.all(ins >= 0.0) and np.all(ins <= 1.0)
    assert not np.isfinite(a).any() or np.nanmin(a) >= 0.0
    assert np.nanmax(a) <= 1.0
    if variant == "allfinite":
        assert np.isfinite(a).all()
    else:
        assert np.isnan(a[~fp]).all()


@pytest.mark.parametrize("variant", ["allfinite", "nan"])
def test_catalogue_pixels_are_zeroed(tmp_path, masks, variant):
    """Forum 11516 post 2: known-fault pixels are excluded from evaluation, so
    emitting there buys nothing and spends budget."""
    fp, cat, rng = masks
    v = np.ones((H, W), np.float32)
    w, p = _write(tmp_path, v, fp, cat, variant)
    a = rasterio.open(p).read(1)
    assert (a[cat] == 0.0).all()
    assert w.checks["positive_on_catalogue"] == 0


def test_profile_matches_the_official_template(tmp_path, masks):
    fp, cat, _ = masks
    w, p = _write(tmp_path, np.ones((H, W), np.float32), fp, cat, "allfinite")
    with rasterio.open(p) as ds:
        assert ds.count == 1 and ds.dtypes[0] == "float32"
        assert str(ds.crs).upper().endswith("32611")
        assert tuple(ds.transform)[:6] == sub.TRANSFORM


def test_shape_mismatch_raises(masks):
    fp, cat, _ = masks
    with pytest.raises(ValueError):
        sub.sanitise(np.ones((H + 1, W), np.float32), fp)


def test_content_id_ignores_the_outside_convention(masks):
    fp, cat, rng = masks
    v = (rng.random((H, W)) > 0.9).astype(np.float32)
    assert sub.content_id(v, fp, cat) == sub.content_id(v, fp, cat)
    w = v.copy(); w[~fp] = 7.0        # junk outside the footprint
    assert sub.content_id(w, fp, cat) == sub.content_id(v, fp, cat)


def test_zip_contains_a_single_geotiff(tmp_path, masks):
    fp, cat, _ = masks
    w, p = _write(tmp_path, np.ones((H, W), np.float32), fp, cat, "allfinite")
    import zipfile
    z = sub.zip_submission(p)
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert len(names) == 1 and names[0].endswith(".tif")


def test_published_files_pass_every_hard_check():
    """Guard on the actual artefacts served from docs/downloads/gems22/.

    Two modes, because the official 419 MB rasters are gitignored and therefore
    absent on a fresh clone and on every CI runner:

    * WITH `data/sample_submission.tif` + `data/labels.tif`: run the full
      `verify_file()` against the official footprint and catalogue (12 hard
      checks, including "no positive pixel lands on a known fault").
    * WITHOUT them: fall back to a template-free check that still proves a lot --
      profile, dtype, CRS, shape, geotransform, the [0, 1] range, and the SHA-256
      recorded in evidence/submission_build.json.  Skipping entirely here would
      leave the published artefacts completely unguarded on CI, which is where a
      bad commit is most likely to slip through.
    """
    import hashlib
    import json

    d = REPO / "docs/downloads/gems22"
    tifs = sorted(d.glob("*.tif"))
    if not tifs:
        pytest.skip("no submission built yet")

    sample = REPO / "data/sample_submission.tif"
    labels = REPO / "data/labels.tif"

    if sample.exists() and labels.exists():
        fp = np.isfinite(rasterio.open(sample).read(1))
        cat = rasterio.open(labels).read(1) == 1
        for t in tifs:
            c = sub.verify_file(t, fp, cat)
            assert c["hard_all_pass"], (
                f"{t.name}: failed {[k for k in c['hard_keys'] if not c['checks'][k]]}")
            assert c["checks"]["footprint_min_ge_0"] and c["checks"]["footprint_max_le_1"]
            assert c["positive_on_catalogue"] == 0, f"{t.name}: emits onto known faults"
        return

    # ---- template-free fallback -------------------------------------------
    ev_path = REPO / "evidence/submission_build.json"
    recorded = {}
    if ev_path.exists():
        for r in json.loads(ev_path.read_text()).get("files") or []:
            recorded[r["file"]] = r
    for t in tifs:
        with rasterio.open(t) as ds:
            a = ds.read(1)
            assert ds.count == 1, f"{t.name}: {ds.count} bands, expected 1"
            assert ds.dtypes[0] == "float32", f"{t.name}: dtype {ds.dtypes[0]}"
            assert str(ds.crs).upper().endswith("32611"), f"{t.name}: CRS {ds.crs}"
            assert (ds.height, ds.width) == (sub.HEIGHT, sub.WIDTH), (
                f"{t.name}: shape {(ds.height, ds.width)}")
            assert tuple(ds.transform)[:6] == sub.TRANSFORM, f"{t.name}: geotransform"
        finite = np.isfinite(a)
        assert a[finite].min() >= 0.0, f"{t.name}: value below 0"
        assert a[finite].max() <= 1.0, f"{t.name}: value above 1 -> would trip the range error"
        if "allfinite" in t.name:
            assert finite.all(), f"{t.name}: allfinite variant contains {int((~finite).sum())} NaN"
            assert set(np.unique(a).tolist()) <= {0.0, 1.0}, f"{t.name}: not binary"
        rec = recorded.get(t.name)
        if rec:
            h = hashlib.sha256(t.read_bytes()).hexdigest()
            assert h == rec["sha256"], f"{t.name}: sha256 {h[:16]} != recorded {rec['sha256'][:16]}"
            assert t.stat().st_size == rec["bytes"], f"{t.name}: size changed"
