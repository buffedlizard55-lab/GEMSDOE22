"""Writing, zipping, checking and naming submission files."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest
import rasterio

from gems.footprint import HEIGHT, WIDTH
from gems.submission import check_variants, make_filename, make_note, sanitize, scored_content_id, write_submission, zip_single


@pytest.fixture()
def pred(footprint):
    rng = np.random.default_rng(3)
    a = (rng.random(footprint.shape) < 0.02).astype(np.float32)
    a[~footprint] = np.nan
    return a


def test_sanitize_removes_nan_inf_and_out_of_range():
    a = np.array([np.nan, np.inf, -np.inf, -0.3, 1.7, 0.4], dtype=np.float32)
    s = sanitize(a)
    assert s.dtype == np.float32 and list(s) == [0.0, 1.0, 0.0, 0.0, 1.0, np.float32(0.4)]


def test_official_format_file_passes_every_hard_check_and_mirrors_the_template(template_tif, pred, tmp_path):
    p = write_submission(pred, template_tif, tmp_path / "a.tif", outside="nan")
    r = check_variants(p, template_tif)
    assert r["ok_to_upload"] and r["official_format_compliant"] and not r["hard_failures"]
    assert r["checks"]["profile_matches_official_sample"]["pass"]          # NaN-aware nodata comparison
    with rasterio.open(p) as s, rasterio.open(template_tif) as t:
        assert (s.width, s.height) == (WIDTH, HEIGHT) and s.dtypes == ("float32",) and s.count == 1
        assert s.profile["compress"].lower() == t.profile["compress"].lower() and np.isnan(s.nodata)
        a, b = s.read(1), t.read(1)
        assert np.array_equal(np.isnan(a), np.isnan(b))                     # identical NaN footprint


def test_all_finite_twin_carries_the_same_in_footprint_prediction(template_tif, pred, footprint, tmp_path):
    a = write_submission(pred, template_tif, tmp_path / "n.tif", outside="nan")
    b = write_submission(pred, template_tif, tmp_path / "z.tif", outside="zero")
    with rasterio.open(a) as s1, rasterio.open(b) as s2:
        x, y = s1.read(1), s2.read(1)
    assert np.array_equal(x[footprint], y[footprint]) and np.isfinite(y).all()
    assert check_variants(b, template_tif)["ok_to_upload"] and not check_variants(b, template_tif)["official_format_compliant"]


def test_nan_inside_the_footprint_is_sanitised_by_the_writer_and_caught_by_the_checker(template_tif, footprint, tmp_path):
    bad = np.where(footprint, np.float32(0.2), np.float32(np.nan)).astype(np.float32)
    idx = np.flatnonzero(footprint.ravel())
    bad.ravel()[idx[:5]] = np.nan
    bad.ravel()[idx[5]] = 1.05
    fixed = write_submission(bad, template_tif, tmp_path / "fixed.tif")
    assert check_variants(fixed, template_tif)["ok_to_upload"]
    # write a deliberately broken file bypassing the writer
    with rasterio.open(template_tif) as t:
        prof = t.profile.copy()
    broken = tmp_path / "broken.tif"
    with rasterio.open(broken, "w", **prof) as d:
        d.write(bad, 1)
    r = check_variants(broken, template_tif)
    assert not r["ok_to_upload"] and {"footprint_all_finite", "footprint_range_0_1"} <= set(r["hard_failures"])


def test_wrong_dtype_is_a_hard_failure(template_tif, footprint, tmp_path):
    with rasterio.open(template_tif) as t:
        prof = t.profile.copy()
    prof.update(dtype="uint8", nodata=None)
    p = tmp_path / "u8.tif"
    with rasterio.open(p, "w", **prof) as d:
        d.write(np.where(footprint, 255, 0).astype("uint8"), 1)
    r = check_variants(p, template_tif)
    assert "dtype_float32" in r["hard_failures"] and "footprint_range_0_1" in r["hard_failures"]


def test_zip_contains_exactly_one_geotiff(template_tif, pred, tmp_path):
    p = write_submission(pred, template_tif, tmp_path / "x.tif")
    z = zip_single(p)
    with zipfile.ZipFile(z) as zf:
        assert zf.namelist() == ["x.tif"]


def test_content_id_depends_on_scored_pixels_not_container_or_known_fault_pixels(footprint):
    rng = np.random.default_rng(1)
    cat = np.zeros_like(footprint); cat[1000:1003, 1000:1500] = footprint[1000:1003, 1000:1500]
    base = np.where(footprint, (rng.random(footprint.shape) < 0.02), False).astype(np.float32)
    with_cat = base.copy(); with_cat[cat] = 1.0                      # only known-fault pixels differ
    changed = base.copy()
    flip = np.flatnonzero((footprint & ~cat).ravel())[:50]
    changed.ravel()[flip] = 1.0 - changed.ravel()[flip]              # flip 50 scored pixels
    assert scored_content_id(base, footprint, cat) == scored_content_id(with_cat, footprint, cat)
    assert scored_content_id(base, footprint, cat) != scored_content_id(changed, footprint, cat)


def test_filename_and_note_conventions():
    n = make_filename("gems16", "H18-3a topo+geophys", "20260930", "c502dfab", "nan")
    assert n == "gems16-h18-3a-topo-geophys-20260930-c502dfab-nan.tif"
    note = make_note("H18-3a", "x" * 300, "c502dfab")
    assert len(note) <= 200
    assert make_note("H18-3a", "thin ridges", "c502dfab").endswith("not yet live-scored")


def test_wrong_shape_is_reported_not_raised(template_tif, tmp_path):
    with rasterio.open(template_tif) as t:
        prof = t.profile.copy()
    prof.update(width=100, height=80, transform=rasterio.transform.from_origin(0, 0, 100, 100), blockysize=1)
    p = tmp_path / "small.tif"
    with rasterio.open(p, "w", **prof) as d:
        d.write(np.zeros((80, 100), np.float32), 1)
    r = check_variants(p, template_tif)                     # must not raise on the mismatched boolean index
    assert r["ok_to_upload"] is False
    assert {"shape_matches_template", "footprint_all_finite"} <= set(r["hard_failures"])
