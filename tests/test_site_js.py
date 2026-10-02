"""The browser checker (docs/js/gems-tiff.js) is exercised under Node against Python-written files and compared with numpy."""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
import rasterio

from gems.paths import ROOT, SITE_DATA_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")
CLI = ROOT / "docs" / "js" / "check_cli.js"
FP_BIN = SITE_DATA_DIR / "footprint.bin"


def run_cli(path):
    r = subprocess.run([NODE, str(CLI), str(path), str(FP_BIN)], capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def base(footprint):
    rng = np.random.default_rng(11)
    a = (rng.random(footprint.shape) < 0.03).astype(np.float32) * rng.random(footprint.shape).astype(np.float32)
    return np.where(footprint, a, np.float32(np.nan)).astype(np.float32)


def write(path, arr, template_tif, **kw):
    with rasterio.open(template_tif) as t:
        prof = t.profile.copy()
    prof.update(kw)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype(prof["dtype"]), 1)
    return path


@pytest.mark.parametrize("kw", [dict(compress="lzw"), dict(compress="deflate"), dict(compress="deflate", predictor=3), dict(compress="none"),
                                dict(compress="lzw", tiled=True, blockxsize=256, blockysize=256)], ids=["lzw", "deflate", "deflate-pred3", "none", "tiled-lzw"])
def test_every_supported_encoding_decodes_to_the_same_numbers(tmp_path, base, footprint, template_tif, kw):
    if kw.get("tiled"):
        kw = dict(kw, blockysize=256)
    p = write(tmp_path / "t.tif", base, template_tif, **kw)
    res = run_cli(p)
    assert res["ok"] is True, res
    st = res["stats"]
    ins = base[footprint]
    assert st["inNaN"] == 0 and st["positives"] == int((ins > 0).sum())
    assert abs(st["inMax"] - float(ins.max())) < 1e-6 and st["outNaN"] == int(np.isnan(base[~footprint]).sum())


def test_browser_checker_flags_nan_and_range_violations(tmp_path, base, footprint, template_tif):
    bad = base.copy()
    idx = np.flatnonzero(footprint.ravel())
    bad.ravel()[idx[:7]] = np.nan
    bad.ravel()[idx[10]] = 1.05
    bad.ravel()[idx[11]] = -0.2
    res = run_cli(write(tmp_path / "bad.tif", bad, template_tif, compress="lzw"))
    assert res["ok"] is False and {"footprint_nan", "range"} <= set(res["hardFailures"])
    assert res["stats"]["inNaN"] == 7


def test_zero_outside_twin_is_ok_but_advisory_flagged(tmp_path, base, footprint, template_tif):
    z = np.where(footprint, base, 0.0).astype(np.float32)
    res = run_cli(write(tmp_path / "z.tif", z, template_tif, compress="lzw", nodata=None))
    assert res["ok"] is True
    adv = {c["id"]: c for c in res["checks"]}
    assert adv["outside_nan"]["pass"] is False and adv["outside_nan"]["hard"] is False


def test_wrong_dtype_is_rejected(tmp_path, footprint, template_tif):
    p = write(tmp_path / "u8.tif", np.where(footprint, 255, 0), template_tif, compress="lzw", dtype="uint8", nodata=None)
    res = run_cli(p)
    assert res["ok"] is False and "float32" in res["hardFailures"]


def test_published_downloads_pass_the_browser_checker():
    files = sorted((ROOT / "docs" / "downloads").glob("*.tif"))
    assert files, "no published downloads found"
    for f in files:
        res = run_cli(f)
        assert res["ok"] is True, (f.name, res["hardFailures"])
