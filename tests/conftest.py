"""Shared fixtures.  Tests must pass on a fresh clone with NO large rasters: the scored footprint is reconstructed from the
committed run-length payload (docs/data/footprint.bin).  Tests that need the competition rasters are marked ``data`` and skip
when GEMS_DATA_DIR does not hold them."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.footprint import load_footprint, write_template  # noqa: E402
from gems.paths import DATA_DIR  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the competition rasters in GEMS_DATA_DIR (skipped otherwise)")


@pytest.fixture(scope="session")
def footprint():
    return load_footprint()


@pytest.fixture(scope="session")
def template_tif(tmp_path_factory):
    return write_template(tmp_path_factory.mktemp("tmpl") / "template.tif")


@pytest.fixture(scope="session")
def real_data():
    need = [DATA_DIR / n for n in ("sample_submission.tif", "labels.tif", "training_features.tif")]
    if not all(p.exists() for p in need):
        pytest.skip("competition rasters not present (set GEMS_DATA_DIR)")
    return DATA_DIR
