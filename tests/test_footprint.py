from __future__ import annotations

import hashlib

import numpy as np
import pytest
import rasterio

from gems.footprint import HEIGHT, WIDTH, decode_runs, load_footprint, meta
from gems.paths import SITE_DATA_DIR


def test_payload_hash_and_counts_match_the_recorded_metadata():
    m = meta()
    payload = (SITE_DATA_DIR / "footprint.bin").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == m["payload_sha256"]
    fp = load_footprint()
    assert fp.shape == (HEIGHT, WIDTH) and int(fp.sum()) == m["footprint_pixels"] == 5_167_373


def test_truncated_payload_is_rejected():
    with pytest.raises(ValueError):
        decode_runs((SITE_DATA_DIR / "footprint.bin").read_bytes()[:-3])


@pytest.mark.data
def test_payload_equals_the_official_template_footprint(real_data):
    with rasterio.open(real_data / "sample_submission.tif") as s:
        ref = np.isfinite(s.read(1))
    assert np.array_equal(load_footprint(), ref)
