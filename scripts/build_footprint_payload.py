"""Encode the official template's finite-pixel footprint as alternating uleb128 runs (first run = outside)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gems.paths import SITE_DATA_DIR, TEMPLATE_PATH  # noqa: E402
from gems.validator import sha256_file  # noqa: E402


def uleb(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def main() -> None:
    with rasterio.open(TEMPLATE_PATH) as s:
        fp = np.isfinite(s.read(1)).ravel()
    change = np.flatnonzero(np.diff(fp.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [fp.size]])
    runs = np.diff(bounds)
    if fp[0]:
        runs = np.concatenate([[0], runs])
    payload = b"".join(uleb(int(r)) for r in runs)
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DATA_DIR / "footprint.bin").write_bytes(payload)
    meta = {
        "width": 3292,
        "height": 3730,
        "footprint_pixels": int(fp.sum()),
        "outside_pixels": int((~fp).sum()),
        "runs": int(len(runs)),
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "template_sha256": sha256_file(TEMPLATE_PATH),
        "encoding": "alternating uleb128 run lengths over row-major pixels, first run = outside the footprint",
        "source": "finite pixels of sample_submission.tif (the official template as bridged; see README for provenance caveat)",
    }
    (SITE_DATA_DIR / "footprint.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta)


if __name__ == "__main__":
    main()
