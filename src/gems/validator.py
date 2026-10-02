"""Small shared helper. Submission validation lives in :mod:`gems.submission` (``check_variants``, ``write_submission``)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
