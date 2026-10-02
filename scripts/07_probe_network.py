#!/usr/bin/env python3
"""Measure, and record, exactly what this sandbox can and cannot reach.

Why this exists.  Every external figure quoted anywhere in this repository has to
come from somewhere, and a reviewer needs to know which sources were fetched
directly from the sandbox and which were reached through the agent's separate
page-fetch path.  Those two egress routes do NOT behave the same way: several
official hosts that the page-fetch path reads fine are unreachable from a socket
inside the sandbox (TLS handshake aborted).  Asserting "verified from the
official source" without recording which route was used would be misleading, so
the routes are measured and written to evidence/network_reachability.json.

Nothing here is cached or hand-typed: each row is a live attempt, timestamped,
with the exception text that caused the failure preserved verbatim.

Usage:  python3 scripts/07_probe_network.py
"""
from __future__ import annotations

import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems22.spec import REPO                                  # noqa: E402

OUT = REPO / "evidence" / "network_reachability.json"
TIMEOUT = 12

# host -> why it matters to this project
HOSTS = {
    "www.drivendata.org": "competition pages, metric spec, leaderboard, forum",
    "community.drivendata.org": "forum threads 11516 / 11527 (organiser clarifications)",
    "github.com": "repository, releases",
    "api.github.com": "repo/org enumeration, metadata",
    "codeload.github.com": "tarball download of predecessor repos + official rasters",
    "raw.githubusercontent.com": "raw file fetch from a commit",
    "buffedlizard55-lab.github.io": "predecessor sessions' published GitHub Pages sites",
    "pypi.org": "python dependencies",
    "files.pythonhosted.org": "python dependency wheels",
    "www.usgs.gov": "USGS 3DEP / general programme pages",
    "www.sciencebase.gov": "GeoDAWN radiometrics item 657e1d85d34e23d3533209f7",
    "doi.org": "DOIs for Bour & Davy 1999, Marrett et al. 2018, and the rest",
    "gdr.openei.org": "Geothermal Data Repository submission 1391 (INGENIOUS)",
    "mrdata.usgs.gov": "State Geologic Map Compilation shapefiles NV.zip / CA.zip",
    "tnmaccess.nationalmap.gov": "3DEP 1 m tile inventory and download API",
    "opentopography.org": "alternative DEM source",
    "drivendata-public-assets.s3.amazonaws.com": "the two metric worked-example diagrams",
    "www.dropbox.com": "some competition data mirrors",
    "services.arcgisonline.com": "basemap tiles (context only)",
}


def probe(host: str) -> dict:
    """One live attempt: DNS, then TLS+HTTP GET over 443."""
    row: dict = {"host": host, "purpose": HOSTS[host]}
    t0 = time.time()
    try:
        row["dns"] = sorted({ai[4][0] for ai in socket.getaddrinfo(host, 443, socket.AF_INET)})
    except Exception as exc:                              # noqa: BLE001
        row["dns"] = None
        row["dns_error"] = f"{type(exc).__name__}: {exc}"
    try:
        req = urllib.request.Request(f"https://{host}/", headers={"User-Agent": "gems22-probe/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            row["https_status"] = r.status
            row["reachable"] = True
            row["note"] = "HTTPS GET / succeeded"
    except urllib.error.HTTPError as exc:
        # an HTTP error means the TLS/HTTP round trip DID complete -- the host is
        # reachable, it just dislikes our request (403/404 on a bare / is normal)
        row["https_status"] = exc.code
        row["reachable"] = True
        row["note"] = f"host answered with HTTP {exc.code}; network path is open"
    except Exception as exc:                              # noqa: BLE001
        row["https_status"] = None
        row["reachable"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.time() - t0, 2)
    return row


def main() -> int:
    print(f"probing {len(HOSTS)} hosts (timeout {TIMEOUT}s each)...\n", flush=True)
    rows = []
    for h in HOSTS:
        r = probe(h)
        rows.append(r)
        mark = "REACHABLE" if r["reachable"] else "BLOCKED  "
        detail = (f"HTTP {r['https_status']}" if r["reachable"]
                  else r.get("error", "?")[:64])
        print(f"  {mark} {h:42s} {detail}", flush=True)

    ok = [r["host"] for r in rows if r["reachable"]]
    bad = [r["host"] for r in rows if not r["reachable"]]
    doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/07_probe_network.py",
        "timeout_seconds": TIMEOUT,
        "summary": {
            "n_hosts_probed": len(rows),
            "n_reachable": len(ok),
            "n_blocked": len(bad),
            "reachable": ok,
            "blocked": bad,
        },
        "interpretation": (
            "Two egress routes exist and they are NOT equivalent. (1) A direct "
            "socket from this sandbox -- what this script measures. (2) The agent's "
            "page-fetch/search tooling, which routes through a different network "
            "path and DID return the text of several hosts listed here as blocked, "
            "including the drivendata.org metric page, the forum threads, the live "
            "leaderboard and buffedlizard55-lab.github.io. Every quotation in this "
            "repository from a host that is blocked on route (1) was obtained on "
            "route (2) as TEXT ONLY; no binary could be transferred that way, which "
            "is why the competition rasters were instead reassembled from "
            "codeload.github.com and pinned by SHA-256. This asymmetry is recorded "
            "as irregularity flag F-13."
        ),
        "consequences": [
            "The live leaderboard score of any file produced here cannot be "
            "observed from the sandbox, so no submission slot is spent on an idea "
            "that has not beaten the holdout best.",
            "The two metric worked-example diagrams live on "
            "drivendata-public-assets.s3.amazonaws.com and could not be retrieved "
            "by either route; the worked example is therefore validated "
            "arithmetically only (flag F-02).",
            "External layers already committed by predecessor sessions are reused "
            "and re-hashed rather than re-downloaded, and their provenance JSONs "
            "carry the original DOI / ScienceBase / GDR identifiers.",
        ],
        "probes": rows,
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"\nreachable {len(ok)} / blocked {len(bad)} -> wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
