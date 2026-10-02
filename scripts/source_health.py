"""Check that every OFFICIAL source URL in registry/sources.json is still reachable (weekly CI job; also runnable by hand).

Deliberately skips drivendata.org hosts: DrivenData's Terms of Use forbid robots/spiders, so those links are checked by a
person.  Writes docs/data/source_health.json.  One polite request per URL, sequential, identifying user-agent.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SKIP_HOSTS = ("drivendata.org",)
UA = "16GEMSDOE-source-health/1.0 (+https://github.com/buffedlizard55-lab/16GEMSDOE; weekly link check of official data sources)"


def probe(url: str) -> dict:
    out = {"url": url, "ok": False, "status": None, "final_url": None, "content_type": None, "error": None}
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url.split("#")[0], method=method, headers={"User-Agent": UA, "Range": "bytes=0-0"} if method == "GET" else {"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out.update(ok=200 <= r.status < 400, status=r.status, final_url=r.geturl(), content_type=r.headers.get("Content-Type"), error=None)
                return out
        except urllib.error.HTTPError as e:
            out.update(status=e.code, error=f"HTTP {e.code}")
            if method == "HEAD" and e.code in (403, 405, 400, 501):
                continue
            return out
        except Exception as e:  # noqa: BLE001
            out.update(error=repr(e)[:160])
            if method == "HEAD":
                continue
            return out
    return out


def main() -> int:
    rows = json.loads((ROOT / "registry" / "sources.json").read_text())["rows"]
    seen, results, skipped = set(), [], []
    for r in rows:
        url = r["url"].split("#")[0]
        host = urlparse(url).netloc
        if any(host.endswith(h) for h in SKIP_HOSTS):
            skipped.append({"id": r["id"], "url": url, "reason": "DrivenData Terms of Use forbid automated access"})
            continue
        if url in seen:
            continue
        seen.add(url)
        res = probe(url)
        res["id"] = r["id"]
        results.append(res)
        time.sleep(0.5)
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": results, "skipped_drivendata": skipped}
    (ROOT / "docs" / "data" / "source_health.json").write_text(json.dumps(out, indent=2) + "\n")
    bad = [r for r in results if not r["ok"]]
    print(f"{len(results) - len(bad)}/{len(results)} reachable; skipped (ToU): {len(skipped)}")
    for b in bad:
        print("  UNREACHABLE", b["id"], b["url"], b["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
