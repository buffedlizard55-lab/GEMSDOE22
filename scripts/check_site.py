"""Static checks for the generated site: local links/assets/anchors resolve, JS parses, no unresolved placeholders."""
from __future__ import annotations

import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.dup_ids = [], set(), []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            if a["id"] in self.ids:
                self.dup_ids.append(a["id"])        # duplicate ids break getElementById and anchors
            self.ids.add(a["id"])
        for k in ("href", "src"):
            if k in a and a[k]:
                self.links.append((tag, a[k]))
        if tag == "a" and a.get("name"):
            self.ids.add(a["name"])


def main() -> int:
    errors: list[str] = []
    pages = sorted(DOCS.glob("*.html")) + [ROOT / "index.html"]
    parsed = {}
    for p in pages:
        t = p.read_text()
        if re.search(r"\{\{\w+\}\}", t):
            errors.append(f"{p.name}: unresolved placeholder")
        q = P()
        q.feed(t)
        parsed[p.resolve()] = q
    for p, q in parsed.items():
        for d in q.dup_ids:
            errors.append(f"{p.name}: duplicate id={d!r}")
        for tag, href in q.links:
            if re.match(r"^(https?:|mailto:|tel:|data:|javascript:)", href):
                continue
            path, frag = urldefrag(href)
            target = (p.parent / unquote(path)).resolve() if path else p
            if not target.exists():
                errors.append(f"{p.name}: broken {tag} -> {href}")
                continue
            if frag and target.suffix == ".html" and target in parsed and frag not in parsed[target].ids:
                errors.append(f"{p.name}: missing anchor #{frag} in {target.name}")
    for js in DOCS.glob("js/*.js"):
        r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
        if r.returncode:
            errors.append(f"{js.name}: {r.stderr.strip()[:200]}")
    for e in errors:
        print("ERROR", e)
    print(f"checked {len(pages)} pages, {sum(len(q.links) for q in parsed.values())} links; errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
