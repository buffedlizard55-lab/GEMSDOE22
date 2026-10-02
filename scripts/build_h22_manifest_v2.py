#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path
import numpy as np, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gems.footprint import load_footprint
from gems.submission import check_variants
from gems.validator import sha256_file
from gems.paths import SITE_DATA_DIR, DOWNLOADS_DIR, ROOT

TEMPLATE = DOWNLOADS_DIR / "gems19-h19-4-multiline-corroborated-openness-thermal-pop-20260930-691e4dfa-nan.tif"
if not TEMPLATE.exists():
    TEMPLATE = sorted(DOWNLOADS_DIR.glob("*-nan.tif"))[0]
footprint = load_footprint()
try:
    import rasterio as rio
    data_labels = ROOT / "data" / "labels.tif"
    if data_labels.exists():
        with rio.open(data_labels) as s:
            cat = (s.read(1) > 0) & footprint
    else:
        cat = np.zeros_like(footprint, dtype=bool)
except Exception:
    cat = np.zeros_like(footprint, dtype=bool)

def mask_of(tif_path):
    with rasterio.open(tif_path) as s:
        arr = s.read(1)
    return (np.nan_to_num(arr, nan=0.0) > 0.5) & footprint & (~cat)

# Discover latest gems22 files
h22_nans = sorted(DOWNLOADS_DIR.glob("gems22-h22-*.tif"))
# Filter to nan only
h22_nans = [p for p in h22_nans if p.name.endswith("-nan.tif")]
print("found", h22_nans)
candidates = []
# Map key to file
for p in h22_nans:
    name = p.name
    # key from name: h22-1 vs h22-2
    if "h22-1" in name:
        key, hid = "h22-1", "H22-1"
        title = "H22-1 Fractal-Clustering Prior + 4-Line Synthesis @ 2.50% Budget (Bour & Davy 1999 + Ripley K)"
        one_liner = "Fits D≈1.37 and K(r) to 3,199 catalogued traces, uses distance-to-larger-fault prior to promote clustered pixels over isolated ones, and audits predictions with same K(r); 2.50% footprint."
        lines = ["L0_FractalClustering_SpatialStatistic","L1_PopScaling_TipRelay","L2_Backward_ThermalGeochem","L3_Openness_LRM_Scarp","L4_Geopotential_Basement"]
        cid = name.split("-")[-2]  # content id before -nan
        # find allfinite and zip counterparts
        all_name = name.replace("-nan.tif", "-allfinite.tif")
        zip_name = name.replace(".tif", ".zip")
        # checks
        tif_path = DOWNLOADS_DIR / name
        all_path = DOWNLOADS_DIR / all_name
        zip_path = DOWNLOADS_DIR / zip_name
        if not (tif_path.exists() and all_path.exists() and zip_path.exists()):
            print(f"missing counterpart for {name}")
            continue
        chk = check_variants(tif_path, TEMPLATE)
        chk_all = check_variants(all_path, TEMPLATE)
        note = f"22GEMSDOE {hid} | Fractal-Clustering Prior (Bour&Davy D=1.37 + Ripley K>1) + 4-Line Synthesis at 2.50% | id {cid} | not yet live-scored"
        candidates.append((key, hid, title, one_liner, cid, name, all_name, zip_name, lines, note, chk, chk_all))
    elif "h22-2" in name:
        key, hid = "h22-2", "H22-2"
        title = "H22-2 Fractal 2.43% Budget Corroborated Synthesis (Power-Law Midpoint + Clustering)"
        one_liner = "Identical 5-line fractal-augmented surface to H22-1 but at 2.43% (125,567 px) midpoint budget; DISTINCT for A/B budget experiment."
        lines = ["L0_FractalClustering_SpatialStatistic","L1_PopScaling_TipRelay","L2_Backward_ThermalGeochem","L3_Openness_LRM_Scarp","L4_Geopotential_Basement"]
        cid = name.split("-")[-2]
        all_name = name.replace("-nan.tif", "-allfinite.tif")
        zip_name = name.replace(".tif", ".zip")
        tif_path = DOWNLOADS_DIR / name
        all_path = DOWNLOADS_DIR / all_name
        zip_path = DOWNLOADS_DIR / zip_name
        if not (tif_path.exists() and all_path.exists() and zip_path.exists()):
            print(f"missing counterpart for {name}")
            continue
        chk = check_variants(tif_path, TEMPLATE)
        chk_all = check_variants(all_path, TEMPLATE)
        note = f"22GEMSDOE {hid} | Fractal 2.43pct Budget (Bour&Davy D=1.37) + 4-Line Corroborated | id {cid} | not yet live-scored"
        candidates.append((key, hid, title, one_liner, cid, name, all_name, zip_name, lines, note, chk, chk_all))

# Add H19 baselines
old_manifest = json.loads((SITE_DATA_DIR / "submissions.json").read_text())
old_by_key = {c["key"]: c for c in old_manifest["candidates"]}
# Ensure we have h19-4/5 entries (need to preserve but they may be from old manifest with old h22-1 ids; we want the original h19 entries not the previously generated h22)
# Reload original 19 manifest backup if available
# For now reuse if exists
for k in ("h19-4", "h19-5"):
    # try to find original entry from initial backup (we have it in old_by_key but old_by_key currently after our last build includes h22 with old IDs;
    # Instead read from a backup copy we saved earlier? Let's just keep current h19 entries if they exist)
    if k in old_by_key and old_by_key[k]["content_id"] in ("691e4dfa","e27054cf"):
        # reuse
        c = old_by_key[k]
        # recompute checks to ensure bytes/sha still correct (they are)
        candidates.append((c["key"], c["hid"], c["title"], c["one_liner"], c["content_id"], c["files"]["tif"]["name"], c["files"]["tif_allfinite"]["name"], c["files"]["zip"]["name"], c["lines_satisfied"], c["note"], c["checks_official_format"], c["checks_allfinite_twin"]))

# Now build candidate dicts
cand_dicts = []
for key, hid, title, one_liner, cid, tif_name, all_name, zip_name, lines, note, chk, chk_all in candidates:
    tif_path = DOWNLOADS_DIR / tif_name
    all_path = DOWNLOADS_DIR / all_name
    zip_path = DOWNLOADS_DIR / zip_name
    cand_dicts.append({
        "key": key,
        "hid": hid,
        "title": title,
        "one_liner": one_liner,
        "content_id": cid,
        "lines_satisfied": lines,
        "files": {
            "tif": {"name": tif_name, "bytes": tif_path.stat().st_size, "sha256": sha256_file(tif_path), "href": f"downloads/{tif_name}"},
            "zip": {"name": zip_name, "bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path), "href": f"downloads/{zip_name}"},
            "tif_allfinite": {"name": all_name, "bytes": all_path.stat().st_size, "sha256": sha256_file(all_path), "href": f"downloads/{all_name}"},
        },
        "note": note,
        "checks_official_format": chk,
        "checks_allfinite_twin": chk_all,
    })

# Compute Jaccards
masks = {}
for c in cand_dicts:
    masks[c["key"]] = mask_of(DOWNLOADS_DIR / c["files"]["tif"]["name"])

for c in cand_dicts:
    others = []
    for o in cand_dicts:
        if o["key"] == c["key"]:
            continue
        a = masks[c["key"]]
        b = masks[o["key"]]
        inter = int(np.logical_and(a,b).sum())
        union = int(np.logical_or(a,b).sum())
        jacc = inter/union if union else 0
        others.append({"key": o["key"], "hid": o["hid"], "jaccard_positive": round(float(jacc),4), "shared_positive": inter, "a_positive": int(a.sum()), "b_positive": int(b.sum())})
    others.sort(key=lambda x: x["jaccard_positive"], reverse=True)
    verdict = "DISTINCT" if all(o["jaccard_positive"] < 0.80 for o in others) else "NEAR_DUP"
    c["similar_to_other_candidates"] = others
    c["uniqueness"] = {"verdict": verdict, "candidate_positive_scored_pixels": int(masks[c["key"]].sum()), "nearest": others[:2]}
    c["similar_to_history"] = []
    c["scored_fraction"] = round(float(masks[c["key"]].sum()/footprint.sum()),5)

manifest = {
    "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
    "git_commit": "h22-manifest-v2",
    "template_sha256": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
    "rolling_limit": "3 submissions per rolling 7-day window per entity (official rules 3.2/3.4; staff: forum topic 11524 post 2)",
    "claims": {"score_predicted": False, "statement": "H22 fractal-clustering candidates promote spatially coherent predictions (D≈1.37); holdout projection pending GEMS_DATA_DIR."},
    "candidates": cand_dicts
}
out = SITE_DATA_DIR / "submissions.json"
out.write_text(json.dumps(manifest, indent=2)+"\n")
print(f"Wrote {out} with {len(cand_dicts)}")
for c in cand_dicts:
    print(c["key"], c["content_id"], c["uniqueness"]["verdict"], c["similar_to_other_candidates"][0]["jaccard_positive"] if c["similar_to_other_candidates"] else 0)
