"""Everything the site publishes is re-verified here: hashes, format, uniqueness verdicts, links, and evidence-derived claims."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import rasterio

from gems import forensics as F
from gems.holdout import gate
from gems.paths import DOCS_DIR, EVIDENCE_DIR, ROOT, SITE_DATA_DIR
from gems.submission import check_variants

SUBS = json.loads((SITE_DATA_DIR / "submissions.json").read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("cand", SUBS["candidates"], ids=lambda c: c["key"])
def test_published_files_match_manifest_hashes_and_pass_all_hard_checks(cand, template_tif):
    for kind in ("tif", "zip", "tif_allfinite"):
        f = cand["files"][kind]
        path = DOCS_DIR / f["href"]
        assert path.exists() and path.stat().st_size == f["bytes"], f["name"]
        assert sha(path) == f["sha256"]
    tif = DOCS_DIR / cand["files"]["tif"]["href"]
    r = check_variants(tif, template_tif)
    assert r["ok_to_upload"] and r["official_format_compliant"]
    twin = check_variants(DOCS_DIR / cand["files"]["tif_allfinite"]["href"], template_tif)
    assert twin["ok_to_upload"]
    with zipfile.ZipFile(DOCS_DIR / cand["files"]["zip"]["href"]) as z:
        assert z.namelist() == [tif.name] and hashlib.sha256(z.read(tif.name)).hexdigest() == cand["files"]["tif"]["sha256"]
    assert cand["content_id"] in tif.name and cand["content_id"] in cand["note"] and "not yet live-scored" in cand["note"]
    assert len(cand["note"]) <= 200


def test_candidates_are_unique_against_history_and_against_each_other():
    ids = [c["content_id"] for c in SUBS["candidates"]]
    assert len(set(ids)) == len(ids)
    for c in SUBS["candidates"]:
        assert c["uniqueness"]["verdict"] == "DISTINCT", c["key"]
        assert all(o["jaccard_positive"] < F.NEAR_DUP_JACCARD for o in c["similar_to_other_candidates"])


def test_twin_files_have_identical_in_footprint_predictions(footprint):
    for c in SUBS["candidates"]:
        with rasterio.open(DOCS_DIR / c["files"]["tif"]["href"]) as a, rasterio.open(DOCS_DIR / c["files"]["tif_allfinite"]["href"]) as b:
            assert np.array_equal(a.read(1)[footprint], b.read(1)[footprint])


def test_candidate_names_are_unique_and_not_reused_from_history():
    reg = json.loads((ROOT / "registry" / "submissions.json").read_text())["entries"]
    names = {c["files"]["tif"]["name"] for c in SUBS["candidates"]}
    assert not names & {e["repo_path"].split("/")[-1] for e in reg}
    assert len(names) == len(SUBS["candidates"])


def test_h18_gate_is_reproducible_from_the_recorded_fold_numbers():
    rep = json.loads((EVIDENCE_DIR / "hypothesis_h18_validation.json").read_text())
    assert rep["submission_slot_spent"] is False and rep["submission_geotiff_created"] is False
    for arm, res in rep["arms"].items():
        again = gate(res, rep["baseline"])
        assert again["passed"] == rep["gate"][arm]["passed"], arm
        assert again["delta_mean_dense"] == pytest.approx(rep["gate"][arm]["delta_mean_dense"], abs=2e-5)
    assert rep["gate"]["H18_3a_complexity_prior"]["passed"] is True
    assert not any(rep["gate"][a]["passed"] for a in ("H18_1_PoE_scarp_x_geophysics", "H18_3b_oblique_prior", "H18_3c_both"))
    assert rep["baseline"]["reproduced_exactly"] is True


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_preregistration_commit_is_an_ancestor_of_the_first_results_commit():
    """The H18 arms and constants were committed before any H18 result file existed (guards against cherry-picking)."""
    pre = json.loads((EVIDENCE_DIR / "hypothesis_h18_validation.json").read_text())["preregistration"]["git_commit_before_results"]
    if _git("cat-file", "-e", f"{pre}^{{commit}}").returncode != 0:
        pytest.skip("pre-registration commit not in this clone's history (shallow clone or rewritten history)")
    assert "docs/research/preregistration_h18.md" in _git("show", "--name-only", "--format=", pre).stdout
    first = _git("log", "--diff-filter=A", "--format=%H", "--", "evidence/hypothesis_h18_validation.json").stdout.split()
    if not first:
        pytest.skip("results file not yet committed")
    assert _git("merge-base", "--is-ancestor", pre, first[-1]).returncode == 0


def test_controls_show_the_clustering_caveat_that_the_site_states():
    e = json.loads((EVIDENCE_DIR / "hypothesis_h18_exploratory.json").read_text())
    s = e["resampled_sparse_summary"]
    lift_total = s["H18_3a_complexity_prior"]["mean"] - s["H16_1_baseline"]["mean"]
    lift_density = s["E2_plain_density_prior"]["mean"] - s["H16_1_baseline"]["mean"]
    assert lift_density / lift_total > 0.7          # the page says density explains most of the lift
    assert e["draws_where_3a_beats_baseline"] == 5


def test_labels_provenance_claim_is_backed_by_the_ci_evidence():
    ci = json.loads((EVIDENCE_DIR / "ci" / "external_verification.json").read_text())
    a = ci["steps"]["A_labels_provenance_vs_GDR_qfaults"]["versions"]["gdr_qfaults_v2"]["all_touched_false"]
    assert a["exact_overlap_px"] / a["labels_pixels"] > 0.999 and a["labels_within_1px_of_raster_frac"] == 1.0


def test_tc_band_identity_claim_is_backed_by_the_profile():
    p = json.loads((EVIDENCE_DIR / "feature_profile.json").read_text())["tc_identity_check"]
    assert p["corr_band6_vs_external_geodawn_TC"] > 0.99 and abs(p["corr_band6_vs_computed_magnetic_tilt"]) < 0.05 and p["band6_negative_values"] == 0


def test_forensic_claims_hold_in_the_evidence():
    sim = json.loads((EVIDENCE_DIR / "submission_similarity.json").read_text())
    assert sim["integrity_problems"] == []
    groups = [set(g) for g in sim["identical_on_scored_pixel_groups"]]
    assert any({"GEMSDOE1", "5GEMSDOE", "8GEMSDOE"} <= g for g in groups) and {"12GEMSDOE", "12GEMSDOE-allfinite"} in groups
    ens = next(l for l in sim["lineage_identical_bytes_across_repos"] if l["git_blob_sha1"].startswith("812e61b740"))
    assert ens["copies"] == 8 and len(ens["repos"]) == 6


def test_site_builds_cleanly_and_links_resolve():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_site.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_home_page_leads_with_the_download_and_states_the_limits():
    html = (DOCS_DIR / "index.html").read_text()
    first_dl = html.index("Download submission")
    assert first_dl < html.index("Why the score 0.1563 kept repeating")
    assert "Dec 3, 2026 23:59 UTC" in html and "rolling" in html and "not live-scored" in html.lower()
    # 22GEMSDOE promotes H22 on the index; historic H19 candidates remain in the manifest/downloads but may not be hero cards
    cands_by_key = {c["key"]: c for c in SUBS["candidates"]}
    # The hero cards mirror whatever build_site.py nominates as primary/secondary. Assert that the
    # FIRST candidate in the manifest (the recommended upload) is genuinely on the home page, and
    # that its DrivenData note is too -- that is the property that actually matters, not which
    # hypothesis happens to hold the top slot this chapter.
    primary = SUBS["candidates"][0]
    assert primary["files"]["tif"]["href"] in html, f"primary {primary['key']} missing from index"
    assert primary["note"].split("|")[0].strip() in html
    assert "Download submission" in html
    assert not re.search(r"\{\{\w+\}\}", html)


def test_no_automated_access_to_drivendata_is_implemented():
    """DrivenData Terms of Use prohibit robots/spiders: no script or workflow may fetch drivendata.org."""
    offenders = []
    for p in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").rglob("*.py")) + list((ROOT / ".github").rglob("*.yml")):
        t = p.read_text()
        for m in re.finditer(r"(urlopen|requests\.(get|post|head)|curl|wget|fetch\()[^\n]*drivendata\.org", t):
            offenders.append(f"{p.relative_to(ROOT)}: {m.group(0)[:80]}")
    assert not offenders, offenders


def test_metric_rule_sensitivity_is_negligible_and_does_not_change_conclusions():
    s = json.loads((EVIDENCE_DIR / "hypothesis_h18_sensitivity.json").read_text())
    worst = max(abs(d[k]["registered_rule"] - d[k]["strict_rule"]) for d in s["draws"].values() for k in d)
    assert worst < 0.001
    assert s["draws_3a_beats_baseline_strict"] == 6 and s["draws_3a_beats_density_strict"] == 6


def test_readme_counts_match_the_registries_and_the_original_request_is_embedded():
    readme = (ROOT / "README.md").read_text()
    m = re.search(r"(\d+) sourced claims, (\d+) flags", readme)
    assert m, "README must state the audit counts"
    assert int(m.group(1)) == len(json.loads((ROOT / "registry" / "sources.json").read_text())["rows"])
    assert int(m.group(2)) == len(json.loads((ROOT / "registry" / "irregularities.json").read_text())["flags"])
    # the verbatim request (first and last sentences) must be present so every session starts from it
    assert "There should be an easy to download submission tif file as described by the prompt." in readme
    assert "Work line by line verify everything no hallucinations." in readme.rsplit("Original request", 1)[1]
    assert (ROOT / "AGENTS.md").read_text().startswith("# Instructions for automated agents")


def test_downloads_folder_contains_only_files_listed_in_the_manifest():
    listed = {pathlib.Path(f["href"]).name for c in SUBS["candidates"] for f in c["files"].values()}
    present = {p.name for p in (DOCS_DIR / "downloads").iterdir() if p.is_file()}
    assert present == listed, (present - listed, listed - present)


def test_proxy_calibration_evidence_supports_the_weak_gate_statement():
    c = json.loads((EVIDENCE_DIR / "proxy_calibration_vs_lb.json").read_text())
    r = c["correlation_with_public_score"]
    assert c["n_unique_scored_files"] == 15
    assert abs(r["known_dense"]["spearman_rho"]) < 0.3 and r["known_dense"]["spearman_p"] > 0.2      # dense proxy is uninformative here
    assert r["sgmc_gap"]["spearman_rho"] > 0 and r["sgmc_gap"]["spearman_p"] < 0.06
    keys = {x["key"] for x in c["candidates_on_the_same_scales"]}
    assert keys == {"h19-4", "h19-5", "h18-3a", "h16-1", "sgmc-gap"}
    readme = (ROOT / "README.md").read_text()
    assert f"{r['known_dense']['spearman_rho']:+.2f}" in readme and f"{r['sgmc_gap']['spearman_rho']:+.2f}" in readme


def test_signal_attribution_is_reported_as_exploratory_with_no_corrected_significance():
    a = json.loads((EVIDENCE_DIR / "lb_signal_attribution.json").read_text())
    assert a["status"] == "EXPLORATORY_HYPOTHESIS_GENERATION" and a["n_files"] == 15
    assert min(r["p_uncorrected"] for r in a["top_by_abs_rho"]) > a["bonferroni_p_threshold_0_05"]
    feats = {r["feature"]: r["spearman_rho"] for r in a["top_by_abs_rho"]}
    assert feats["lid1m_antislope"] > 0 and feats["depth_base_grad"] < 0            # the two directions the register cites
    reg = (DOCS_DIR / "research.html").read_text()
    assert "Nothing is significant after correction" in reg
