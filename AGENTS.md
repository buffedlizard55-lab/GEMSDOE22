# Instructions for automated agents (and humans) — 22GEMSDOE (fractal-clustering update)

1. **Read `README.md` first** — the *Status*, *Limitations* and *Next steps* sections, then the verbatim **Original request** at the bottom (including the Bour & Davy / Ripley K fractal-clustering paragraph). Re-read the request's closing rules before finishing: line-by-line verification from official sources, links for manual review, flag irregularities, no hallucinations, multiple review passes. The clustering prior & audit (`src/gems/clustering.py`) is the *new* 2026-10-02 requirement — fit `D` and `K(r)` to the catalogued `3,199` traces before touching the model, use `D` as a geometric prior, and audit predictions with the same `K(r)`.
2. **Run `python -m pytest -q`** before and after any change (set `GEMS_DATA_DIR` to also run the `data`-marked tests).
3. **Evidence over memory.** A claim on the site must trace to `registry/sources.json` (fetched source) or `evidence/*.json` (computed). Label statements OBSERVED / COMPUTED / INFERENCE. Add irregularities to `registry/irregularities.json`.
4. **Gate before slots.** No candidate is recommended unless it beats the holdout best under a pre-registered protocol (`docs/research/preregistration_*.md`, committed *before* results) **and** passes `gems.forensics.gate_candidate` (not a duplicate) **and** `gems.submission.check_variants`. The weekly limit is 3 uploads per **rolling** 7 days per entity.
5. **Never automate drivendata.org** (Terms of Use forbid robots/spiders; `tests/test_site_integrity.py::test_no_automated_access_to_drivendata_is_implemented` enforces it). Scores are copied in by a person with `scripts/record_score.py`.
6. **Do not commit rasters** from `GEMS_DATA_DIR`; only published candidates in `docs/downloads/` and small evidence files.
7. The agent sandbox cannot reach USGS/GDR/DrivenData; verify external data in GitHub Actions (`.github/workflows/external-verification.yml`) and read the committed results in `evidence/ci/`.
8. After changing data or evidence, run `python scripts/build_site.py && python scripts/check_site.py` and commit the regenerated HTML.
9. **After a sandbox restore, verify binaries before trusting the tree.** Once, a restore left `docs/data/footprint.bin` text-decoded (U+FFFD bytes, 31,029 B instead of 15,681 B; flag F21) and lost the local git history. `python -m pytest -q tests/test_footprint.py` detects it. Repair: `gh api -H 'Accept: application/vnd.github.raw' repos/buffedlizard55-lab/GEMSDOE/contents/data/bridge/example_submission.tif > "$GEMS_DATA_DIR/sample_submission.tif"` (its SHA-256 must be `2176d08e…`), then `python scripts/build_footprint_payload.py`; the payload must hash to `9ec93feb…` and leave `docs/data/footprint.json` unchanged. Also re-hash `docs/downloads/*` against `docs/data/submissions.json`.

---

# Part B — the parallel `gems22` workstream (merged 2026-10-02)

A second, independent analysis was developed in parallel and merged here. It lives
in its own namespace so it cannot silently overwrite the trunk:

| trunk (`src/gems`) | gems22 workstream |
|---|---|
| `src/gems/` | `src/gems22/` |
| `scripts/build_site.py`, `scripts/check_site.py` | `scripts/00…07`, `scripts/05b` |
| `scripts/record_score.py` | `scripts/record_score_gems22.py` |
| `evidence/clustering_fit.json` | `evidence/gems22_clustering_fit.json` |
| `registry/submissions.json` | `registry/gems22_submissions.json` |
| `docs/index.html` and friends | `docs/gems22-*.html` (9 pages) |
| `tests/test_*.py` | `tests/test_gems22_*.py` |
| — | `LIMITATIONS.md`, `NEXT_STEPS.md`, `.github/workflows/ci.yml` |

**Both analyses are retained, and they disagree on three points.** See
`README.md` §0A for the disagreements and what would settle them. When editing
either side, do not "fix" the other side's numbers to match — the disagreement is
the informative part.

Run both suites: `python -m pytest tests -q` covers trunk *and* gems22 (98 tests
in the gems22 set alone).

## Non-negotiable invariants

These are load-bearing. Changing one silently invalidates artefacts that are
already written, and the contradiction will not be obvious.

### I-1. The fold partition is pinned by a test

`holdout.trace_cluster_folds(cat | gap, n_folds=4, n_clusters=48, seed=22)` must
reproduce exactly the per-fold pixel counts recorded in
`evidence/holdout_union.json`:

| fold | gt px (head A) | gt px (head B) |
|---|---|---|
| T0 | 21,344 | 28,445 |
| T1 | 13,147 | 17,411 |
| T2 | 11,727 | 10,254 |
| T3 | 14,639 | 23,478 |

`tests/test_gems22_holdout.py::test_real_data_folds_reproduce_recorded_evidence` asserts
this. If it fails, **every** cached OOF map, holdout sweep and budget conclusion
in the repo is stale and must be regenerated before anything is claimed.

Do not "improve" the cluster→fold assignment. A greedy-by-size variant was tried
and produced a badly imbalanced fold (T0 = 875 px of 60,834) while invalidating
all evidence for no measured benefit. The round-robin is retained deliberately,
with a repair step that guarantees no empty fold and only fires on degenerate
small inputs.

### I-2. Never let a catalogue-derived feature into the shared cache

`LEAKY_PREFIXES` in `scripts/03_build_features.py` exists because
`struct_dist_cat_log`, `struct_dist_cat_inv`, `struct_cat_dilate3` and
`struct_bour_davy_prior(_valid)` are functions of the known-fault map. A held-out
fault sits at distance 0 from itself, which produced a **fold NW DTI of 0.977**.
Rebuilding them per fold from `catalogue & train_mask` dropped it to 0.033
(flag F-10).

If you add a feature, ask: *is this computable from `labels.tif` alone?* If yes,
it must be built per fold, not cached globally.

### I-3. Do not spend a submission slot on an idea that has not beaten the holdout best

This is from the brief, not a local preference. The gate lives in
`evidence/submission_build.json` → `gate`. Currently
`gbm_heads_passed: false`, which is why the delivered file contains **no learned
detector**.

**H26 addendum (2026-10-02) — the gate is now measured *relative to the no-skill
floor*.** A random binary emission of `n` pixels scores `floor_dti(n, |G|, N)` on
this metric, which at the group's habitual ~121k budget is ≈ 0.18 and at 550k is
≈ 0.29 (`src/gems/floor.py`, validated to ±3 % against the eight measured random
arms in `evidence/holdout_union.json`; `FLOOR_ANALYSIS.md`). Therefore:
(i) candidates are ranked by **lift = DTI − floor_dti(n, |G|, N) at their own
budget**, never by raw DTI, and never across instruments; (ii) a budget increase
must be justified as a *lift* increase, because the fold's flat floor (~0.10 at
every budget) does not transfer to the live density (0.11 → 0.29); (iii) no slot
is spent on a file whose projected DTI is at or below the floor for its budget.

### I-4. Submissions must be binary and must never leave the [0, 1] range

Two derived facts, both unit-tested in `tests/test_gems22_metric.py`:

* `FN_w = |G| − TP_w` **exactly**, so `DTI = A/(0.2A + 0.2B + 0.8|G|)`.
* For a fixed support, DTI is strictly increasing in a uniform confidence scale,
  therefore **optimal submissions are binary {0, 1}**. Do not submit graded
  probabilities.

The `0.8|G|` term is immovable: predicting 1.0 everywhere reaches only
≈ `c/(c + 0.2(1 − c))` ≈ 0.10. Coverage dominates; there is no way to buy score
with confidence scaling.

### I-5. Every emitted file is written through `submission.write_submission`

It sanitises (sentinel, `±inf`, out-of-range), zeroes the catalogue, writes, then
**re-reads the file from disk** and asserts 12 hard checks. Never write a
submission GeoTIFF by hand and never bypass `sanitise()` — the 3,073 in-footprint
sentinel pixels (flag F-03) are the most plausible root cause of the rejection
`Predicted values must be in range [0, 1]`.

After copying artefacts anywhere, run `scripts/05b_reverify_published.py`. It
re-hashes and re-verifies the bytes **actually served** from `docs/downloads/` and
`submissions/`, including zip integrity. It exits non-zero on any mismatch and is
the release gate.

## Running things

```bash
pip install -r requirements.txt          # add --break-system-packages if needed
bash scripts/run_all.sh                  # 0..9, full rebuild, ~30 min
python3 -m pytest tests -q               # 98 tests, ~30 s, no GPU, no network
python3 scripts/05b_reverify_published.py # release gate on served artefacts
python3 scripts/06_build_site.py         # regenerate docs/ from evidence/*.json
```

Individual stages, all idempotent:

| script | what it does | cost |
|---|---|---|
| `00_verify_data.py` | hash + geometry check of the official rasters | seconds |
| `01_calibrate_anchors.py` | rank the 3 live-scored anchors on offline proxies | ~1 min |
| `02_fit_clustering.py` | Bour & Davy + Marrett NCC, **before** any model | ~2 min |
| `03_build_features.py` | 156-layer cache | 193 s, **3.83 GB** |
| `04_train_and_optimize.py` | trace-cluster CV + exact DTI budget sweep | **1322 s** |
| `04b_infer_G_and_rescale.py` | fit `\|G\|` from live scores, rescale the curve | ~1 min |
| `04c_head_to_head.py` | our maps vs the anchors → the gate | ~200 s |
| `05_build_submission.py` | build, audit (NCC), write | 434 s |
| `05b_reverify_published.py` | re-verify served bytes | ~2 s |
| `06_build_site.py` | generate `docs/` | <1 s |
| `07_probe_network.py` | measure egress, write `evidence/network_reachability.json` | ~5 s |
| `record_score.py --id <content_id> --score <X>` | log an observed live score | instant |

## Environment gotchas, learned the hard way

* **Memory.** `03_build_features.py` was killed with `EXIT=137` (OOM) when it
  accumulated in float64. Keep the streaming float32/float16 path. Peak resident
  set is close to the sandbox limit.
* **Network.** Two egress routes, not equivalent. Direct sockets reach only
  `github.com`, `api.github.com`, `codeload.github.com`, `pypi.org`,
  `files.pythonhosted.org`. Everything else aborts TLS — including
  `drivendata.org`, `usgs.gov`, `sciencebase.gov`, `doi.org`, `gdr.openei.org`,
  `nationalmap.gov`, `*.github.io`, `raw.githubusercontent.com`,
  `*.s3.amazonaws.com` (flag F-13). The agent's page-fetch tooling uses a
  *different* route and returns **text only** — never a binary. That is why the
  competition rasters were reassembled from `codeload.github.com` parts and pinned
  by SHA-256 instead of being downloaded.
* **Large artefacts stay out of git.** `data/*.tif` (419 MB), `data/derived/*`
  (4.4 GB), the three big `assets/external/*.tif` (91 MB) and `submissions/` are
  gitignored. `docs/downloads/` **is** committed — it is the GitHub Pages
  deliverable. Total tracked size ≈ 15 MB; keep it that way.
* **`fetch_page` on a proxy-mangled URL** returns an Alibaba OSS
  `SignatureDoesNotMatch` error. Always pass the real target URL.
* **No GPU.** Hence no U-Net, hence `NEXT_STEPS` item 3.

## Style and honesty requirements from the brief

* Verify line-by-line from official trusted sources and **provide links** for
  manual review. Every external figure on the site carries its URL.
* **No hallucinations.** If a number is not measured, the site prints `pending`
  rather than a plausible value — and a genuine `pending` should be treated as a
  bug to fix, not left in place. (Three apparent ones currently in the HTML are
  substring false positives: "de**pending**", "**spending**".)
* **Flag all irregularities.** New surprises go into `IRREGULARITIES` in
  `scripts/06_build_site.py` with an `id`, `severity`, `detail`, `disposition`
  and `source`, then the site is rebuilt. F-01…F-13 exist; the next is F-14.
* **Three passes.** Implement, then review for bugs and edge cases, then re-check
  against the original brief. Do not stop after pass 1.
* Negative results get recorded with the same prominence as positive ones. The
  0/6 head-to-head failure and the degenerate head A are in `LIMITATIONS.md` L-3,
  not in a footnote.

## What "done" looks like for the next session

A live score for content id `f6777492` recorded in `registry/gems22_submissions.json`
via `scripts/record_score_gems22.py`, and `LIMITATIONS.md` L-1 updated with the observed
value — confirming or refuting the live-rescaling method that the whole budget
decision rests on.
