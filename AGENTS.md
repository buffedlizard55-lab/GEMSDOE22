# AGENTS.md

Operating instructions for an agent (or human) continuing this repository.

## Read these first, in this order

1. **`README.md`** — it contains the *verbatim* original brief and the Arena Core
   Values (**Maximize P(Win)**; **Own the Outcome**). The brief instructs that it
   be re-read at the start of every session as the starting point. Do not work
   from a summary of it, including this file.
2. **`LIMITATIONS.md`** — what the evidence does *not* support. The most
   important entry is L-1/L-2: no live score for the delivered file has ever been
   observed, and the offline proxy is known to invert against the live leaderboard
   at the top.
3. **`NEXT_STEPS.md`** — ranked work, plus a table of ideas already evaluated and
   rejected with their evidence. Do not re-litigate those without new data.
4. **`docs/irregularities.html`** (or `IRREGULARITIES` in
   `scripts/06_build_site.py`) — flags F-01…F-13.

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

`tests/test_holdout.py::test_real_data_folds_reproduce_recorded_evidence` asserts
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

### I-4. Submissions must be binary and must never leave the [0, 1] range

Two derived facts, both unit-tested in `tests/test_metric.py`:

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

A live score for content id `74cb4afe` recorded in `registry/submissions.json`
via `scripts/record_score.py`, and `LIMITATIONS.md` L-1 updated with the observed
value — confirming or refuting the live-rescaling method that the whole budget
decision rests on.
