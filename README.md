# GEMSDOE22 — DOE GEMS Geothermal Fault Discovery Challenge

**DrivenData competition #306 · The Geologic Enhanced Mapping System (GEMS) Prize Challenge**

> ### ⬇️ SUBMISSION FILES ARE HERE — one click, already range-verified
>
> **[`docs/downloads/`](docs/downloads/)** · primary recommendation is the
> **`-allfinite`** variant (finite everywhere, so it cannot trip the
> `"Predicted values must be in range [0, 1]"` rejection).
>
> Full step-by-step upload guide, copyable unique filename and copyable
> DrivenData note: **[`docs/executive_summary.html`](docs/executive_summary.html)**
> Site index: **[`docs/index.html`](docs/index.html)**

---

## 0. Read this first — the standing project brief

This section is the verbatim brief for the project. **Re-read it at the start of
every session** before changing anything: it is the specification the work is
measured against, and every later section exists to serve it.

### 0.1 Arena core values (focal point for all work)

- **Maximize P(Win)** — "Maximize the Probability of Winning": our decision
  making framework. In every decision, we weigh tradeoffs, assess risk, and
  choose the path that maximizes the probability that Arena succeeds. We set
  aside our emotions and make tough decisions in order to maximize P(Win).
  "Maximize P(Win)" frees us from constraints and clarifies that we must put
  Arena first.
- **Own the Outcome** — We own results end to end — not just our individual slice
  of the work. When problems arise and we have the means to act, we do so without
  waiting for permission or assignment. We treat failure and success as signals
  and use them to improve. At Arena, we stay accountable to the final outcome.

### 0.2 The brief, verbatim

```text
Review the repo.

There should be an easy to download submission tif file as described by the
prompt.  Read the entire prompt.

Treat the fault population as a spatial statistic, not a pile of independent
pixels. Real fault networks show documented fractal clustering: Bour and Davy
(Geophysical Research Letters, 1999) establish a direct mathematical link between
a fault network's clustering dimension and the exponent of its length-frequency
distribution, measured through the distance from each fault to its nearest larger
neighbor, and later structural-geology studies apply a normalized correlation
count to test, at a given length scale, whether faults in a population are
clustered, randomly spaced, or regularly spaced. Fit this clustering statistic to
the known INGENIOUS/INGENIOUS USGS traces inside the GeoDAWN footprint before
touching the model, and use it two ways: as a geometric prior that favors a
candidate pixel lying along the extrapolated clustering pattern of a known larger
fault over an equally-scored but spatially isolated one, and as a post-hoc audit —
compute the same statistic on your own predicted raster, and flag any submission
whose predicted spatial arrangement diverges sharply from the population
statistics actually measured in this region as a likely detection artifact
(survey-line aliasing, acquisition-block edges) rather than genuine geology.
Nothing in a per-pixel loss function checks whether the output looks like a real
fault population; this does.

WE NEED TO STUDY, ANALYZE, AND UNDERSTAND THE HIGHEST SCORE FROM THE GEMDOE SITE
WHERE THE SUBMISSION TIF IS DOWNLOADED FROM WHICH IS THE FOLLOWING:
https://buffedlizard55-lab.github.io/19GEMSDOE/docs/index.html
h19-4-multiline-corroborated-openness-thermal-pop-20260930-691e4dfa-nan: 0.1894
h19-5-powerlaw-budget-multiline-corroborated-20260930-e27054cf-nan: 0.1922
Why and how did this get the highest score and are we able to generate a
submission that scores higher than 0.1894?  Answer the question using Phd level
experience, knowledge, and judgement.

The following is the leaderboard for the competition:
https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/
0.3049 is the highest score right now so we need to design a new strategy,
research, testing, analyzing, and generating submission system than the current
website.  It should be unique, take unique approaches to generating a submission
that can score higher than .3049.

Put this prompt into the repo readme and read it everytime we work on the project
as a starting point to make sure we are building what we are aiming for and have
a strong base to continue building and improving on making something useful for
everyday use.  It should solve the problem of having to manually check everything
ourselves and having an up to date current feed.

Before implementing, generate 3-5 candidate geological hypotheses we haven't
tried yet, each naming: the specific layer(s) involved, the physical signature
being targeted (e.g., an edge-detection or curvature transform), why it should
catch a fault missing from the USGS/INGENIOUS catalogue rather than one already
in it, and how it differs from anything already implemented in this repo.  Rank
them by expected DTI improvement and implementation cost.  Validate the top
candidate on our spatially-blocked holdout set before touching a weekly submission
slot — do not spend a submission slot on an idea that hasn't beaten the current
holdout best.  If a candidate can't be validated without new external data, name
the specific free, official source needed and check it's obtainable before
proposing the idea as viable.

Work line by line verifying from official verified trusted sources, provide links
for manual review.  There should be no manual input, work on your own to complete
tasks.  Flag any irregularities for review.  No hallucinations.  Verify no
hallucinations.  The goal of this project is to get a full list that follow our
requirements.  No hallucinations.  Verify line by line.

We need to focus on being able to generate a submission into the competition.
The site should be able to generate a TIF file that is required for submission.
It should be as easy as download to click a File to submit into the competition.
This needs to be in the executive summary or the very beginning of the site.  it
should be obvious when you visit the site.

I tried to submit the document that i downloaded from the site but it returned
this error on the submission form: "Predicted values must be in range [0, 1]"

Also we need to give it a unique name and A short comment to help you or your
team tell submissions apart later e.g. clustering with k=25

Create a executive summary subpage that explains exactly how to make a
submission into the contest.

We need to create a project that can compete and place top of the leaderboard.
We need to understand the problem, collect all the data and organize it into a
clean easily auditable table with official verified links for manual
verification.  Get familiar with the problem through the overview and problem
description https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
and the about page https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/
Download the data from the data tab.  Create and train your own model — the
reference solution https://github.com/drivendataorg/gems-prize-reference-solution
implements a simple approach.  Use your model to generate predictions that match
the submission format.  Tell me what are your limitations and what you need
access to during this project.  We will need to find free publicly available
sources and data from official and verified sources if we are to use 3rd party or
external data.

You must be able to do your own research, deep research, scientific literature
research and organize the knowledge so that we can critically think through the
problem and generate a solution through scientific and free publicly available
information.  this must be done autonomously and must be constantly reviewed and
improved upon.  Provide suggestions and improvements and implement them.

Site creation: Create a github page for this repo that has clean ui, user
friendly, simple and easy to use.  It should be organized and clean.  It should
include all relevant information in an easy to read format with official verified
links as sources for review.  Work line by line verify everything no
hallucinations.

Run this task through multiple passes.
Pass 1: Implement the task completely and verify the result.
Pass 2: Review your work for bugs, missing requirements, incorrect assumptions,
        and edge cases. Fix everything you find.
Pass 3: Re-check the entire implementation against the original request. Improve
        accuracy, reliability, completeness, and code quality. Fix any remaining
        issues.
Do not stop after the first pass.  Each pass must build on the previous one.
Before finishing, verify that the final result fully satisfies the original
request.  Work line by line verify everything no hallucinations.

Go ahead and create a pull request and then merge the pull request onto the main.
Make suggestions for what work still needs to be done and any limitations that is
in the way of a successful project.  It should be worked on in this next session
or the next session.  Work line by line verify everything no hallucinations.
```

### 0.3 Competition facts verified in this repo (with links for manual review)

| # | Fact | Value measured / quoted here | Official source |
|---|------|------------------------------|-----------------|
| 1 | Metric | distance-weighted Tversky index, α=0.2 (FP), β=0.8 (FN), triangular kernel R=300 m | [problem description §Performance metric](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) |
| 2 | Grid | 3292 × 3730 px @ 100 m, EPSG:32611, transform (100, 0, 243350, 0, −100, 4508550) | measured from `sample_submission.tif` |
| 3 | Footprint ("the bounds") | 5,167,373 finite px; 7,111,787 NaN outside | measured from `sample_submission.tif` |
| 4 | Submission dtype | single band `float32`, values in [0, 1] | [§Submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) |
| 5 | Known-fault masking | "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms." | [DrivenData staff, forum 11516 post 2](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/2) |
| 6 | Including known faults is score-neutral | "for scoring purposes it should not matter whether these known faults are included with predictions or not" | [forum 11516 post 2](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/2) |
| 7 | Final round also masks known faults | "Re-evaluation will also mask/exclude the existing USGS/INGENIOUS faults." | [forum 11516 post 2](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/2) |
| 8 | Test-set construction is not disclosed | "We're not sharing details about the data sources, fault types, or coverage behind the test faults beyond what's in the problem description." | [DrivenData staff, forum 11527 post 7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7) |
| 9 | Two prize rounds | Initial $50,000 (top 5 × $10k) on a fixed private label set; Final $250,000 ($100k/$70k/$40k/$25k/$15k) re-scored against an expert-expanded label set | [§Competition structure](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) |
| 10 | Upload limit | 3 uploads per rolling 7 days | [competition page](https://www.drivendata.org/competitions/306/competition-doe-gems/) |
| 11 | Reference solution | U-Net (resnet18 encoder) + `TverskyLoss(alpha=0.2, beta=0.8)`, 5 Monte-Carlo random 50/50 patch splits | [drivendataorg/gems-prize-reference-solution](https://github.com/drivendataorg/gems-prize-reference-solution) |
| 12 | Live leaderboard top (fetched 2026-10-01) | #1 `DARD` **0.3168** (11 subs); #2 `alexoktaba` 0.3042; #3 Batik Shirt Brothers 0.2998; #4 joeyfezster 0.2919; #5 xiaofanhu 0.2901; … #24 `smrtdoog5` **0.1922** | [leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) |
| 13 | Bour & Davy relation | `x = (a − 1)/D`, where `x` is the exponent of ⟨d⟩ = distance to the nearest **larger** neighbour, `a` the length-frequency exponent, `D` the fractal dimension | [GRL 26(13):2001–2004, doi:10.1029/1999GL900419](https://doi.org/10.1029/1999GL900419) |
| 14 | Normalized correlation count | NCC(r) > 1 clustered, ≈1 random, < 1 regularly spaced; log-log slope = (correlation dimension − 1) | [Marrett et al. 2018, J. Struct. Geol. 108:16–33, doi:10.1016/j.jsg.2017.06.012](https://doi.org/10.1016/j.jsg.2017.06.012) |
| 15 | NCC applied to faults | [Wang, Laubach, Gale & Ramos 2019, Petrol. Geosci. 25:415–428, doi:10.1144/petgeo2018-146](https://doi.org/10.1144/petgeo2018-146) |
| 16 | Scaling review | [Bonnet et al. 2001, Rev. Geophys. 39(3):347–383, doi:10.1029/1999RG000074](https://doi.org/10.1029/1999RG000074) |
| 17 | GeoDAWN survey | [USGS data release, doi:10.5066/P93LGLVQ, ScienceBase item 657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7) |
| 18 | INGENIOUS / GDR 1391 | [gdr.openei.org/submissions/1391, DOI 10.15121/1881483](https://gdr.openei.org/submissions/1391) |
| 19 | USGS state geologic map compilation (faults) | [mrdata.usgs.gov/geology/state/shp/NV.zip](https://mrdata.usgs.gov/geology/state/shp/NV.zip) · [CA.zip](https://mrdata.usgs.gov/geology/state/shp/CA.zip) |
| 20 | USGS 3DEP 1 m DEM | [nationalmap.gov / 3DEP](https://www.usgs.gov/3d-elevation-program) |

---

## 1. Data — where it came from and how it is verified

**No manual input is required.** `bash scripts/fetch_data.sh` reconstructs
`data/` entirely from public sources; every byte is then checked against a
SHA-256 pin.

| File | Bytes | SHA-256 (first 16) | How this repo obtained it |
|------|-------|--------------------|---------------------------|
| `data/training_features.tif` | 418,912,844 | `4371c82e3b8339b8` | Reassembled from the 5 GitHub-transported parts committed in [`buffedlizard55-lab/6GEMSDOE`](https://github.com/buffedlizard55-lab/6GEMSDOE/tree/main/data/bridge) and hash-verified against that repo's `manifest.json`, which pins the official DrivenData mirrors. **Byte-exact.** |
| `data/labels.tif` | 425,830 | `7ba308ccdc4418b3` | Same manifest; identical bytes to the official `existing_faults.tif` mirror. |
| `data/sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd` | Same manifest; identical bytes to the official `example_submission.tif` mirror. |

**Network egress is measured, not assumed.** `scripts/07_probe_network.py` probes
19 official hosts live: **5 answer** a direct HTTPS request (`github.com`,
`api.github.com`, `codeload.github.com`, `pypi.org`, `files.pythonhosted.org`) and
**14 abort the TLS handshake**, including `www.drivendata.org`,
`community.drivendata.org`, `www.usgs.gov`, `www.sciencebase.gov`, `doi.org`,
`gdr.openei.org`, `mrdata.usgs.gov`, `tnmaccess.nationalmap.gov` and
`buffedlizard55-lab.github.io`. The Arena `fetch_page` tool uses a *different*
route and did return the text of several blocked hosts, which is how every live
web page quoted in this README was read — **text only, never a binary**. That is
why the competition rasters were reassembled from `codeload.github.com` parts and
pinned by SHA-256 rather than downloaded. Recorded as flag **F-13**; raw probes in
[`evidence/network_reachability.json`](evidence/network_reachability.json).

External layers under `assets/external/` are all USGS / DOE public-domain
products; their provenance, DOIs and SHA-256 pins are in
[`evidence/data_provenance.json`](evidence/data_provenance.json).

---

## 2. The metric, exactly — and what it implies

Writing `A = TP_w`, `B = FP_w`, `G = ` number of ground-truth pixels:

```
FN_w = G − A                       (exact identity: the same max term appears in both)
DTI  = A / (0.2·A + 0.2·B + 0.8·G)
```

Three consequences drive every design decision in this repo. All three are
unit-tested in [`tests/test_metric.py`](tests/test_metric.py).

1. **The denominator has an immovable floor of `0.8·G`.** Coverage, not
   precision, dominates the achievable score. Predicting `p=1` on every scored
   pixel still only reaches `≈ c/(c + 0.2(1−c))` where `c` is the ground-truth
   fraction — about 0.10 here, which is why the group's weakest submissions sit
   at 0.01–0.05 rather than at zero.

2. **For a fixed support, DTI is strictly increasing in a uniform confidence
   scale.** `DTI(q) = qA₁/(0.2q(A₁+B₁) + 0.8G)` has `d/dq > 0`. Therefore an
   optimal submission is **binary**: emit 1.0 on the chosen support and 0.0
   elsewhere. Soft probabilities are never better than saturating them.

3. **The break-even posterior is very low.** Adding a pixel-set with marginal
   gains `(dA, dB)` improves DTI iff `dA > τ·(dA + dB)` with
   `τ = 0.2·DTI/(1 − 0.2·DTI)`. For an isolated pixel with posterior `π`,
   `dA = π`, `dB = 1 − π`, so emit iff `π > τ/(1+τ)`.
   At our live anchor DTI = 0.1894, `τ = 0.0403` and **π\* = 3.9 %**.
   Any pixel with better than a ~4 % chance of sitting on an unmapped fault is
   worth emitting. Every prior GEMSDOE session instead hard-coded a "top 2.4–2.5 %
   of the footprint" budget, which is a *quantity* rule, not a *value* rule.

`src/gems22/metric.py` is validated to 7×10⁻¹⁵ against an independent
triple-loop transcription of the published formula over 60 randomised grids, and
reproduces the published worked example's arithmetic (3.00 / 1.89 / 2.00 → 0.60).

---

## 3. Fault population as a spatial statistic

Implemented in [`src/gems22/clustering.py`](src/gems22/clustering.py); fitted by
[`scripts/02_fit_clustering.py`](scripts/02_fit_clustering.py) **before** any
model was trained, on the known INGENIOUS/USGS traces inside the GeoDAWN
footprint. Results: [`evidence/clustering_fit.json`](evidence/clustering_fit.json).

| Population | n traces | median L | a (length-frequency) | x (nearest-larger) | D (correlation dim.) | x / ((a−1)/D) | NCC verdict 2–40 px |
|---|---|---|---|---|---|---|---|
| A · `labels.tif` raster catalogue | 3,199 | 1,200 m | 1.83–2.16 (R² 0.97–0.99) | 0.822 (R² 0.956) | 2.29 ⚠ | 1.62 | **clustered**, mean NCC 5.72, range 1.83–22.5, slope −0.749 |
| B · GDR 1391 INGENIOUS *vector* traces inside the footprint | 376 | 9,990 m | 0.91–1.20 (R² 0.87–0.93) | 0.437 (R² 0.844) | 1.471 (R² 0.982) | 3.21 | — |

⚠ `D = 2.29 > 2` for population A is **not** a fractal dimension: the
correlation integral saturates once every box is occupied, so a fit over
1–140 px is biased upward. Population B's `D = 1.47` is the usable estimate.
Both are reported, and the saturation is flagged as irregularity **F-07**.

Used two ways, exactly as the brief requires:

- **Geometric prior.** `bour_davy_prior_field()` inverts ⟨d(l)⟩ = A·l^x into an
  *expected unmapped length* `l*(r) = (r/A)^(1/x)` at distance `r` from each of
  the 200 longest known traces, and weights a candidate pixel by how strongly
  `l*(r)` lands in the length range where the catalogue is demonstrably
  incomplete. It uses **only trace geometry** — no feature band — so it is an
  independent re-ranking term that favours a candidate lying along the
  extrapolated clustering pattern of a known larger fault over an equally-scored
  but spatially isolated one.
- **Post-hoc audit.** `run_audit()` in `scripts/05_build_submission.py` computes
  the same NCC on our own predicted raster and compares it to the catalogue's.
  A predicted population whose mean NCC is < 0.5× or > 2× the measured regional
  value, or whose clustered/random/regular verdict disagrees with the
  catalogue's, is **flagged as a likely detection artefact** (survey-line
  aliasing, acquisition-block edges) rather than genuine geology. No per-pixel
  loss checks this; the NCC does.

---

## 4. How to run everything

```bash
bash scripts/fetch_data.sh            # reconstruct data/ from public sources + verify hashes
python3 scripts/00_verify_data.py     # re-assert every published constant
python3 scripts/01_calibrate_anchors.py   # calibrate proxies vs the 3 live-scored anchors
python3 scripts/02_fit_clustering.py      # Bour & Davy + Marrett NCC fits (pre-model)
python3 scripts/03_build_features.py      # 156-layer feature cache (3.83 GB, ~195 s)
python3 scripts/04_train_and_optimize.py --grid union_traces   # blocked CV + budget sweep
python3 scripts/04b_infer_G_and_rescale.py # fit |G| from live scores, rescale the curve
python3 scripts/04c_head_to_head.py       # the gate: our maps vs the live-scored anchors
python3 scripts/05_build_submission.py    # prior re-rank, emit, NCC audit, write
python3 scripts/05b_reverify_published.py # RELEASE GATE: re-verify the bytes actually served
python3 scripts/06_build_site.py          # regenerate docs/ from evidence/*.json
python3 scripts/07_probe_network.py       # measure egress -> evidence/network_reachability.json
python3 scripts/record_score.py --id <content_id> --score <X>   # log an observed live score
python3 -m pytest tests -q                # 98 invariants (~30 s, no GPU, no network)
bash scripts/run_all.sh                   # the whole chain, steps 0..9
```

Requirements: `numpy scipy rasterio scikit-learn scikit-image shapely pyproj pytest`.
Sizing verified against the actual sandbox: **2 CPU, 3.8 GB RAM, no GPU**. The
feature cache is float16 and streamed to disk one layer at a time so peak RAM
stays near one full-grid layer.

---

## 5. Limitations and what is needed next

See [`LIMITATIONS.md`](LIMITATIONS.md) and [`NEXT_STEPS.md`](NEXT_STEPS.md).
Headline blockers, in priority order:

1. **No DrivenData login in this sandbox**, so the private test set and the
   live score of any new file cannot be observed here. Three uploads per rolling
   7 days are the only ground-truth feedback channel, and the brief forbids
   spending one on an idea that has not beaten the holdout best.
2. **No GPU.** The official reference solution is a U-Net ensemble; a
   competitive deep model cannot be trained in this sandbox. Everything here is
   a gradient-boosted multi-scale filter bank, which is a deliberate CPU-only
   choice, not an oversight.
3. **The offline proxy is only weakly valid.** Across the 15 group files with
   real scores, the SGMC-gap proxy correlates with the public score at
   Spearman ρ = +0.52 (p = 0.048) — but among the three best anchors it
   **inverts** (ρ = −1.000, p = 0.333, n = 3). Irregularity **F-01**. All
   gating here therefore uses the trace-cluster holdout as primary and treats
   SGMC-gap as secondary.
4. **1 m LiDAR coverage is partial** — 75.4 % of the grid has 3DEP 1 m tiles,
   and the openness/LRM fields cached here cover only 71,974 cells (1.4 %) over
   eight 10 km tiles. Extending them needs network access to USGS 3DEP.

---

## 6. Repository map

```
src/gems22/    metric.py      exact DTI + the three identities + break-even rule
               spec.py        grid/footprint/masking constants, SHA-256 pins
               clustering.py  Bour & Davy 1999 + Marrett 2018 NCC
               features.py    streamed 156-layer multi-scale feature bank
               holdout.py     geographic-block AND trace-cluster folds
               submission.py  GeoTIFF writer + 12-check pre-flight verifier
scripts/       00…07          verify → calibrate → cluster → features → train → |G| →
                              gate → submit → re-verify → site → probe network
               05b            re-verify the SERVED bytes (SHA-256 + 12 checks + zip)
               fetch_data.sh  reconstruct data/ from reachable public hosts
               run_all.sh     the whole chain, idempotent
               record_score.py  log an observed live score into registry/
tests/         test_metric.py test_holdout.py test_submission.py test_clustering.py
                              98 tests; the holdout one pins the exact fold partition
evidence/      *.json         every measurement, with the source that produced it
registry/      group_geometry.json  the |G| fit over 19 live-scored files
               submissions.json     every file the group has scored, with its geometry
docs/          index.html executive_summary.html downloads/ …   (GitHub Pages)
               assets/tifcheck.js   dependency-free in-browser GeoTIFF pre-flight checker
assets/external/  USGS/DOE public-domain layers + provenance JSONs
assets/lb_anchors/ the three GeoTIFFs whose live public scores we know
submissions/   local staging copy of docs/downloads/ (gitignored)
.github/workflows/ci.yml   tests + artefact integrity + site freshness
```

Top-level documents: [`README.md`](README.md) (the brief, verbatim, plus verified
facts), [`AGENTS.md`](AGENTS.md) (invariants that must not change silently),
[`LIMITATIONS.md`](LIMITATIONS.md) (what the evidence does not support),
[`NEXT_STEPS.md`](NEXT_STEPS.md) (ranked work and rejected ideas).
