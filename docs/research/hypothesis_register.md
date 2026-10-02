# Hypothesis register v2 — DOE GEMS (2026-09-30)

**Labels used on this page.** **OBSERVED** = read from a cited source. **COMPUTED** = calculated here from files in this repository (path given). **INFERENCE** = reasoning, not established fact. Numbers on this page are filled from the evidence JSON when the site is built, so they cannot drift from the evidence.

## 1. What the official sources say that changes the strategy (OBSERVED)

| # | Fact | Source |
|---|---|---|
| 1 | Known USGS/INGENIOUS fault pixels are **masked pixel-exactly** in both prize rounds; predictions on them can never score. | [DrivenData staff, forum 11516 #2 and #4](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516/4) |
| 2 | A prediction near a known fault but far from a new-fault pixel is **fully penalised** (no buffer for known faults). New-fault pixels *can* lie within 300 m of a known trace ("corrections"). | same, post #4 |
| 3 | "New fault" = any fault pixel not already captured by USGS/INGENIOUS, **including newly mapped geometry of an existing fault system**. | [forum 11536 #2](https://community.drivendata.org/t/where-do-you-draw-the-line/11536/2) |
| 4 | The organizers will not reveal what data or fault types the test faults come from; Phase 2 labels are built by expert review of **all** submissions. | [forum 11527 #7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7) |
| 5 | The USGS Quaternary database holds faults with evidence of coseismic surface deformation in the past 1.6 million years. | [USGS](https://www.usgs.gov/tools/interactive-us-fault-map) |
| 6 | USGS Quaternary faults can be up to ~400 m from lidar-based labels; database mapping density is heterogeneous. | [Hermant et al. 2025](https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2025/Hermant.pdf) |
| 7 | Step-overs/relay ramps host ~32 % and fault terminations ~25 % of categorised Great Basin geothermal systems. | [Faulds & Hinz 2015 (OSTI)](https://www.osti.gov/servlets/purl/1724082) |
| 8 | Fault intersections and terminations locate upflow along faults. | [Siler et al. 2019 (USGS)](https://pubs.usgs.gov/publication/70202167) |
| 9 | A ~15 km left-step between range-front faults hosts numerous ENE-striking intra-basin faults. | [USGS, Earney et al. 2024](https://www.usgs.gov/publications/geophysical-modeling-a-possible-blind-geothermal-system-near-battle-mountain-nv) |

**INFERENCE from 1–6:** the catalogue is a hazard-oriented Quaternary database, so faults that are older, shorter, secondary, buried or mis-located are plausibly what the experts add. Nothing official confirms which of these dominate the test set.

## 2. What the group's own leaderboard history says (COMPUTED)

Source: [`evidence/submission_similarity.json`](https://github.com/buffedlizard55-lab/16GEMSDOE/blob/main/evidence/submission_similarity.json) (21 registered files, all Git-blob hashes verified against the live repositories).

* **One prediction, four entries.** GEMSDOE1, 5GEMSDOE and 8GEMSDOE are identical on every scored pixel (the first two are byte-identical); GEMSDOE2 overlaps them at Jaccard {{dup_gemsdoe2_jaccard}}. That is 4 of the 18 scored entries (LB 0.1563/0.1560).
* **Why it keeps happening.** The ens12 file (blob `812e61b740…`) sits at {{ens12_copies}} paths across {{ens12_repos}} repositories: each new repo was seeded from earlier repos' evidence folders whose default `submission.tif` *is* ens12.
* **A second repeated idea.** 6GEMSDOE, 11GEMSDOE and GEMSDOE10-H16 independently tried near-catalogue top-k predictions (45–68 % within 300 m of known faults) and scored 0.0286, 0.0202 and 0.0461.
* **No geometry statistic explains the score.** Rank correlations over {{corr_n}} unique scored contents: near-catalogue fraction ρ = {{corr_near}}, far-field fraction ρ = {{corr_far}}; none significant. The hunch "far-field is better" is **not** supported by this sample.

## 3. Ranked hypotheses

**Five candidate geological hypotheses** (H18-3a, H18-3b, H18-4, H18-5, H18-6) and **one methodological fusion arm** (H18-1) are listed. Rank = directional prior on DTI gain *and* cost. "Tested" rows report measured holdout results (Section 4); nothing else is quantified.

| Rank | ID | Layers | Physical signature | Why it could find a fault missing from the catalogue | Differs from repo work | Cost / data | Status |
|---|---|---|---|---|---|---|---|
| 1 | **H18-3a** structural-discontinuity density prior | catalogue geometry (`labels.tif`) + H16-1 surface | smoothed density of fault **endpoints and junctions** (σ = 5 km) | Unmapped secondary faults (horse-tails, relay-breaching faults, damage zones) concentrate where mapped faults end or cross (Faulds & Hinz; Siler et al.). | No earlier arm uses catalogue geometry at km scale. | Low; no new data | **Tested — passed gate**, with caveat (Section 4) |
| 2 | **H18-4** geologic-map fault gap | USGS SGMC fault lines (NV, CA) | bedrock/older faults drawn on state geologic maps that are absent from the Quaternary catalogue | {{sgmc_off_px}} SGMC fault pixels lie >300 m from every catalogued fault; if experts used geologic maps these are candidates. | Uses an independent official map, not a model. **Not new to the group:** GEMSDOE3 already rasterised the same SGMC faults (Jaccard 0.63 with our file; flag F22). | Medium; data staged by CI (obtainable) | Explored; **not gate-eligible** (proxy cannot test "new") |
| 3 | **H18-5** thermal-anchor linking | GDR springs/wells, sinter/tufa, Quaternary volcanics | hot springs and vents mark active fluid conduits | {{springs_far_pct}} % of footprint springs are >3 km from any catalogued fault: candidate unmapped conduits. | No geothermal point data used before. | Medium; data obtainable (CI) | Premise measured; operator not built |
| 4 | **H18-6** cultural-lineament suppression | Census TIGER roads/rails | roads, rails and canals create scarp-like lineaments in DEM detectors | Removes false positives rather than finding faults. | Not used before. | Low–medium; obtainable ([TIGER2023 `ROADS/`, `PRISECROADS/`, `RAILS/`](https://www2.census.gov/geo/tiger/TIGER2023/)) | Proposed only |
| 5 | H18-1 product-of-experts fusion | OOF scarp arm × geophysics arm | joint presence of independent evidence | No label prior exists for new faults, so require agreement. | H16-1 blends arithmetically. | Very low | **Tested — failed** |
| 6 | H18-3b/c oblique-strike prior | catalogue strike field + ridge strike | cross-faults oblique to range fronts (USGS Argenta Rise) | Step-over intra-basin faults strike obliquely. | New orientation operator. | Low | **Tested — failed**; the mirrored parallel prior also failed (post-hoc) |

Carried over from the prior register and **not** run in this pass (ranked below the above because they overlap existing scalar features or need heavy raw-DEM work): H17-2 drainage deflection/knickzones, H17-3 strain/seismic lineament coherence, H17-4 multi-element radiometric alteration, H17-5 basement/contact concordance. H17-1 (magnetic–gravity edge agreement) was tested earlier and failed.

## 4. Validation — pre-registered, then controlled

Protocol, constants and information rule: [`preregistration_h18.md`](https://github.com/buffedlizard55-lab/16GEMSDOE/blob/main/docs/research/preregistration_h18.md) (committed before any result). Evaluator: `src/gems/holdout.py`, verified to reproduce H16-1 exactly. Baseline H16-1: dense {{h16_dense}} / sparse {{h16_sparse}}.

{{h18_table}}

**Gate rule:** higher mean dense **and** sparse DTI than H16-1; higher sparse DTI in ≥ 3/4 folds; no fold losing > 0.01.

### Controls run after the pass (exploratory, not gate-eligible)
{{explore_table}}

**Reading the controls (INFERENCE, stated plainly).** The endpoint/junction prior beats the baseline in {{draws_beat_base}}/5 re-drawn sparse proxies (mean {{rd_3a}} vs {{rd_base}}). A *plain fault-density* prior, with no endpoint or junction information, recovers most of that lift ({{rd_dens}}), so the gain is mainly **fault clustering**; the structural increment is small but consistent (3a beats density in {{draws_beat_dens}}/5). The random-20 %-of-components proxy rewards clustering by construction, and known faults are masked in the real scoring, so this is necessary evidence, **not** evidence of a leaderboard gain.

### Does any proxy predict the public score? (post-hoc calibration on the group's own files)
The group's {{calib_n}} distinct scored files have real public scores, so the evaluator itself can be evaluated: rank-correlate each file's DTI against a proxy truth (known faults masked) with its public score.

{{calib_table}}

**Reading it (INFERENCE).** The known-fault proxy that holdout gating rests on has **no detectable rank relationship** with the public score of our own files (ρ = {{calib_dense_rho}}, p = {{calib_dense_p}}); it is also contaminated for files trained on the catalogue, so this does not prove the spatial holdout is useless, only that it cannot be *validated* with these files. The SGMC-gap truth does somewhat better (ρ = {{calib_gap_rho}}, p = {{calib_gap_p}}) but is weak, and is driven by far-field-heavy files that still scored poorly. **Treat every holdout number as weak evidence and use the first live A/B to recalibrate.** On the SGMC-gap scale the new candidates sit at H16-1 {{calib_h161}} and H18-3a {{calib_h183a}}, bracketing ens12 ({{calib_ens12}}); that is descriptive, not a forecast.

### Signal attribution across the group's scored files (exploratory, hypothesis generation only)
For each distinct scored file, the mean percentile rank of every feature channel at its scored pixels was correlated with the public score ({{attr_n}} files, {{attr_m}} comparisons; [`evidence/lb_signal_attribution.json`](https://github.com/buffedlizard55-lab/16GEMSDOE/blob/main/evidence/lb_signal_attribution.json)). **Nothing is significant after correction** (Bonferroni p < {{attr_bonf}}; best uncorrected p = {{attr_bestp}}, and about {{attr_fp}} false positives are expected at p < 0.05). The only pattern worth a pre-registered test is directional: files enriched in 1 m-lidar scarp indices (`lid1m_antislope` ρ = {{attr_antislope}}, `lid1m_tect_vs_fluv` ρ = {{attr_tectfluv}}) tend to score higher, and files enriched on steep basement-depth gradients, i.e. basin margins (`depth_base_grad` ρ = {{attr_depth}}), tend to score lower. **INFERENCE, not a finding:** emphasising tectonic-scarp topography and de-emphasising already-catalogued basin-margin structure may be worth a matched-budget experiment.

### Metric-rule sensitivity (post-hoc check of an implementation assumption)
Staff wording ("it should not matter whether these known faults are included with predictions or not") implies predictions on known-fault pixels cannot score at all. The registered evaluator made them neutral for false positives but still let them earn true-positive credit. Re-scoring six sparse proxies (the registered draw plus five re-draws) with those predictions dropped changes any arm's mean sparse DTI by at most {{sens_max}}, and the conclusions are unchanged: H18-3a beats H16-1 in {{sens_beat_base}}/6 draws and the plain-density control in {{sens_beat_dens}}/6 ([`evidence/hypothesis_h18_sensitivity.json`](https://github.com/buffedlizard55-lab/16GEMSDOE/blob/main/evidence/hypothesis_h18_sensitivity.json)).

### SGMC exploratory arm (external data; bias warning)
{{sgmc_table}}

SGMC alone reaches sparse {{sgmc_gap_sparse}} with no training, but adding it to H16-1 raises the dense proxy and lowers the sparse proxy, so it fails the "both must improve" rule. The proxy's hidden faults are Quaternary catalogue faults, which geologic maps tend to include; it therefore **cannot** tell whether SGMC faults are *new* faults.

## 5. External data needed, and whether it is obtainable (checked)

| Data | Free official source | Licence | Obtainable? | Evidence |
|---|---|---|---|---|
| Geologic-map faults (H18-4) | [USGS SGMC state shapefiles](https://mrdata.usgs.gov/geology/state/) (`NV.zip`, `CA.zip`) | USGS public data | **Yes** — downloaded 69 MB + 25 MB on a GitHub runner; nominal scale 1:1,000,000 | [`evidence/ci/external_verification.json`](https://github.com/buffedlizard55-lab/16GEMSDOE/blob/main/evidence/ci/external_verification.json) |
| Springs, wells, sinter/tufa, volcanics, 2 m probes (H18-5) | [GDR 1391 INGENIOUS](https://gdr.openei.org/submissions/1391) | CC BY 4.0 | **Yes** — all downloaded on a runner | same file |
| Structural inventory of 426 geothermal systems (H18-5) | [OSTI/GDR 355](https://www.osti.gov/dataexplorer/biblio/dataset/1148722) | DOE open | Record read; file not yet downloaded | — |
| Roads/rails (H18-6) | [Census TIGER/Line 2023](https://www2.census.gov/geo/tiger/TIGER2023/) | public domain | **Listed** at the official Census directory (listing read 2026-09-30; path names corrected from an earlier typo). A file download has **not** been tested yet | [S36](audit.html) |
| Raw 10 m DEM (H17-2) | [USGS 3DEP tiles](https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n39w119/USGS_13_n39w119.tif) | public domain | **Yes** from runners (HTTP 200, `image/tiff`); **not** from the agent sandbox | [`evidence/ci/egress_probe.txt`](https://github.com/buffedlizard55-lab/16GEMSDOE/blob/main/evidence/ci/egress_probe.txt) |

All of these hosts are blocked from the agent sandbox but open from GitHub-hosted runners, which is why verification runs in CI and commits its results back.

## 6. Recommended use of the three weekly slots

1. **H18-3a** — the only candidate that passed the pre-registered gate; distinct from all 21 registered files (max Jaccard {{cand_max_jaccard}}).
2. **H16-1** — the benchmark, never live-scored. Uploading it beside H18-3a is a clean A/B on the real leaderboard for the clustering prior, and the first calibration of whether the holdout predicts the public score.
3. **H18-4 SGMC gap** — *owner decision*: it cannot pass the gate by construction, so it is a measurement, not a validated bet. It is the most different candidate from every *registered* file, but it is the same idea as GEMSDOE3's unregistered `gapfinder-v2-sgmc-gap` file (Jaccard 0.63, flag F22): upload at most one of the two.

The limit is three per **rolling** week per entity; choose the single final submission only after live scores are in.

### 6b. Decision rules for the first live scores (written before any live result exists)
These are **proposed decision rules, not evidence**; the thresholds are judgement calls recorded now so they cannot be bent after seeing scores.

| Live outcome (public score) | Interpretation | Next action |
|---|---|---|
| #1 H18-3a > #2 H16-1 by ≥ 0.005 | the clustering prior transfers to new faults | adopt it; pre-register a small grid over σ and λ on the holdout *before* using a slot |
| \|#1 − #2\| < 0.005 | the prior neither helps nor hurts on the public set | keep H16-1 as the base; spend effort on signal quality, not priors |
| #1 < #2 by ≥ 0.005 | the holdout proxy rewards clustering that the real test does not | down-weight the sparse proxy; add a catalogue-distance-profile constraint to the gate |
| #3 (SGMC gap) ≥ 0.10 | public geologic-map faults overlap the expert labels strongly | add higher-resolution map data (NBMG quadrangles), combine with detectors under the marginal bar τ |
| #3 between 0.03 and 0.10 | partial overlap | use SGMC only as a corroboration weight on detector output |
| #3 < 0.03 | geologic-map faults are not what the experts added | drop H18-4 |
| both #1 and #2 below the best historical 0.1563 | topography/geophysics OOF surfaces are not beating the CNN ensemble | fuse with ens12 only after a marginal-precision test (next row) |

**Marginal-precision test for unions.** A candidate set *S* added to a scored file with public score *s* helps only if its kernel-weighted true-positive gain per unit of false-positive mass exceeds τ = 0.2·s / (1 − 0.2·s) (τ = 0.0323 at s = 0.1563). The cheapest way to measure it is a paired upload of the base and base ∪ *S*; do that only for a set large enough to move a four-decimal score.

**Cadence.** The allowance is three uploads per *rolling* seven days per entity: upload #1 and #2 first, hold the third slot for the follow-up the table selects (or #3 if the owner decides), and never upload a file the uniqueness gate calls a duplicate.

## 7. Limits

* The holdout recovers **known** faults; the real scoring ignores them. Proxy gains are necessary, not sufficient.
* No score is predicted anywhere in this repository.
* Hypotheses that assume the test faults are geologic-map faults, thermal-conduit faults, or step-over faults are **untested assumptions**; the organizers disclose nothing about the test set.
