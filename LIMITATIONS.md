# Limitations

What this repository does **not** establish. Every item below is a real bound on
what can be claimed from the evidence in `evidence/*.json`, ordered by how much
it should change a reader's confidence. Nothing here is hedging for its own sake:
each entry states what would resolve it.

The headline: **no number in this repository is a measured live score for the
file it produces.** The submission's live effect is a *directional expectation*
derived from an offline holdout, and the offline holdout is known to invert
against the live leaderboard at the top (flag F-01). Read everything else in
that light.

---

## L-1. The live score of the delivered file is unobserved, and cannot be observed from here

`evidence/network_reachability.json` measures 14 of 19 official hosts as
TLS-unreachable from this sandbox, including `www.drivendata.org`. There is no
login, no upload path and no score feedback. Three consequences:

* The expected live range quoted on the site (≈0.198–0.208) is a **live-rescaled
  holdout projection**, not a measurement. It could be wrong in either direction.
* The `|G| ≈ 125,000` inference, the τ threshold and the π\* emission rule all
  rest on the group's *historical* 19 live scores, which were observed by
  predecessor sessions, not by this one.
* The Phase-2 re-scoring set is expert-expanded from Phase-1 submissions
  ([forum 11527 post 7](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527/7)).
  A file optimised against the Phase-1 geometry can move in the opposite
  direction in Phase 2. Nothing here can test that.

**Resolved by:** uploading the file and recording the score with
`scripts/record_score_gems22.py`. One slot, three per rolling 7 days.

## L-2. The offline proxy cannot *resolve* the top three files (flag F-01) — corrected after the merge

`evidence/anchor_calibration.json`: across all four SGMC-gap proxy variants the
three authenticated anchors rank **h16-1 > h19-4 > h19-5**, i.e. Spearman
ρ = **−1.000** against the live ordering **h19-5 (0.1922) > h19-4 (0.1894) >
h16-1 (0.1855)**. n = 3, p = 0.333.

**This was originally written up as "the proxy inverts", and that framing was
wrong.** The merged trunk carries stronger evidence:
`evidence/proxy_calibration_vs_lb.json` reports the SGMC-gap proxy correlating
with public score at **Spearman ρ = +0.518, p = 0.048, n = 15**. The proxy works.

The two results are consistent, and the distinction matters:

* The trunk measures the proxy across a **wide** score range (≈0.15–0.19).
* This measurement uses only the top three files, whose live scores span
  **0.0067** (0.1855 → 0.1922), while the proxy spread across the same three is
  0.0106 in the opposite direction.

A proxy can be monotone over a wide range and still be unable to *resolve* a
0.0067 band. The correct claim is therefore: **the offline proxy cannot rank the
top three files**, not "the proxy is broken". It remains reliable for coarse
comparisons (a collapsed model scores 0.033 vs an anchor's 0.107).

This is still why the delivered submission contains **no learned detector**: with
no offline instrument fine enough to separate 0.1855 from 0.1922, the only
defensible base is a map whose live score is already authenticated. Hence
`gate.gbm_heads_passed: false` and
`this_file_basis: "value-based re-emission of the highest authenticated live map"`.

**Resolved by:** any offline instrument that separates the top three correctly.
`registry/gems22_submissions.json` (19 observations) and the trunk's
`registry/submissions.json` (15 scored files) are the calibration sets for any
future attempt.

## L-3. Both supervised heads fail the gate; head A is degenerate

`evidence/head_to_head.json` → `verdict`: **0 of 6** fold×head combinations beat
the anchors, at matched budget *or* at their own optimum. Held-out SGMC-gap
efficiency was 0.004–0.008 for our heads vs 0.028–0.042 for the live-scored
anchors.

Head A specifically collapsed: `oof_A_T0/T1.f32` had `n_unique = 1` (a constant
6.32e-07) across the whole scored domain, because `struct_dist_cat_inv` is a
*perfect separator of the training positives* and boosting locked onto it. Head A
efficiency 0.0000–0.0078. It was abandoned, not repaired.

So: **156 engineered layers and two gradient-boosting heads did not beat a
four-line corroboration recipe from a previous session.** That is a negative
result and it is recorded as one rather than being buried.

**Resolved by:** a U-Net on the same cache and the same folds (the official
reference solution's architecture), which needs a GPU this sandbox does not have.

## L-4. `|G| ≈ 125,000` is a fitted point estimate, not an observation

`registry/group_geometry.json` → `G_fit`: `G = 125000.0`, objective `0.757`,
`band_frac = 0.842` (84.2% of the 19 files land in efficiency 0.02–0.35),
`corr_log_n_eff = −0.585`. Implied per-file efficiencies span **0.002 to 0.219**
— two orders of magnitude.

Everything downstream inherits that uncertainty:

| quantity | at \|G\| = 125,000 | sensitivity |
|---|---|---|
| τ (marginal DTI rule) at DTI 0.1894 | 0.0403 | scales with DTI |
| π\* emit threshold at DTI 0.1894 | 3.87% | τ/(1+τ) |
| π\* emit threshold at DTI 0.3168 | 6.34% | τ/(1+τ) |

If the true private `|G|` is 60,000 or 250,000, the optimal budget moves. The
chosen 550,000 sits on a broad plateau (300k–700k), which is deliberate — a
plateau is the right answer to take when the calibration is soft — but the
plateau's *centre* is not sharply determined.

**Resolved by (IMPLEMENTED OFF-LINE):** `src/gems22/metric.py` (`fit_G_mle` and
`propagate_G_uncertainty`) and `scripts/04b_infer_G_and_rescale.py` now perform a
maximum-likelihood joint fit of `|G|` over all `22` unique binary `(n_i, DTI_i)`
observations (after collapsing byte/scored-pixel duplicates), yielding
`G_mle = 107,000`, `G_mean = 116,106.1 ± 20,458.8`, 68% CI `[96,255, 134,963]`,
95% CI `[84,294, 165,151]` (`registry/group_geometry.json`), and propagating
`[p16, p50, p84]` intervals into `τ`, `π*`, and live-rescaled DTI predictions
(`f6777492`: `[0.16277, 0.17783, 0.19560]`). Remaining epistemic uncertainty
will be collapsed once `f6777492` (`n = 550,000`) is live-scored on DrivenData.

## L-5. The chosen base is h19-5, but the offline evidence marginally prefers h19-4

`evidence/submission_build.json` records `best_base_by_live_gain: "h19-4"` while
`base_anchor: "h19-5"`. This is a deliberate decision, not an oversight, and the
numbers are:

| anchor | authenticated live | mean live-rescaled gain @ its optimum | folds gaining | median live optimum |
|---|---|---|---|---|
| h16-1 | 0.1855 | +0.01828 | 4/4 | 550,000 |
| h19-4 | 0.1894 | **+0.01852** | 4/4 | 430,000 |
| h19-5 | **0.1922** | +0.01794 | 4/4 | 625,000 |

h19-4's advantage is **+0.00058** — inside the noise of an offline proxy that is
already known to invert (L-2). h19-5's advantage is **+0.0028 authenticated
live score**. The authenticated quantity was preferred.

The budget 550,000 is the **median of the three per-anchor median live optima**
{430k, 550k, 625k}, i.e. it is not tuned to the base that was selected.

**Resolved by:** the same upload that resolves L-1, or a proxy that ranks
correctly (L-2).

## L-6. Bour & Davy's scaling relation is measured here, not validated

Population A (raster catalogue, 3,199 traces): `a` = 1.83–2.16 depending on
`L_min`, nearest-larger-neighbour `x = 0.8223`, `A = 6.268 m`, `R² = 0.956`,
`D = 2.289`. Population B (GDR/INGENIOUS vector traces, 376): `a` = 0.908–1.200,
`x = 0.4367`, `A = 177.27 m`, `R² = 0.844`, `D = 1.471`.

The relation under test is `x = (a−1)/D` ([Bour & Davy 1999](https://doi.org/10.1029/1999GL900419)).
Measured ratios `x/((a−1)/D)` are **1.62** (A) and **3.21** (B). Neither
population reproduces the relation, and this repo does not claim it does:

* **`D = 2.289 > 2` is not a valid fractal dimension for a planar point set**
  (flag F-07). The correlation integral saturates once every box is occupied —
  Bour et al. (2002) note this explicitly — so any fit spanning 1–140 px is
  biased upward. Population B's `D = 1.471` is the only usable estimate.
* Population A's length distribution is dominated by **mapping incompleteness**,
  not fractal geometry. `a ≈ 2` on a catalogue whose median trace is 1,200 m is
  what truncation looks like.
* The fitted `a` does not reproduce the predecessor's 1.762 (flag F-08); the
  difference is the length estimator, and both are internally consistent.

The prior is still used, at weight `w = 0.3`, as a **geometric re-ranking term
among equally-scored candidates** — which needs only the monotone statement
"large faults have their nearest larger neighbour farther away", not the full
`x = (a−1)/D` identity. That weaker claim is supported by the data and is unit
tested.

**Resolved by:** estimating `D` over a lag range where the correlation integral
has not saturated, or from the vector population only.

## L-7. The Marrett NCC audit is a sanity gate, not a quality measure

`evidence/submission_build.json` → `audit`: catalogue mean NCC 5.890 (verdict
*clustered*), base anchor 3.474 (*clustered*), predicted 3.597 (*clustered*),
ratio 0.61 (inside the [0.5, 2.0] band), no verdict mismatch, no flags →
**PASS**.

What this proves: the predicted raster is not spatially *pathological* — it is
not regular (which would indicate survey-line aliasing) and not wildly more
clustered than the catalogue (which would indicate collapse onto acquisition
block edges). What it does **not** prove: that the predictions are faults. A map
can be perfectly clustered and entirely wrong; a random map restricted to a
clustered template would also pass.

## L-8. External layer coverage is incomplete and unevenly distributed

* **1 m 3DEP DEM openness/LRM**: 71,974 cells cached over eight 10 km tiles —
  about 1.4% of the footprint.
* **Derived 12-band LiDAR scarp product**: 3,892,964 grid cells = 31.7% of the
  full grid = 75.4% of the footprint, from 706 of 716 tiles. The missing 24.6%
  is concentrated in the NE quadrant, so any scarp-based feature is *systematically
  absent* there rather than randomly missing.
* **GeoDAWN radiometrics**: 5,166,085 valid px.
* The 19 competition bands contain **no radiometric family** (flag F-06), so K/Th/U
  evidence enters only through the externally fetched GeoDAWN layers.

A detector trained on these layers is blind in the NE quadrant in a way that no
cross-validation fold will reveal, because the folds are trace clusters, not
coverage classes.

**Resolved by:** fetching the remaining ~10 tiles (`NEXT_STEPS` item 2). Needs a
machine that can reach `tnmaccess.nationalmap.gov`, which this sandbox cannot.

## L-9. The official worked example cannot be reproduced from the published text (flag F-02)

[The metric page](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
gives TP = 3.00, FP = 1.89, FN = 2.00 → DTI 0.60. The arithmetic is consistent:
`3.00/(3.00 + 0.2·1.89 + 0.8·2.00) = 0.6027 → 0.60`. But the *geometry* is not
reproducible: the identity `FN = |G| − TP` forces `|G| = 5`, and no contiguous
5-pixel vertical ground-truth line admits 3 fully-covered plus 2 entirely
uncovered pixels under `k(d) = max(1 − d/300, 0)`.

The two schematic diagrams that would resolve this live on
`drivendata-public-assets.s3.amazonaws.com`, unreachable by both egress routes.

`src/gems22/metric.py` is instead validated to **7e-15** against an independent
triple-loop transcription over 60 random grids, and asserts the arithmetic
consistency of the published example. The implementation is trusted; the example's
geometry is flagged as unexplained.

## L-10. Sentinel pixels inside the official footprint (flag F-03)

3,073 pixels that are *inside* the official footprint carry the float32 nodata
sentinel `−3.4028235e+38`. A naive `np.clip(x, 0, 1)` maps them to 0.0 silently;
a naive `x[x < 0] = 0` does too; but any path that computes with them first
produces `inf`/`NaN` and is the most plausible root cause of the rejection
message `Predicted values must be in range [0, 1]`.

`submission.sanitise()` handles them explicitly, and `tests/test_gems22_submission.py`
feeds the exact sentinel value plus `±inf` and out-of-range values through both
output conventions. This is a fix for a *hypothesised* cause — the actual
rejected file is not in the registry, so the diagnosis is inferred from the data
plus the message, not confirmed by reproduction.

## L-11. What is deliberately not attempted here

* No GPU model, no U-Net (L-3).
* No Phase-2-specific optimisation; the re-scoring set is unknown by design.
* No use of the INGENIOUS 2 m temperature-probe / geothermometer inversion
  (`NEXT_STEPS` item 5) — data is local, the inversion is not built.
* No ensemble across hypotheses H22-3 (radiometric alteration halo) — the layers
  are cached but the transform is unimplemented and unvalidated.
* The leaderboard feed is a build-time snapshot (fetched 2026-10-01), not live;
  the site cannot refresh itself from inside this sandbox (`NEXT_STEPS` item 6).

## L-12. Reproducibility caveats

* `data/derived/` (4.4 GB) and the three large `assets/external/*.tif` layers
  (91 MB) are **gitignored**. A fresh clone reproduces them only via
  `scripts/fetch_data.sh` on a machine that can reach the blocked hosts.
* `trace_cluster_folds()` is pinned by a regression test against the exact
  per-fold pixel counts in `evidence/holdout_union.json`. If scikit-learn's
  KMeans changes behaviour across versions, that test fails loudly rather than
  silently invalidating every holdout number in the repo.
* The fold partition, the leakage-control prefix list and the `|G|` value are
  mutually load-bearing. Changing one without regenerating `evidence/*.json`
  produces a repo whose documentation contradicts its artefacts.

## L-13. The live no-skill floor is conditional on a *fitted* `|G|`

`src/gems/floor.py` computes what a random emission of `n` pixels scores
(`FLOOR_ANALYSIS.md`). At the fitted live density it is ≈ 0.181 at the group's
habitual 121k budget and ≈ 0.290 at 550k, and the closed form reproduces the
repository's own measured random arms to within −1.8 % … +3.1 %. But `|G|` itself
is a fit (`fit_G_mle`: MLE 107,000, 95 % CI 84,294–165,151), not an observation,
so the floor is a *band*, not a number. If the true live label set were ~60k px the
121k floor would be ≈ 0.154 and every historic submission would gain ~0.03 of real
lift. The floor's dominant uncertainty is therefore `|G|`, and the only clean way
to remove it is a deliberately random control upload (or the paired
`base` vs `base ∪ S` upload), which costs a weekly slot. Until then, all
lift-over-floor statements in this repository are bracketed by the `|G|` CI.

## L-14. No offline instrument in this repository has demonstrated resolving power for the live task

Three instruments disagree at the top of the table. On the trace-cluster folds with
the live masking convention and a size-matched random control
(`evidence/h26_instrument_consistency.json`): `h19-4` +0.0003, `h19-5` −0.0007,
`h22-1` −0.0002 (mean over 8 fold x head cells). Live: `h19-5` 0.1922 > `h19-4`
0.1894 > `h16-1` 0.1855. The four-quadrant instrument (flagged invalid, F-09)
prefers `h19-4`. The SGMC proxy inverts (`rho = -1.000` over the top three). Even a
perfect instrument must beat the metric's own no-skill floor, which at the group's
budget is 0.181 (p50 `|G|`) against a live best of 0.1922. The only way to settle
it is live: a random control upload pins the floor, a paired `base` vs `base ∪ S`
upload pins `tau`.

