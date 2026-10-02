# Pre-registration — H23 hypothesis set and the E0 emission-policy experiment (22GEMSDOE, 2026-10-02)

**Committed before any H23 result was measured.** Git commit at registration time is recorded in
`evidence/hypothesis_h23_validation.json::preregistration.git_commit_before_results`.
Any analysis not described here is labelled `POST_HOC` in the evidence file and is **not** allowed
to promote a candidate to a submission slot.

---

## 0. Why this pre-registration exists

The group has spent 21 sessions improving *probability surfaces* while holding the *emission policy*
fixed at ~2.5 % of the footprint. Section E0 below pre-registers the hypothesis that the emission
policy — how many pixels you emit and how you allocate them in space — is a larger lever than any
remaining surface improvement, and gives the exact decision rule and the exact sweep to be run.
Sections H23-1..H23-5 pre-register five new *geological* channels, each with its layer list,
physical transform, catalogue-gap rationale, novelty claim and expected-gain/cost rank.

## 1. Data, splits and scoring used for gating

* Competition rasters (SHA-256 pinned): `training_features.tif 4371c82e…`,
  `labels.tif 7ba308cc…` (60,988 catalogue pixels), `sample_submission.tif 2176d08e…`
  (5,167,373 footprint pixels, 7,111,787 outside).
* **Official metric**: distance-weighted Tversky index, `DTI = TP_w / (TP_w + 0.2·FP_w + 0.8·FN_w)`,
  kernel `k(d) = max(1 − d/300 m, 0)`, `FP_w = Σ p·(1 − k_to_truth)`.
* **Primary gate (NEW this session) — off-catalogue SGMC holdout.**
  Truth = USGS State Geologic Map Compilation fault pixels **minus** the INGENIOUS catalogue,
  rasterised to the 100 m grid (`evidence/ci/derived_sgmc_faults_100m_u8.tif`,
  83,593 px total, **79,615 px off-catalogue**). This is the only available proxy for the thing the
  prize actually scores — faults the catalogue does not contain. `evidence/proxy_calibration_vs_lb.json`
  shows this family of proxy correlates with the public leaderboard (Spearman ρ = 0.518, p = 0.048 for
  `sgmc_gap`; ρ = 0.454, p = 0.090 for `sgmc_offcat`) whereas the *known-catalogue* proxy does not
  (ρ = 0.107, p = 0.704). Four spatially blocked quadrants (NW / NE / SW / SE from the footprint
  row/column medians) are reported separately.
* **Secondary gate — the inherited 4-quadrant catalogue holdout** (dense + 20 %-of-components sparse),
  retained for continuity with H16/H18/H19/H22. It is *not* sufficient on its own for promotion,
  because it only measures the ability to re-find scarp-bearing catalogue faults.
* **Uniqueness gate** (`src/gems/forensics.py::gate_candidate`): Jaccard < 0.80 against every historic
  group submission and against every sibling candidate.
* **Format gate** (`src/gems/submission.py::check_variants`): all 9 hard checks, values in `[0, 1]`.

## 2. E0 — emission policy (pre-registered sweep)

### 2.1 Decision rule (derived, not fitted)

Let `g` be the expected weighted true-positive credit added by emitting one more pixel and `f` its
false-positive weight. Because `FN_w = n_truth − TP_w`, adding `g` changes the denominator by
`0.2·g + 0.2·f`, so

```
dDTI > 0   ⟺   g > 0.2 · DTI · (g + f)
```

With `f ≈ 1` (a pixel farther than the 300 m kernel radius from every truth pixel) this reduces to

```
E[credit] > 0.2 · DTI          (≈ 0.038 at the current group best DTI ≈ 0.19)
```

i.e. **any pixel with more than ~4 % probability of being (or lying within 300 m of) a hidden fault
pixel is worth emitting.** The group's 2.5 % budget is not derived from this rule; it was inherited
from a power-law *length-deficit* argument (H19-1) that counts **traces**, not **pixels**, and
therefore under-counts by roughly the ratio (fault length / 100 m).

### 2.2 Independent corroboration already on record

`GEMSDOE10/reports/budget_density_sweep.json` (arm H20, blocked 806-system holdout, 3 draws) measured
this directly. At a catalogue-like truth density of 1.34 % the mean DTI rose monotonically with the
emitted fraction:

| policy | emitted px | mean DTI |
|---|---|---|
| topk01 | 9,292 | 0.0865 |
| topk02 | 18,583 | 0.1148 |
| topk03 | 27,875 | 0.1320 |
| topk04 | 37,167 | 0.1422 |
| topk06 | 55,750 | 0.1537 |
| topk08 | 74,334 | 0.1553 |
| **thin12** | **37,782** | **0.1664** |

Two things follow. (i) The optimum is well above the group's 2.5 %. (ii) At an *identical* pixel
count, `thin12` (37,782 px, DTI 0.1664) beats `topk04` (37,167 px, DTI 0.1422) by **+0.0243 DTI** —
the allocation, not just the count, matters, because the true-positive credit is a **MAX** over the
300 m kernel while every pixel pays false-positive cost.

### 2.3 Sweep to be run

Grid over `pool_frac ∈ {0.025, 0.035, 0.05, 0.065, 0.08, 0.10, 0.125}` ×
`policy ∈ {topk, thinned}` on the baseline H16-1 out-of-fold surface, evaluated on **both** gates.
**Promotion rule**: the policy that maximises the *mean off-catalogue SGMC DTI* over the 4 quadrants,
subject to the catalogue-holdout mean dense DTI not falling below the H16-1 baseline by more than
0.005. The chosen policy is then applied to whatever surface wins §3.

## 3. H23-1..H23-5 — new geological channels (registered before measurement)

Full machine-readable register: `src/gems/hypotheses_h23.py::spec_table()`.

| Rank | ID | Family | New layer(s) first used here | Cost |
|---|---|---|---|---|
| 1 | H23-1 | topographic / fractal | — (new transform on existing 1 m / 10 m / 100 m edge fields) | medium |
| 2 | H23-2 | morphotectonic | `topo_u8` band 9 `dem_mean` (100 m elevation) | medium |
| 3 | H23-5 | topographic fabric | `topo_u8` bands 7 `aspect_coherence`, 8 `hs_lineament` | low |
| 4 | H23-4 | radiometric | `radiometric_u8` 7-band K/Th/U stack | low |
| 5 | H23-3 | potential field | — (TDR / THDR / tilt-depth transform on existing GeoDAWN bands) | low |

**Evaluation protocol for each channel** (identical for all five, no tuning after the fact):

1. Each channel is computed **label-free** on the full 100 m grid, then rank-normalised inside the
   footprint.
2. *Standalone arm*: the channel alone is used as the emission score at the E0-optimal policy.
3. *Fused arm*: the baseline H16-1 out-of-fold surface is multiplied by
   `w = 1 + β·(q − 0.5)` with a **single fixed** `β = 0.6` and `q` the in-footprint rank of the
   channel, i.e. a ±30 % modulation. `β` is **not** tuned per channel.
4. *Supervised arm* (only for any channel that passes step 3): append the channel as extra columns to
   the `H16_3_Antislope_Piedmont_Scarp_1m_10m` column set and retrain the 4-fold
   `HistGradientBoostingClassifier` with the identical hyper-parameters used for H16/H19
   (`max_iter=150, max_leaf_nodes=31, min_samples_leaf=80, learning_rate=0.06,
   l2_regularization=2.0, random_state=20260929+fold`).

**Promotion rule for a geological channel**: it must (a) improve the mean off-catalogue SGMC DTI over
the baseline on ≥ 3 of 4 quadrants, **and** (b) not reduce the catalogue-holdout mean dense DTI by more
than 0.005, **and** (c) keep the resulting submission DISTINCT (J < 0.80).
A channel that fails is reported **REJECTED** with the measured numbers; it does not get a slot.

## 4. Slot discipline

The rules allow **3 uploads per rolling 7 days per entity** (official rules §3.2/§3.4; staff forum
topic 11524 post 2). No H23 candidate is uploaded unless it has passed §1–§3 on the pre-registered
protocol. Untested channels are published to the site as **research artifacts**, explicitly labelled
`not yet live-scored`, and never in the "Upload #1" position.

## 5. Declared limitations (unchanged, re-stated)

* The SGMC proxy is a *different* fault compilation from the INGENIOUS catalogue: it shares its
  scarp-detection bias and differs in position by up to a few hundred metres at its nominal
  1:1,000,000 source scale. It is a better proxy than the catalogue, not the hidden test set.
* Neither proxy reproduces the private test geography, the hidden fault population, the 300 m
  evaluation kernel applied to *hidden* truth, or the leaderboard's public/private split.
* Ranking by a proxy that is itself scarp-biased will systematically *under-rate* H23-2, whose whole
  premise is that it fires where the local scarp is absent. This is pre-registered as an expected
  failure mode, not excused after the fact.
