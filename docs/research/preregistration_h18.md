# Pre-registration — H18 arms (committed BEFORE any H18 result was computed)

**Date:** 2026-09-30 (UTC). **Evaluator:** `src/gems/holdout.py` (verified to reproduce H16-1 exactly: 0.21272 dense / 0.08541 sparse, all four folds identical to the committed evidence).
**Comparator ("current holdout best"):** H16-1, per-fold dense `[0.20774, 0.23898, 0.15464, 0.24951]`, sparse `[0.089, 0.07794, 0.06622, 0.10847]`.
**Promotion gate (unchanged from the prior register):** higher mean dense **and** mean sparse DTI than H16-1; higher sparse DTI in ≥ 3 of 4 folds; no fold losing more than 0.01 DTI (dense or sparse).
**Rules for this pass:** every arm below is run once, with the constants below, and **all** results are reported (including failures). No constant is tuned on held-out labels. No arm is added after seeing a result without being labelled *post-hoc/exploratory* and excluded from the gate.

## Arms

| Arm | Hypothesis (short) | Inputs | Fixed constants |
|---|---|---|---|
| **H18-1 PoE** | A fault with no catalogue entry needs independent lines of evidence. Fuse the topographic-scarp expert and the geophysical-context expert by **summing log-odds** (product-of-experts) instead of H16-1's hand-weighted arithmetic blend. | OOF probabilities of `H16_3_ScarpPure_1m_10m` (topography only) and `Baseline_Bands19_DeReg` (geophysics + coarse topography); seam-free quantile calibration identical to H16-1 | equal weights; probabilities clipped to [1e-4, 1-1e-4] |
| **H18-3a Complexity prior** | Hidden-system faults concentrate where mapped faults terminate, intersect or step (USGS/INGENIOUS structural-setting literature). Boost H16-1 by a smoothed density of catalogue **endpoints + junctions**. | H16-1 OOF surface; skeleton of the *known* fault raster available to the evaluator (see "information rule") | Gaussian σ = 50 px (5 km); `p' = p · (1 + λ·z)`, λ = 1, `z` = density rescaled to [0, 1] by its 99th percentile |
| **H18-3b Oblique prior** | New intra-basin/cross faults strike obliquely to the local dominant range-front trend (USGS Argenta Rise step-over description). Boost candidate ridges whose strike is oblique to the local dominant known-fault strike. | H16-1 OOF surface + local dominant strike from the known fault raster structure tensor | tensor smoothing σ = 100 px (10 km); boost `1 + μ·sin²(Δθ)`, μ = 1 |
| **H18-3c Both** | Product of the two boosts above. | as above | λ = μ = 1 |

**Information rule (prevents leakage of held-out labels into the prior).** For the *sparse-proxy* score of a quadrant, the prior may use the training quadrants' faults **plus that quadrant's masked 80 % "known" components** (this mirrors the real task: all catalogued faults are known, new ones are hidden). For the *dense* score of a quadrant, the prior may use **only the other three quadrants' faults**. The 20 % held-out components are never used. H18-1 uses no labels beyond those in the OOF models.

## What each outcome means
* **PASS** → the arm is a *candidate*. It still needs the uniqueness gate and a format check before any file is shared, and a pass is **not** evidence of a leaderboard gain (the holdout recovers *known* faults; the hidden set is *new* faults).
* **FAIL** → recorded in `evidence/hypothesis_h18_validation.json` and in the register; the arm is not promoted and no file is generated for it.

## Hypotheses that are proposed but *not* run here (need new external data; obtainability checked in the register)
H18-4 geologic-map (SGMC) faults as a bedrock-fault prior; H18-5 thermal-feature anchoring (GDR INGENIOUS springs/wells, sinter/tufa, Quaternary volcanics); H18-6 cultural-lineament (road) suppression for DEM-based detectors.
