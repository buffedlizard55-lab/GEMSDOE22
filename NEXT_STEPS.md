# Next steps

Ranked work for the next session, with the gate each item must clear before it is
allowed to consume a submission slot. The site renders the same list at
[`docs/gems22-next_steps.html`](https://buffedlizard55-lab.github.io/GEMSDOE22/docs/gems22-next_steps.html);
this file is the long form and is the one to edit — the site is generated.

## The standing rule

> Never spend a submission slot on an idea that has not beaten the current
> holdout best.

Three uploads per rolling 7 days is the **only** ground-truth feedback channel,
and `evidence/anchor_calibration.json` shows the offline proxy currently inverts
against the live leaderboard at the top (ρ = −1.000, n = 3). Slots are the
scarcest resource in this project. Every item below is marked with whether it
needs one.

**H26 addendum (2026-10-02).** The rule is now enforced relative to the metric's
**no-skill floor**: a candidate must beat the incumbent *and* show a positive
**lift = DTI − floor_dti(n, |G|, N)** at its own budget
(`src/gems/floor.py`, `FLOOR_ANALYSIS.md`, `evidence/h26_floor_model.json`;
validated to ±3 % against the repo's own measured random arms). At the fitted
`|G|` the 121k floor is 0.181 and the 550k floor is 0.290, while the best live
score is 0.1922 — a +0.011 lift. Read §1 below with that in mind.

Nothing in items 2–6 needs a slot. Only item 1 does.

---

## 0. Spend the next slot on a *measurement* — a random binary control — *needs 1 slot*

Nothing shipped has a demonstrated lift over a size-matched random control on the
valid trace-cluster instrument (`evidence/h26_instrument_consistency.json`), and the
live scores sit on the metric's own no-skill floor (`FLOOR_ANALYSIS.md`). The single
upload that would unlock the most is a **random binary control at a fixed budget**
(e.g. 121,131 px, same format and masking as a normal submission). It measures the
live floor and `|G|` directly, and retro-calibrates all ~20 historical scores. The
second is the paired upload (`base` vs `base ∪ S`) which measures `tau` live. These
are deliberate exceptions to the "beat the holdout best" rule because they are
measurements, not hypotheses — the operator decides.

## 1. Spend one slot on the value-based emission budget — *needs 1 slot* — **HELD (H26 floor analysis)**

> **H26 verdict: do not upload `f6777492` first.** Its projected live range is
> 0.1628–0.1956, which lies *below* the 550,000-px no-skill floor of 0.2900
> (p50 `|G|`; band 0.2414–0.3446) — the projection and the floor cannot both
> describe the same map. The cheap resolution is the paired upload
> (`H19-4` vs `H19-4 ∪ S`), which measures the live floor and `τ` in one slot and
> then makes every later budget decision evidence-based. The material below is
> retained as the record of how the budget was derived, not as a live
> recommendation.

**The single largest measured, replicated, data-free gain available.**

Evidence: `evidence/submission_build.json` → `live_rescaled_budget_analysis`.
Across **3 authenticated anchors × 4 trace-cluster folds = 12/12 combinations**,
re-scaling the emission budget from the group's habitual ~120k to its live-rescaled
optimum improves projected DTI, by a mean of **+0.0179 to +0.0185** per anchor.
The three per-anchor median optima are {430,000; 550,000; 625,000}; the median of
those is the **550,000** used in the delivered file (10.771% of the 5,106,385-px
scored domain).

The gate it already cleared:
- `audit` → Marrett NCC verdict *clustered* for catalogue (5.890), base (3.474)
  and prediction (3.597); ratio 0.61 inside [0.5, 2.0]; no flags → **PASS**.
- `scripts/05b_reverify_published.py` → 8/8 served artefacts, SHA-256 match,
  12/12 hard checks, zip integrity → **PASS**.

The gate it did **not** clear, and why it is still the right bet: the GBM heads
failed the head-to-head (`gate.gbm_heads_passed: false`), so this submission uses
**no unvalidated detector**. It is a re-emission of the highest *authenticated*
live map (h19-5, 0.1922) at a budget justified by rescaling that same map's own
held-out behaviour. Under flag F-01 that is a directional expectation
(≈0.198–0.208), not a guarantee.

**Then:** record the outcome immediately with
`python3 scripts/record_score_gems22.py --id f6777492 --score <X>`. That single
observation is worth more than any further offline work, because it is the first
data point that can confirm or refute the live-rescaling method itself.

**If it scores below 0.1922:** the rescaling is refuted, `|G| ≈ 125,000` is
wrong, and the next session should revert to the 120k regime and attack L-2
(a correctly-ranking proxy) before anything else.

## 2. Extend 1 m 3DEP DEM openness/LRM coverage from 1.4% to the full footprint

Only 71,974 cells over eight 10 km tiles are cached. 7GEMSDOE already fetched
706 of 716 tiles and recorded the URLs in `knowledge/dem_tiles.json`; the derived
12-band scarp product covers 75.4% of footprint grid cells. Closing the remaining
**24.6% — concentrated in the NE quadrant** — is the highest-value data work left,
because the current gap is systematic, not random (see `LIMITATIONS.md` L-8).

Needs: a machine that can reach `tnmaccess.nationalmap.gov` (blocked here;
`evidence/network_reachability.json`). Payoff ≈ +0.01 to +0.03 DTI. Cost: medium.

## 3. Train a U-Net on the existing 156-layer cache

The official reference solution
([`drivendataorg/gems-prize-reference-solution`](https://github.com/drivendataorg/gems-prize-reference-solution))
is a U-Net ensemble: resnet18 encoder, `TverskyLoss(alpha=0.2, beta=0.8)`,
5 random 50/50 patch splits, patch 128, batch 32, 5 epochs, AdamW lr 1e-4. This
repo could not train one — no GPU — and its gradient-boosting heads lost to the
anchors 0/6 (`LIMITATIONS.md` L-3).

The feature cache is already on disk in the right shape. **Keep the folds, the
targets and the emission rule identical** so the comparison stays like-for-like;
swap only the learner. Reuse `holdout.trace_cluster_folds(cat | gap, 4, 48, seed=22)`
— it is pinned by a regression test, and reusing it means a U-Net result is
directly comparable to every number already in `evidence/`.

Needs: GPU. Payoff ≈ +0.02 to +0.08 DTI. Cost: high.

## 4. Calibrate `|G|` properly; turn the holdout into a score predictor (**IMPLEMENTED OFF-LINE; AWAITING LIVE A/B**)

Currently `|G| = 125,000` comes from a **grid search** that makes the group's 23
live scores self-consistent (`G_fit` in `registry/group_geometry.json`). Per-file
implied efficiencies span 0.002–0.319, so we also implemented joint maximum-likelihood
estimation (`src/gems22/metric.py::fit_G_mle` and `propagate_G_uncertainty`) in
`scripts/04b_infer_G_and_rescale.py`.

Collapsing identical-on-scored-pixel duplicates leaves `22` unique binary
`(n_i, DTI_i)` observations, yielding `G_mle = 107,000`, `G_mean = 116,106.1 ± 20,458.8`,
68% CI `[96,255, 134,963]`, 95% CI `[84,294, 165,151]` (`registry/group_geometry.json`
and `registry/gems22_submissions.json`), with propagated `[p16, p50, p84]` intervals on
`τ = 0.2·DTI/(1 − 0.2·DTI)` (`[0.03299, 0.03837, 0.04352]`), `π* = τ/(1+τ)`
(`[0.03194, 0.03695, 0.04170]`), and live-rescaled DTI (`f6777492` at `n = 550,000`:
`[0.16277, 0.17783, 0.19560]`).

Remaining step: once `f6777492` (`n = 550,000`) is live-scored, append its score and
re-run `python3 scripts/04b_infer_G_and_rescale.py`.

## 5. Add the INGENIOUS 2 m temperature-probe / geothermometer inversion (**IMPLEMENTED IN `src/gems22/hypotheses.py` & `src/gems22/features.py`**)

27,092 spring/well records (`7,859` thermal/geochemical anomalies, `75.71% >500 m`
from any mapped fault), 2,782 2 m temperature probes (`594` with `F2mDAB >= +1.5°C`,
`86.87% >500 m` from any mapped fault), 281 paleo-geothermal sinter/travertine/tufa
sites (`72.95% >500 m`), and 21 Quaternary volcanic vents (`85.71% >500 m`) are now
inverted into 4 backward conduit requirement rasters (`thermal_wellspring_conduit`,
`thermal_probe2m_conduit`, `thermal_paleo_vent_conduit`, `thermal_backward_composite`)
via `src/gems22/hypotheses.py::build_thermal_conduit_layers()` and wired into
`src/gems22/features.py::build_stream(..., include_thermal_inversion=True)` with
verified metrics in `evidence/thermal_conduit_evaluation.json`.

Remaining step: train a U-Net / multi-scale head on an unrestricted GPU host
including these 4 backward thermal conduit layers and gate it on the held-out
trace-cluster folds before spending a submission slot.

## 6. Automate the leaderboard and site feed

`scripts/record_score_gems22.py` plus a GitHub Action that re-reads the public
leaderboard and rebuilds `docs/` closes the loop the brief asks for, so nothing
has to be checked by hand. `.github/workflows/ci.yml` already runs the tests and
the artefact re-verification; extending it to fetch the leaderboard needs the
network access this sandbox lacks (GitHub Actions has it).

Needs: GitHub Actions. Payoff: removes all manual checking. Cost: low.

---

## 7. New hypotheses ranked by expected lift over the floor (H26) — *no slot until gated*

Full register: [`HYPOTHESES_H26.md`](HYPOTHESES_H26.md). Order:
`H26-A` offset markers on Quaternary surfaces (medium cost, highest expected
lift) > `H26-B` anisotropic tip continuation with DEM strike + conjugate
confirmation (low cost, testable on the pinned trace-cluster folds) > `H26-C`
silica-vs-carbonate blind-fault feeder inversion > `H26-D` conjugate X-pattern
nodes > `H26-E` drainage-knickpoint alignment (most orthogonal, lowest expected
lift). Runnable now with the data in this repository; `H26-A` additionally needs
the mirrored 10 m 3DEP tiles to be turned into channels, and the 1 m tiles beyond
the eight cached ones remain blocked from this sandbox
(`evidence/network_reachability.json`).

The running `scripts/h26_geometry_holdout.py` experiment already measures the
coarse versions of `H26-B` (`strike_tip`, with and without 1-px dilation) against
isotropic `halo_isotropic` and `bour_davy_ring` arms at five budgets, on the
pinned folds, for both heads.

## Ideas evaluated and rejected — do not re-litigate without new evidence

Recorded so the next session does not spend a slot rediscovering them.

| idea | why it was rejected | evidence |
|---|---|---|
| Trim the anchors to a smaller budget | live-rescaled gain at n = 60,000 is **−0.0748** — catastrophic | `evidence/submission_build.json` |
| Build the submission from the GBM | 0/6 folds beat the anchors; head A degenerate (`n_unique = 1`) | `evidence/head_to_head.json` |
| Use geographic quadrant folds | structurally invalid: they delete every known fault from the held-out region, but the live task keeps the catalogue visible everywhere. Measured DTI 0.033 vs 0.165 for the same features under object-level holdout | flag F-09 |
| Keep catalogue-derived features in the shared cache | they leak: a held-out fault sits at distance 0 from itself, giving fold NW DTI **0.977**. Removing them dropped it to 0.033 | flag F-10 |
| Claim the large-budget gain as ours | most of it is available to a **random** predictor (0.080 at n = 120k → 0.156 at its own optimum). The claim must be stated as *relative to the random control* | flag F-11 |
| Score a head against the full SGMC gap | it trained on the gap; DTI 0.310/0.324 was spurious. Held-out fold traces only → 0.0012–0.0044 | `evidence/head_to_head.json` |
| Treat `sample_submission.tif` as "predict total fault absence" | it contains **60,988 ones** — exactly the catalogue count | flag F-04 |
| Prefer NaN-outside over 0.0-outside (or vice versa) | score-identical: `12GEMSDOE` and `12GEMSDOE-allfinite` both = 0.1294 | flag F-05 |
| Assert `x = (a−1)/D` holds for the catalogue | measured ratios 1.62 and 3.21; `D = 2.289 > 2` is saturated and biased | flag F-07, `LIMITATIONS.md` L-6 |

## Suggested order for the next session

1. Read `README.md` — it contains the verbatim brief and the Arena Core Values.
2. Read `AGENTS.md` — the invariants that must not be broken silently.
3. Do item **4** (low cost, no slot, improves every later decision).
4. Do item **1** (one slot) and record the score.
5. Branch on the outcome as described in item 1, then take item **2** or **3**
   depending on whether a GPU or network access is available.
