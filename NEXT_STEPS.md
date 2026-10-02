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

Nothing in items 2–6 needs a slot. Only item 1 does.

---

## 1. Spend one slot on the value-based emission budget — *needs 1 slot*

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
`python3 scripts/record_score_gems22.py --id 74cb4afe --score <X>`. That single
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

## 4. Calibrate `|G|` properly; turn the holdout into a score predictor

Currently `|G| = 125,000` comes from a **grid search** that makes the group's 19
live scores self-consistent (objective 0.757, 84.2% of files in efficiency
0.02–0.35). Per-file implied efficiencies span 0.002–0.219, so it is a point
estimate wearing a uniform (`LIMITATIONS.md` L-4).

Fit `|G|` jointly by maximum likelihood over all 19 `(A_i, B_i, DTI_i)` triples
with an explicit residual model, and propagate the resulting uncertainty into
τ = 0.2·DTI/(1 − 0.2·DTI) and π\* = τ/(1+τ). That converts the holdout from a
*ranking* device into a *score predictor*, which is what makes slot allocation
rational instead of merely consistent.

Needs: nothing new — the 19 observations are already in `registry/gems22_submissions.json`.
Payoff: better slot allocation. Cost: **low**. Best effort-to-value ratio in this
list; do it before spending any further slots.

## 5. Add the INGENIOUS 2 m temperature-probe / geothermometer inversion

27,092 spring/well records and 21 volcanic vents are cached in
`assets/external/`; 13GEMSDOE committed the full `2m_temperature_probe` dbf.
75.7% of the spring/well points lie **>500 m from any mapped fault**, which is
the observation that makes this worth building.

Backward conduit inversion — a near-surface thermal anomaly in an amagmatic
extensional setting requires a permeable pathway — is a **physically independent**
line of evidence that no head in this repo currently uses. It is hypothesis
H22-3's sibling and, unlike the radiometric halo, it needs no new download.

Needs: nothing new. Payoff ≈ +0.005 to +0.02 DTI. Cost: medium.

## 6. Automate the leaderboard and site feed

`scripts/record_score_gems22.py` plus a GitHub Action that re-reads the public
leaderboard and rebuilds `docs/` closes the loop the brief asks for, so nothing
has to be checked by hand. `.github/workflows/ci.yml` already runs the tests and
the artefact re-verification; extending it to fetch the leaderboard needs the
network access this sandbox lacks (GitHub Actions has it).

Needs: GitHub Actions. Payoff: removes all manual checking. Cost: low.

---

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
