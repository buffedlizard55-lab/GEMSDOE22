# The metric's no-skill floor — why 0.1922 happened, and what it changes

Session H26, 2026-10-02. Computed by `scripts/h26_floor_model.py`; reusable code in
`src/gems/floor.py`; validated by `tests/test_floor.py`; raw numbers in
`evidence/h26_floor_model.json`.

## 1. The floor is not zero, and it is large

The competition metric is

```
DTI = A / (0.2 A + 0.2 B + 0.8 |G|)        A = TP_w, B = FP_w, FN_w = |G| - A
```

(`src/gems/metric.py::dti_components_exact`, verified against the organisers'
worked example: TP 3.00 / FP 1.89 / FN 2.00 → 0.60).

A **random** binary emission of `n` pixels is not worth zero: one emitted pixel
credits every ground-truth pixel within `R = 3 px`. With `f = n / N` and kernel
rings `k_i` (ascending, `m_i` offsets above each level),

```
E[credit per gt pixel] = sum_i (k_i - k_{i-1}) (1 - (1-f)^{m_i})
E[A] = |G| * E[credit per gt pixel]
E[B] = n * (1 - union_mass / N)
```

`union_mass` is the max-kernel-weighted area of the union of the 25 R-balls around
the ground-truth pixels. For independent gt pixels it is `|G| * K_sum` with
`K_sum = 9.38030`; real fault traces are clustered, so the measured value is lower.
The script calibrates `union_ratio` on this repository's **own measured random
arms** and then reproduces their DTI to within −1.8 % … +3.1 %:

| arm (`evidence/holdout_union.json` → `folds.*.null_random`) | n | measured DTI | closed form | error |
|---|---|---|---|---|
| T0/A | 200,000 | 0.10394 | 0.10224 | −1.6 % |
| T0/B | 250,000 | 0.12445 | 0.12538 | +0.8 % |
| T1/A | 200,000 | 0.07150 | 0.07151 | +0.0 % |
| T1/B | 250,000 | 0.08710 | 0.08785 | +0.9 % |
| T2/A | 200,000 | 0.06420 | 0.06528 | +1.7 % |
| T2/B | 150,000 | 0.05750 | 0.05930 | +3.1 % |
| T3/A | 200,000 | 0.07910 | 0.07765 | −1.8 % |
| T3/B | 200,000 | 0.11120 | 0.10939 | −1.6 % |

Measured union-mass ratio: mean **0.3651**, range 0.354–0.382 (clustered gt traces,
as expected for fault maps).

## 2. What the floor is on the live task

Live scored domain `N = 5,106,385` px (footprint), `|G|` from
`src/gems22/metric.py::fit_G_mle` (MLE 107,000; posterior mean 116,106; 95 % CI
84,294–165,151). `evidence/h26_floor_model.json` → `live_no_skill_floor`:

| n emitted | floor at `|G|`=84,294 | floor at 116,106 | floor at 165,151 |
|---|---|---|---|
| 60,000 | 0.1056 | 0.1101 | 0.1139 |
| 121,131 (group's habitual budget) | 0.1683 | **0.1812** | 0.1928 |
| 250,000 | 0.2198 | 0.2458 | 0.2711 |
| 464,424 (`|G|`=p50 floor peak) | — | **0.290** | — |
| 428,000 (`|G|`=MLE floor peak) | — | 0.278 | — |
| 550,000 (delivered rescale budget) | 0.2414 | **0.2900** | 0.3446 |
| 5,106,385 (all-ones) | 0.0804 | 0.1098 | – |

The curve is not monotone: it peaks near 430k–465k (0.278 at the `|G|` MLE,
0.290 at the p50) and falls, because `A` saturates at `|G|` while `B` keeps growing.

## 3. Why this matters: the top of the live table is at the floor

| file | n | live score | floor at p50 `|G|` | lift band over the 95 % `|G|` floor band |
|---|---|---|---|---|
| `h19-5` `e27054cf` | 121,131 | **0.1922** | 0.1812 | **−0.001 … +0.024** |
| `h19-4` `691e4dfa` | 123,779 | 0.1894 | 0.1836 | −0.006 … +0.019 |
| `h16-1` | 123,939 | 0.1855 | 0.1837 | −0.010 … +0.015 |
| `h28-dotted-ridge` | 120,966 | 0.1839 | 0.1810 | −0.009 … +0.016 |
| `7GEMSDOE` (best small-budget file) | 76,859 | 0.1461 | 0.1330 | +0.007 … +0.020 |

At `|G|` = 116,106 the lift is positive for only two of the nineteen scored files in
the proxy-calibration table — `7GEMSDOE` (+0.0131 at 76,859 px) and `16GEMSDOE`
(+0.0018 at 123,939 px) — and the four top anchors sit between −0.0006 and +0.011.
Read against the full 95 % `|G|` band, every one of those lifts spans zero (or goes
negative). **The live scores recorded so far do not by themselves demonstrate skill
above a budget-matched random control.**

The historical "2.2× the floor" readings for `h19-4`/`h19-5` came from the
four-quadrant Dense/Sparse instrument, which this repository itself flags as
structurally invalid (F-09). `scripts/h26_instrument_consistency.py` re-measures
every published file on the *valid* trace-cluster instrument, with the live masking
convention (`domain = footprint & ~visible`), against a random control of the same
size (`evidence/h26_instrument_consistency.json`):

| file | n (median) | mean DTI | mean floor | lift vs floor | lift vs size-matched random control | cells > 0 |
|---|---|---|---|---|---|---|
| `h19-5` `e27054cf` | 117,982 | 0.08376 | 0.08397 | −0.00021 | −0.00067 | 4/8 |
| `h19-4` `691e4dfa` | 120,574 | 0.08504 | 0.08436 | +0.00069 | +0.00029 | 4/8 |
| `h22-1` `7fd2f28b` | 120,433 | 0.08381 | 0.08434 | −0.00054 | −0.00018 | 5/8 |
| `h22-2` `00a4a807` | 122,137 | 0.08388 | 0.08460 | −0.00071 | −0.00024 | 4/8 |
| `h23-a` (6 %) | 328,870 | 0.07465 | 0.08577 | −0.01112 | −0.01067 | 2/8 |
| `h23-b` (10 %) | 506,802 | 0.06399 | 0.07609 | −0.01210 | −0.01187 | 1/8 |

On the instrument the repository trusts, **no published candidate beats a
size-matched random control**, and the two large-budget files lose clearly. The
instrument also ranks `h19-4` above `h19-5`, the reverse of the live order — the
same `ρ = −1.000` pathology as the SGMC proxy, and for the same reason: the live
spread (0.0067) is smaller than the floor's own budget sensitivity (0.0024 between
121k and 124k plus the `|G|` band).

Caveat, stated because it cuts both ways: the trace-cluster instrument holds out a
spatially coherent *cluster*, so it penalises proximity priors harder than the live
task does (live new faults are often continuations of mapped systems). "Does not
beat random here" therefore does **not** prove "no skill live"; it proves there is
no *validated* skill in anything shipped so far, on either instrument.

## 4. Consequences adopted

1. **Ranking rule (new).** Offline and live candidates are compared by
   **lift = DTI − floor_dti(n, |G|, N) at the candidate's own budget**, never by raw
   DTI, and never across instruments. Where a candidate is compared on the fold,
   the comparator is a random control of the same size *on that fold*
   (`evidence/h26_instrument_consistency.json`), not the floor formula alone.
1b. **The top slot is a measurement, not an idea.** Nothing shipped has a
   demonstrated lift. The single most informative upload available is a random
   binary control at a fixed budget: it measures the live floor and `|G|` directly
   and retro-calibrates every historical score. The second is the paired upload
   (`base` vs `base ∪ S`), which measures `τ` live. Both are exceptions to the
   "beat the holdout best" rule and should be decided explicitly by the operator.
2. **The fold's budget sweep is not transferable.** On the fold the floor is flat
   (0.096 at 120k, 0.088 at 550k); live it rises to 0.29 at 550k. The measured
   "+0.018 for rescaling to 550k" therefore cannot be read as a live gain without
   an independent live-floor measurement.
3. **The delivered 550k rescale must not be uploaded yet.** Its projected
   live range (0.1628–0.1956, `evidence/submission_build.json` →
   `live_rescaled_budget_analysis`) lies *below* the 550k floor at `|G|` ≥ 116k
   (0.29) and only above it if `|G|` is near the low end of the CI. The projection
   and the floor cannot both be right about the same map: one of the two models is
   wrong. The cheap way to find out is the paired-upload experiment
   (`base` vs `base ∪ S`), which also measures `τ` directly.
4. **New hypotheses are ranked by expected lift over the floor** (see
   `HYPOTHESES_H26.md`), which favours mechanisms that find faults the catalogue
   does not contain.

## 5. What is *not* claimed

* `|G|` is a **fit** (`fit_G_mle`), not an observation. If the true live label set
  were ~60k px, the 121k floor would be ≈0.154 and the 0.1922 score would be a real
  +0.038 lift. The floor argument is therefore conditional, and its uncertainty is
  dominated by `|G|`, not by the closed form (which is validated to ~3 %).
* The `union_ratio` 0.3651 is calibrated on *catalogue* trace clustering; the live
  expert label set is not observable from here. The script reports both the
  independent-pixel bound (ratio 1.0) and the calibrated value.
* A low live score does not prove the map is worthless — it proves the metric, at
  this density and budget, cannot distinguish it from a random control.
