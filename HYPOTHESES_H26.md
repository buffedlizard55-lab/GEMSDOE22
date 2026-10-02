# H26 hypothesis register — faults that are *not* in the catalogue

Session H26, 2026-10-02. Every number below is either copied from a named file in
this repository or marked as an inference. Nothing here has been submitted.

## Why this register looks different from H16–H23

H26 measured the **no-skill floor** of the competition metric
(`src/gems/floor.py`, `evidence/h26_floor_model.json`). The closed form

```
E[credit/gt px] = sum_i (k_i - k_{i-1}) * (1 - (1-f)^{m_i}),   f = n / N
E[A] = |G| * E[credit/gt px]        E[B] = n * (1 - union_mass / N)
DTI  = A / (0.2 A + 0.2 B + 0.8 |G|)        (FN_w = |G| - A)
```

reproduces this repository's **own measured random-control arms** (the eight
`folds.*.null_random` arms in `evidence/holdout_union.json`) to within −1.8 % …
+3.1 % on DTI, and it says:

| quantity (scored domain 5,106,385 px) | value |
|---|---|
| live no-skill floor at the group's habitual ~121k budget, `|G|` = 116,106 | **0.1812** (95 % `|G|` band 0.1683–0.1928) |
| live no-skill floor at the delivered 550,000-px budget | **0.2900** (band 0.2414–0.3446) |
| fold (`T0/A`) no-skill floor at 120k | 0.0950–0.0966 |
| best live group score (`h19-5` `e27054cf`, 0.1922) as a multiple of its live floor | **1.06×** |
| the same file's fold score as a multiple of the fold floor | **2.2×** |

Consequences that govern everything below:

1. **Rank hypotheses by *lift over a budget-matched random control*, never by raw DTI.**
   A fold DTI of 0.2162 is 2.2× the fold floor; a live DTI of 0.1922 is 1.06× the live
   floor. Raw DTI across instruments is not comparable.
2. **Budget alone buys score at live density.** The floor rises from 0.11 at 60k to a
   peak of **0.278–0.290 at n ≈ 430k–465k** (depending on `|G|`) and falls again. Any "bigger budget" result measured
   on the fold (where the floor is flat, ~0.09–0.10) does not transfer; it must be
   re-measured as a *lift*.
3. **The live target is defined against the catalogue.** Staff state the test faults are
   expert-mapped faults *not* in USGS/INGENIOUS data, and declined to name sources or
   fault types (Forum 11527). The measured 2.2×→1.06× collapse is exactly what a
   catalogue-reproducing detector should look like on a catalogue-complement target.
   So the hypotheses worth building are the ones that find structures the catalogue
   does **not** already contain.

The standing rule still applies: no submission slot until a candidate beats the current
holdout best **measured as lift over the floor at the candidate's own budget**.

## Ranked candidates

| # | ID | mechanism | new vs H16–H23 | data | cost | expected lift over floor |
|---|---|---|---|---|---|---|
| 1 | **H26-A** | Offset markers: detect *displacement* of Quaternary depositional surfaces (fan toes, terrace risers, bar-and-swale axes), not ridges along known strike | H19-3/H22-1 detect scarps aligned with mapped structures; none test marker displacement independently of a known trace | 10 m 3DEP DEM (mirrored), 1 m lidar openness npz (cached) | medium | **high (inference)** — attacks the exact class the organisers describe, and is the only candidate whose signal is *not* catalogue-derived |
| 2 | **H26-B** | Anisotropic tip continuation: strike taken from the local DEM structure tensor, angular dispersion measured from the DFN orientation histogram, confirmed by a second independent orientation family | H19-1 extrapolates catalogue strike unidirectionally; H22-1 uses an *isotropic* halo (`halo_isotropic` λ = 18 px) | none new | **low** | medium (inference); directly testable on trace-cluster folds |
| 3 | **H26-C** | Blind-fault feeder inversion with a silica vs carbonate geothermometer discriminant, projected updip through basin fill | H19-2 built a conduit field but did not separate quartz (>70 °C) from tufa/carbonate systems (registered as H22-4, never implemented) | GDR 1391 CSVs (mirrored) | low-medium | medium (inference) |
| 4 | **H26-D** | Conjugate X-pattern nodes: emit where two *independently derived* lineament families cross at 50–80° | registered as H22-3 and never implemented | gravity/magnetic extensions + DEM (cached) | low | medium (inference) |
| 5 | **H26-E** | Drainage-knickpoint alignment: a line of knickpoints at consistent distance downstream in independent catchments | never tried in any repo in the group | 3DEP 10 m DEM (mirrored) | medium | medium-low (inference), but the most orthogonal to every existing map |

Cost is implementation + CPU in this repository, not scientific difficulty.

### H26-A — Offset markers on Quaternary surfaces (rank 1)

* **Layers:** `dem10` channels (`dem10_{openness,lrm,slope,curvature,...}`), 1 m lidar
  `openness`/`lrm` npz, `slope_10m`, the SGMC-gap trace raster
  (`derived_sgmc_faults_100m_u8.tif`, 83,593 positive px), and the catalogue
  (`labels.tif`, 60,988 px).
* **Physical signature:** a *marker* (fan-toe break, terrace riser, bar-and-swale axis)
  is offset across a line. The test is on the marker's plan-form continuity: fit the
  marker's local trend on both sides of a candidate fault line; a fault is where the
  trend breaks with a consistent sense along ≥ 1 km. This is a *differential* signal and
  is therefore insensitive to the absolute scarp brightness that the existing
  openness/LRM detectors key on.
* **Why it finds catalogue-missing faults:** the catalogue is a compilation of *mapped
  traces*; a fault whose only expression is a shear of young fan surfaces may never have
  been drawn, while it is exactly the class "expert-mapped new faults" should contain.
* **Difference from existing work:** H19-3/H22-1 emit where a ridge *looks* like a scarp
  and lies in the fractal halo of a mapped fault; H26-A emits where an *independent
  surface* is measurably displaced, and does not require a mapped neighbour.
* **Validation before any slot:** hold out *fans* (drainage-basin clusters), not traces —
  the trace-cluster folds (`holdout.trace_cluster_folds(cat | gap, 4, 48, seed=22)`)
  cannot test it, because the training signal is the marker geometry, not the trace set.
  Gate: lift over `floor_dti(n, |G|_fold, N_fold)` must exceed the best existing arm's
  lift in ≥ 3 of 4 fan folds.
* **Data obtainability:** 10 m DEM tiles for this footprint are already mirrored (tag
  `ext/dem10-36343078537`, commit `91d6566ecc`, GEMSDOE10; upstream public-domain
  `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/...`).
  The 1 m lidar tiles are **not** mirrored beyond the eight cached 10 km tiles; the
  remaining 706 tile URLs are recorded in 7GEMSDOE `knowledge/dem_tiles.json` but the
  TNM host is unreachable from this sandbox (`evidence/network_reachability.json`:
  14 of 19 official hosts TLS-unreachable). Fetch path: the user's own network, or a
  GitHub Actions run on a mirror repo.

### H26-B — Anisotropic tip continuation (rank 2)

* **Layers:** trace geometry only (`labels.tif`, `derived_sgmc_faults_100m_u8.tif`),
  plus the 10 m DEM structure tensor for strike.
* **Physical signature:** fault tips propagate along the local strike; angular scatter
  around the tip direction is a *measured* quantity, not a constant. Combine with a
  conjugate-partner test (both families must be present within 2–4 km).
* **Why it finds catalogue-missing faults:** the fractal clustering fit
  (`evidence/clustering_fit.json`; nearest-larger median 1,630.9 m, p90
  6,047.1 m; `D_correlation = 1.624`, `D_Bour–Davy = 1.489`, `alpha_ols = 2.0224`,
  `l_min = 1,650 m`) says the population is spatially organised; the *next* fault is
  most likely an along-strike or conjugate continuation of an existing trunk, and the
  live set is described as "newly mapped geometry of existing systems".
* **Difference from existing work:** `strike_tip_field` exists in
  `src/gems/geometry_prior.py` but uses catalogue strike and a fixed reach (≤ 20 px);
  `halo_isotropic` ignores orientation entirely. H26-B re-derives strike from the DEM and
  calibrates the angular dispersion from the measured orientation histogram.
* **Validation:** trace-cluster holdout (valid for this hypothesis). The running H26
  experiment (`scripts/h26_geometry_holdout.py`) already measures a coarse version
  (`strike_tip` arm with and without 1-px dilation, budgets 0.5–10 % of the fold mass).
* **Data:** none new.

### H26-C — Silica vs carbonate blind-fault feeder inversion (rank 3)

* **Layers:** GDR 1391 well/spring geothermometry, 2 m probes and paleo-thermal deposits
  (already parsed by `src/gems22/hypotheses.py::build_thermal_conduit_layers`), plus
  `tc`/`tmi_hg` bands and the SGMC-gap raster as the structural prior.
* **Physical signature:** a blind fault feeding a deep (quartz, >70 °C) system produces
  a *silica* signature whose spatial offset from the spring cluster gives the updip
  projection direction; shallow carbonate/tufa systems do not. Discriminating them
  removes the tufa false positives that sit on paleo-shorelines.
* **Why it finds catalogue-missing faults:** the feeder is inferred to be *between* the
  deep equilibrium and the surface expression, i.e. under basin fill where no trace is
  mapped — the classic blind-fault case.
* **Difference from existing work:** H19-2 accumulated a conduit field; no repository has
  separated the two geothermometer families (registered as H22-4, never implemented).
* **Validation:** SGMC-gap blocked holdout, scored as lift over floor.
* **Data:** GDR 1391 CSVs are mirrored (GEMSDOE24); upstream `https://gdr.openei.org`
  is TLS-unreachable from the sandbox.

### H26-D — Conjugate X-pattern nodes (rank 4)

* **Layers:** GeoDAWN radiometric/extension 4-band rasters, `iso_grav_anom`,
  `tmi_hg`, `tmi_vg`, and the DEM structure tensor.
* **Physical signature:** two lineament families crossing at 50–80° host short
  intra-basin faults (the ~15 km Argenta Rise step-over cited in the README is the type
  locality). Emit at intersections weighted by both families' amplitude.
* **Why it finds catalogue-missing faults:** intersections are *nodes* of the population,
  not continuations of mapped traces, and are systematically under-mapped because each
  individual segment is short and low-amplitude.
* **Difference from existing work:** registered as H22-3 and never implemented.
* **Validation:** trace-cluster holdout, lift over floor, plus a Jaccard gate (< 0.80)
  against all 22 historic group submissions.
* **Data:** none new.

### H26-E — Drainage-knickpoint alignment (rank 5)

* **Layers:** 10 m DEM channels (`dem10_elev`, `_slope`, `_curvature`), stream network
  derived from the same DEM.
* **Physical signature:** a line of knickpoints at a consistent downstream position in
  *independent* catchments implies a single tectonic line crossing all of them; the
  signal is in the longitudinal profile and is invisible to openness/LRM.
* **Why it finds catalogue-missing faults:** the fault need not produce a topographic
  scarp at 100 m sampling; it only needs to have reorganised drainage.
* **Difference from existing work:** never tried in any repository in the group.
* **Validation:** trace-cluster holdout; the Jaccard gate matters most here.
* **Data:** none new.

## Sources and obtainability (checked 2026-10-02 from the sandbox)

| source | official URL | reachable from sandbox | mirrored here |
|---|---|---|---|
| USGS 3DEP 1/3 arc-second DEM | `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/…` | no (TLS abort) | yes — GEMSDOE10 tag `ext/dem10-36343078537` |
| USGS Qfaults | `https://earthquake.usgs.gov/catalog/qfaults/` | no | yes — 7GEMSDOE (`qfaults_prior_u8.tif`) |
| GDR 1391 geothermal data | `https://gdr.openei.org` | no | yes — GEMSDOE24 |
| competition rasters | DrivenData | no (curl); yes via `fetch_page` for HTML only | yes — GitHub bridge parts, SHA-256 pinned in `scripts/prepare_data.py` |

Reachability is measured, not assumed: `evidence/network_reachability.json`
(14 of 19 official hosts TLS-unreachable from this sandbox). Anything new must arrive by
the user's network or a GitHub Actions run.
