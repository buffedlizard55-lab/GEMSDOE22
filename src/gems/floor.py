"""Closed-form no-skill floor of the competition distance-weighted Tversky metric.

Why this module exists
----------------------
``DTI = A / (0.2 A + 0.2 B + 0.8 |G|)`` with ``A = TP_w`` (kernel credit summed
over ground-truth pixels), ``B = FP_w`` (``sum over emitted pixels of
1 - kernel_to_gt``) and ``FN_w = |G| - A``.  A random binary emission of ``n``
pixels therefore does *not* score zero: at the competition's own ground-truth
density a random field scores about as much as the repository's best live
submissions do.  Any ranking of candidate submissions that ignores this floor is
measuring budgets, not detectors.

The floor is available in closed form.  With ``N`` the scored domain, ``f = n/N``,
kernel rings ``k_i`` (ascending, with ``m_i`` offsets above each level),

    E[credit per gt pixel] = sum_i (k_i - k_{i-1}) * (1 - (1-f)^{m_i})
    E[A] = |G| * E[credit per gt pixel]
    E[B] = n * (1 - union_mass / N)

where ``union_mass`` is the area of the union of the R-balls around every gt
pixel weighted by the max kernel.  For independent gt pixels it equals
``|G| * KERNEL_SUM``; real fault traces are clustered, so the measured union mass
is smaller.  ``UNION_RATIO_MEASURED`` is calibrated from this repository's own
measured random-control arms (``evidence/holdout_union.json``,
``folds.*.null_random``) and the model then reproduces those arms' DTI to within
3% (``evidence/h26_floor_model.json``).  Pass ``union_ratio=1.0`` for the
independent-pixel bound.

Everything here is pure NumPy; the validator in ``tests/test_floor.py`` checks
the closed form against ``gems.metric.dti_components_exact`` on a padded
synthetic grid (no edge effects).
"""
from __future__ import annotations

import numpy as np

ALPHA: float = 0.2
BETA: float = 0.8

#: distinct kernel values inside R = 3 px on the 100 m grid, ascending, paired
#: with the number of kernel offsets whose weight is *above* the value.  The
#: kernel itself is ``k(d) = max(1 - d/3, 0)`` over integer offsets ``d <= 3``.
KERNEL_LEVELS: tuple[tuple[float, int], ...] = (
    (0.0571909584, 25),   # d = sqrt(8)
    (0.2546440075, 21),   # d = sqrt(5)
    (1.0 / 3.0, 13),      # d = 2
    (0.5285954792, 9),    # d = sqrt(2)
    (2.0 / 3.0, 5),       # d = 1
    (1.0, 1),             # d = 0
)

#: sum of the 25 non-zero kernel weights
KERNEL_SUM: float = float(
    sum(v * c for v, c in
        ((1.0, 1), (2.0 / 3.0, 4), (0.5285954792, 4), (1.0 / 3.0, 4),
         (0.2546440075, 8), (0.0571909584, 4)))
)

#: union-mass ratio measured on this repository's random-control arms
#: (mean 0.3651, range 0.354..0.382 over 8 fold x head arms)
UNION_RATIO_MEASURED: float = 0.3651


def expected_credit_per_gt(density: float) -> float:
    """E[max kernel credit] for one ground-truth pixel at emission ``density``."""
    if density <= 0.0:
        return 0.0
    if density >= 1.0:
        return 1.0
    total, lo = 0.0, 0.0
    for k_i, m_i in KERNEL_LEVELS:
        total += (k_i - lo) * (1.0 - (1.0 - density) ** m_i)
        lo = k_i
    return float(total)


def expected_components(n: float, G: float, N: float,
                        union_ratio: float = UNION_RATIO_MEASURED) -> dict:
    """Expected (A, B, dti) of a random binary emission of ``n`` pixels.

    ``union_ratio=1.0`` gives the independent-ground-truth bound (an upper bound
    on the floor); ``UNION_RATIO_MEASURED`` matches measured fault-trace clustering.
    """
    f = min(float(n) / float(N), 1.0)
    credit = expected_credit_per_gt(f)
    A = min(G * credit, G)
    union_mass = min(union_ratio * G * KERNEL_SUM, N)
    B = float(n) * (1.0 - union_mass / N)
    dti = A / (ALPHA * A + ALPHA * B + BETA * G + 1e-12)
    return {"n": float(n), "G": float(G), "N": float(N), "density": f,
            "expected_credit_per_gt": credit, "A": A, "B": B, "dti": float(dti),
            "union_mass": union_mass, "union_ratio": union_ratio}


def floor_dti(n: float, G: float, N: float,
              union_ratio: float = UNION_RATIO_MEASURED) -> float:
    """Expected DTI of a no-skill (random) binary emission of ``n`` pixels."""
    return float(expected_components(n, G, N, union_ratio)["dti"])


def floor_band(n: float, G_lo: float, G_hi: float, N: float,
               union_ratio: float = UNION_RATIO_MEASURED) -> tuple[float, float]:
    """Floor at the low and high ends of a ground-truth-size uncertainty band."""
    return (floor_dti(n, G_lo, N, union_ratio), floor_dti(n, G_hi, N, union_ratio))


def lift_over_floor(dti: float, n: float, G: float, N: float,
                    union_ratio: float = UNION_RATIO_MEASURED) -> float:
    """Observed score minus the score a random emission of the same size gets."""
    return float(dti) - floor_dti(n, G, N, union_ratio)


def peak_floor(G: float, N: float,
               union_ratio: float = UNION_RATIO_MEASURED,
               n_max: float | None = None) -> tuple[float, float]:
    """(n, dti) of the highest no-skill score: bigger budgets buy floor, not skill."""
    hi = n_max if n_max is not None else min(N, max(4.0 * G, 100_000.0))
    ns = np.linspace(max(hi / 200.0, 1.0), hi, 600)
    vals = [floor_dti(float(n), G, N, union_ratio) for n in ns]
    i = int(np.argmax(vals))
    return float(ns[i]), float(vals[i])


def implied_a_bounds(dti: float, n: float, G: float) -> tuple[float, float]:
    """Exact bracket on A = TP_w given an observed score, from ``0 <= B <= n``.

    ``A = dti * (0.2 B + 0.8 G) / (1 - 0.2 dti)`` is increasing in ``B``, so the
    bounds are attained at ``B = 0`` (a perfectly clean emission) and ``B = n``
    (every emitted pixel pure junk).  Both ends are inferences, not observations.
    """
    denom = 1.0 - ALPHA * float(dti)
    lo = float(dti) * BETA * float(G) / denom
    hi = float(dti) * (ALPHA * float(n) + BETA * float(G)) / denom
    return lo, hi
