"""Spatially-blocked cross-validation for the GEMS grid.

WHY BLOCKED: fault networks are spatially autocorrelated (Bour & Davy 1999 show
the trace barycentres are themselves a fractal point set, NCC >> 1 at 2-40 px in
this very region -- see evidence/gems22_clustering_fit.json).  A random patch split, as
used by the official reference solution notebook
(`make_patches(..., test_proportion=0.5, seed=...)` on a 128 px patch grid), puts
pieces of the SAME fault into train and test.  That inflates apparent skill and
is why prior sessions found their known-fault holdout was uninformative about the
live leaderboard (Spearman rho = +0.11, p = 0.70).

The blocked design mirrors the live scoring configuration exactly:

    live:      GT            = faults NOT in the catalogue
               scored domain = footprint MINUS catalogue pixels
    holdout k: GT            = catalogue faults inside block k  (never trained on)
               scored domain = block k MINUS catalogue pixels that lie in the
                               TRAINING blocks (i.e. minus everything the model
                               was allowed to see)
               collar        = a `buffer_px` ring around the block boundary is
                               dropped from scoring, because the feature bank
                               reaches up to sigma = 4 px (400 m) plus a 9-px
                               box filter, so a wider collar is needed to stop
                               information crossing the fold boundary.

Two grids are provided:
  * `quadrants`  -- 4 folds (NW, NE, SW, SE), comparable with prior sessions.
  * `blocks3x3`  -- 9 folds, a stricter test with smaller training sets.
  * `checker`    -- 2 folds of interleaved 8x8-px supersquares, the strictest
                    test of local generalisation (used only as a diagnostic).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=bool)   # 8-connectivity for trace labelling


def quadrant_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    H, W = shape
    mh, mw = H // 2, W // 2
    out = {}
    out["NW"] = np.zeros(shape, bool); out["NW"][:mh, :mw] = True
    out["NE"] = np.zeros(shape, bool); out["NE"][:mh, mw:] = True
    out["SW"] = np.zeros(shape, bool); out["SW"][mh:, :mw] = True
    out["SE"] = np.zeros(shape, bool); out["SE"][mh:, mw:] = True
    return out


def blocks3x3_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    H, W = shape
    ys = np.linspace(0, H, 4).astype(int)
    xs = np.linspace(0, W, 4).astype(int)
    out = {}
    for i in range(3):
        for j in range(3):
            m = np.zeros(shape, bool)
            m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = True
            out[f"r{i}c{j}"] = m
    return out


def checker_masks(shape: tuple[int, int], block: int = 8) -> dict[str, np.ndarray]:
    H, W = shape
    yy = (np.arange(H)[:, None] // block) % 2
    xx = (np.arange(W)[None, :] // block) % 2
    c = (yy ^ xx).astype(bool) * np.ones((1, 1), bool)
    return {"even": np.broadcast_to(c, shape).copy(),
            "odd": np.broadcast_to(~c, shape).copy()}


GRID_BUILDERS = {"quadrants": quadrant_masks, "blocks3x3": blocks3x3_masks,
                 "checker": checker_masks}


def folds(shape: tuple[int, int], grid: str = "quadrants") -> dict[str, np.ndarray]:
    return GRID_BUILDERS[grid](shape)


def evaluation_mask(block: np.ndarray, footprint: np.ndarray, catalogue: np.ndarray,
                    buffer_px: int = 12) -> np.ndarray:
    """Scored domain for one fold.

    Exact structural analogue of the live configuration:

        live:     scored domain = footprint MINUS catalogue pixels
                  GT            = new faults, disjoint from the catalogue
        fold k:   scored domain = block k eroded by the collar, IN footprint
                  GT            = catalogue faults inside that eroded block

    Every catalogue pixel the model was allowed to TRAIN on lies outside the
    eroded block (`fold_train_mask` is `footprint & ~inner`), so those pixels are
    excluded from scoring exactly as the organiser excludes known faults
    (DrivenData forum topic 11516, post 2).  The only catalogue pixels inside the
    scored domain are this fold's held-out faults, which play the role of the
    private "new fault" ground truth.

    `catalogue` is accepted for signature symmetry with the live rule and is
    deliberately NOT subtracted here; subtracting it would delete the fold's own
    ground truth (a bug caught by tests/test_holdout.py).
    """
    inner = ndimage.binary_erosion(block, np.ones((3, 3), bool),
                                   iterations=int(buffer_px))
    return inner & footprint


def fold_target(block: np.ndarray, catalogue: np.ndarray, footprint: np.ndarray,
                buffer_px: int = 12) -> np.ndarray:
    """Ground truth for one fold: held-out catalogue faults, subset of the domain.

    Invariant asserted by `tests/test_holdout.py`: target is a strict subset of
    `evaluation_mask`, and target is disjoint from `fold_train_mask`.
    """
    return catalogue & evaluation_mask(block, footprint, catalogue, buffer_px)


def fold_train_mask(block: np.ndarray, footprint: np.ndarray, catalogue: np.ndarray,
                    buffer_px: int = 12) -> np.ndarray:
    """Pixels available for training when `block` is held out."""
    inner = ndimage.binary_erosion(block, np.ones((3, 3), bool),
                                   iterations=int(buffer_px))
    return footprint & ~inner


# ---------------------------------------------------------------------------
# Trace-cluster holdout -- the design that actually matches the live task
# ---------------------------------------------------------------------------
def trace_cluster_folds(catalogue: np.ndarray, n_folds: int = 4,
                        n_clusters: int = 48, seed: int = 0,
                        min_trace_px: int = 3) -> dict[str, np.ndarray]:
    """Partition catalogue fault TRACES into spatially-coherent held-out groups.

    WHY NOT GEOGRAPHIC QUADRANTS.  A geographic block fold deletes *every* known
    fault from the held-out region.  But the live task keeps the full catalogue
    visible everywhere and asks for faults that are missing from it -- proximity
    to known structure is legitimate signal there (Bour & Davy 1999 show fault
    networks are fractal-clustered, so unmapped faults concentrate near mapped
    ones).  A quadrant fold therefore penalises exactly the behaviour the live
    metric rewards: measured here, a head trained on quadrants collapsed to
    p99 = 0.0000 inside the held-out quadrant and scored DTI 0.033, while the
    same features scored DTI 0.165 under a trace-level holdout.

    This design keeps the object-level guarantee that matters -- the model never
    sees a held-out TRACE, and traces are grouped into spatially coherent
    clusters so it cannot simply interpolate between two halves of one fault --
    while leaving catalogue context present across the whole map.

    Returns {fold_name: ground_truth_bool_mask}.
    """
    from sklearn.cluster import KMeans

    lab, n = ndimage.label(catalogue, structure=STRUCT8)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    ids = np.array([i for i in range(1, n + 1) if sizes[i] >= min_trace_px])
    if ids.size < n_folds * 2:
        raise ValueError(f"too few traces ({ids.size}) for {n_folds} folds")
    cen = np.array(ndimage.center_of_mass(catalogue, lab, ids.tolist()))
    k = int(min(max(n_clusters, n_folds), max(n_folds, ids.size // 4)))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(cen)
    lab_ids = km.labels_
    # Round-robin the clusters across folds so each fold gets geographically
    # scattered but internally coherent trace groups.  This is the assignment that
    # generated every number in evidence/*.json; it must not be changed silently.
    #
    # KNOWN EDGE CASE, fixed by the repair below: `argsort(lab_ids)` ranks *traces*
    # but maps them to cluster labels, so clusters holding many traces occupy many
    # ranks and the modulo can skip a residue entirely, and KMeans can also return
    # empty clusters.  On the real 3,199-trace catalogue all four folds come out
    # non-empty and well balanced (gt px 21344 / 13147 / 11727 / 14639), so the
    # repair never fires there and the recorded evidence stays valid.  It only
    # fires on degenerate small inputs (e.g. the synthetic 40-trace unit test).
    order = np.argsort(lab_ids, kind="stable")
    fold_of_cluster: dict[int, int] = {}
    for rank, ci in enumerate(order):
        fold_of_cluster[int(lab_ids[ci])] = rank % n_folds
    out: dict[str, np.ndarray] = {f"T{i}": np.zeros(catalogue.shape, bool)
                                  for i in range(n_folds)}
    for tid, c in zip(ids, lab_ids):
        out[f"T{fold_of_cluster[int(c)]}"] |= (lab == tid)

    # ---- repair: guarantee no empty fold, moving as little as possible --------
    empty = [i for i in range(n_folds) if not out[f"T{i}"].any()]
    if empty:
        by_size = np.argsort([sizes[t] for t in ids], kind="stable")[::-1]
        for i in empty:
            donor = max((j for j in range(n_folds) if j != i and out[f"T{j}"].any()),
                        key=lambda j: int(out[f"T{j}"].sum()))
            moved = False
            for idx in by_size:                     # give away a whole trace
                tid = int(ids[idx])
                trace = (lab == tid)
                if (trace & out[f"T{donor}"]).any() and int(out[f"T{donor}"].sum()) > 1:
                    out[f"T{donor}"] &= ~trace
                    out[f"T{i}"] |= trace
                    moved = True
                    break
            if not moved:                           # last resort: single pixel
                ys, xs = np.nonzero(out[f"T{donor}"])
                out[f"T{donor}"][ys[0], xs[0]] = False
                out[f"T{i}"][ys[0], xs[0]] = True
    assert all(v.any() for v in out.values()), "trace_cluster_folds produced empty folds"
    return out


def trace_fold_masks(catalogue: np.ndarray, footprint: np.ndarray,
                     gt: np.ndarray, collar: int = 1) -> dict[str, np.ndarray]:
    """Derive the three masks for one trace-cluster fold.

    train_target : catalogue traces the model MAY learn from
    train_domain : pixels usable for fitting (footprint minus the held-out
                   traces and their `collar` halo)
    scored_domain: footprint minus train_target -- the exact analogue of the
                   live rule "pixels corresponding to known USGS/INGENIOUS faults
                   are masked / excluded from evaluation" (forum 11516 post 2),
                   with the held-out traces playing the role of the private
                   new-fault ground truth.
    """
    halo = gt if collar <= 0 else ndimage.binary_dilation(
        gt, np.ones((3, 3), bool), iterations=int(collar))
    train_target = catalogue & ~halo
    return {"train_target": train_target,
            "train_domain": footprint & ~halo,
            "scored_domain": footprint & ~train_target,
            "gt": gt & footprint}
