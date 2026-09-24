# Kairos Temporal Flow — Design Choices

The design decisions behind the temporal flow layer. Mechanics are in
`Concepts.md` and the parameters in `params.md`.

Components are fixed and only their mixing weights are estimated.

---

## 1. Candidate Periods and Dataset Requirements

`candidate_periods` is set per dataset to the recurrence scale its recordings
can resolve: `[60, 300, 600]` s on TBD, `[1, 12, 24, 168]` h on ATC and HB.
The spectral coefficients are only meaningful when the training span covers at
least 2–3 full cycles of a listed period; the per-cell order gate keeps a
period the data cannot resolve out of the forecast, leaving the mean term.

| Goal | Required dataset length | Suggested periods |
|---|---|---|
| Sub-hour patterns | ≥2 hours | `[600.0, 1800.0, 3600.0]` |
| Daily patterns | ≥3 days | `[3600.0, 14400.0, 43200.0, 86400.0]` |
| Weekly patterns | ≥3 weeks | `[86400.0, 259200.0, 604800.0]` |

---

## 2. Loop Closure / Deformation Alignment

On every backend correction the cells are re-keyed through kimera_pgmo's
stamped `deformPoints` (`deformation_knn` control points within
`deformation_tolerance_s`). The original (pre-deformation) cell index and
creation timestamp are carried per cell, and the deformation graph windows
control points relative to the *original* observation: every loop closure
deforms a cell from its original position, so repeated closures stay
consistent. The yaw part of a correction is re-expressed on the fixed slots: a whole-slot rotation permutes the per-slot state, a
fractional one interpolates the weight predictors between adjacent slots.

---

## 3. Edge Coupling

Each traversable, face-sharing voxel pair carries a coupling between their
slots, estimated as the pointwise mutual information of their
responsibilities. Pairs of detections within `edge_coupling_delta_t_coh` form
co-occurrences; every sampling window yields one log-PMI sample per slot pair
from that window's counts alone (smoothed by a 1e-6 floor), fed to a spectral
predictor. The window is the shortest candidate period divided by
`edge_coupling_samples_per_period` (12), and stays open until it holds
`edge_coupling_window_min_pairs` (2) pairs.

---

## 4. Evidence Sharing

A voxel's mixing weights are its own running means, shrunk toward the
count-weighted mean of its traversable neighbourhood: the voxels in the
support set of its place node and in those of the places joined to it by an
edge. The neighbourhood lends `sharing_lent_crossings` (3) crossings of
evidence, so it holds a share m/(C+m) of the reported weight, falling to zero
as the voxel accumulates crossings C of its own. The fixed regime
indexes slots identically across all cells, so neighbour weights combine slot
by slot. The
shrinkage is applied when the weights are read and only to the mean term:
the stored state and the harmonics stay the voxel's own.
