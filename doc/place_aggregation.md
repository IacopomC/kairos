# Place-level flow aggregation

This document explains, conceptually, how per-voxel flow state is rolled
up into per-place summaries. It is a companion to
`temporal_dynamics.md`; it deliberately does not name specific functions
or files, since those move. The kairos paper is the authoritative
reference for the math.

---

## 1. The role of place aggregation

Per-voxel flow state is the source of truth: every voxel that ever
received a detection maintains its own mixture model and predictor, and
those voxel states persist across active-window archival. Place nodes,
by contrast, are scene-graph entities meant for high-level reasoning ---
planners, archetype classifiers, edge-flow scorers --- and they need a
compact, voxel-free summary of the dynamics inside the spatial region
each place "owns". Place aggregation is the bridge: at every backend
update we re-derive a fresh summary per place from the current voxel
state, write it into node metadata, and discard the intermediate work.
The summary is regenerated each tick rather than incrementally
maintained; voxel state moves much faster than backend cadence, so a
fresh derivation is simpler than tracking deltas.

---

## 2. From voxels to place support

The first decision is: which voxels belong to which place?

The earlier implementation used an axis-aligned 3D box around each
place's centroid, sized roughly by the place's free-space inscribed-
sphere radius and capped at a configurable maximum. That had two
problems. Boxes around adjacent places overlap, so a single voxel could
contribute to multiple summaries and inflate downstream metrics, and the
size cap was a free parameter unrelated to the scene-graph resolution
the places were actually carrying.

The current implementation uses a Voronoi partition. Each flow voxel is
assigned to its single nearest place node; a place's support set is then
exactly the set of voxels closer to it than to any other place. This is
a true partition --- no overlap, no gaps within the partitioned region
--- and it has no resolution-dependent tuning. The one parameter that
remains is a maximum reach: a voxel farther than this cap from its
nearest place is dropped rather than adopted. The cap exists only to
prevent a single boundary voxel (e.g. at the edge of the active window,
where places thin out) from being adopted by an arbitrarily distant
place. Within the densely-covered interior of the scene graph the cap is
not active.

Why nearest-place is the right spatial primitive: places live on a
sparse graph that already encodes which spatial region each one
"controls". Two adjacent places implicitly delimit each other's reach;
the Voronoi assignment is just the mathematical statement of that
delimitation. Whatever shape a region has (corridor, room, intersection)
falls out of the place layout, not a separately-tuned per-place shape.

Implementation note: the partition is computed by 1-NN search from each
flow voxel into the set of place positions. A KD-tree built over place
positions makes this cheap; it is rebuilt from scratch each backend
update rather than maintained incrementally, because the cost is small
compared to the rest of the update and incremental maintenance would add
state-tracking complexity for no measurable gain.

---

## 3. What gets summarised, and how

For each place's support voxels we accumulate weighted sums and produce
a per-place tuple plus a small list of mixture components (capped at
three; weights below a configurable floor are dropped, and components
backed by very few voxels can also be dropped via a per-component voxel-
count gate).

The per-place fields are:

- **probability**: the unweighted mean of `p_present` over support
  voxels. This is the activity variable: how likely is something present
  in the place's region right now. Weighting it by anything else (such
  as concentration) would be circular --- it is the thing we are trying
  to measure.
- **confidence**: the *presence-weighted* mean of each cell's dominant-
  component weight. Note that "confidence" here is not a model-fit
  metric; per-cell, it is the concentration of mass on the strongest
  mixture mode (one if a cell is unimodal, lower if it has spread its
  mass across competing directions). The presence weighting downweights
  cells that are confidently fit but rarely active --- those would
  otherwise inflate the score with stale unimodal noise from voxels that
  have not seen recent activity. The asymmetry against `probability` is
  intentional.
- **support**: the count of contributing voxels. A coarse measure of
  spatial breadth.
- **t_latest**: the timestamp of the most recently updated contributing
  voxel. Lets a downstream consumer ignore stale places (e.g. ones whose
  last detection is older than the planner's horizon) without going back
  to the per-voxel state.
- **dominant theta / rho**: the heading and speed of the place's
  combined dominant flow direction, derived from the component list.
  Cached because the visualizer reads them every frame; storing them
  avoids re-deriving each render tick.

The per-place mixture is a small list of components, each carrying:

- **weight, theta, rho, sigma_theta, sigma_rho, sigma_cross**: standard
  weighted-average Gaussian-mixture statistics.
- **voxel_count**: how many voxels contributed to this component slot.
  Captures *spatial breadth* of evidence for the component --- a
  three-voxel slot at a place's edge looks very different from a
  thirty-voxel slot covering its core, even if both end up with similar
  mixing weights.
- **R**: the slot-level mean resultant length, in [0, 1]. Captures
  *angular agreement across the contributing voxels*: 1 means every
  voxel in this slot agreed on the heading, 0 means they cancelled out.
  This is distinct from `sigma_theta`, which is the within-voxel angular
  uncertainty averaged across the slot. Two corridors might have the
  same `sigma_theta` but very different `R` if one's voxels actually
  agree about a heading and the other's are individually-confident-but-
  inconsistent.

Both `voxel_count` and `R` are computed at aggregation time because
neither can be reconstructed from the reduced summary later.

A note on the regime: in the **fixed-component** mode, every voxel's
mixture has the same K cardinal headings, so within any slot every
contributing voxel reports the same `theta` --- per-slot `R` is
identically 1 and `voxel_count` always equals `support`. The two fields
become discriminative only in the **learned-component** mode, where
different voxels can have different headings within a slot and
disagreement is meaningful. Downstream classifiers that need an
"is the place angularly diffuse?" signal in fixed mode should compute
mixture-level R on the fly (`‖Σ_k w_k (cosθ_k, sinθ_k)‖ / Σ_k w_k`); see
the archetype rescue branch in `UpdateArchetypesFunctor` for an example.

---

## 4. What is deliberately not stored

Several quantities a downstream consumer might want are derivable in
constant time from the components list, so we do not duplicate them:

- The *mixture-level* resultant length (sometimes called the net-flux
  index) is a fixed function of the component weights and headings.
- The dominant-mode count --- how many components exceed a weight
  threshold --- is the input to the archetype layer above places, and
  the threshold belongs to that layer, not here.
- A "fill fraction" or place-size-normalised activity score is just
  `support` divided by the place's geometric extent; downstream code can
  compute that if it cares.

The principle: store what is not derivable; let consumers derive what is.

---

## 5. Edge-level annotations

After per-place summaries are written, the same pass walks place--place
edges and computes forward-flow, reverse-flow, average speed, and
bidirectionality scores per edge, by projecting each endpoint's
component headings onto the edge direction. The edge direction itself is
read from the place-node attributes in the scene graph, not from the
summaries --- the summary deliberately does not duplicate place geometry
that the graph already authoritatively carries. This avoids two
locations going stale relative to each other.

---

## 6. When and why the summary changes

The summary changes for three reasons, in roughly increasing severity:

- A new detection enters a voxel, that voxel's mixture and presence
  model update, and on the next backend tick the place owning that voxel
  picks up the change.
- A place is added, removed, or moved by the place-graph extractor, and
  the Voronoi partition redistributes voxels between neighbouring
  places. This happens whenever the GVD-based extractor reshapes the
  free-space topology --- common during early exploration, rare on
  re-visits.
- A loop closure deforms the scene, voxel positions are remapped, and
  the partition is rebuilt over the deformed positions. This is handled
  by the deformation-alignment pass before aggregation runs; from
  aggregation's point of view the voxel set is just at a different
  location in world coordinates.

Because the partition is recomputed from scratch each backend update,
case (2) requires no special handling --- the next tick already operates
on the new place layout.
