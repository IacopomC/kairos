"""The Kairos flow-DSG *recording* format + render configuration.

A recording is a portable, framework-agnostic snapshot of a Kairos run that the
visualizer turns into a Rerun recording. Any dataset produces one via a small
adapter; the visualizer never imports the Kairos C++ core, so it runs standalone
(numpy + rerun) even on a different Python.

Stored as a single .npz. Fields (all arrays; T = #snapshots, N = #nav nodes,
V = #flow voxels, C = #mixture components kept per node):

REQUIRED
  t0, t1            : float, session start/end (unix s)
  nav_pos           : (N,2) navigational-node xy
  nav_edges         : (E,2) node adjacency (index pairs)
  flow_times        : (T,)  snapshot times (unix s)
  node_active       : (T,N) bool, node has dynamics at that snapshot
  node_dtheta       : (T,N) dominant heading (rad; NaN if inactive)
  node_drho         : (T,N) dominant speed (m/s)

OPTIONAL (rendered only if present)
  nav_radius        : (N,)  support radius per node (m)
  nav_appear_t      : (N,)  time each node/occupancy cell appears (incremental);
                            absent => the whole graph is static from t0
  room_xy           : (2,)  concourse/room centroid
  vox_xy            : (V,2) flow-voxel centres
  vox_create_t      : (V,)  first-detection time per voxel (<0 = never)
  vox_active_t      : (V,)  time voxel reached the active threshold (<0 = never)
  vox_count         : (V,)  final detection count per voxel (drives intensity)
  node_ctheta       : (T,N,C) per-node mixture-component headings (weight-desc)
  node_cweight      : (T,N,C) per-node component weights
  node_arch         : (T,N) motion-archetype code (index into arch_colors);
                            spark_dsg::ArchetypeType (0 STATIC, 1 UNIMODAL,
                            2 BIMODAL, 3 MULTIMODAL, 4 DIFFUSE)
  mesh_verts        : (Mv,3) reconstruction-mesh vertices (world frame)
  mesh_faces        : (Mf,3) triangle vertex indices
  mesh_colors       : (Mv,3) per-vertex RGB uint8 (absent => flat mesh colour)
  voxel_size        : float, flow-voxel edge (m), inherited from the run config
  robot_times       : (Rt,) robot-pose times
  robot_xy          : (Rt,2) robot xy
  det_times         : (Dt,) detection playback times
  det_step,x,y,th   : (D,)  per-arrow: playback-step index, xy, heading
  det_in            : (D,)  bool, inside the robot's active window
  det_vox           : (D,)  voxel index each detection falls in (-1 = none)
  active_window_radius : float, robot sensing radius (m)
  arch_colors       : (A,)  hex strings indexed by node_arch
"""
from dataclasses import dataclass, field
import numpy as np

_REQUIRED = ["t0", "t1", "nav_pos", "nav_edges", "flow_times",
             "node_active", "node_dtheta", "node_drho"]
# Indexed by spark_dsg::ArchetypeType; mirrors UpdateArchetypesFunctor node
# colours. Only a fallback -- a recording normally carries its own arch_colors.
_DEFAULT_ARCH_COLORS = ["#808080", "#00c800", "#0064ff", "#ff3200", "#e6c800"]


@dataclass
class RenderConfig:
    """Layer heights (m), colours, and drawing knobs. Heights are exaggerated in
    z so the DAG reads clearly; nothing depends on them physically."""
    z_mesh: float = -2.0                 # reconstruction mesh sits just below occupancy
    z_occupancy: float = 0.0
    z_voxels: float = 12.0
    z_nav: float = 36.0                  # voxels->nav gap doubled (12 -> 24)
    z_robot_offset: float = 0.6          # robot marker sits nav + this
    z_arrows: float = 48.0
    z_archetypes: float = 68.0
    z_room: float = 84.0                 # pulled toward archetypes to tighten the top span
    # colours (RGB, or RGBA for the translucent window)
    nav_node_color: tuple = (165, 200, 235)
    nav_edge_color: tuple = (120, 140, 170)
    arch_edge_color: tuple = (125, 132, 146, 140)   # archetype web; alpha keeps it a veil on white
    occupancy_color: tuple = (170, 176, 184)
    wall_color: tuple = (30, 30, 30)                 # occupied/wall cells of the occupancy map
    voxel_created_color: tuple = (205, 175, 175)     # in-map, not yet active
    voxel_active_lo: tuple = (200, 120, 120)         # low detection count
    voxel_active_hi: tuple = (95, 12, 12)            # high detection count
    robot_color: tuple = (255, 215, 0)
    detection_color: tuple = (170, 70, 70)           # out-of-window
    detection_inwin_color: tuple = (255, 55, 55)     # sensed
    hitvox_color: tuple = (255, 255, 255)            # highlighted voxel border
    window_color: tuple = (215, 215, 225, 110)       # translucent sphere shell
    background: tuple = (255, 255, 255)                # 3D-view background RGB (None = viewer default)
    # knobs
    voxel_size: float = 0.4
    active_threshold_intensity_pct: float = 90.0     # percentile for colour cap
    arrow_scale: float = 1.8
    # Shaft radii of the dominant component and of the weaker ones. Sized in
    # metres like the arrow length, so a small arrow_scale wants small radii too:
    # an arrow short enough to stay inside its own cell is otherwise all head.
    arrow_radius: float = 0.10
    arrow_radius_minor: float = 0.05
    arrow_color: str = "archetype"       # "archetype" | "heading" (hue wheel over direction)
    # Which heading each arrow takes. "components" draws every mixture component
    # the place holds, each at its own pinned cardinal heading, so a place with
    # several directions shows several arrows. "dominant" draws the single
    # weighted-mean heading instead: one arrow per place, free to point anywhere,
    # so the field varies smoothly at the cost of showing multimodality.
    arrow_source: str = "components"     # "components" | "dominant"
    # Carry component weight in opacity rather than only in length and width, so
    # the weaker directions of a place read as faint rather than as small.
    arrow_alpha_by_weight: bool = False
    arrow_alpha_min: float = 0.25        # opacity floor, so the weakest stays visible
    arrow_hue_sat: float = 0.90          # heading mode: HSV saturation/value of the wheel
    arrow_hue_val: float = 0.95
    arrow_min_frac: float = 0.35
    component_weight_thresh: float = 0.12
    max_components: int = 3
    static_archetype_edges: bool = False  # draw the archetype web for STATIC components
    cover_bin_s: float = 10.0            # appearance-time bin for structural layers
    voxel_bin_s: float = 3.0             # appearance-time bin for flow voxels
    window_shell_stride: int = 3         # sub-sample the sphere shell

    @property
    def z_robot(self):
        return self.z_nav + self.z_robot_offset


class Recording:
    """Loads a recording .npz and exposes fields with defaults for the optional
    ones. Access via attributes (e.g. rec.nav_pos); missing optional fields are
    None so the renderer can skip their layers."""

    def __init__(self, data):
        self._d = data
        missing = [k for k in _REQUIRED if k not in data.files]
        if missing:
            raise ValueError(f"recording missing required fields: {missing}")

    @classmethod
    def load(cls, path):
        return cls(np.load(path, allow_pickle=True))

    def get(self, name, default=None):
        return self._d[name] if name in self._d.files else default

    def __getattr__(self, name):
        d = self.__dict__["_d"]
        if name in d.files:
            v = d[name]
            return float(v) if v.ndim == 0 else v
        return None

    @property
    def n_nodes(self):
        return len(self._d["nav_pos"])

    @property
    def arch_color_list(self):
        c = self.get("arch_colors")
        return list(c) if c is not None else _DEFAULT_ARCH_COLORS
