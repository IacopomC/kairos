"""Optional visualization recorder for a live Kairos run.

Snapshots the growing scene graph + flow field at fixed intervals during a run
and writes a portable *recording* (.npz) that `kairos_visualizer` renders. It
only READS the pipeline (`pipeline.graph`, `pipeline.flow_voxels()`), so it never
affects the run's outputs. Enabled by the driver only when KAIROS_VIZ_RECORDING
is set; pure numpy + spark_dsg (no rerun needed to produce a recording).
"""
import numpy as np

from kairos_python import flow_archetype

_MAXC = 3          # top mixture components kept per node for the flow arrows (viz only)


class VizRecorder:
    """Accumulates snapshots of the DSG + flow field during a run."""

    def __init__(self, interval_s=30.0, voxel_size=0.4, active_n=10,
                 active_window_radius=5.0, mesh_interval_s=30.0):
        # interval_s <= 0 snapshots the graph/flow EVERY frame (finest incremental
        # growth). The mesh is captured on its own coarser cadence (mesh_interval_s)
        # so a per-frame graph run does not re-dump the full mesh every frame.
        self.interval = float(interval_s)
        self.mesh_interval = float(mesh_interval_s)
        self.vs = float(voxel_size)
        self.active_n = int(active_n)
        self.awr = float(active_window_radius)   # robot sensing window (map_window.max_radius_m)
        self._next_t = None
        self._next_mesh_t = None
        self.t0 = None
        self.t1 = None
        # navigational nodes: node-id -> index (grows as the graph grows)
        self._nid = {}
        self.node_pos = []
        self.node_appear_t = []
        self._edges = set()
        self.snap_t = []
        self._snaps = []          # per snapshot: {idx: (dtheta, drho, [cth], [cw])}
        # flow voxels: (vx,vy) -> index
        self._vid = {}
        self.vox_xy = []
        self.vox_create_t = []
        self.vox_active_t = []
        self._vox_count = {}
        # robot + detections
        self.robot_t = []
        self.robot_xy = []
        self.det_t = []
        self.det_x = []
        self.det_y = []
        self.det_th = []
        self.det_pid = []          # per-detection person id (-1 if dataset gives none)
        self._have_pid = False     # True once any real ped_id is seen
        # reconstruction mesh: keep only the latest snapshot (it is cumulative,
        # so storing every snapshot would waste memory for no extra information)
        self._mesh = None

    def _stamp(self, t):
        self.t0 = t if self.t0 is None else min(self.t0, t)
        self.t1 = t if self.t1 is None else max(self.t1, t)

    def robot(self, t, world_t_body):
        self._stamp(t)
        self.robot_t.append(float(t))
        self.robot_xy.append([float(world_t_body[0]), float(world_t_body[1])])

    def detections(self, t, xyz, theta, ped_ids=None):
        """Log raw pushed detections. When the dataset supplies per-person ids
        (`ped_ids`), they are kept so save() can collapse to one arrow per person
        per playback bin; without them every raw sample is kept (a standing
        crowd at high Hz then piles many arrows on one spot)."""
        if ped_ids is None:
            ped_ids = [-1] * len(xyz)
        else:
            self._have_pid = True
        for p, th, pid in zip(xyz, theta, ped_ids):
            self.det_t.append(float(t)); self.det_x.append(float(p[0]))
            self.det_y.append(float(p[1])); self.det_th.append(float(th))
            self.det_pid.append(int(pid))

    def maybe_snapshot(self, pipeline, t):
        self._stamp(t)
        if self.interval > 0:
            if self._next_t is None:
                self._next_t = t
            if t < self._next_t:
                return
            # advance past t so a coarse interval emits at most one snapshot here
            while self._next_t <= t:
                self._next_t += self.interval
        self._snapshot(pipeline, t)          # interval <= 0 => every frame

    def _mesh_due(self, t):
        if self.mesh_interval <= 0:          # capture with every snapshot
            return True
        if self._next_mesh_t is None:        # always grab the first mesh
            self._next_mesh_t = t + self.mesh_interval
            return True
        if t >= self._next_mesh_t:
            while self._next_mesh_t <= t:
                self._next_mesh_t += self.mesh_interval
            return True
        return False

    def _snapshot(self, pipeline, t):
        nav = pipeline.nav_snapshot()          # primitives (no spark_dsg type crossing)
        ids = nav["ids"]; xs = nav["x"]; ys = nav["y"]; sup = nav["support"]
        dth = nav["dtheta"]; dr = nav["drho"]
        cth_all = nav["comp_theta"]; cw_all = nav["comp_weight"]
        # Archetype is read back from the C++ classification (spark_dsg ARCHETYPES
        # layer via UpdateArchetypesFunctor), never recomputed here. Absent on an
        # older binding -> 0 (unclassified) so the recorder still runs.
        arch_all = nav.get("arch") if hasattr(nav, "get") else None
        snap = {}
        for k in range(len(ids)):
            nid = ids[k]
            if nid not in self._nid:
                self._nid[nid] = len(self.node_pos)
                self.node_pos.append([float(xs[k]), float(ys[k])])
                self.node_appear_t.append(float(t))
            idx = self._nid[nid]
            if sup[k] and sup[k] > 0:
                comps = sorted(zip(cth_all[k], cw_all[k]),
                               key=lambda p: -p[1])[:_MAXC]
                arch = int(arch_all[k]) if arch_all is not None else 0
                snap[idx] = (float(dth[k]), float(dr[k]),
                             [float(c[0]) for c in comps],
                             [float(c[1]) for c in comps], arch)
        for s, tg in zip(nav["edge_src"], nav["edge_tgt"]):
            if s in self._nid and tg in self._nid:
                a, b = self._nid[s], self._nid[tg]
                self._edges.add((min(a, b), max(a, b)))
        try:
            fv = pipeline.flow_voxels()
            for vx, vy, c in zip(fv["vx"], fv["vy"], fv["count"]):
                key = (int(vx), int(vy))       # ground-plane cell (ignore vz)
                if key not in self._vid:
                    self._vid[key] = len(self.vox_xy)
                    self.vox_xy.append([(vx + 0.5) * self.vs, (vy + 0.5) * self.vs])
                    self.vox_create_t.append(float(t)); self.vox_active_t.append(-1.0)
                vi = self._vid[key]
                self._vox_count[vi] = int(c)
                if self.vox_active_t[vi] < 0 and c >= self.active_n:
                    self.vox_active_t[vi] = float(t)
        except Exception:
            pass                               # flow_voxels optional
        try:
            m = pipeline.mesh_snapshot() if self._mesh_due(t) else None
            if m and len(m.get("vx", [])):     # keep the latest cumulative mesh
                verts = np.column_stack([m["vx"], m["vy"], m["vz"]]).astype(np.float32)
                faces = np.column_stack([m["f0"], m["f1"], m["f2"]]).astype(np.int32)
                mesh = {"verts": verts, "faces": faces}
                if "r" in m and len(m["r"]) == len(verts):
                    mesh["colors"] = np.column_stack(
                        [m["r"], m["g"], m["b"]]).astype(np.uint8)
                self._mesh = mesh
        except Exception:
            pass                               # mesh optional (older binding)
        self.snap_t.append(float(t)); self._snaps.append(snap)

    def save(self, path, arch_colors=None):
        N = len(self.node_pos); T = len(self.snap_t)
        nav_pos = np.array(self.node_pos, float) if N else np.zeros((0, 2))
        edges = (np.array(sorted(self._edges), np.int32) if self._edges
                 else np.zeros((0, 2), np.int32))
        node_active = np.zeros((T, N), bool)
        node_dtheta = np.full((T, N), np.nan); node_drho = np.zeros((T, N))
        node_ctheta = np.zeros((T, N, _MAXC)); node_cweight = np.zeros((T, N, _MAXC))
        node_arch = np.zeros((T, N), np.int8)
        for ti, snap in enumerate(self._snaps):
            for idx, (dth, drho, cth, cw, arch) in snap.items():
                node_active[ti, idx] = True
                node_dtheta[ti, idx] = dth; node_drho[ti, idx] = drho
                node_arch[ti, idx] = arch
                for m in range(min(_MAXC, len(cth))):
                    node_ctheta[ti, idx, m] = cth[m]; node_cweight[ti, idx, m] = cw[m]
        V = len(self.vox_xy)
        vox_xy = np.array(self.vox_xy, float) if V else np.zeros((0, 2))
        vox_count = np.array([self._vox_count.get(i, 0) for i in range(V)], float)
        # detections binned into a playback grid so det_step indexes det_times
        det = {}
        if self.det_t:
            dt = max(self.interval / 6.0, 0.5)
            dtimes = np.arange(self.t0, self.t1 + dt, dt)
            dstep = np.clip(((np.array(self.det_t) - self.t0) / dt).astype(np.int32),
                            0, len(dtimes) - 1)
            det_x = np.array(self.det_x); det_y = np.array(self.det_y)
            det_th = np.array(self.det_th)
            if self._have_pid:
                # One arrow per person per playback bin (dataset supplies ped_id):
                # keep the latest sample in each (bin, person) so the heading is
                # the freshest, collapsing a standing crowd's high-Hz duplicates.
                pid = np.array(self.det_pid)
                last = {}
                for i in range(len(dstep)):
                    last[(int(dstep[i]), int(pid[i]))] = i
                keep = np.array(sorted(last.values()), dtype=np.int64)
                dstep = dstep[keep]; det_x = det_x[keep]
                det_y = det_y[keep]; det_th = det_th[keep]
            o = np.argsort(dstep, kind="stable")
            det = dict(det_times=dtimes, det_step=dstep[o],
                       det_x=det_x[o], det_y=det_y[o], det_th=det_th[o],
                       det_in=np.zeros(len(o), bool), det_vox=np.full(len(o), -1, np.int32))
        colors = (np.array(arch_colors) if arch_colors is not None else
                  np.array(flow_archetype.color_list()))
        mesh = {}
        if self._mesh is not None:
            mesh = dict(mesh_verts=self._mesh["verts"], mesh_faces=self._mesh["faces"])
            if "colors" in self._mesh:
                mesh["mesh_colors"] = self._mesh["colors"]
        np.savez_compressed(
            path, t0=float(self.t0), t1=float(self.t1),
            nav_pos=nav_pos, nav_edges=edges, nav_appear_t=np.array(self.node_appear_t),
            room_xy=(nav_pos.mean(0) if N else np.zeros(2)),
            vox_xy=vox_xy, vox_count=vox_count,
            vox_create_t=np.array(self.vox_create_t), vox_active_t=np.array(self.vox_active_t),
            flow_times=np.array(self.snap_t),
            node_active=node_active, node_dtheta=node_dtheta, node_drho=node_drho,
            node_ctheta=node_ctheta, node_cweight=node_cweight, node_arch=node_arch,
            robot_times=np.array(self.robot_t), robot_xy=np.array(self.robot_xy),
            arch_colors=colors, active_window_radius=self.awr,
            voxel_size=float(self.vs), **mesh, **det)
        nv = len(self._mesh["verts"]) if self._mesh is not None else 0
        print(f"[viz_recorder] wrote {path}: {T} snapshots, {N} nodes, "
              f"{V} voxels, {nv} mesh verts")
