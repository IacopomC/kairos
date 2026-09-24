"""Render a Kairos flow-DSG recording to a Rerun .rrd.

Dataset-agnostic: everything comes from the recording (see schema.py) and the
RenderConfig. Layers whose data is absent are skipped, so the same renderer
serves a coverage-driven simulated run or a real online reconstruction alike.
Imports rerun lazily so the package is importable without it.
"""
import colorsys

import numpy as np

from .schema import Recording, RenderConfig


def _heading_rgb(angle, cfg):
    """Direction as a hue: opposite headings sit opposite on the wheel, so a
    reversal of the dominant flow reads as a complementary colour."""
    hue = (float(angle) + np.pi) / (2.0 * np.pi) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, cfg.arrow_hue_sat, cfg.arrow_hue_val)
    return (int(r * 255), int(g * 255), int(b * 255))


def _with_alpha(rgb, frac, cfg):
    """Component weight as opacity: the dominant direction is opaque and the
    weaker ones fade toward `arrow_alpha_min`, which keeps the faintest legible."""
    lo = cfg.arrow_alpha_min
    a = lo + (1.0 - lo) * float(np.clip(frac, 0.0, 1.0))
    return (rgb[0], rgb[1], rgb[2], int(round(255 * a)))


def _hex2rgb(h):
    h = str(h).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _xyz(xy, z):
    return np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), z)])


def _cell_mesh(centers, half, z, color):
    """Flat continuous surface: one quad (two triangles) per cell centre at z."""
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    verts = (centers[:, None, :] + corners[None, :, :]).reshape(-1, 2)
    verts = np.column_stack([verts, np.full(len(verts), z)])
    base = np.arange(len(centers)) * 4
    tris = np.empty((len(centers), 2, 3), int)
    tris[:, 0, :] = np.column_stack([base, base + 1, base + 2])
    tris[:, 1, :] = np.column_stack([base, base + 2, base + 3])
    cols = np.tile(np.array(color, np.uint8), (len(verts), 1))
    return verts, tris.reshape(-1, 3), cols


def _tri_marker(x, y, heading, z, size=0.7):
    loc = np.array([[0.6 * size, 0.0], [-0.4 * size, 0.35 * size],
                    [-0.4 * size, -0.35 * size]])
    c, s = np.cos(heading), np.sin(heading)
    w = loc @ np.array([[c, -s], [s, c]]).T + [x, y]
    return np.column_stack([w, [z, z, z]])


def render_recording(rec, out_path, config=None):
    """Render `rec` (a Recording or an .npz path) to `out_path` (.rrd)."""
    try:
        import rerun as rr
    except ImportError as e:  # optional dependency
        raise ImportError("kairos_visualizer rendering needs rerun-sdk: "
                          "`pip install kairos_visualizer` (or rerun-sdk).") from e
    if not isinstance(rec, Recording):
        rec = Recording.load(rec)
    cfg = config or RenderConfig()
    t0 = rec.t0
    vox_nav = rec.nav_pos.astype(float)
    edges = rec.nav_edges
    N = rec.n_nodes
    arch_colors = [_hex2rgb(c) for c in rec.arch_color_list]

    nbr = [[] for _ in range(N)]
    for i, j in edges:
        nbr[i].append(j); nbr[j].append(i)

    rr.init("kairos_dsg")
    rr.save(str(out_path))
    if cfg.background is not None:
        import rerun.blueprint as rrb
        rr.send_blueprint(rrb.Spatial3DView(origin="/world",
                                            background=tuple(cfg.background),
                                            line_grid=rrb.LineGrid3D(visible=False)))
    SOLID = rr.components.FillMode.Solid
    WIRE = getattr(rr.components.FillMode, "MajorWireframe", SOLID)

    def batch_time(idx, times, fn):
        idx = np.asarray(idx)
        if len(idx) == 0:
            return
        b = np.floor((times[idx] - t0) / cfg.cover_bin_s).astype(int)
        for bb in np.unique(b):
            rr.set_time("time_of_day", duration=float(bb * cfg.cover_bin_s))
            fn(int(bb), idx[b == bb])

    # appearance time per node/occupancy cell (static from t0 if not provided)
    appear = rec.nav_appear_t
    if appear is None:
        appear = np.full(N, t0)
    have = np.where(appear >= 0)[0]

    # ---- z_mesh: Hydra reconstruction mesh (real geometry), static ----
    mv = rec.get("mesh_verts")
    mf = rec.get("mesh_faces")
    have_mesh = mv is not None and mf is not None and len(mv) and len(mf)
    if have_mesh:
        mv = mv.astype(float).copy()
        mv[:, 2] += cfg.z_mesh                    # lift to the mesh layer height
        mc = rec.get("mesh_colors")
        rr.log("world/mesh", rr.Mesh3D(
            vertex_positions=mv, triangle_indices=mf.astype(np.int32),
            vertex_colors=(mc.astype(np.uint8) if mc is not None else None)),
            static=True)

    # ---- z_occupancy: nav-footprint floor, incremental. Only a stand-in for the
    # real geometry, so skip it when the recording carries an actual mesh
    # (otherwise the footprint quads and the mesh render as a confusing double floor).
    def occ_batch(bb, sel):
        v, t, c = _cell_mesh(vox_nav[sel], cfg.voxel_size / 2 - 0.01,
                             cfg.z_occupancy, cfg.occupancy_color)
        rr.log(f"world/occupancy/b{bb}", rr.Mesh3D(vertex_positions=v,
               triangle_indices=t, vertex_colors=c))
    if not have_mesh:
        batch_time(have, appear, occ_batch)

    # ---- occupancy-map walls (occupied cells): incremental, appearing as the
    # robot's coverage reaches them, alongside the free-cell floor ----
    wall = rec.get("wall_xy")
    if wall is not None and len(wall):
        wall = np.asarray(wall, float)
        wt = rec.get("wall_appear_t")
        if wt is None:
            wt = np.full(len(wall), t0)

        def wall_batch(bb, sel):
            v, t, c = _cell_mesh(wall[sel], cfg.voxel_size / 2 - 0.01,
                                 cfg.z_occupancy, cfg.wall_color)
            rr.log(f"world/occupancy/walls/b{bb}", rr.Mesh3D(vertex_positions=v,
                   triangle_indices=t, vertex_colors=c))
        batch_time(np.where(wt >= 0)[0], wt, wall_batch)

    # ---- z_nav: navigational nodes (spheres) + 8-connected edges, incremental ----
    def nav_batch(bb, sel):
        rr.log(f"world/nav/nodes/b{bb}", rr.Points3D(_xyz(vox_nav[sel], cfg.z_nav),
               radii=0.22, colors=cfg.nav_node_color))
    batch_time(have, appear, nav_batch)
    ce = (appear[edges[:, 0]] >= 0) & (appear[edges[:, 1]] >= 0)
    edge_t = np.maximum(appear[edges[:, 0]], appear[edges[:, 1]])

    def edge_batch(bb, sel):
        segs = np.stack([_xyz(vox_nav[edges[sel, 0]], cfg.z_nav),
                         _xyz(vox_nav[edges[sel, 1]], cfg.z_nav)], axis=1)
        rr.log(f"world/nav/edges/b{bb}", rr.LineStrips3D(list(segs),
               colors=cfg.nav_edge_color, radii=0.02))
    batch_time(np.where(ce)[0], edge_t, edge_batch)

    # ---- z_room ----
    if rec.room_xy is not None:
        rm = rec.room_xy
        rr.log("world/room", rr.Points3D([[rm[0], rm[1], cfg.z_room]], radii=1.4,
               colors=(239, 85, 59)), static=True)

    _render_voxels(rr, rec, cfg, SOLID, WIRE, t0)
    _render_robot(rr, rec, cfg, WIRE, t0)
    _render_detections(rr, rec, cfg, WIRE, t0, vox_nav)
    _render_flow(rr, rec, cfg, t0, vox_nav, nbr, arch_colors, N)

    print(f"[kairos_visualizer] wrote {out_path}: {len(rec.flow_times)} snapshots, "
          f"{N} nav nodes, {len(edges)} edges")


def _render_voxels(rr, rec, cfg, SOLID, WIRE, t0):
    if rec.vox_xy is None:
        return
    vox = rec.vox_xy.astype(float)
    count = rec.vox_count; create_t = rec.vox_create_t; active_t = rec.vox_active_t
    if count is None or create_t is None:
        return
    cap = max(1.0, float(np.percentile(count[count > 0], cfg.active_threshold_intensity_pct))
              if (count > 0).any() else 1.0)

    def batches(vidx, times, path, colors, half, mode):
        vidx = np.asarray(vidx)
        if len(vidx) == 0:
            return
        b = np.floor((times[vidx] - t0) / cfg.voxel_bin_s).astype(int)
        for bb in np.unique(b):
            sel = vidx[b == bb]
            rr.set_time("time_of_day", duration=float(bb * cfg.voxel_bin_s))
            rr.log(f"{path}/b{int(bb)}", rr.Boxes3D(
                centers=_xyz(vox[sel], cfg.z_voxels),
                half_sizes=np.tile([half, half, 0.1], (len(sel), 1)),
                colors=colors(sel), fill_mode=mode))

    def active_cols(sel):
        frac = np.clip(count[sel] / cap, 0.0, 1.0)[:, None]
        lo = np.array(cfg.voxel_active_lo); hi = np.array(cfg.voxel_active_hi)
        return (lo * (1 - frac) + hi * frac).astype(np.uint8)

    batches(np.where(create_t >= 0)[0], create_t, "world/voxels/created",
            lambda sel: cfg.voxel_created_color, 0.18, WIRE)
    if active_t is not None:
        batches(np.where(active_t >= 0)[0], active_t, "world/voxels/active",
                active_cols, 0.19, SOLID)


def _render_robot(rr, rec, cfg, WIRE, t0):
    rt = rec.robot_times; rxy = rec.robot_xy
    if rt is None or rxy is None:
        return
    rxy = rxy.astype(float)
    la = 3
    fwd = rxy[np.minimum(np.arange(len(rxy)) + la, len(rxy) - 1)] - rxy
    head = np.arctan2(fwd[:, 1], fwd[:, 0])
    for k in range(1, len(head)):
        if fwd[k, 0] ** 2 + fwd[k, 1] ** 2 < 1e-4:
            head[k] = head[k - 1]
    tri = np.tile(cfg.robot_color, (3, 1)).astype(np.uint8)
    R = rec.active_window_radius
    if R:                                # a see-through blocky sphere shell
        n = int(np.ceil(R / cfg.voxel_size))
        g = np.arange(-n, n + 1) * cfg.voxel_size
        gx, gy, gz = np.meshgrid(g, g, g)
        off = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        dist = np.linalg.norm(off, axis=1)
        shell = off[(dist > R - cfg.voxel_size) & (dist <= R)][::cfg.window_shell_stride]
        rr.log("world/window/cells", rr.Boxes3D(centers=shell,
               half_sizes=np.tile([0.19, 0.19, 0.19], (len(shell), 1)),
               colors=cfg.window_color, fill_mode=WIRE), static=True)
    for k, tt in enumerate(rt):
        rr.set_time("time_of_day", duration=float(tt - t0))
        rr.log("world/robot", rr.Mesh3D(
            vertex_positions=_tri_marker(float(rxy[k, 0]), float(rxy[k, 1]),
                                         float(head[k]), cfg.z_robot),
            triangle_indices=[[0, 1, 2]], vertex_colors=tri))
        if R:
            rr.log("world/window", rr.Transform3D(
                translation=[float(rxy[k, 0]), float(rxy[k, 1]), cfg.z_robot]))


def _render_detections(rr, rec, cfg, WIRE, t0, vox_nav):
    dT = rec.det_times
    if dT is None:
        return
    step = rec.det_step; dx = rec.det_x; dy = rec.det_y; dth = rec.det_th
    din = rec.det_in; dvox = rec.det_vox
    Z = cfg.z_robot
    bounds = np.searchsorted(step, np.arange(len(dT) + 1))

    def arrows(sel, px, py, th, dl, col, r):
        n = int(sel.sum())
        v = np.column_stack([np.cos(th[sel]) * dl, np.sin(th[sel]) * dl, np.zeros(n)])
        return rr.Arrows3D(origins=np.column_stack([px[sel], py[sel], np.full(n, Z)]),
                           vectors=v, colors=col, radii=r)

    for k in range(len(dT)):
        rr.set_time("time_of_day", duration=float(dT[k] - t0))
        a, b = int(bounds[k]), int(bounds[k + 1])
        if b <= a:
            for e in ("detections", "detections/inwin", "detections/hitvox"):
                rr.log(f"world/{e}", rr.Clear(recursive=False))
            continue
        px = dx[a:b]; py = dy[a:b]; th = dth[a:b]; inw = din[a:b]; dv = dvox[a:b]
        out = ~inw
        rr.log("world/detections", arrows(out, px, py, th, 0.7, cfg.detection_color, 0.04)
               if out.any() else rr.Clear(recursive=False))
        if inw.any():
            rr.log("world/detections/inwin",
                   arrows(inw, px, py, th, 1.15, cfg.detection_inwin_color, 0.10))
            hv = np.unique(dv[inw & (dv >= 0)])
            if len(hv):
                rr.log("world/detections/hitvox", rr.Boxes3D(
                    centers=_xyz(vox_nav[hv], cfg.z_voxels),
                    half_sizes=np.tile([0.22, 0.22, 0.13], (len(hv), 1)),
                    colors=cfg.hitvox_color, fill_mode=WIRE))
            else:
                rr.log("world/detections/hitvox", rr.Clear(recursive=False))
        else:
            for e in ("detections/inwin", "detections/hitvox"):
                rr.log(f"world/{e}", rr.Clear(recursive=False))


def _render_flow(rr, rec, cfg, t0, vox_nav, nbr, arch_colors, N):
    ft = rec.flow_times
    dtheta = rec.node_dtheta; drho = rec.node_drho; active = rec.node_active
    ctheta = rec.node_ctheta; cweight = rec.node_cweight; arch = rec.node_arch
    room = rec.room_xy
    S, THRESH, MAXC = cfg.arrow_scale, cfg.component_weight_thresh, cfg.max_components
    RAD, RAD_MINOR = cfg.arrow_radius, cfg.arrow_radius_minor
    for ti in range(len(ft)):
        rr.set_time("time_of_day", duration=float(ft[ti] - t0))
        idx = np.where(active[ti])[0]
        if len(idx) == 0:
            for e in ("nav/flow", "archetypes", "archetypes/edges"):
                rr.log(f"world/{e}", rr.Clear(recursive=False))
            continue
        speed = np.clip(drho[ti], 0.4, 2.0)
        aorg, avec, acol, arad = [], [], [], []
        for i in idx:
            # Static nodes carry no reliable dominant flow; drawing their (grey)
            # arrows clutters and covers the classified ones, so skip them.
            if arch is not None and int(arch[ti, i]) == 0:
                continue
            base = speed[i] * S
            col = arch_colors[int(arch[ti, i])] if arch is not None else (200, 60, 60)
            if ctheta is not None and cweight is not None:
                cw = cweight[ti, i]; cth = ctheta[ti, i]; wsum = float(cw.sum())
            else:
                cw = None; wsum = 0.0
            # Each item is (heading, length, shaft radius, weight relative to the
            # dominant component); the last drives opacity when that is enabled.
            if cfg.arrow_source == "dominant" or wsum <= 0:
                items = [(dtheta[ti, i], base, RAD, 1.0)]
            else:
                p = cw / wsum
                items = [(cth[m], base * max(cfg.arrow_min_frac, p[m] / p[0]),
                          RAD if m == 0 else RAD_MINOR, p[m] / p[0])
                         for m in range(len(cw)) if cw[m] > 0 and (m == 0 or p[m] >= THRESH)]
                items = items[:MAXC] or [(dtheta[ti, i], base, RAD, 1.0)]
            ox, oy = vox_nav[i]
            for aa, ll, rd, frac in items:
                if np.isnan(aa):
                    continue
                aorg.append([ox, oy, cfg.z_arrows])
                avec.append([np.cos(aa) * ll, np.sin(aa) * ll, 0.0])
                rgb = _heading_rgb(aa, cfg) if cfg.arrow_color == "heading" else col
                acol.append(_with_alpha(rgb, frac, cfg) if cfg.arrow_alpha_by_weight
                            else rgb)
                arad.append(rd)
        rr.log("world/nav/flow", rr.Arrows3D(origins=np.array(aorg), vectors=np.array(avec),
               colors=np.array(acol, np.uint8), radii=np.array(arad))
               if aorg else rr.Clear(recursive=False))

        if arch is None or room is None:
            continue
        seen = np.zeros(N, bool); centers, ccols, radii, edges = [], [], [], []
        for i in idx:
            if seen[i]:
                continue
            c = arch[ti, i]; stack, comp = [i], []; seen[i] = True
            while stack:
                nn = stack.pop(); comp.append(nn)
                for m in nbr[nn]:
                    if active[ti, m] and not seen[m] and arch[ti, m] == c:
                        seen[m] = True; stack.append(m)
            cx = float(vox_nav[comp, 0].mean()); cy = float(vox_nav[comp, 1].mean())
            centers.append([cx, cy, cfg.z_archetypes]); ccols.append(arch_colors[int(c)])
            radii.append(0.6)
            if int(c) == 0 and not cfg.static_archetype_edges:
                continue
            for m in comp:
                edges.append([[vox_nav[m, 0], vox_nav[m, 1], cfg.z_nav], [cx, cy, cfg.z_archetypes]])
            edges.append([[cx, cy, cfg.z_archetypes], [room[0], room[1], cfg.z_room]])
        rr.log("world/archetypes", rr.Points3D(centers, colors=np.array(ccols, np.uint8),
               radii=radii))
        rr.log("world/archetypes/edges", rr.LineStrips3D(edges, colors=cfg.arch_edge_color,
               radii=0.02))
