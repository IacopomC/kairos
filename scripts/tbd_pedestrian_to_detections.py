#!/usr/bin/env python3
# Copyright 2026, Iacopo Catalano.
# All Rights Reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Convert TBD `ped_labels_w_time/<sess>_<seg>_t.txt` pedestrian label dumps
into a single sorted `tracks.csv` keyed by wall-clock timestamp, ready to be
fed into the Kairos flow module by `kairos run --tracks`.

Each txt file has space-separated rows with no header:

    timestamp_us  frame  ped_id  x  y

where timestamp_us is a Unix microsecond timestamp, frame is a 0-based
segment-local frame index (0..5999 for 10-min segments at 10 fps), and
(x, y) are pedestrian positions in BEV-world coordinates (metres).

Heading and speed are computed by finite differences along each
pedestrian's trajectory with a small smoothing window. Z is set to a
constant `--z-floor` because TBD's pedestrian tracks are 2D (overhead
camera, ground plane).

Coordinate frames: the txt trajectories are in the BEV camera "world" frame
(defined by the static `real_world.mat` calibration shared across all
sessions). The Kairos voxel grid is in the robot's `map` frame, which is
session-specific (cabot's SLAM picks the origin at startup). Without a
transform, BEV-world detections land in the wrong region of the voxel grid.
By default this script applies the BEV-world -> robot-map transform from
`Robot_data/convert.py` (origin (11.64182, -8.52854) m, yaw (-117.6-61.1)°)
to the trajectories before computing velocities and writing CSV. Pass
`--no-bev-to-map` to skip the transform if you need raw BEV-world coords.

Output schema (one row per pedestrian per frame):

    t_ns,ped_id,x_world,y_world,z_world,theta_rad,rho_mps

Usage (recommended — full accuracy, skips dead-time intervals):

    python3 tbd_pedestrian_to_detections.py \\
        --ped-labels-dir    /path/to/Robot_data/ped_labels_w_time \\
        --sync-file         /path/to/Robot_data/zed_sync_times.txt \\
        --cabot-time-file   /path/to/Robot_data/2022-12/cabot_2022-12-07-12-15-36time.txt \\
        --useful-times-file /path/to/Robot_data/svo_useful_times.txt \\
        --scene-dir         /path/to/scene_tbd_2022-12-07-12-15-36 \\
        --output            /path/to/scene_tbd_2022-12-07-12-15-36/tracks.csv

    --sync-file         selects the correct TBD session and anchors TBD frame
                        indices to wall-clock time via --cabot-time-file.
    --cabot-time-file   maps SVO frame numbers to Unix timestamps (µs).
    --useful-times-file filters pedestrian detections to only the SVO intervals
                        where the robot was actually moving (from svo_useful_times.txt).
                        Requires --cabot-time-file and --sync-file.

Run for all scenes in a loop (all months together):

    ROBOT_DATA=/path/to/Robot_data
    for MONTH in 2022-12 2023-01 2023-02; do
        for SCENE in /path/to/tbd_dataset/$MONTH/scene_tbd_*; do
            DATE=$(basename $SCENE | sed 's/scene_tbd_//')
            python3 tbd_pedestrian_to_detections.py \\
                --ped-labels-dir    $ROBOT_DATA/ped_labels_w_time \\
                --sync-file         $ROBOT_DATA/zed_sync_times.txt \\
                --cabot-time-file   $ROBOT_DATA/$MONTH/cabot_${DATE}time.txt \\
                --useful-times-file $ROBOT_DATA/svo_useful_times.txt \\
                --scene-dir         $SCENE \\
                --output            $SCENE/tracks.csv
        done
    done

Dependencies: numpy.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np


# BEV-world -> robot-map transform, copied verbatim from
# `Robot_data/convert.py:180-183` (the origin Pose). It encodes the BEV
# camera world frame's origin expressed in the robot's `map` frame for the
# building all cabot sessions in this dataset were recorded in. Same for
# every cabot run; calibrated once.
BEV_TO_MAP_TX = 11.64182
BEV_TO_MAP_TY = -8.52854
BEV_TO_MAP_YAW_RAD = (-117.6 - 61.1) / 180.0 * math.pi

# TBD pedestrian labels are at 10 fps → one frame every 100 ms.
_TBD_FRAME_DT_NS: int = 100_000_000


def apply_bev_to_map(xy_world: np.ndarray) -> np.ndarray:
    """Rotate+translate Nx2 (x,y) from BEV world frame into robot map frame.

    Inverse of `convert.py.Pose.transform`: pose_map = R(+yaw) * pose_world + t.
    """
    c = math.cos(BEV_TO_MAP_YAW_RAD)
    s = math.sin(BEV_TO_MAP_YAW_RAD)
    R = np.array([[c, -s], [s, c]])
    return xy_world @ R.T + np.array([BEV_TO_MAP_TX, BEV_TO_MAP_TY])


def parse_sync_file(sync_path: Path) -> dict[str, tuple]:
    """Parse zed_sync_times.txt -> {scene_date: (svo_time_origin, tbd_sess, rgb_time_origin)}.

    Format: <date>, <svo_time_origin>, <tbd_sess>, <rgb_time_origin>
    e.g.    2022-12-07-12-44-46, 8350, 13, 1643
      svo_time_origin : SVO frame at the sync point
      tbd_sess        : TBD overhead-camera session ID
      rgb_time_origin : TBD frame at the sync point (simultaneous with svo_time_origin)
    """
    result = {}
    with sync_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                result[parts[0]] = (int(parts[1]), parts[2], int(parts[3]))
    return result


def read_sync_ts_us(cabot_time_path: Path, svo_time_origin: int) -> int:
    """Return the Unix timestamp (µs) of SVO frame `svo_time_origin` from cabot_*time.txt."""
    with cabot_time_path.open() as f:
        for i, line in enumerate(f):
            if i == svo_time_origin:
                return int(line.split(",")[1].strip())
    raise ValueError(f"Frame {svo_time_origin} not found in {cabot_time_path}")


def load_cabot_time_dict(cabot_time_path: Path) -> dict[int, int]:
    """Load cabot_*time.txt into {frame_number: timestamp_us}.

    Format: `frame_number, timestamp_us` (one line per ZED frame, ~30 fps).
    """
    result = {}
    with cabot_time_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            result[int(parts[0].strip())] = int(parts[1].strip())
    return result


def parse_useful_times(useful_times_path: Path, scene_date: str) -> list[tuple[int, int]]:
    """Parse svo_useful_times.txt and return [(start_svo_frame, end_svo_frame)] for scene_date.

    Format:  cabot_<date>_data3d[suffix]  <TAB>  start-end, start-end, ...
    Returns empty list if scene not found or listed as N/A.
    """
    with useful_times_path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"^(cabot_\S+?_data3d\S*)\s+(.*)", stripped)
            if m is None:
                continue
            col0, rest = m.group(1), m.group(2).strip()
            # Extract date from cabot_<date>_data3d
            dm = re.match(r"^cabot_(.+?)_data3d", col0)
            if dm is None or dm.group(1) != scene_date:
                continue
            if rest == "N/A" or not rest:
                return []
            intervals = []
            for chunk in rest.split(","):
                chunk = chunk.strip()
                if "-" in chunk:
                    a, b = chunk.split("-", 1)
                    intervals.append((int(a.strip()), int(b.strip())))
            return intervals
    return []


def parse_txt_filename(name: str):
    """Pull (sess, seg) out of `<sess>_<seg>_t.txt`. Returns (sess: str, seg: int) or None."""
    m = re.match(r"^(?P<sess>.+?)_(?P<seg>\d+)_t\.txt$", name)
    if m is None:
        return None
    return m.group("sess"), int(m.group("seg"))


def load_txt_file(txt_path: Path):
    """Load a ped_labels_w_time txt file.

    Columns: timestamp_us  frame  ped_id  x  y  (space-separated, no header).
    Returns an Nx5 float64 array.
    """
    return np.loadtxt(str(txt_path), dtype=np.float64)


def smoothed_velocities(xy_m: np.ndarray, dt: float, window: int):
    """Per-row (theta_rad, rho_mps) using centered finite differences smoothed
    over `window` adjacent samples. Output length matches input.
    """
    n = xy_m.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)

    # Centered diff: v[i] = (x[i+1] - x[i-1]) / (2*dt) for interior; endpoints
    # use forward/backward diff.
    v = np.zeros_like(xy_m)
    v[1:-1] = (xy_m[2:] - xy_m[:-2]) / (2.0 * dt)
    v[0] = (xy_m[1] - xy_m[0]) / dt
    v[-1] = (xy_m[-1] - xy_m[-2]) / dt

    if window > 1:
        # Simple moving average on velocity components.
        kernel = np.ones(window, dtype=np.float64) / float(window)
        pad = window // 2
        for axis in range(2):
            padded = np.pad(v[:, axis], pad, mode="edge")
            v[:, axis] = np.convolve(padded, kernel, mode="valid")[: n]

    rho = np.linalg.norm(v, axis=1)
    theta = np.arctan2(v[:, 1], v[:, 0])
    return np.column_stack((theta, rho))


def expand_txt_segment(
    data: np.ndarray,
    *,
    tbd_epoch_ns: int,
    z_floor: float,
    smoothing_window: int,
    apply_bev_to_map_transform: bool,
):
    """Expand one segment's ped_labels_w_time data to per-frame rows.

    `data` is an Nx5 array with columns: timestamp_us, frame, ped_id, x, y.
    Positions (x, y) are already in metres in BEV world frame.

    Output timestamps are derived as:
        t_ns = scene_start_ns + int(frame) * _TBD_FRAME_DT_NS
    The txt-file timestamp column is NOT used for output — its absolute epoch
    differs from the robot recording epoch. The frame column is the global
    (cross-segment) TBD frame index at 10 fps; frame 0 ≈ robot frame 0.

    Yields tuples: (t_ns, ped_id, x_m, y_m, z_m, theta_rad, rho_mps).
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    for ped_id_raw in np.unique(data[:, 2]):
        mask = data[:, 2] == ped_id_raw
        ped_data = data[mask]
        # Sort by frame index to ensure temporal order.
        ped_data = ped_data[np.argsort(ped_data[:, 1])]

        xy_m = ped_data[:, 3:5].copy()
        if ped_data.shape[0] < 2:
            continue

        if apply_bev_to_map_transform:
            # Transform positions BEFORE velocity differentiation so headings
            # come out in map frame too. Equivalent to rotating the velocities
            # afterward, but cleaner.
            xy_m = apply_bev_to_map(xy_m)

        velocities = smoothed_velocities(xy_m, _TBD_FRAME_DT_NS * 1e-9, smoothing_window)

        ped_id = int(ped_id_raw)
        for k in range(ped_data.shape[0]):
            t_ns = tbd_epoch_ns + int(round(ped_data[k, 1])) * _TBD_FRAME_DT_NS
            theta, rho = velocities[k]
            yield (
                t_ns,
                ped_id,
                float(xy_m[k, 0]),
                float(xy_m[k, 1]),
                float(z_floor),
                float(theta),
                float(rho),
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--ped-labels-dir", required=True, type=Path,
                    help="Directory containing <sess>_<seg>_t.txt files.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output tracks.csv path.")
    ap.add_argument("--scene-dir", type=Path, default=None,
                    help=("Path to a FileDataLoader scene directory. "
                          "Required together with --sync-file. Reads poses.csv "
                          "to get the scene epoch (first timestamp = TBD frame 0)."))
    ap.add_argument("--sync-file", type=Path, default=None,
                    help=("Path to zed_sync_times.txt. When given with --scene-dir, "
                          "only the TBD session matching this scene is loaded, "
                          "ensuring correct pedestrian-to-scene pairing."))
    ap.add_argument("--cabot-time-file", type=Path, default=None,
                    help=("Path to cabot_<time>time.txt. Used with --sync-file to anchor "
                          "TBD frame indices to real wall-clock time (col 1/col 3 of sync file)."))
    ap.add_argument("--useful-times-file", type=Path, default=None,
                    help=("Path to svo_useful_times.txt. Filters detections to only the SVO "
                          "intervals where the robot was moving. Requires --cabot-time-file "
                          "and --sync-file."))
    ap.add_argument("--smoothing-window", type=int, default=3,
                    help="Moving-average window size on velocity (samples).")
    ap.add_argument("--z-floor", type=float, default=0.0,
                    help="Z value (meters) assigned to all 2D pedestrian samples.")
    ap.add_argument("--max-segments", type=int, default=None,
                    help="Cap on number of segments processed (debug).")
    ap.add_argument("--no-bev-to-map", dest="bev_to_map", action="store_false",
                    help=("Skip the BEV-world -> robot-map transform from "
                          "convert.py. Default applies it; without it, "
                          "tracks land in the BEV calibration frame, NOT in "
                          "the cabot map frame the hydra voxel grid lives in."))
    ap.set_defaults(bev_to_map=True)
    args = ap.parse_args()

    if not args.ped_labels_dir.is_dir():
        sys.exit(f"ERR: --ped-labels-dir not found: {args.ped_labels_dir}")

    if args.bev_to_map:
        print(
            f"[transform] applying BEV-world -> robot-map: "
            f"t=({BEV_TO_MAP_TX}, {BEV_TO_MAP_TY}) m, "
            f"yaw={math.degrees(BEV_TO_MAP_YAW_RAD):.2f} deg"
        )
    else:
        print("[transform] skipping BEV->map (raw BEV coords; --no-bev-to-map)")

    # --- Scene epoch from poses.csv ---
    scene_start_ns: int | None = None
    scene_end_ns: int | None = None
    if args.scene_dir is not None:
        poses_path = args.scene_dir / "poses.csv"
        if not poses_path.is_file():
            sys.exit(f"ERR: poses.csv not found in scene-dir: {poses_path}")
        with poses_path.open() as f:
            f.readline()  # header
            first_row = f.readline().strip()
            last_row = first_row
            for line in f:
                line = line.strip()
                if line:
                    last_row = line
        if not first_row:
            sys.exit(f"ERR: poses.csv is empty: {poses_path}")
        scene_start_ns = int(first_row.split(",")[0])
        scene_end_ns = int(last_row.split(",")[0])
        duration_s = (scene_end_ns - scene_start_ns) * 1e-9
        print(f"[scene] epoch: {scene_start_ns} ns, duration {duration_s:.1f}s "
              f"(≈ {int(duration_s / (_TBD_FRAME_DT_NS * 1e-9))} TBD frames)")

    # --- Session filter from zed_sync_times.txt ---
    scene_date: str | None = (
        re.sub(r"^scene_tbd_", "", args.scene_dir.name) if args.scene_dir is not None else None
    )
    tbd_sess_filter: str | None = None
    tbd_epoch_ns: int | None = None
    if args.sync_file is not None:
        if args.scene_dir is None:
            sys.exit("ERR: --sync-file requires --scene-dir")
        if not args.sync_file.is_file():
            sys.exit(f"ERR: --sync-file not found: {args.sync_file}")
        sess_map = parse_sync_file(args.sync_file)
        if scene_date not in sess_map:  # type: ignore[operator]
            print(f"[sync] scene '{scene_date}' has no TBD session in {args.sync_file.name} — "
                  f"writing empty tracks.csv.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w") as f:
                f.write("t_ns,ped_id,x_world,y_world,z_world,theta_rad,rho_mps\n")
            return 0
        svo_time_origin, tbd_sess_filter, rgb_time_origin = sess_map[scene_date]
        print(f"[sync] scene '{scene_date}' → TBD session '{tbd_sess_filter}' "
              f"(svo_origin={svo_time_origin}, tbd_frame_origin={rgb_time_origin})")
        if args.cabot_time_file is not None:
            if not args.cabot_time_file.is_file():
                sys.exit(f"ERR: --cabot-time-file not found: {args.cabot_time_file}")
            sync_ts_us = read_sync_ts_us(args.cabot_time_file, svo_time_origin)
            tbd_epoch_ns = (sync_ts_us - rgb_time_origin * 100_000) * 1_000
            print(f"[sync] sync_ts={sync_ts_us} µs → tbd_epoch_ns={tbd_epoch_ns}")
    elif args.scene_dir is not None:
        print("[sync] WARNING: --sync-file not given; loading ALL sessions. "
              "Pass --sync-file zed_sync_times.txt for correct scene pairing.")

    # --- Collect txt files, optionally filtered by session ---
    txt_files = []
    for p in sorted(args.ped_labels_dir.glob("*_t.txt")):
        parsed = parse_txt_filename(p.name)
        if parsed is None:
            print(f"  skip (unrecognized name): {p.name}")
            continue
        sess, seg = parsed
        if tbd_sess_filter is not None and sess != tbd_sess_filter:
            continue
        txt_files.append((sess, seg, p))

    if not txt_files:
        sys.exit(f"ERR: no <sess>_<seg>_t.txt files found "
                 f"(session filter: {tbd_sess_filter or 'none'})")

    if args.max_segments is not None:
        txt_files = txt_files[: args.max_segments]

    # Compute the effective TBD epoch and frame window for this scene.
    # tbd_epoch_ns is the real wall-clock ns of TBD frame 0 (requires --cabot-time-file).
    # Without it, falls back to scene_start_ns (frame 0 ≈ scene start — approximate).
    import math as _math
    effective_epoch_ns: int = tbd_epoch_ns if tbd_epoch_ns is not None else (scene_start_ns or 0)

    scene_frame_min: int | None = None
    scene_frame_max: int | None = None
    if scene_start_ns is not None and scene_end_ns is not None:
        scene_frame_min = max(0, _math.ceil((scene_start_ns - effective_epoch_ns) / _TBD_FRAME_DT_NS))
        scene_frame_max = _math.floor((scene_end_ns - effective_epoch_ns) / _TBD_FRAME_DT_NS)
        print(f"[scene] TBD frame window: [{scene_frame_min}, {scene_frame_max}] "
              f"({scene_frame_max - scene_frame_min} frames)")

    # --- Useful SVO intervals -> per-segment TBD frame ranges ---
    # Each entry is (tbd_frame_min, tbd_frame_max) for one useful SVO interval.
    useful_tbd_intervals: list[tuple[int, int]] | None = None
    if args.useful_times_file is not None:
        if args.cabot_time_file is None:
            sys.exit("ERR: --useful-times-file requires --cabot-time-file")
        if tbd_epoch_ns is None:
            sys.exit("ERR: --useful-times-file requires --sync-file + --cabot-time-file for epoch anchoring")
        if not args.useful_times_file.is_file():
            sys.exit(f"ERR: --useful-times-file not found: {args.useful_times_file}")
        svo_intervals = parse_useful_times(args.useful_times_file, scene_date)  # type: ignore[arg-type]
        if not svo_intervals:
            print(f"[useful-times] no intervals for '{scene_date}' — keeping full window")
        else:
            cabot_times = load_cabot_time_dict(args.cabot_time_file)
            useful_tbd_intervals = []
            for svo_s, svo_e in svo_intervals:
                ts_s = cabot_times.get(svo_s)
                ts_e = cabot_times.get(svo_e)
                if ts_s is None or ts_e is None:
                    print(f"  WARNING: SVO frames {svo_s}-{svo_e} missing from cabot_time, skipped")
                    continue
                f_s = int(_math.ceil((ts_s * 1_000 - tbd_epoch_ns) / _TBD_FRAME_DT_NS))
                f_e = int(_math.floor((ts_e * 1_000 - tbd_epoch_ns) / _TBD_FRAME_DT_NS))
                useful_tbd_intervals.append((f_s, f_e))
            print(f"[useful-times] {len(useful_tbd_intervals)} intervals mapped to TBD frames: "
                  + ", ".join(f"[{a},{b}]" for a, b in useful_tbd_intervals))

    rows = []
    for sess, seg, p in txt_files:
        print(f"[{p.name}] loading...")
        data = load_txt_file(p)
        if data.size == 0:
            print(f"  empty, skipping")
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Filter by TBD frame index (column 1).
        # With --useful-times-file: union of per-interval masks (skips dead-time gaps).
        # Without it: single [scene_frame_min, scene_frame_max] window.
        if useful_tbd_intervals is not None:
            mask = np.zeros(data.shape[0], dtype=bool)
            for f_s, f_e in useful_tbd_intervals:
                mask |= (data[:, 1] >= f_s) & (data[:, 1] <= f_e)
            data = data[mask]
        elif scene_frame_min is not None and scene_frame_max is not None:
            mask = (data[:, 1] >= scene_frame_min) & (data[:, 1] <= scene_frame_max)
            data = data[mask]
        if data.shape[0] == 0:
            print(f"  no frames in scene window, skipping")
            continue

        n_before = len(rows)
        for row in expand_txt_segment(
            data,
            tbd_epoch_ns=effective_epoch_ns,
            z_floor=args.z_floor,
            smoothing_window=args.smoothing_window,
            apply_bev_to_map_transform=args.bev_to_map,
        ):
            rows.append(row)
        n_peds = len(np.unique(data[:, 2]))
        print(f"  pedestrians={n_peds} samples_added={len(rows) - n_before}")

    if not rows:
        sys.exit("ERR: no detection samples generated")

    rows.sort(key=lambda r: (r[0], r[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("t_ns,ped_id,x_world,y_world,z_world,theta_rad,rho_mps\n")
        for r in rows:
            f.write(
                f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},"
                f"{r[5]:.6f},{r[6]:.6f}\n"
            )

    t_min_s = rows[0][0] * 1e-9
    t_max_s = rows[-1][0] * 1e-9
    print(f"\nWrote {len(rows)} samples to {args.output}")
    print(f"  time span: {t_min_s:.3f}s ... {t_max_s:.3f}s "
          f"(duration {t_max_s - t_min_s:.1f}s)")
    print(f"  unique pedestrians: {len({r[1] for r in rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
