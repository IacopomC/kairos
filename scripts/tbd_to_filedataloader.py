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
Convert one scene of the CMU TBD pedestrian dataset (CaBot robot, ZED2) into
the Hydra `FileDataLoader` folder layout documented in
`doc/non_ros_usage.md`.

A TBD scene on disk:

    cabot_<time>_data3d/
        rgb/NNNNNN.jpg          # 1280x720, sparse (RGB only at some frames)
        depth/NNNNNN.png        # 1280x720, int32 depth in mm, dense
        timestamps.txt          # `# out_idx svo_frame ts_ms` then rows
    cabot_<time>-trajectory.yaml  # cartographer trajectory (base_link in map)

The RGB folder is the bottleneck (typically ~6% of frames are extracted), so
we iterate it and look up the matching depth + nearest pose for each.

Pipeline:
    1. Parse cabot_*-trajectory.yaml into Nx8 (t_ns, x, y, z, qw, qx, qy, qz).
    2. Parse timestamps.txt into {out_idx -> ts_ns}.
    3. For each rgb_<out_idx>.jpg:
         - read matching depth_<out_idx>.png
         - find nearest robot pose by ts_ns (within --ts-tolerance-ms)
         - apply REP-103 rotation to convert base_link pose -> optical pose
         - write rgb_NNNNNNN.png, depth_NNNNNNN.tiff (uint16 mm),
           labels_NNNNNNN.png (zeros; a labels file already present is kept).
    4. Write camera_info.yaml (ZED2 HD720 left intrinsics) and poses.csv.

Usage (full extraction):
    python3 tbd_to_filedataloader.py \
        --data3d     ./Robot_data/2022-12/cabot_2022-12-07-12-15-36_data3d \
        --trajectory ./Robot_data/2022-12/cabot_2022-12-07-12-15-36-trajectory.yaml \
        --output     ./scene_tbd_2022-12-07-12-15-36 \
        [--max-frames 500] [--ts-tolerance-ms 100]

Regenerate poses.csv only (no images written), with data3d available:
    python3 tbd_to_filedataloader.py \
        --data3d     ./Robot_data/2022-12/cabot_2022-12-07-12-15-36_data3d \
        --trajectory ./Robot_data/2022-12/cabot_2022-12-07-12-15-36-trajectory.yaml \
        --output     ./scene_tbd_2022-12-07-12-15-36 \
        --pose-only

Regenerate poses.csv only, without data3d folder (only timestamps.txt needed):
    python3 tbd_to_filedataloader.py \
        --timestamps-file ./Robot_data/2022-12/cabot_2022-12-07-12-15-36_data3d/timestamps.txt \
        --trajectory      ./Robot_data/2022-12/cabot_2022-12-07-12-15-36-trajectory.yaml \
        --output          ./scene_tbd_2022-12-07-12-15-36 \
        --pose-only

After full extraction, run scripts/generate_labels.py on the output dir to populate
ade20k_outdoor semantic labels (the converter writes zeros).

Dependencies: numpy, pyyaml, opencv-python, scipy, pillow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Calibration constants (provided by the user).
# Intrinsics: ZED2 HD720 left camera factory calibration.
# Extrinsics: base_link -> zed_left_camera_optical.
#   - Rotation:    REP-103 (base x-fwd, y-left, z-up) -> optical (x-right,
#                  y-down, z-fwd). Quaternion in xyzw form, well-known constant.
#   - Translation: 0,0,0 placeholder. The TBD release ships no static TF for
#                  base_link -> ZED; refine when CaBot URDF is available.
# ---------------------------------------------------------------------------
ZED2_HD720_INTRINSICS = {
    "fx": 521.6384887695312,
    "fy": 521.6384887695312,
    "cx": 637.798828125,
    "cy": 349.9053955078125,
    "width": 1280,
    "height": 720,
}
T_BASE_TO_OPTICAL = np.array([0.0, 0.0, 0.0])
Q_BASE_TO_OPTICAL_XYZW = np.array([-0.5, 0.5, -0.5, 0.5])


# ============================================================================
# Trajectory parsing
# ============================================================================

def load_trajectory(path: Path) -> np.ndarray:
    """
    Return Nx8 array sorted by timestamp:
        [t_ns, x, y, z, qw, qx, qy, qz]

    The yaml has the cartographer wire format:
        trajectory:
          - header: {stamp: {secs: ..., nsecs: ...}, frame_id: "map"}
            pose:
              position: {x, y, z}
              orientation: {x, y, z, w}     <-- xyzw ROS order
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    nodes = data.get("trajectory", [])
    if not nodes:
        raise ValueError(f"no trajectory nodes in {path}")

    rows = []
    for node in nodes:
        stamp = node["header"]["stamp"]
        t_ns = int(stamp["secs"]) * 1_000_000_000 + int(stamp["nsecs"])
        p = node["pose"]["position"]
        o = node["pose"]["orientation"]
        # cartographer publishes xyzw; convert to wxyz for storage.
        rows.append((t_ns, p["x"], p["y"], p["z"],
                     o["w"], o["x"], o["y"], o["z"]))

    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    # Normalise quaternions defensively.
    qn = np.linalg.norm(arr[:, 4:8], axis=1, keepdims=True)
    arr[:, 4:8] /= np.clip(qn, 1e-9, None)
    return arr


# ============================================================================
# Frame index <-> timestamp map
# ============================================================================

def load_timestamps(path: Path) -> dict[int, int]:
    """
    Parse timestamps.txt with header `# out_idx svo_frame ts_ms`.
    Return {out_idx -> ts_ns}.
    """
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            out_idx = int(parts[0])
            ts_ms = int(parts[2])
            out[out_idx] = ts_ms * 1_000_000  # ms -> ns
    return out


# ============================================================================
# Pose composition + matching
# ============================================================================

_R_BASE_OPTICAL = R.from_quat(Q_BASE_TO_OPTICAL_XYZW)


def compose_world_optical(qwxyz_base: np.ndarray, txyz_base: np.ndarray):
    """
    world_T_optical = world_T_base * base_T_optical, where base_T_optical is
    a pure REP-103 rotation (no translation). Inputs are wxyz/xyz arrays.

    Returns (q_wxyz, t_xyz) for world -> optical.
    """
    # scipy expects xyzw
    q_xyzw = np.array([qwxyz_base[1], qwxyz_base[2], qwxyz_base[3], qwxyz_base[0]])
    R_world_base = R.from_quat(q_xyzw)
    R_world_optical = R_world_base * _R_BASE_OPTICAL
    t_world_optical = R_world_base.apply(T_BASE_TO_OPTICAL) + txyz_base
    qx, qy, qz, qw = R_world_optical.as_quat()
    return np.array([qw, qx, qy, qz]), t_world_optical


def nearest_pose_idx(pose_ts_ns: np.ndarray, query_ns: int) -> int:
    i = int(np.searchsorted(pose_ts_ns, query_ns))
    if i == 0:
        return 0
    if i == len(pose_ts_ns):
        return len(pose_ts_ns) - 1
    return i if abs(pose_ts_ns[i] - query_ns) < abs(pose_ts_ns[i - 1] - query_ns) else i - 1


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--data3d", required=False, default=None, type=Path,
                    help="Path to cabot_<time>_data3d/ (contains rgb/, depth/, timestamps.txt). "
                         "Optional when --pose-only is set together with --timestamps-file.")
    ap.add_argument("--trajectory", required=True, type=Path,
                    help="Path to cabot_<time>-trajectory.yaml.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory in FileDataLoader layout.")
    ap.add_argument("--timestamps-file", type=Path, default=None,
                    help="Path to timestamps.txt. Used with --pose-only when --data3d is absent.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap on frames written (useful for smoke testing).")
    ap.add_argument("--ts-tolerance-ms", type=float, default=100.0,
                    help="Max allowed pose<->frame timestamp mismatch (TBD trajectory is ~10 Hz).")
    ap.add_argument("--pose-only", action="store_true",
                    help="Only write poses.csv and camera_info.yaml; skip image extraction.")
    args = ap.parse_args()

    if not args.trajectory.is_file():
        sys.exit(f"ERR: trajectory yaml not found: {args.trajectory}")

    # Resolve timestamps.txt and optional image directories.
    if args.data3d is not None:
        if not args.data3d.is_dir():
            sys.exit(f"ERR: data3d not found: {args.data3d}")
        rgb_dir = args.data3d / "rgb"
        depth_dir = args.data3d / "depth"
        ts_file = args.data3d / "timestamps.txt"
        if args.pose_only:
            if not ts_file.exists():
                sys.exit(f"ERR: timestamps.txt not found: {ts_file}")
        else:
            for p in (rgb_dir, depth_dir, ts_file):
                if not p.exists():
                    sys.exit(f"ERR: expected {p} inside data3d")
    else:
        if not args.pose_only:
            sys.exit("ERR: --data3d is required unless --pose-only is set")
        if args.timestamps_file is None:
            sys.exit("ERR: --pose-only without --data3d requires --timestamps-file")
        if not args.timestamps_file.is_file():
            sys.exit(f"ERR: --timestamps-file not found: {args.timestamps_file}")
        ts_file = args.timestamps_file
        rgb_dir = None
        depth_dir = None

    args.output.mkdir(parents=True, exist_ok=True)
    tol_ns = int(args.ts_tolerance_ms * 1e6)

    print(f"[1/4] Loading trajectory: {args.trajectory}")
    poses = load_trajectory(args.trajectory)
    pose_ts_ns = poses[:, 0].astype(np.int64)
    print(f"      {len(poses):,} poses; "
          f"t0={pose_ts_ns[0] / 1e9:.3f}s  tN={pose_ts_ns[-1] / 1e9:.3f}s")

    print(f"[2/4] Loading timestamps: {ts_file}")
    idx_to_ts_ns = load_timestamps(ts_file)
    print(f"      {len(idx_to_ts_ns):,} (out_idx -> ts) pairs")

    print(f"[3/4] Writing camera_info.yaml")
    intr = dict(ZED2_HD720_INTRINSICS)
    with open(args.output / "camera_info.yaml", "w") as f:
        yaml.safe_dump(intr, f, default_flow_style=False)

    h, w = intr["height"], intr["width"]
    labels_blank = np.zeros((h, w), dtype=np.uint16)
    n_labels_kept = 0

    pose_rows = []
    frame_idx = 0
    skipped_no_depth = 0
    skipped_no_ts = 0
    skipped_no_pose = 0
    skipped_size_mismatch = 0

    if rgb_dir is not None:
        print(f"[4/4] Iterating {rgb_dir}")
        frame_iter = sorted(rgb_dir.glob("*.jpg"))
        if not frame_iter:
            sys.exit(f"ERR: no rgb/*.jpg in {rgb_dir}")
    else:
        print(f"[4/4] Iterating timestamps (pose-only, no data3d)")
        # Iterate all entries in timestamps.txt as sentinel Path objects keyed by out_idx.
        frame_iter = sorted(idx_to_ts_ns.keys())

    for item in frame_iter:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        if rgb_dir is not None:
            rgb_path = item
            try:
                out_idx = int(rgb_path.stem)
            except ValueError:
                print(f"      skip (non-numeric stem): {rgb_path.name}")
                continue
        else:
            out_idx = item

        ts_ns = idx_to_ts_ns.get(out_idx)
        if ts_ns is None:
            skipped_no_ts += 1
            continue

        if not args.pose_only:
            depth_path = depth_dir / f"{out_idx:06d}.png"
            if not depth_path.exists():
                skipped_no_depth += 1
                continue

        # Pose matching
        j = nearest_pose_idx(pose_ts_ns, ts_ns)
        if abs(int(pose_ts_ns[j]) - ts_ns) > tol_ns:
            skipped_no_pose += 1
            continue

        # Compose world_T_optical
        q_wxyz_base = poses[j, 4:8]
        t_xyz_base = poses[j, 1:4]
        q_wxyz_opt, t_xyz_opt = compose_world_optical(q_wxyz_base, t_xyz_base)

        if not args.pose_only:
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)  # BGR uint8
            depth_raw = np.array(Image.open(depth_path))       # int32 mm

            if rgb is None or depth_raw is None:
                print(f"      skip (read failure): {rgb_path.name}")
                continue
            if rgb.shape[:2] != (h, w) or depth_raw.shape != (h, w):
                skipped_size_mismatch += 1
                continue

            depth_mm = np.clip(depth_raw, 0, 65535).astype(np.uint16)

            stem = f"{frame_idx:07d}"
            cv2.imwrite(str(args.output / f"rgb_{stem}.png"), rgb)
            cv2.imwrite(str(args.output / f"depth_{stem}.tiff"), depth_mm)
            label_path = args.output / f"labels_{stem}.png"
            if label_path.exists():
                n_labels_kept += 1   # keep labels already generated for this frame
            else:
                cv2.imwrite(str(label_path), labels_blank)

        pose_rows.append((
            int(ts_ns),
            float(t_xyz_opt[0]), float(t_xyz_opt[1]), float(t_xyz_opt[2]),
            float(q_wxyz_opt[0]), float(q_wxyz_opt[1]),
            float(q_wxyz_opt[2]), float(q_wxyz_opt[3]),
        ))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"      ... {frame_idx} frames written")

    with open(args.output / "poses.csv", "w") as f:
        f.write("timestamp_ns,px,py,pz,qw,qx,qy,qz\n")
        for row in pose_rows:
            f.write(f"{row[0]},{row[1]:.9g},{row[2]:.9g},{row[3]:.9g},"
                    f"{row[4]:.9g},{row[5]:.9g},{row[6]:.9g},{row[7]:.9g}\n")

    print(f"\nDone. Wrote {frame_idx} frames to {args.output}")
    if skipped_no_ts:
        print(f"  skipped {skipped_no_ts} rgb frames with no timestamps.txt entry")
    if skipped_no_depth:
        print(f"  skipped {skipped_no_depth} rgb frames with no matching depth")
    if skipped_no_pose:
        print(f"  skipped {skipped_no_pose} rgb frames with no pose within {args.ts_tolerance_ms} ms")
    if skipped_size_mismatch:
        print(f"  skipped {skipped_size_mismatch} rgb frames with unexpected image size")

    print(f"\nNext steps:")
    if n_labels_kept:
        print(f"  1) Generate semantic labels for the new frames ({n_labels_kept} "
              f"existing labels_*.png kept; blank placeholders are filled):")
        print(f"     python scripts/generate_labels.py {args.output}")
    else:
        print(f"  1) Generate semantic labels (replaces the blank labels_*.png):")
        print(f"     python scripts/generate_labels.py {args.output} --overwrite")
    print(f"  2) Run the pipeline (the tbd config is in the kairos repo):")
    print(f"     kairos run {args.output} -c tbd -l ade20k_outdoor -m 100 -o <out_dir> --no-publish --no-enable-lcd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
