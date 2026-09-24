"""The `kairos run` command: run the Kairos pipeline incrementally over a scene.

Builds the scene graph and trains the flow model, writing `dsg.json`,
`flow_state.bin` and the resolved config to `<output>/<scene name>/`. Each
scene runs in its own pipeline; a multi-session deployment is reproduced by
running the sessions in order, each with the previous session's
`<output>/<scene name>/temporal/flow_state.bin` as `--load-flow-state`.

    kairos run /data/tbd/scene_01 -c tbd -l ade20k_outdoor \
        -o /out/run01 --no-publish --no-enable-lcd

`-c` names a file in `config/datasets/` without the `.yaml`; `-l` names one in
`config/label_spaces/` without the `_label_space.yaml`. `kairos run --help`
lists every option. See the repository README for the Docker wrappers.
"""

import csv
import os
import pathlib
import sys
import time

import click
import hydra_python
import spark_dsg
from hydra_python.commands import resolve_output_path
from hydra_python.data_loader import DataLoader, DataLoaderIter
from kairos_python.pipeline import load_pipeline, get_config_path, load_configs, viz_params_from_config


def _default_orb_vocab():
    """Discover an ORB vocabulary for the offline LCD bow producer (so LCD
    actually fires instead of logging "0 loop closures"). Returns the first
    existing candidate path, else None."""
    candidates = []
    env = os.environ.get("KAIROS_ORB_VOCAB")
    if env:
        candidates.append(pathlib.Path(env))
    try:
        cfg = get_config_path()  # <repo>/kairos/config (devel) or <pkg>/config
        candidates += [cfg.parent.parent / "orb_vocab.txt",
                       cfg.parent.parent.parent / "orb_vocab.txt"]
    except Exception:
        pass
    candidates += [pathlib.Path("/ws/src/orb_vocab.txt"),
                   pathlib.Path.home() / "3dsg-kairos" / "orb_vocab.txt"]
    for c in candidates:
        try:
            if c and c.is_file():
                return str(c)
        except OSError:
            pass
    return None


def _load_tracks(path):
    """Load tracks.csv (output of tbd_pedestrian_to_detections.py) grouped by t_ns.

    Returns sorted list of (t_ns, xyz_world, theta, rho, ped_ids). ped_ids is a
    per-detection person id when the CSV carries a `ped_id` column (TBD does),
    else None so downstream consumers fall back to raw per-sample handling.
    """
    grouped = {}
    with pathlib.Path(path).expanduser().open() as f:
        reader = csv.DictReader(f)
        has_pid = reader.fieldnames is not None and "ped_id" in reader.fieldnames
        for row in reader:
            t_ns = int(row["t_ns"])
            entry = grouped.setdefault(
                t_ns, {"xyz": [], "theta": [], "rho": [], "ped_id": []})
            entry["xyz"].append(
                (float(row["x_world"]), float(row["y_world"]), float(row["z_world"]))
            )
            entry["theta"].append(float(row["theta_rad"]))
            entry["rho"].append(float(row["rho_mps"]))
            if has_pid:
                entry["ped_id"].append(int(row["ped_id"]))
    return [
        (t_ns, grouped[t_ns]["xyz"], grouped[t_ns]["theta"], grouped[t_ns]["rho"],
         grouped[t_ns]["ped_id"] if has_pid else None)
        for t_ns in sorted(grouped)
    ]


def _first_frame_ts_ns(data):
    """Peek the first frame's timestamp without consuming the iterator."""
    data.reset()
    packet = data.next()
    if packet is None:
        return None
    data.reset()
    return int(packet.timestamp)


def _maybe_run_scoring(
    pipeline, score_tracks_path, scene_output_path, label, holdout_frac=0.0,
    elapsed_s=None,
):
    """Drive score_csv + record_stats over a test tracks file at the end of a
    run. No-op when score_tracks_path is None.

    Writes preds.csv directly under scene_output_path, so a scorer finds it
    without needing to know the temporal subdir layout. The matching
    `preds.stats.json` sidecar lands in the same directory.
    """
    if not score_tracks_path:
        return
    if scene_output_path is None:
        raise click.ClickException("--score-tracks requires -o/--output-path")
    if not _resolve_bench_dir():
        raise click.ClickException(
            "--score-tracks needs the kairos-suite checkout: place it next to "
            "kairos in the workspace src/, or set KAIROS_SUITE_DIR to its "
            "evaluation/ directory")
    try:
        from score_helper import score_csv  # noqa: E402
        from run_stats import record_stats  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"failed to import kairos-suite helpers: {exc}")

    scene_output_path.mkdir(parents=True, exist_ok=True)
    preds_path = scene_output_path / "preds.csv"
    click.secho(
        f"[score] running score_csv → {preds_path}", fg="cyan"
    )
    try:
        stats = score_csv(
            pipeline,
            test_tracks=str(score_tracks_path),
            output=str(preds_path),
            holdout_frac=holdout_frac,
        )
        click.echo(
            f"[score] matched {stats['n_matched']}/{stats['n_total']} "
            f"({stats['match_rate']:.1%})"
            + (
                f" (held-out tail {holdout_frac:.0%}, "
                f"skipped {stats['n_skipped_holdout']})"
                if holdout_frac > 0.0
                else ""
            )
        )
        record_stats(
            pipeline,
            predictions_csv=str(preds_path),
            label=label,
            elapsed_s=elapsed_s,
        )
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"scoring failed: {exc}")


def _resolve_bench_dir():
    """Locate the benchmark evaluation package on sys.path, mirroring
    _maybe_run_scoring's resolution order (env override → already-importable →
    walk up the tree). The scoring helpers live in kairos-suite/evaluation/.
    KAIROS_SUITE_DIR, if set, must point at the directory containing
    score_helper.py (i.e. .../kairos-suite/evaluation).
    Returns True if run_stats becomes importable, False otherwise."""
    bench_dir = os.environ.get("KAIROS_SUITE_DIR")
    if not bench_dir:
        for parent in pathlib.Path(__file__).resolve().parents:
            candidate = parent / "kairos-suite" / "evaluation" / "score_helper.py"
            if candidate.is_file():
                bench_dir = str(candidate.parent)
                break
    if bench_dir and bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    return bench_dir is not None


def _maybe_record_train_stats(pipeline, scene_output_path, label, n_pushed, elapsed_s=None):
    """Snapshot runtime/footprint after a training-only run (no --score-tracks).

    The add_observation timing accumulates during the training push loop. The
    scoring path's record_stats only fires when scoring, and a scoring run
    pushes no detections — so without this call the per-detection model-update
    cost the paper's Runtime column needs would never be recorded in the
    held-out protocol's training phase. Writes a `train.stats.json` sidecar
    next to flow_state.bin (distinct from the scoring `preds.stats.json`, so
    nothing is clobbered). No-op when nothing was pushed or no output dir.
    """
    if scene_output_path is None or n_pushed <= 0:
        return
    try:
        if not _resolve_bench_dir():
            click.secho(
                "[stats] kairos-suite not found; skipping train stats.",
                fg="yellow",
            )
            return
        from run_stats import record_stats  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        click.secho(f"[stats] failed to import run_stats: {exc}", fg="red")
        return

    scene_output_path.mkdir(parents=True, exist_ok=True)
    # record_stats appends `.stats.json` to a non-.csv path → train.stats.json.
    stats_stub = scene_output_path / "train"
    try:
        rec = record_stats(pipeline, predictions_csv=str(stats_stub), label=label,
                           elapsed_s=elapsed_s)
        timing = rec.get("flow_timing", {})
        click.secho(
            f"[stats] wrote {stats_stub}.stats.json "
            f"(add_observation_mean_ms={timing.get('add_observation_mean_ms')}, "
            f"sharing_mean_ms={timing.get('sharing_mean_ms')})",
            fg="cyan",
        )
    except Exception as exc:  # noqa: BLE001
        click.secho(f"[stats] record_stats failed: {exc}", fg="red")


def _run_one(
    scene_path,
    config_name,
    labelspace,
    output_path,
    config_verbosity,
    place_feature_strategy,
    publish,
    progress,
    max_steps,
    callbacks,
    tracks_path,
    auto_align_tracks,
    tracks_time_shift_ns,
    enable_lcd,
    load_flow_state,
    score_tracks_path=None,
    score_label=None,
    score_holdout_frac=0.0,
    mem_profile_csv=None,
    mem_profile_every=50,
    dsg_dump_dir=None,
):
    scene_output_path = (
        None if output_path is None else output_path / scene_path.stem
    )
    data = hydra_python.get_dataloader(scene_path)

    tracks = []
    shift_ns = int(tracks_time_shift_ns)
    if tracks_path is not None:
        tracks = _load_tracks(tracks_path)
        if not tracks:
            click.secho(f"[WARN] tracks file is empty: {tracks_path}", fg="yellow")
        else:
            t_first = tracks[0][0]
            t_last = tracks[-1][0]
            n_samples = sum(len(t[1]) for t in tracks)
            click.echo(
                f"[tracks] {len(tracks)} frame-times, {n_samples} samples, "
                f"span {(t_last - t_first) * 1e-9:.1f}s"
            )

            frame0_ns = _first_frame_ts_ns(data)
            if frame0_ns is None:
                click.secho("[WARN] data loader returned no frames", fg="yellow")
            else:
                offset_s = (frame0_ns - t_first) * 1e-9
                click.echo(
                    f"[tracks] first track t_ns={t_first}, "
                    f"first frame t_ns={frame0_ns}, "
                    f"offset (frame - track) = {offset_s:+.3f}s"
                )
                if auto_align_tracks:
                    auto_shift = frame0_ns - t_first
                    click.secho(
                        f"[tracks] --auto-align-tracks: shifting all tracks by "
                        f"{auto_shift * 1e-9:+.3f}s",
                        fg="cyan",
                    )
                    shift_ns += auto_shift
                elif abs(offset_s) > 1.0:
                    click.secho(
                        "[tracks] tracks and frames don't share an epoch "
                        "(off by >1s). Pass --auto-align-tracks (smoke test) "
                        "or --tracks-time-shift-ns N (manual) or regenerate "
                        "tracks.csv with --scene-dir to derive the right "
                        "epoch — otherwise no detections will be drained.",
                        fg="yellow",
                    )

    pipeline = load_pipeline(
        data,
        config_name,
        labelspace,
        output_path=scene_output_path,
        config_verbosity=config_verbosity,
        place_feature_strategy=place_feature_strategy,
        zmq_url=None if not publish else "tcp://127.0.0.1:8001",
        enable_lcd=enable_lcd,
    )

    if not pipeline:
        raise click.ClickException(
            f"failed to load pipeline for {scene_path} (config '{config_name}', "
            f"label space '{labelspace}')")

    # Optional warm start: restore a previously-trained flow state before any
    # frames are pushed. Cell indices line up with the freshly-constructed
    # voxel grid only if voxel_size + world frame agree with the saved
    # session, so chained training works within a single physical
    # environment (e.g. all TBD month-block sessions in the same hall) but
    # is unsafe across environments. Load failures abort the run rather
    # than silently fall back to a fresh start, so a typo in the path
    # doesn't quietly produce a from-scratch training run.
    if load_flow_state:
        ok, err = pipeline.load_flow_state(str(load_flow_state))
        if not ok:
            click.secho(
                f"[load-flow-state] failed to restore {load_flow_state}: {err}",
                fg="red",
            )
            return
        try:
            stats = pipeline.flow_stats()
            click.secho(
                f"[load-flow-state] restored {stats.get('cells', '?')} cells, "
                f"{stats.get('total_components', '?')} components "
                f"from {load_flow_state}",
                fg="cyan",
            )
        except AttributeError:
            click.secho(
                f"[load-flow-state] restored from {load_flow_state}",
                fg="cyan",
            )

    mem_profiler = None
    if mem_profile_csv is not None:
        from kairos_python.mem_profile import MemProfiler

        mem_profiler = MemProfiler(mem_profile_csv, every=mem_profile_every,
                                   dsg_dump_dir=dsg_dump_dir)
        click.secho(
            f"[mem-profile] sampling every {mem_profiler.every} frames -> "
            f"{mem_profiler.path}",
            fg="cyan",
        )

    track_idx = 0
    n_pushed = 0
    n_pushed_frames = 0
    elapsed_s = None
    t_run_start = time.time()
    # Presence-grid export for presence-AUC eval. Only during a scoring run
    # (--score-tracks), gated by KAIROS_PRESENCE_GRID=1. Emits, at a 1 s cadence
    # over the replayed test scene, every in-frustum cell's predicted presence
    # (spectral + static lambda, 5 s/10 s horizons) using the same active-voxel
    # visibility that drives the Poisson exposure beta.
    grid_fh = None
    grid_writer = None
    next_grid_t_s = None
    if (
        score_tracks_path is not None
        and os.environ.get("KAIROS_PRESENCE_GRID") == "1"
        and scene_output_path is not None
    ):
        scene_output_path.mkdir(parents=True, exist_ok=True)
        grid_fh = open(scene_output_path / "presence_grid.csv", "w", newline="")
        grid_writer = csv.writer(grid_fh)
        grid_writer.writerow(
            ["t_seconds", "vx", "vy", "vz", "p5", "p10"]
        )
    # Optional visualization recorder (KAIROS_VIZ_RECORDING=1). Read-only: it
    # only snapshots pipeline.graph / pipeline.flow_voxels(), so it never changes
    # the run's outputs. Off by default; renders with kairos_visualizer.
    viz_rec = None
    if os.environ.get("KAIROS_VIZ_RECORDING") == "1" and scene_output_path is not None:
        from kairos_python.viz_recorder import VizRecorder
        # Inherit the spatial params (flow-voxel edge, robot sensing radius) from
        # the same resolved config the run uses, so the recording can never drift
        # from the pipeline. Re-resolving here (only when recording) keeps
        # load_pipeline's signature untouched for every other caller.
        vp = viz_params_from_config(
            load_configs(config_name, labelspace=labelspace, enable_lcd=enable_lcd))
        viz_rec = VizRecorder(
            interval_s=float(os.environ.get("KAIROS_VIZ_INTERVAL_S", "30")),
            mesh_interval_s=float(os.environ.get("KAIROS_VIZ_MESH_INTERVAL_S", "30")),
            voxel_size=vp["voxel_size"],
            active_window_radius=vp["active_window_radius"])
        scene_output_path.mkdir(parents=True, exist_ok=True)

    try:
        data_iter = DataLoaderIter(data)
        if progress:
            import tqdm
            data_iter = tqdm.tqdm(data_iter, total=len(data))

        for step_idx, packet in enumerate(data_iter):
            if max_steps and step_idx >= max_steps:
                break

            for func in callbacks:
                func(packet)

            if tracks:
                frame_ts_ns = int(packet.timestamp)
                while (
                    track_idx < len(tracks)
                    and (tracks[track_idx][0] + shift_ns) <= frame_ts_ns
                ):
                    t_ns, xyz, theta, rho, ped_ids = tracks[track_idx]
                    ok = pipeline.push_detections(
                        (t_ns + shift_ns) * 1e-9, xyz, theta, rho,
                        track_ids=ped_ids if ped_ids is not None else [],
                    )
                    if ok:
                        n_pushed += len(xyz)
                        n_pushed_frames += 1
                    if viz_rec is not None:
                        viz_rec.detections((t_ns + shift_ns) * 1e-9, xyz, theta, ped_ids)
                    track_idx += 1

            pipeline.step(
                packet.timestamp,
                packet.world_t_body,
                packet.world_q_body,
                packet.depth,
                packet.labels,
                packet.color,
                **packet.extras,
            )

            if viz_rec is not None:
                try:                                # viz is best-effort, never fatal
                    t_s = packet.timestamp * 1e-9
                    viz_rec.robot(t_s, packet.world_t_body)
                    viz_rec.maybe_snapshot(pipeline, t_s)
                except Exception as e:
                    click.echo(f"[warn] viz recorder disabled: {e}", err=True)
                    viz_rec = None

            if grid_writer is not None:
                t_s = packet.timestamp * 1e-9
                if next_grid_t_s is None:
                    next_grid_t_s = t_s
                if t_s >= next_grid_t_s:
                    g = pipeline.presence_grid(t_s)
                    for i in range(len(g["vx"])):
                        grid_writer.writerow(
                            [t_s, g["vx"][i], g["vy"][i], g["vz"][i],
                             g["p5"][i], g["p10"][i]]
                        )
                    next_grid_t_s += 1.0

            if mem_profiler is not None:
                mem_profiler.maybe_sample(pipeline, step_idx, packet.timestamp)
    finally:
        elapsed_s = time.time() - t_run_start
        if grid_fh is not None:
            grid_fh.close()
        if viz_rec is not None:
            try:
                viz_rec.save(scene_output_path / "viz_recording.npz")
            except Exception as e:
                click.echo(f"[warn] viz recording save failed (non-fatal): {e}", err=True)
        if mem_profiler is not None:
            mem_profiler.close()
        if tracks_path is not None:
            click.echo(
                f"[tracks] pushed {n_pushed} samples across "
                f"{n_pushed_frames} frame-times"
            )
        # Score test detections BEFORE pipeline.save() so the same pipeline
        # instance (still holding live in-memory NUDFT variance counters) is
        # the one being scored. record_stats reads those counters directly.
        _maybe_run_scoring(
            pipeline,
            score_tracks_path,
            scene_output_path,
            label=score_label or config_name,
            holdout_frac=score_holdout_frac,
            elapsed_s=elapsed_s,
        )
        # When scoring ran it already snapshotted timing (same pipeline, after
        # the training pushes). For a training-only run (no --score-tracks),
        # capture the per-update cost here so the runtime is recorded.
        if score_tracks_path is None:
            _maybe_record_train_stats(
                pipeline,
                scene_output_path,
                label=score_label or config_name,
                n_pushed=n_pushed,
                elapsed_s=elapsed_s,
            )
        try:
            pipeline.save()
        except RuntimeError as e:
            click.echo(f"[warn] pipeline.save() failed (non-fatal): {e}", err=True)


@click.command(name="run")
@click.argument("scenes", type=click.Path(exists=True), nargs=-1)
@click.option("-c", "--config-name", required=True,
              help="dataset config in config/datasets/ without .yaml (atc, hb, tbd)")
@click.option("-l", "--labelspace", default="ade20k_outdoor",
              help="label space in config/label_spaces/ without _label_space.yaml")
@click.option("-o", "--output-path", default=None)
@click.option("-g", "--glog-level", default=0, help="minimum glog level")
@click.option("-y", "--force", is_flag=True, help="overwrite previous output")
@click.option("-m", "--max-steps", default=None, type=int, help="total number of steps")
@click.option("--openset-model", default=None, type=str, help="clip model to use")
@click.option("-v", "--verbosity", default=0, help="glog verbosity")
@click.option("--show-images", default=False, help="show semantics", is_flag=True)
@click.option("--show-config", default=False, help="show hydra config", is_flag=True)
@click.option("--publish/--no-publish", default=True, help="publish over zmq")
@click.option("--progress/--no-progress", default=True, help="show progress bar")
@click.option(
    "--tracks",
    "tracks_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Optional tracks.csv (output of "
        "scripts/tbd_pedestrian_to_detections.py). When given, detections are "
        "interleaved into the flow temporal module before each frame step. "
        "Requires `enable_flow_temporal_module: true` in the dataset YAML."
    ),
)
@click.option(
    "--auto-align-tracks/--no-auto-align-tracks",
    default=False,
    help=(
        "Shift every track timestamp so the first track aligns with the "
        "first dataloader frame. Useful for smoke tests when tracks.csv "
        "was generated with the wrong session epoch."
    ),
)
@click.option(
    "--tracks-time-shift-ns",
    default=0,
    type=int,
    help="Add this offset (nanoseconds) to every track timestamp.",
)
@click.option(
    "--enable-lcd/--no-enable-lcd",
    default=None,
    help=(
        "Run Hydra's visual loop-closure detection (LCD). DEFAULT (unset): "
        "LCD is OFF. Pass --enable-lcd to activate. When LCD is on, an ORB "
        "vocab is auto-discovered (see --bow-vocab) so loop closures "
        "actually fire. Kairos re-keys its flow cells on every correction "
        "the backend applies either way."
    ),
)
@click.option(
    "--bow-vocab",
    "bow_vocab",
    default=None,
    type=click.Path(dir_okay=False),
    help=(
        "Path to an ORB vocabulary (DBoW2/3 .yml.gz). When set, the "
        "bow_producer data callback runs ORB + DBoW3 per accepted keyframe "
        "and pushes a BowQuery onto Hydra's bow_queue, unblocking "
        "agent-layer LCD. Run hydra-kairos/scripts/fetch_orb_vocab.sh to download the "
        "Kimera-VIO ORBvoc.yml.gz to the default cache path."
    ),
)
@click.option(
    "--bow-features",
    "bow_features",
    default=1000,
    type=int,
    help="Number of ORB features per frame (only used when --bow-vocab is set).",
)
@click.option(
    "--bow-color-is-rgb/--bow-color-is-bgr",
    "bow_color_is_rgb",
    default=True,
    help=(
        "Channel order of dataloader.color. FileDataLoader reads PNGs with "
        "imageio.v3.imread which decodes RGB-ordered, so the default is RGB "
        "and BowProducer flips to BGR before ORB. Pass --bow-color-is-bgr "
        "if a custom dataloader already hands you BGR."
    ),
)
@click.option(
    "--load-flow-state",
    "load_flow_state",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a flow_state.bin saved by an earlier run. When set, the "
        "kairos per-cell state (NUDFTs, Poisson, timestamps, voxel grid) "
        "is restored before any frames or "
        "detections are ingested -- so this run continues training the "
        "loaded model instead of starting fresh. Use to chain training "
        "across multiple TBD sessions within the same physical environment. "
        "Aborts if the file is missing or schema version mismatches."
    ),
)
@click.option(
    "--score-tracks",
    "score_tracks_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Test-set tracks.csv to score AFTER training. When set, the run "
        "drives the suite's evaluation/score_helper.py score_csv against this file "
        "and writes preds.csv + preds.stats.json under -o/--output-path. "
        "Without it the runner only saves flow_state.bin. Pass the same "
        "file as --tracks for an in-sample "
        "score, or a held-out file for a generalisation score."
    ),
)
@click.option(
    "--score-label",
    "score_label",
    default=None,
    type=str,
    help=(
        "System label embedded in preds.stats.json (used by evaluate.py "
        "to title the column). Defaults to --config-name."
    ),
)
@click.option(
    "--score-holdout-frac",
    "score_holdout_frac",
    default=0.0,
    type=float,
    help=(
        "When > 0, restrict scoring to the last `score_holdout_frac` of "
        "the [min_t_ns, max_t_ns] range in the --score-tracks CSV "
        "(deterministic time-split). Default 0.0 scores every detection. "
        "Use 0.2 for an 80/20 in-sample "
        "holdout; the held-out-week protocol passes a different test "
        "CSV instead and leaves this at 0.0."
    ),
)
@click.option(
    "--mem-profile-csv",
    "mem_profile_csv",
    default=None,
    type=str,
    help=(
        "Write per-layer memory instrumentation to this CSV (RSS + DSG node "
        "counts per layer split active/archived + object trajectory-history "
        "lengths + flow dynamics size). Used to pin which layer's growth "
        "tracks the within-scene RSS climb."
    ),
)
@click.option(
    "--mem-profile-every",
    "mem_profile_every",
    default=50,
    type=int,
    help="Sample interval in frames for --mem-profile-csv (default 50).",
)
@click.option(
    "--dsg-dump-dir",
    "dsg_dump_dir",
    default=None,
    type=str,
    help=(
        "If set, dump the backend DSG (JSON + mesh) into this dir at each "
        "mem-profile sample. Set --mem-profile-every to ~one pass (e.g. 333) "
        "to get one DSG per pass for visual inspection. Requires "
        "--mem-profile-csv."
    ),
)
def cli(
    scenes,
    config_name,
    output_path,
    labelspace,
    glog_level,
    force,
    max_steps,
    openset_model,
    verbosity,
    show_images,
    show_config,
    publish,
    progress,
    tracks_path,
    auto_align_tracks,
    tracks_time_shift_ns,
    enable_lcd,
    bow_vocab,
    bow_features,
    bow_color_is_rgb,
    load_flow_state,
    score_tracks_path,
    score_label,
    score_holdout_frac,
    mem_profile_csv,
    mem_profile_every,
    dsg_dump_dir,
):
    """Run Hydra against various scenes."""
    hydra_python.set_glog_level(glog_level, verbosity)
    output_path = resolve_output_path(output_path, force=force)

    if tracks_path is not None and len(scenes) != 1:
        raise click.UsageError(
            "--tracks expects exactly one scene argument; got "
            f"{len(scenes)}."
        )

    # LCD is OFF by default; pass --enable-lcd to activate.
    enable_lcd = bool(enable_lcd)
    click.secho(
        f"[lcd] Hydra loop closure {'ON' if enable_lcd else 'OFF'} "
        f"(config '{config_name}')",
        fg="cyan",
    )

    callback_specs = {}
    if openset_model is not None:
        callback_specs["clip"] = {"model_name": openset_model}
    if show_images:
        callback_specs["image_viewer"] = {}

    # LCD needs ORB BoW descriptors fed per keyframe by bow_producer; without
    # them LCD is silently inert ("0 loop closures"). Resolve the vocab robustly
    # so loop closures actually fire wherever the run happens.
    if not enable_lcd:
        if bow_vocab is not None:
            click.secho(
                "[lcd] --bow-vocab ignored: LCD is off (pass --enable-lcd).",
                fg="yellow",
            )
            bow_vocab = None
    else:
        if bow_vocab is not None and not pathlib.Path(bow_vocab).is_file():
            click.secho(
                f"[lcd] --bow-vocab '{bow_vocab}' not found; auto-discovering.",
                fg="yellow",
            )
            bow_vocab = None
        if bow_vocab is None:
            bow_vocab = _default_orb_vocab()
        if bow_vocab is not None:
            click.secho(f"[lcd] using ORB vocab: {bow_vocab}", fg="cyan")
            callback_specs["bow_producer"] = {
                "vocab_path": bow_vocab,
                "n_features": bow_features,
                "color_is_rgb": bow_color_is_rgb,
            }
        else:
            click.secho(
                "[lcd] WARNING: LCD is ON but no ORB vocab found -> loop "
                "closures will NOT fire. Set --bow-vocab / $KAIROS_ORB_VOCAB or "
                "place orb_vocab.txt at the repo root.",
                fg="yellow",
            )
    callbacks = list(hydra_python.DataCallbackRegistry.create(callback_specs))

    if load_flow_state is not None and len(scenes) != 1:
        # Multi-scene with one load path would chain the SAME starting state
        # into every scene rather than progressively accumulating, which is
        # almost never what someone wants. Force one scene per call so the
        # user is explicit about chaining (re-run the CLI with the previous
        # run's flow_state as --load-flow-state).
        raise click.UsageError(
            "--load-flow-state expects exactly one scene argument; got "
            f"{len(scenes)}. To chain multiple sessions, invoke run once per "
            "scene and pass each session's saved flow_state.bin to the next."
        )

    for scene_path in scenes:
        scene_path = pathlib.Path(scene_path).expanduser().absolute()
        _run_one(
            scene_path=scene_path,
            config_name=config_name,
            labelspace=labelspace,
            output_path=output_path,
            config_verbosity=0 if show_config else 1,
            place_feature_strategy=None if openset_model is None else "fusion",
            publish=publish,
            progress=progress,
            max_steps=max_steps,
            callbacks=callbacks,
            tracks_path=tracks_path,
            auto_align_tracks=auto_align_tracks,
            tracks_time_shift_ns=tracks_time_shift_ns,
            enable_lcd=enable_lcd,
            load_flow_state=load_flow_state,
            score_tracks_path=score_tracks_path,
            score_label=score_label,
            score_holdout_frac=score_holdout_frac,
            mem_profile_csv=mem_profile_csv,
            mem_profile_every=mem_profile_every,
            dsg_dump_dir=dsg_dump_dir,
        )
