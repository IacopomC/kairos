# kairos_visualizer

An **optional, standalone** Rerun visualizer for Kairos flow scene-graph runs.

It renders a layered, 3D view of a run: the reconstruction geometry, the incrementally-created flow voxels (coloured by activity), the navigational nodes and their edges, per-node flow arrows and motion archetypes, the robot, and the pedestrian detections. Node archetypes are the Kairos C++ classification (`spark_dsg::ArchetypeType`), read back from the run.

## Design

- **Decoupled.** Its only interface with the Kairos C++ core is a portable *recording* file (a single `.npz`, see `kairos_visualizer/schema.py`) that any run emits through a small adapter.
- **Optional.** Install it for the Rerun view; it is a separate package from the Kairos install. The ROS/rviz viewer is the primary viewer.
- **Portable.** Pure Python (numpy + rerun), so it runs anywhere rerun installs (Python 3.9+). Only rendering needs rerun.

## Use

Rendering needs Python 3.9+ (rerun's floor), so it runs in its own environment, separate from the Python 3.8 pipeline. Either install the package:

```bash
pip install ./kairos_visualizer          # pulls numpy + rerun-sdk (Python 3.9+)
kairos-visualize my_run.npz -o my_run.rrd
# or: python -m kairos_visualizer my_run.npz -o my_run.rrd
```

or render with the prebuilt image (no local Python 3.9+ needed):

```bash
docker build -t kairos-viz .   # from this directory (kairos/kairos_visualizer), once
docker run --rm -v "$PWD":/out kairos-viz /out/my_run.npz -o /out/my_run.rrd
```

**Viewing the `.rrd`.** Open it with the Rerun viewer, `rerun my_run.rrd` (`pip install rerun-sdk` provides the binary), and scrub the `time_of_day` timeline. app.rerun.io also opens small recordings; for large files use the native `rerun` binary. Use a rerun-sdk version matching the one that rendered the file.

Programmatic:

```python
from kairos_visualizer import render_recording, RenderConfig
render_recording("my_run.npz", "my_run.rrd", RenderConfig(z_nav=12.0))
```

## Producing a recording

Two ways to emit a recording that this renders:

- **From a live run (any dataset).** Set `KAIROS_VIZ_RECORDING=1` (optionally `KAIROS_VIZ_INTERVAL_S=<seconds>`, default 30) when running `kairos run`. It snapshots the growing scene graph + flow field at intervals and writes `viz_recording.npz` under the scene output. Spatial params it needs (flow-voxel size, robot sensing radius) are read from the resolved run config. The recorder is off by default; the run's outputs are byte-identical with or without it, and a viz error stays inside the recorder.
- **From a dataset adapter.** A dataset harness can write the recording directly (e.g. a simulated-robot driver that snapshots the online model).

## Recording format

See `kairos_visualizer/schema.py` for the full field list. A recording needs the navigational nodes/edges and per-snapshot node dynamics; the floor, flow voxels,
robot, and detections are optional layers that are drawn only if present. Layer heights and
colours are controlled by `RenderConfig`.
