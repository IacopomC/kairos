# Kairos

[![Static Badge](https://img.shields.io/badge/-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2609.27467)
[![Project Page](https://img.shields.io/badge/Project-Page-147575)](https://iacopomc.github.io/kairos/)
[![ROS Version](https://img.shields.io/badge/ROS_2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository is the official implementation of the paper:

> **Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs**
>
> [Iacopo Catalano](https://scholar.google.com/citations?hl=en&user=VnPwRvkAAAAJ&view_op=list_works&sortby=pubdate), [Julio Placed](https://scholar.google.com/citations?hl=en&user=1ho6W5EAAAAJ&view_op=list_works&sortby=pubdate), [Javier Civera](https://scholar.google.com/citations?hl=en&user=j_sMzokAAAAJ&view_op=list_works&sortby=pubdate), and [Jorge Pena-Queralta](https://scholar.google.com/citations?hl=en&user=J1SHJeMAAAAJ&view_op=list_works&sortby=pubdate). <br>
>
> *arXiv preprint arXiv:2609.27467*, 2026

Kairos is a *predictive directional-flow memory* that extends a 3DSG to a 4D scene graph (4DSG).
A spectral predictor on each mixing weight of a per-voxel directional mixture forecasts the entire distribution over time, anchored to the observed voxels of the robot's TSDF reconstruction and kept consistent under map re-optimization.

At any traversable location and any future time, Kairos answers two questions a static map cannot: whether agents are likely to be present there, and if so, how they are likely to be moving, as a full directional distribution. A presence channel kept separate from flow distinguishes *no one passes here* from *no one has yet been observed*. Because the forecast covers the whole reconstructed environment and is exposed on the navigational graph, a robot can weigh entire routes before committing to one, comparing how each will flow at the time it would traverse it rather than planning against a snapshot.

<div align="center">
    <img src="doc/media/kairos.gif">
</div>

## Prerequisites

- **Docker**, or **Ubuntu 24.04** with **ROS 2 Jazzy** for a native build
- A pedestrian detection source: a tracks CSV per scene for the Python
  pipeline, or `odom_array_msgs/msg/OdometryArray` messages for the ROS node

## Installation

Kairos runs on Ubuntu 24.04 and ROS 2 Jazzy. Docker is the recommended way to use it on any other system.

### Docker

**Build the image.** The build fetches the other repositories of the workspace
itself (`kairos.repos`), so this repository is all it needs:

```bash
git clone https://github.com/IacopomC/kairos.git
kairos/docker/build_image.sh      # -> kairos:latest
```

**Run Kairos on a scene.**

```bash
docker run --rm -v /path/to/datasets:/data -v /path/to/output:/out kairos:latest \
    kairos run /data/tbd/<month>/<scene> -c tbd -l ade20k_outdoor \
    --tracks /data/tbd/<month>/<scene>/tracks.csv \
    -o /out/run01 --no-publish
```

**What you get** in `/path/to/output/run01/<scene>/`: the scene graph
(`backend/dsg_with_mesh.json`), the trained flow state (`temporal/flow_state.bin`) and
the run's config. `flow_state.bin` is the trained model: pass it to the next
session with `--load-flow-state`, or evaluate it with `--score-tracks`.

### Native (Ubuntu 24.04 + ROS 2 Jazzy)

**Set up the workspace:**

```bash
mkdir -p ~/kairos_ws/src && cd ~/kairos_ws/src
git clone https://github.com/IacopomC/kairos.git
vcs import . < kairos/kairos.repos   # needs python3-vcstool
```

**System dependencies:**

```bash
sudo apt install -y python3-pip python3-colcon-common-extensions python3-rosdep \
    python3-vcstool nlohmann-json3-dev libgoogle-glog-dev libcli11-dev libeigen3-dev
cd ~/kairos_ws
sudo rosdep init 2>/dev/null; rosdep update          # once per machine
rosdep install --from-paths src --ignore-src -r -y
```

**Serialize the GTSAM solve** in Kimera-PGMO (see the note below):

```bash
python3 src/kairos/docker/serialize_gtsam_solve.py src
```

**Build the C++**, from the workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to kairos kairos_ros \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCONFIG_UTILS_ENABLE_ROS=OFF
source install/setup.bash
```

**Install the Python packages**, which builds the Kairos Python bindings. numpy
stays at the system version the ROS packages are built against:

```bash
pip install --break-system-packages "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" \
    click distinctipy imageio networkx pandas pyqtgraph pyyaml rosbags scipy \
    sortedcontainers tqdm
pip install --break-system-packages --no-deps \
    "spark_dataset_interfaces @ git+https://github.com/MIT-SPARK/spark_dataset_interfaces"
pip install --break-system-packages --no-deps ./src/spark_dsg ./src/hydra ./src/kairos
```

**Run it.** In every new shell, source the workspace first:

```bash
source ~/kairos_ws/install/setup.bash
kairos run --help
```

> **GTSAM and TBB.** The packaged GTSAM runs its solve in parallel with TBB and crashes nondeterministically inside `OptimizeClique` when many loop closures accumulate. `serialize_gtsam_solve.py` makes that solve single-threaded. Re-apply it after a fresh checkout of Kimera-PGMO; the Docker image applies it during the build.

## Running Kairos

```bash
kairos run <scene>... -c <config> -l <labelspace> -o <output-dir>
```

| Option | Meaning |
|---|---|
| `-c, --config-name` | a file in [`config/datasets/`](config/datasets/) without the `.yaml` (e.g. `tbd`, `atc`) |
| `-l, --labelspace` | a file in [`config/label_spaces/`](config/label_spaces/) without the `_label_space.yaml` |
| `-o, --output-path` | where the scene graph, flow state, and config are written |
| `-m, --max-steps` | stop after N frames; useful for a quick test |
| `--tracks` | pedestrian tracks CSV to feed the flow layer, when detections are not produced by the perception stack |
| `--load-flow-state` | resume from a previously trained `flow_state.bin` instead of starting empty (in case of multi-sessions scenes) |
| `--score-tracks` | evaluates a held-out tracks CSV at the end of the run |
| `--enable-lcd/--no-enable-lcd` | loop-closure detection; needs an ORB vocabulary (`scripts/fetch_orb_vocab.sh`, or `--bow-vocab`) |
| `--publish/--no-publish` | publish over ZMQ; turn off for offline runs |

`kairos run --help` lists the rest.

Each scene runs in its own pipeline and writes to `<output-dir>/<scene name>/`.
A multi-session deployment is reproduced by running the sessions in order, each
with the previous session's `<output-dir>/<scene name>/temporal/flow_state.bin`
as `--load-flow-state`: the 3DSG is rebuilt per session and the flow state
carries over.

**Quick test** (200 frames, no evaluation):

```bash
docker run --rm -v /path/to/datasets:/data -v /path/to/output:/out kairos:latest \
    kairos run /data/tbd/<month>/<scene> -c tbd -l ade20k_outdoor \
    -o /out/smoke -m 200 --no-publish
```

## Grounded simulation

In its *grounded simulation* in [`kairos-suite`](https://github.com/IacopomC/kairos-suite), a simulated robot patrols a navigation graph authored on the dataset's floor map with a bounded field of view, and the detections it sees feed the flow model as a live sensor would. This allows to train Kairos over months of real pedestrian traffic (ATC, Havlíčkův Brod).

## Visualization

We use rerun.io for an interactive, time-scrubbable 3D recording of a run.

**1. Produce a recording (`.npz`).** Set `KAIROS_VIZ_RECORDING=1` on a `kairos run`.

**2. Render it to `.rrd`** with [`kairos_visualizer/`](kairos_visualizer/)
(needs Python 3.9+ and numpy 2), which the Kairos image carries in its own
environment as `kairos-visualize`:

```bash
docker run --rm -v "$PWD":/out --entrypoint kairos-visualize kairos:latest \
    /out/run.npz -o /out/run.rrd
```

In alternative, install the package:

```bash
pip install ./kairos_visualizer
python -m kairos_visualizer run.npz -o run.rrd
```

**3. View it.** The viewer alone needs the `rerun-sdk`, at the version that
rendered the file (the Kairos image renders with 0.34.1):

```bash
pip install "rerun-sdk==0.34.1" && rerun run.rrd
```

Without Python 3.9, use the standalone viewer binary
shipped inside the Linux wheel on
[PyPI](https://pypi.org/project/rerun-sdk/0.34.1/#files): download the
`manylinux` `x86_64` `.whl`, unzip, and run the bundled binary.

```bash
unzip rerun_sdk-0.34.1-*_x86_64.whl -d rerun_pkg
chmod +x rerun_pkg/rerun_sdk/rerun_cli/rerun
./rerun_pkg/rerun_sdk/rerun_cli/rerun run.rrd
```

Small `.rrd` files can also be dragged into app.rerun.io.

## Documentation

| Document | Contents |
|---|---|
| [`doc/pipeline_overview.md`](doc/pipeline_overview.md) | how the flow pipeline fits together, end to end |
| [`doc/temporal_dynamics.md`](doc/temporal_dynamics.md) | the flow model: mixture, spectral forecasting, presence, coupling |
| [`doc/params.md`](doc/params.md) | every configuration parameter |
| [`doc/place_aggregation.md`](doc/place_aggregation.md) | how voxel flow is summarized onto place nodes |
| [`doc/Concepts.md`](doc/Concepts.md) | vocabulary used across the code |
| [`doc/timing.md`](doc/timing.md) | how the cost measurements are taken |

The C++ headers under `include/kairos/` carry Doxygen comments. For a browsable
HTML reference:

```bash
apt install doxygen graphviz    # graphviz for the diagrams
doxygen Doxyfile
```

Output lands in `doc/api/html/index.html`.

## Citation

If you use this code in your work, please cite:

```
@article{catalano2026kairos,
  title={Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs},
  author={Catalano, Iacopo and Placed, Julio A and Civera, Javier and Pe{\~n}a-Queralta, Jorge},
  journal={arXiv preprint arXiv:2609.27467},
  year={2026}
}
```
