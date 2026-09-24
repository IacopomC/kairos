#!/usr/bin/env bash
# One-time RUNTIME setup, run INSIDE the kairos container after the C++ workspace
# is built (`catkin_build.sh hydra kairos`). The build image + catkin only give
# you the compiled libs; `kairos run` additionally needs the Python packages and
# a few runtime libs that are not baked into the image. Idempotent: safe to re-run.
#
#   kairos/docker/run.sh bash catkin_build.sh hydra kairos   # compile C++ (once)
#   kairos/docker/run.sh bash /ws/src/kairos/docker/setup_runtime.sh
#   kairos/docker/run.sh                                     # interactive shell
#     > source /ws/src/kairos/docker/kenv.sh
#     > kairos run /data/<scene> -c tbd -l ade20k_outdoor -o /out/run -m 1000 \
#         --no-publish --no-enable-lcd
set -euo pipefail

# ROS's setup.bash references unbound vars (e.g. ROS_MASTER_URI) which trip
# `set -u`; relax nounset just while sourcing it, then restore. (Without this,
# a fresh container with no committed kairos:runtime image aborts setup here.)
set +u
source /opt/ros/noetic/setup.bash
source /ws/devel/setup.bash
set -u

# 1. libzmq.so.5 -- the hydra python bindings link it; the build image (older
#    than this script) may not have it. Newer images install it via Dockerfile.
if ! ldconfig -p | grep -q 'libzmq\.so\.5'; then
  apt-get update && apt-get install -y --no-install-recommends libzmq3-dev
fi

# 2. spark_dsg python bindings -- build the LOCAL fork. (The PyPI `spark_dsg[viz]`
#    dependency pulls open3d>=0.17, which has no cp38 wheel and fails to build.)
python3 -c 'import spark_dsg' 2>/dev/null || pip install /ws/src/spark-dsg-kairos

# 3. pure-python runtime deps + a numpy new enough for matplotlib (focal apt
#    ships numpy 1.17; numpy is forward-ABI-compatible with the 1.17-built libs).
pip install --no-deps imageio networkx scipy tqdm distinctipy click pyqtgraph "numpy==1.24.4"

# 4. hydra_python (data loaders + _hydra_bindings). --no-deps so it does not try
#    to re-pull spark_dsg[viz]. Recompiles _hydra_bindings from the fixed source
#    (glog double-init guard in python/bindings/src/glog_utilities.cpp).
pip install --no-deps /ws/src/hydra-kairos

# 5. kairos bindings. The package declares no metadata (installs as UNKNOWN) and
#    its default relative HYDRA_FLOW_DIR breaks under pip's temp copy, so build
#    the extension directly and expose kairos_python via PYTHONPATH (see kenv.sh).
rm -rf /tmp/kairos_pybuild && mkdir -p /tmp/kairos_pybuild
cmake -S /ws/src/kairos/python -B /tmp/kairos_pybuild \
  -DHYDRA_FLOW_DIR=/ws/src/hydra-kairos \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build /tmp/kairos_pybuild -j"$(nproc)"
_install_dir=/ws/src/kairos/python/src/kairos_python
for _so in /tmp/kairos_pybuild/_kairos_bindings*.so; do
  _name=$(basename "$_so")
  cp "$_so" "${_install_dir}/.${_name}.$$"
  mv "${_install_dir}/.${_name}.$$" "${_install_dir}/${_name}"
done
unset _install_dir _so _name

echo
echo "[setup_runtime] OK. Activate the env and run, e.g.:"
echo "  source /ws/src/kairos/docker/kenv.sh"
echo "  kairos run /data/<scene> -c tbd -l ade20k_outdoor -o /out/run -m 1000 --no-publish --no-enable-lcd"
