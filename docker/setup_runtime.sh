#!/usr/bin/env bash
# One-time RUNTIME setup, run INSIDE the kairos:dev container after the C++
# workspace is built (`colcon_build.sh`). colcon only gives you the compiled
# libraries; `kairos run` additionally needs the Python packages. Idempotent:
# safe to re-run, and re-run it after changing C++ so the bindings are rebuilt.
#
#   kairos/docker/run.sh bash colcon_build.sh                    # compile C++ (once)
#   kairos/docker/run.sh bash /ws/src/kairos/docker/setup_runtime.sh
#   kairos/docker/run.sh                                         # interactive shell
#     > source /ws/src/kairos/docker/kenv.sh
#     > kairos run /data/<scene> -c tbd -l ade20k_outdoor -o /out/run -m 1000 \
#         --no-publish --no-enable-lcd
set -euo pipefail

# ROS's setup scripts reference unbound variables, which trip `set -u`; relax
# nounset just while sourcing them.
set +u
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
set -u

# The dev image installs the third-party dependencies; numpy stays at the
# system version (see Dockerfile.dev).
pip() { pip3 install --break-system-packages --no-cache-dir -c /etc/kairos-pip-constraints.txt "$@"; }

# 1. spark_dsg and hydra_python, from the workspace's forks
pip --no-deps /ws/src/spark_dsg
pip --no-deps /ws/src/hydra

# 2. kairos bindings, built into the source tree so kairos_python runs from
#    there (see kenv.sh)
rm -rf /tmp/kairos_pybuild && mkdir -p /tmp/kairos_pybuild
cmake -S /ws/src/kairos/python -B /tmp/kairos_pybuild -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/kairos_pybuild -j"${KAIROS_BUILD_JOBS:-4}"
_install_dir=/ws/src/kairos/python/src/kairos_python
for _so in /tmp/kairos_pybuild/_kairos_bindings*.so; do
  _name=$(basename "$_so")
  cp "$_so" "${_install_dir}/.${_name}.$$"
  mv "${_install_dir}/.${_name}.$$" "${_install_dir}/${_name}"
done
unset _install_dir _so _name

# 3. the Rerun renderer, into its own environment (see Dockerfile.dev)
/opt/kairos_viz/bin/pip install --no-cache-dir --no-deps /ws/src/kairos/kairos_visualizer

echo
echo "[setup_runtime] OK. Activate the env and run, e.g.:"
echo "  source /ws/src/kairos/docker/kenv.sh"
echo "  kairos run /data/<scene> -c tbd -l ade20k_outdoor -o /out/run -m 1000 --no-publish --no-enable-lcd"
