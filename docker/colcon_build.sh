#!/usr/bin/env bash
# Run INSIDE the container (see run.sh). Builds the colcon workspace mounted at
# /ws/src into /ws/build and /ws/install (named volumes, so builds stay
# incremental across runs).
#
#   colcon_build.sh                 # build kairos, kairos_ros and their dependencies
#   colcon_build.sh hydra kairos    # build only these (+ their dependencies)
#
# Extra CMake flags go after `--`, e.g. `colcon_build.sh kairos -- -DKAIROS_ENABLE_TESTS=ON`.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
cd /ws

pkgs=()
while [[ $# -gt 0 && "$1" != "--" ]]; do pkgs+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
[[ ${#pkgs[@]} -eq 0 ]] && pkgs=(kairos kairos_ros)

# Parallelism is capped so the build fits in about 16 GB of RAM.
MAKEFLAGS="-j${KAIROS_BUILD_JOBS:-4}" colcon --log-base /ws/logs build \
  --build-base /ws/build --install-base /ws/install \
  --parallel-workers "${KAIROS_BUILD_WORKERS:-3}" \
  --packages-up-to "${pkgs[@]}" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCONFIG_UTILS_ENABLE_ROS=OFF \
    --no-warn-unused-cli "$@"
