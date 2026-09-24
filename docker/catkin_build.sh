#!/usr/bin/env bash
# Run INSIDE the container (see run.sh). Initialises the catkin workspace,
# resolves system deps with rosdep, and builds the requested packages.
#
#   catkin_build.sh                 # build everything
#   catkin_build.sh hydra kairos    # build only these (+ their deps)
set -euo pipefail
source /opt/ros/noetic/setup.bash

cd /ws
catkin init >/dev/null 2>&1 || true
# Release for speed (per project requirement). BUILD_TESTING=OFF skips the
# vendored deps' own unit tests (e.g. config_utilities' yaml-cpp test mismatch);
# KAIROS's own tests are built explicitly when needed.
catkin config --extend /opt/ros/noetic \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCONFIG_UTILS_BUILD_DEMOS=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null

# Resolve apt deps declared in the vendored package.xml files. GTSAM is already
# installed from the borglab PPA in the image; ignore anything unresolved.
rosdep install --from-paths /ws/src --ignore-src -r -y || true

# GTSAM here is built with TBB, whose parallel multifrontal elimination races on
# the dense deformation-graph structure many loop closures produce. Serialize
# that one solve (a few ms), as docker/Dockerfile does; only
# tbb::global_control constrains GTSAM's arena.
f=/ws/src/Kimera-RPGO/src/RobustSolver.cpp
if [ -f "$f" ] && ! grep -q global_control "$f"; then
  sed -i '1i #include <tbb/global_control.h>' "$f"
  sed -i 's|^\(void RobustSolver::optimize().*{\)|\1\n  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);|' "$f"
fi

# -j limited to avoid the GCC-killed OOM that Hydra's README warns about.
NPROC="$(( $(nproc) > 6 ? 6 : $(nproc) ))"
catkin build --no-status --summarize -j"${NPROC}" "$@"
