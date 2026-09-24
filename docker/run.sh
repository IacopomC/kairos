#!/usr/bin/env bash
# Run a command (default: interactive shell) inside the build image with the
# workspace mounted at /ws/src. Build artifacts persist in named volumes so
# `catkin build` stays incremental across runs.
#
#   ./run.sh                      # interactive shell
#   ./run.sh bash catkin_build.sh kairos   # run the build helper
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_SRC="$(cd "${HERE}/../.." && pwd)"   # catkin workspace src (parent of kairos/)

docker run --rm -it \
  -v "${WS_SRC}:/ws/src" \
  -v "${HERE}/catkin_build.sh:/usr/local/bin/catkin_build.sh:ro" \
  -v kairos_ws_build:/ws/build \
  -v kairos_ws_devel:/ws/devel \
  -v kairos_ws_logs:/ws/logs \
  kairos:dev "$@"
