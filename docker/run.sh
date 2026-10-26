#!/usr/bin/env bash
# Run a command (default: interactive shell) inside the build image with the
# workspace mounted at /ws/src. Build artifacts persist in named volumes so
# `colcon build` stays incremental across runs.
#
#   ./run.sh                              # interactive shell
#   ./run.sh bash colcon_build.sh kairos   # run the build helper
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_SRC="$(cd "${HERE}/../.." && pwd)"   # colcon workspace src (parent of kairos/)

# Attach a terminal only when there is one, so scripts can call this too.
TTY=(-i); [ -t 0 ] && [ -t 1 ] && TTY=(-it)

docker run --rm "${TTY[@]}" \
  -v "${WS_SRC}:/ws/src" \
  -v "${HERE}/colcon_build.sh:/usr/local/bin/colcon_build.sh:ro" \
  -v kairos_colcon_build:/ws/build \
  -v kairos_colcon_install:/ws/install \
  -v kairos_colcon_logs:/ws/logs \
  kairos:dev "$@"
