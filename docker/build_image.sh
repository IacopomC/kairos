#!/usr/bin/env bash
# Build the Kairos image. Run from anywhere.
#
#   kairos/docker/build_image.sh          # -> kairos:latest (self-contained)
#   kairos/docker/build_image.sh --dev    # -> kairos:dev (toolchain, source mounted)
#
# The self-contained image takes this repository as its build context, fetches
# every other repository from its public remote (kairos.repos) and builds the
# workspace, so it runs with no mounts but your data and output. The dev image
# carries only the toolchain; see run.sh and setup_runtime.sh for that flow. It
# builds on Hydra-ROS's `minimal` image (hydra-minimal), which is built first,
# from the workspace, if it doesn't exist.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

if [[ "${1:-}" == "--dev" ]]; then
  WS_SRC="$(cd "${REPO}/.." && pwd)"
  BASE_IMAGE="${KAIROS_BASE_IMAGE:-hydra-minimal}"
  if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    # Hydra-ROS's Dockerfile copies the workspace src/ for rosdep, from the
    # workspace root
    docker build --network host -t "$BASE_IMAGE" \
      -f "${WS_SRC}/hydra_ros/docker/minimal/Dockerfile" "$(dirname "$WS_SRC")"
  fi
  docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t "${KAIROS_IMAGE:-kairos:dev}" \
    -f "${HERE}/Dockerfile.dev" "${HERE}"
else
  docker build --network host -t "${KAIROS_IMAGE:-kairos:latest}" \
    -f "${HERE}/Dockerfile" "${REPO}"
fi
