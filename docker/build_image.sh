#!/usr/bin/env bash
# Build the Kairos image. Run from anywhere.
#
#   kairos/docker/build_image.sh          # -> kairos:latest (self-contained)
#   kairos/docker/build_image.sh --dev    # -> kairos:dev (toolchain, source mounted)
#
# The self-contained image takes this repository as its build context, fetches
# every other repository from its public remote (install/kairos.rosinstall) and
# builds the workspace, so it runs with no mounts but your data and output. The
# dev image carries only the toolchain; see run.sh and setup_runtime.sh for that
# flow.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

if [[ "${1:-}" == "--dev" ]]; then
  docker build -t "${KAIROS_IMAGE:-kairos:dev}" -f "${HERE}/Dockerfile.dev" "${HERE}"
else
  docker build -t "${KAIROS_IMAGE:-kairos:latest}" -f "${HERE}/Dockerfile" "${REPO}"
fi
