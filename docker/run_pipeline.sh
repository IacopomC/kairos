#!/usr/bin/env bash
# Run `kairos run` in a PERSISTENT, self-healing container so the one-time runtime
# install (setup_runtime.sh) survives across invocations and reboots.
#
#   kairos/docker/run_pipeline.sh /data/<scene> -c tbd -l ade20k_outdoor \
#       -o /out/run -m 1000 --no-publish --no-enable-lcd
#
# Env overrides: KAIROS_IMAGE, KAIROS_CONTAINER, KAIROS_DATASETS, KAIROS_OUT.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_SRC="$(cd "${HERE}/../.." && pwd)"
DATASETS="${KAIROS_DATASETS:-$HOME/datasets}"
OUT="${KAIROS_OUT:-$HOME/kairos_exp_out}"
NAME="${KAIROS_CONTAINER:-kairos-colcon}"

IMAGE="${KAIROS_IMAGE:-}"
if [[ -z "$IMAGE" ]]; then
  if docker image inspect kairos:runtime >/dev/null 2>&1; then IMAGE="kairos:runtime"; else IMAGE="kairos:dev"; fi
fi
mkdir -p "$OUT"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" \
    -v "${WS_SRC}:/ws/src" \
    -v kairos_colcon_build:/ws/build -v kairos_colcon_install:/ws/install -v kairos_colcon_logs:/ws/logs \
    -v "${DATASETS}:/data" -v "${OUT}:/out" \
    "$IMAGE" bash -lc 'sleep infinity' >/dev/null
fi

# Self-heal: if the Python runtime is not present yet, install it once.
if ! docker exec "$NAME" bash -lc 'source /ws/src/kairos/docker/kenv.sh && python3 -c "import kairos_python, hydra_python, spark_dsg"' >/dev/null 2>&1; then
  echo "[run_pipeline] first run: installing Python runtime (one-time) ..."
  docker exec "$NAME" bash /ws/src/kairos/docker/setup_runtime.sh
fi

docker exec "$NAME" bash -lc 'source /ws/src/kairos/docker/kenv.sh; python3 -m kairos_python run "$@"' _ "$@"
