#!/usr/bin/env bash
# Download an ORB vocabulary for the offline bow_producer pipeline.
# Idempotent: skips if the target file is already present.
#
# Source: ORB_SLAM2's ORBvoc.txt.tar.gz (DBoW2 text format, ~50 MB, ~145 MB
# extracted). DBoW3::Vocabulary::load() reads .txt format directly. We avoid
# Kimera-VIO's ORBvoc.yml.gz because that file is stored via Git LFS and
# the raw URL only serves the LFS pointer.
#
# Usage: scripts/fetch_orb_vocab.sh [output_path]
#   default output_path: ${XDG_CACHE_HOME:-$HOME/.cache}/kairos/orb_vocab.txt

set -euo pipefail

DEFAULT_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/kairos"
OUT_PATH="${1:-$DEFAULT_DIR/orb_vocab.txt}"
URL="https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz"

if [[ -s "$OUT_PATH" ]]; then
  echo "[fetch_orb_vocab] vocabulary already present: $OUT_PATH"
  exit 0
fi

mkdir -p "$(dirname "$OUT_PATH")"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[fetch_orb_vocab] downloading $URL"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "$TMP_DIR/voc.tar.gz" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TMP_DIR/voc.tar.gz" "$URL"
else
  echo "ERR: need curl or wget to download the vocabulary" >&2
  exit 1
fi

echo "[fetch_orb_vocab] extracting -> $OUT_PATH"
tar -xzf "$TMP_DIR/voc.tar.gz" -C "$TMP_DIR"
EXTRACTED="$(find "$TMP_DIR" -name 'ORBvoc.txt' -type f -print -quit)"
if [[ -z "$EXTRACTED" ]]; then
  echo "ERR: ORBvoc.txt not found in tarball" >&2
  exit 1
fi
mv "$EXTRACTED" "$OUT_PATH"

echo "[fetch_orb_vocab] done. kairos run finds it there; or pass it with:"
echo "  kairos run --enable-lcd --bow-vocab \"$OUT_PATH\" ..."
