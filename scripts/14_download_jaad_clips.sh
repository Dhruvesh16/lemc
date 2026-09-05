#!/usr/bin/env bash
# Download JAAD video clips (~3.1 GB) into data_external/JAAD/JAAD_clips/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAAD="${LEMC_JAAD_ROOT:-$ROOT/data_external/JAAD}"
cd "$JAAD"
mkdir -p JAAD_clips

if ls JAAD_clips/*.mp4 >/dev/null 2>&1; then
  echo "JAAD_clips already present ($(ls JAAD_clips/*.mp4 | wc -l) mp4 files)"
  exit 0
fi

ZIP="$JAAD/JAAD_clips.zip"
if [[ ! -s "$ZIP" ]]; then
  echo "Trying YorkU mirror..."
  wget -c -O "$ZIP" "http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip" || true
fi
if [[ ! -s "$ZIP" ]]; then
  echo "Trying Google Drive mirror..."
  python3 -m pip install -q gdown
  gdown "1HCFLBO9fJutCKG11FtjKfdLvME6Qe_5L" -O "$ZIP"
fi
unzip -o "$ZIP" -d .
echo "Done. Clips in $JAAD/JAAD_clips/"
