#!/usr/bin/env bash
# Extract JAAD_clips/*.mp4 -> images/video_XXXX/00000.png (official layout)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAAD="${LEMC_JAAD_ROOT:-$ROOT/data_external/JAAD}"
PY="${ROOT}/.venv/bin/python"
cd "$JAAD"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv at $PY — run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ -d images ]] && [[ "$(find images -name '*.png' 2>/dev/null | head -1)" ]]; then
  echo "images/ already populated"
  exit 0
fi

if [[ ! -d JAAD_clips ]] || [[ "$(find JAAD_clips -name '*.mp4' 2>/dev/null | wc -l)" -eq 0 ]]; then
  echo "No mp4 in JAAD_clips/ — run scripts/14_download_jaad_clips.sh first"
  exit 1
fi

echo "Extracting $(find JAAD_clips -name '*.mp4' | wc -l) clips with $PY ..."
"$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from jaad_data import JAAD
JAAD(data_path=os.getcwd()).extract_and_save_images()
print("extracted to", os.path.join(os.getcwd(), "images"))
PY
