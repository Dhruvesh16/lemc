#!/usr/bin/env bash
# One-shot: build (if needed) + GPU check + JAAD ORB grid on RX 7600.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! docker image inspect lemc:rocm >/dev/null 2>&1; then
  echo "Image lemc:rocm not found — building..."
  bash docker/build.sh
fi

echo "=== GPU smoke test ==="
docker/run.sh python3 -c "
import torch
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
"

echo "=== JAAD grid (ORB extract + fixed ORB x3) ==="
docker/run.sh bash scripts/10_run_jaad_grid.sh
