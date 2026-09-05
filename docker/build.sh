#!/usr/bin/env bash
# Build lemc:rocm on top of AMD's ROCm 7.14 + PyTorch 2.12 image (~20GB pull).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Pulling base rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0 ..."
docker pull rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0

echo "Building lemc:rocm (code bind-mounted at run time; image is deps + base) ..."
docker build -t lemc:rocm -f docker/Dockerfile .

echo "Done. Test GPU:"
echo "  docker/run.sh python3 -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\""
