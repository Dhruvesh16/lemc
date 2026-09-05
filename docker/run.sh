#!/usr/bin/env bash
# Run LEMC in Docker on AMD Radeon (RX 7600 / RDNA3) — ROCm device passthrough.
# Matches flags from pedestrian-behaviour/RESULTS.md (pie-pcpa-rocm).
#
#   bash docker/build.sh
#   docker/run.sh python3 -c "import torch; print(torch.cuda.is_available())"
#   SKIP_JAAD_BASELINE=1 docker/run.sh bash scripts/10_run_jaad_grid.sh
#
# CPU only: LEMC_DOCKER_CPU=1 docker/run.sh ...
# Privileged (if HIP init fails): LEMC_DOCKER_PRIVILEGED=1 docker/run.sh ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${LEMC_DOCKER_IMAGE:-lemc:rocm}"

PIE_ROOT="${LEMC_PIE_ROOT:-$ROOT/../pedestrian-behaviour/data/PIE}"
JAAD_ROOT="${LEMC_JAAD_ROOT:-$ROOT/data_external/JAAD}"
BENCH="${LEMC_BENCHMARK_REPO:-$ROOT/../pedestrian-behaviour/models/PedestrianActionBenchmark}"

# RX 7600 = gfx1102 (Navi 33). KFD reports gfx_target_version 110002.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1102}"
# Discrete GPU first (node 1 / 0x7480); iGPU is node 2.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

RENDER_GID="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3 || true)"

GPU_ARGS=()
USER_ARGS=()
if [[ "${LEMC_DOCKER_CPU:-0}" == "1" ]]; then
  echo "LEMC_DOCKER_CPU=1 — CPU only"
elif [[ -e /dev/kfd && -d /dev/dri ]]; then
  GPU_ARGS=(
    --device=/dev/kfd
    --device=/dev/dri
    --ipc=host
    --shm-size=8g
    --ulimit memlock=-1:-1
    --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined
  )
  [[ -n "$VIDEO_GID" ]] && GPU_ARGS+=(--group-add "$VIDEO_GID")
  [[ -n "$RENDER_GID" ]] && GPU_ARGS+=(--group-add "$RENDER_GID")
  [[ "${LEMC_DOCKER_PRIVILEGED:-0}" == "1" ]] && GPU_ARGS+=(--privileged)
  if [[ "${LEMC_DOCKER_AS_ROOT:-0}" != "1" ]]; then
    USER_ARGS=(--user "$(id -u):$(id -g)")
  fi
  echo "ROCm: /dev/kfd + /dev/dri (HSA=$HSA_OVERRIDE_GFX_VERSION HIP=$HIP_VISIBLE_DEVICES)"
else
  echo "WARNING: no /dev/kfd or /dev/dri — running CPU-only."
  echo "  Host: ls -la /dev/kfd /dev/dri && groups | grep -E 'video|render'"
  echo "  Add render group: sudo usermod -aG render,video \$USER  (re-login)"
fi

TTY_ARGS=()
if [[ -t 0 ]]; then
  TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  "${USER_ARGS[@]}" \
  -v "$ROOT:/workspace/lemc" \
  -v "$PIE_ROOT:/data/PIE:ro" \
  -v "$JAAD_ROOT:/data/JAAD" \
  -v "$BENCH:/data/PedestrianActionBenchmark:ro" \
  -e LEMC_PIE_ROOT=/data/PIE \
  -e LEMC_JAAD_ROOT=/data/JAAD \
  -e LEMC_BENCHMARK_REPO=/data/PedestrianActionBenchmark \
  -e HSA_OVERRIDE_GFX_VERSION \
  -e PYTORCH_ROCM_ARCH \
  -e HIP_VISIBLE_DEVICES \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -w /workspace/lemc \
  "$IMAGE" \
  "$@"
