#!/usr/bin/env bash
# One-shot JAAD ORB grid (skip baseline if already trained):
#   SKIP_JAAD_BASELINE=1 docker/run.sh bash scripts/10_run_jaad_grid.sh
# Or use wrapper:
#   bash docker/run_jaad_grid.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
SEEDS=(0 1 2)

echo "=== JAAD baseline (no LEMC) ==="
if [[ "${SKIP_JAAD_BASELINE:-0}" == "1" ]]; then
  echo "SKIP_JAAD_BASELINE=1 — skipping baseline train/eval"
else
for seed in "${SEEDS[@]}"; do
  $PY scripts/03_train.py --config configs/jaad_nolemc.yaml --seed "$seed"
  $PY scripts/04_evaluate.py --config configs/jaad_nolemc.yaml \
    --checkpoint-dir "checkpoints/jaad_nolemc_jaad_base_seed${seed}" --split test \
    | tee "results/eval_jaad_nolemc_seed${seed}.txt"
done
fi

echo "=== JAAD fixed ORB affine ==="
$PY scripts/09b_extract_orb_affine.py --config configs/jaad_fixed_orb_affine.yaml
for seed in "${SEEDS[@]}"; do
  $PY scripts/12_fixed_orb_affine.py --config configs/jaad_fixed_orb_affine.yaml --seed "$seed"
done

echo "=== JAAD sign gate (fixed ORB seed0 — run after 12_fixed_orb_affine) ==="
echo "Sign gate is reported by scripts/12_fixed_orb_affine.py (frac_flatten_* in eval_*.txt)"
