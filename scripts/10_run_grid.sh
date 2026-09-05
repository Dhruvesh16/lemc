#!/usr/bin/env bash
# Step 10 — main experiment grid (3 seeds). Run after pie_orb_shifts.pkl exists.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SEEDS=(0 1 2)

# Row 1: baseline (no LEMC, no speed)
# Row 2: GT OBD speed input
# Row 3: OBD aux (attempt-3 ablation)
# Row 4: ORB aux (main)
# Row 5: LEMC on, no supervision
CONFIGS=(
  configs/pie_nolemc_augmented.yaml
  configs/pie_nolemc_augmented_speed.yaml
  configs/pie_lemc_on_augmented_auxobd.yaml
  configs/pie_lemc_on_augmented_orb.yaml
  configs/pie_lemc_on_augmented_nosup.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "======== TRAIN $cfg seed=$seed ========"
    $PY scripts/03_train.py --config "$cfg" --seed "$seed"
  done
done

echo "======== EVAL all ========"
for cfg in "${CONFIGS[@]}"; do
  tag=$($PY -c "from lemc.utils.config import load_config, run_name; c=load_config('$cfg'); print(run_name(c,0).rsplit('_seed',1)[0])")
  for seed in "${SEEDS[@]}"; do
    ckpt="checkpoints/${tag}_seed${seed}"
    echo "---- eval $ckpt ----"
    $PY scripts/04_evaluate.py --config "$cfg" --checkpoint-dir "$ckpt" --split test \
      | tee "results/eval_${tag}_seed${seed}.txt"
  done
done
