# Reproducibility — LEMC / Fixed ORB (conference package)

## Environment

**Recommended (AMD Radeon RX 7600 / ROCm):** Docker image with ROCm 7.14 + PyTorch 2.12
(same stack as the PIE YOLO density run in `DECISIONS.md`).

```bash
# Host: you must be in render+video groups for GPU (re-login after usermod)
groups | grep -E 'render|video' || sudo usermod -aG render,video "$USER"

bash docker/build.sh
docker/run.sh python3 -c "import torch; print('hip/cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If `torch.cuda.is_available()` is False but `rocm-smi` works inside the container, try:
```bash
LEMC_DOCKER_PRIVILEGED=1 docker/run.sh rocm-smi
LEMC_DOCKER_PRIVILEGED=1 HSA_OVERRIDE_GFX_VERSION=11.0.0 docker/run.sh python3 -c "import torch; print(torch.cuda.is_available())"
```

CPU fallback: `LEMC_DOCKER_CPU=1 docker/run.sh ...`

Host venv (CPU PyTorch) works for JAAD ORB extract + small GRU training:
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
SKIP_JAAD_BASELINE=1 bash scripts/10_run_jaad_grid.sh
```

## Data layout

### PIE
Official PIE tree under `LEMC_PIE_ROOT` with `images/`, annotations, `data_cache/pie_database.pkl`.

### JAAD 2.0
```bash
git clone --depth 1 --branch JAAD_2.0 https://github.com/ykotseruba/JAAD.git data_external/JAAD
bash scripts/14_download_jaad_clips.sh   # ~3.1 GB video zip
bash scripts/15_extract_jaad_frames.sh   # -> images/video_XXXX/*.png
python scripts/01_build_track_store.py --dataset jaad
python scripts/02_density_check.py --dataset jaad
```

## PIE pipeline (complete)

```bash
# Track store + density
python scripts/07_merge_detections.py   # if using augmented store
python scripts/02_density_check.py --dataset pie --track-store-file pie_track_store_augmented.pkl

# ORB affines (window-scoped)
python scripts/09b_extract_orb_affine.py --config configs/pie_fixed_orb_affine.yaml

# Main grid (3 seeds × 5 rows)
bash scripts/10_run_grid.sh

# Fixed ORB (primary result)
python scripts/12_fixed_orb_affine.py --config configs/pie_fixed_orb_affine.yaml --seed 0

# Sign gate
python scripts/08_sign_variance_test.py --config configs/pie_lemc_on_augmented_auxobd.yaml \
  --checkpoint-dir checkpoints/pie_lemc_augmented_auxobd_seed0

# Aggregate tables + significance
python scripts/16_aggregate_grid_metrics.py --pattern 'results/eval_pie_*.txt'
python scripts/17_paired_significance.py \
  --baseline-ckpt checkpoints/pie_nolemc_augmented_base_seed0 \
  --orb-ckpt checkpoints/pie_fixed_orb_affine_v2_seed0

# Figures + PDF report
python scripts/13_make_figures.py
python scripts/generate_paper_report.py
```

## JAAD pipeline

```bash
python scripts/01_build_track_store.py --dataset jaad
# Baseline already trained? Skip retrain:
SKIP_JAAD_BASELINE=1 bash scripts/10_run_jaad_grid.sh
# Or inside Docker (GPU):
SKIP_JAAD_BASELINE=1 docker/run.sh bash scripts/10_run_jaad_grid.sh
```

## Tests

```bash
pytest tests/ -q
```

## Checkpoints naming

`{dataset}_{lemc|nolemc}_{run_tag}_seed{N}` — e.g. `jaad_nolemc_jaad_base_seed0`, `pie_fixed_orb_affine_v2_seed0` (legacy PIE fixed ORB).
