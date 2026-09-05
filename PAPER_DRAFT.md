# LEMC: Sensor-Free Ego-Motion Compensation for Egocentric Pedestrian Prediction

**Draft with results** (2026-08-25). Related-work notes in `NOTES_RELATED_WORK.md`.

## Abstract

Egocentric pedestrian tracks mix pedestrian motion with camera motion. Zhang et al. (TIP 2024) decompose that mixture using ego-vehicle speed as both input and soft supervisor; LIM (TITS 2025) deletes speed to avoid causal confusion. We study sensor-free compensation. A learned multi-agent residual (LEMC) with magnitude–speed or translation-ORB losses fails a standing-pedestrian variance gate and hurts ADE. Replacing global translation with **full partial-affine ORB** ego-motion yields a **93%** flatten rate on standing pedestrians while the car moves, and is **ADE-neutral** vs a bbox-only GRU under the field protocol (τ=16, η=45, TTE∈[30,60]). GT speed as input *hurts* ADE but *helps* intent—consistent with LIM. **JAAD** (no OBD, 295 test windows): baseline ADE 83.1±2.0 px; fixed ORB ADE **62.7±2.3** px, standing+moving flatten **96.9%**.

## 1. Introduction

Moving-camera contamination of image-plane coordinates is first-order for ego-view prediction. Sensor fixes are unavailable on JAAD and leaky on PIE (driver reaction ↔ crossing intent). We ask whether multi-agent geometry and/or classical ORB can recover camera motion without putting a speed sensor in the model.

## 2. Related Work

**Zhang, Ding & Tian (IEEE TIP 2024).** Two-tower model; action-aware loss `RMSE + ω(v)·RMSE_vehicle` with `ω(v)=v` (best). Ego action is an **input**. Protocol: τ=16, η=45, TTE∈[30,60], 50% overlap. PIE ADE **17.41** with I3D vision. ARB/FRB at 30-frame horizon.

**LIM (Chen et al., TITS 2025).** Ego speed causes causal confusion; **delete** it; skeleton-only + adversarial scrubbing.

**Ours.** Zhang uses the sensor; LIM deletes it; we recover geometry without a sensor at inference. Offline ORB affines supervise or replace the residual; never consumed as a live speed feature.

Attempt-3 OBD magnitude correlation (R²=0.297) is structurally similar to Zhang’s speed weighting and **failed** Steps 1–2 (variance gate 0.75%; ADE 65.9 vs 43.9 old-protocol baseline)— motivating directional, position-dependent ORB.

## 3. Method

**LEMC (learned).** Robust multi-agent displacement stats → GRU → residual → `cumsum` → subtract from ego track. Ego excluded from voting; ≥2 other agents required.

**ORB affine (fixed, main positive result).** Offline ORB + RANSAC `estimateAffinePartial2D` on consecutive frames with pedestrian boxes masked (~10% dilation). For ego centre `(cx,cy)`, the transition target is `M@[cx,cy,1]−[cx,cy]` (not global `(tx,ty)`). Cumsum from window start compensates **observation and future**; the GRU predicts in the stabilized frame; ego-view ADE adds the shift back (identical numerically to compensated-space ADE).

**Why full affine matters.** Translation-only ORB flattened only ~21% of standing+moving windows; full affine on the ego point flattened **89%** (diagnostic, n=300).

## 4. Setup

| Setting | Value |
|---------|-------|
| Protocol | T_obs=16, T_pred=45, TTE∈[30,60], overlap 50% |
| PIE windows | train 2227 / val 571 / test 1773 |
| Store | Augmented (YOLO pseudo-agents); median 8 agents/frame |
| Metrics | ADE/FDE (centres); ARB/FRB @ 30f; AUC/F1 |
| Seeds | 0,1,2 — mean±std |
| JAAD | 336 / 52 / 295 train/val/test (TTE); baseline 3 seeds done |

## 5. Results

Figures live in `results/figures/` (png + pdf):

| Fig | File | Content |
|-----|------|---------|
| 1 | `fig1_teaser` | Standing ped: raw vs ORB-affine compensated |
| 2 | `fig2_architecture` | Fixed ORB → GRU pipeline |
| 3 | `fig3_qualitative` | Raw vs compensated across speed percentiles |
| 4 | `fig4_orb_obd_scatter` | ORB \|t\| vs OBD (R²≈0.41) |
| 5 | `fig5_stratified_ade` | ADE by ego state |
| — | `fig_main_grid_ade` | Main grid ADE bars |
| — | `fig_sign_gate` | Sign-gate comparison |

### Table 1 — Dataset (PIE, TTE protocol)

| Split | Windows |
|-------|---------|
| Train | 2227 |
| Val   | 571 |
| Test  | 1773 |

*(Zhang: 3980 train — we additionally require a full 45-frame future.)*

### Table 2 — Main grid (PIE test)

| # | Method | ADE | FDE | AUC | F1 |
|---|--------|-----|-----|-----|-----|
| 1 | Baseline (no LEMC) | **60.81±3.31** | **121.75±9.87** | 0.89±0.01 | 0.74±0.03 |
| 2 | GT OBD speed input | 66.96±1.28 | 136.89±3.02 | **0.93±0.01** | **0.85±0.02** |
| 3 | LEMC + OBD aux | 85.77±11.19 | 164.25±11.33 | 0.80±0.01 | 0.60±0.03 |
| 4 | LEMC + learned ORB affine | 81.73±4.61 | 163.54±7.55 | 0.81±0.01 | 0.63±0.00 |
| 5 | LEMC, no aux | 81.68±5.13 | 159.91±2.39 | 0.77±0.05 | 0.64±0.01 |
| 6 | **Fixed ORB affine** | **60.49±1.83** | 130.28±1.31 | 0.80±0.01 | 0.60±0.01 |

ARB/FRB (baseline seed0): 35.1 / 61.3. Fixed ORB is ADE-neutral vs baseline; learned LEMC variants are strictly worse.

### Table 2b — JAAD (test, n=295, 3 seeds)

| Method | ADE | FDE | AUC | F1 |
|--------|-----|-----|-----|-----|
| Baseline bbox GRU | 83.10±2.00 | 156.25±4.31 | 0.47±0.05 | 0.90±0.00 |
| **Fixed ORB affine** | **62.68±2.34** | **120.11±2.82** | 0.44±0.00 | 0.90±0.00 |

Sign gate (fixed ORB): flatten all 71.9%; standing+moving **96.9%** (n=98). Unlike PIE (ADE-neutral), JAAD ADE drops ~20 px.

### Sign / variance gate (Step 1)

| Method | Flatten all | Standing + moving |
|--------|-------------|-------------------|
| Attempt-3 OBD aux (old protocol) | 0.7% | 0.8% — FAIL |
| Fixed ORB affine | 76.7% | **93.1% — PASS** |

ORB |tx,ty| vs OBD: Pearson r=0.64 (R²=0.41); stopped median |t|=0.09 px.

### Table 3 — Supervision ablation

Rows 3–5 of Table 2: neither OBD-correlation nor learned ORB regression recovers a useful residual under joint prediction loss. Fixed geometric compensation (row 6) is required for the variance gate.

### Stratified ADE/FDE (seed 0)

| Ego state | n | Baseline | GT speed | Fixed ORB |
|-----------|---|----------|----------|-----------|
| Accelerating | 251 | 58.5 / 127.8 | 88.6 / 173.2 | **52.2 / 122.6** |
| Constant | 1364 | 64.5 / 130.6 | 62.2 / 120.3 | **59.4 / 130.1** |
| Decelerating | 158 | **65.0 / 139.8** | 82.0 / 174.7 | 67.5 / 151.5 |

Fixed ORB helps most under acceleration (geometry-dominated). GT speed is worst on accelerating/decelerating for ADE while best for intent overall—speed encodes driver reaction more than pedestrian dynamics.

### Table 5 — Positioning

| | Sensor at inference? | Route | Pixels? | JAAD-ready? |
|--|----------------------|-------|---------|---------------|
| Zhang et al. | Yes (speed/action) | Two-tower + action-aware loss | Vision+bbox | Needs behaviour re-encoding |
| LIM | No (deleted) | Skeleton only | Pose | Yes |
| **Ours (fixed ORB)** | No | Offline ORB affine → compensate | Bbox only | Yes (ORB needs no speed) |
| Learned LEMC | No | Multi-agent residual | Bbox (+det) | Yes in principle; failed gate here |

### Table 6 — Efficiency

| | Params | CPU latency |
|--|--------|-------------|
| Baseline GRU | 210,523 | — |
| + LEMC module | 214,621 | 1.58 ms/seq (batch=1) |
| CPU | AMD Ryzen 9 7950X | |

### Old-protocol diagnostic (attempt 3, pre-fix)

| | ADE | FDE | AUC | F1 |
|--|-----|-----|-----|-----|
| Baseline | 43.94 | 81.41 | 0.756 | 0.703 |
| LEMC-on | 65.94 | 105.88 | 0.739 | 0.663 |

## 6. Limitations

Single GRU backbone (Zhang’s ADE 17.41 uses I3D); no on-vehicle hardware study; learned LEMC does not match ORB under joint training; ORB depth confound; hold-last for missing future affines.

## 7. Conclusion

Sensor-free ego-view compensation is possible with full-affine ORB and passes a standing-pedestrian variance gate, matching bbox-only ADE under the field protocol. Learned multi-agent residuals and OBD-magnitude losses do not. GT ego speed improves intention but degrades trajectory ADE—supporting LIM’s causal-confusion critique while showing that *geometry*, not the speed sensor, is the right ego-motion signal for trajectory.

## Reproducibility

```bash
# ORB affines (obs+pred frames)
python scripts/09b_extract_orb_affine.py --config configs/pie_lemc_on_augmented_orb_affine.yaml

# Main positive result
python scripts/12_fixed_orb_affine.py --seed 0

# Grid (learned ablations)
bash scripts/10_run_grid.sh

# Sign gate / OBD validation
python scripts/08_sign_variance_test.py --config configs/pie_lemc_on_augmented_orb_affine.yaml \
  --checkpoint-dir checkpoints/pie_fixed_orb_affine_v2_seed0
```
