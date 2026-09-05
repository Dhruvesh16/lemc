# Related-work notes (Step 3) — how LEMC differs

Written 2026-08-25 after reading Zhang, Ding & Tian (IEEE TIP 2024) and LIM
(Chen et al., IEEE TITS 2025). Keep this paragraph-ready for Section 2.

## Zhang, Ding & Tian — "Decouple Ego-View Motions…" (IEEE TIP 2024)

**Protocol (adopt as field standard):**
- Observation τ = 16 frames; prediction η = 45 frames
- Time-to-event (TTE) ∈ [30, 60] frames (1–2 s before crossing event)
- 50% overlap when sampling sequences
- PIE: 3,980 train sequences (995 crossing); JAAD: 3,955 train (805 crossing)
- Metrics: ADE/FDE on bbox **centres**; ARB/FRB on full box coords at **30-frame** horizon

**Their numbers (reference):** ADE **17.41** on PIE (multi-modal: I3D vision + bbox motion + ego speed/action). Our bbox-only GRU baseline is expected to sit well above this.

**Action-aware loss (exact form):**
```
loss = RMSE(ℓ_η, ℓ̂_η) + ω(a) · RMSE(ℓ_η, ℓ̂_vehicle_η)
ω(v) = v^p   with p=1 (linear in normalized ego speed) best in ablation
```
Assumption: at high ego speed, vehicle-caused ego-view motion dominates, so the
vehicle tower is pulled toward the full observed trajectory. Ego action is an
**input** to the vehicle tower (PIE: OBD speed km/h; JAAD: re-encoded driver
behaviours 0–3). Decomposition is learned via this weighted RMSE — they do
**not** supervise against an independent geometric ego-motion estimate.

**Overlap with our attempt-3 OBD aux loss:** both up-weight / correlate with
ego speed so the model attributes more of the observed track to vehicle motion
when the car is fast. Structurally similar idea; theirs is tower-decomposition
with sensor as input, ours was residual-magnitude Pearson vs OBD without a
directional geometric target. That similarity is why ORB (directional,
sensor-free) is now the novelty requirement.

## LIM — "Causal Confusion…" (Chen et al., IEEE TITS 2025)

Ego-vehicle speed as an *input* causes train/test distribution shift and causal
confusion (bidirectional causality with crossing intention). LIM **deletes**
the speed modality: skeleton-only uni-modal model + adversarial scrubbing of
latent speed from skeletons. Competitive intent prediction without using speed.

## Positioning paragraph (for Related Work)

Zhang et al. decompose ego-view motion using the ego-vehicle sensor as both a
feature and a soft supervisor (speed-weighted action-aware loss). LIM treats
that sensor as a causal confounder and removes it. LEMC takes a third route:
recover the shared camera-motion component from multi-agent track geometry
alone, with offline ORB affine shifts as a directional training target that is
available on JAAD (no speed sensor) and never consumed at inference. Where
Zhang needs the sensor on every forward pass and LIM discards ego-motion
compensation entirely, we compensate without putting the sensor in the model.

## Implication of Steps 1–2 (recorded here so Section 5 can cite it)

Attempt-3 OBD-correlation aux loss achieved R²=0.297 against OBD but failed
the variance/sign gate (fraction var_comp < var_raw ≈ 0.7%) and **worsened**
ADE (65.9 vs 43.9). Magnitude–speed correlation is not sufficient supervision;
directional ORB regression is the intended fix.
