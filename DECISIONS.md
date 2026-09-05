# Locked decisions

Decisions that change downstream code/config and must not be silently
re-litigated later. Update this file, don't just remember it.

## Density gate (Step 2) — PIE

Ran on the full official train split (sets 01/02/04), not a single-video
spot check: **median 2 agents/frame** (mean 2.59, p95 7, max 11).

**Verdict: AMBER.** LEMC's feature set includes the scale-change cue
(bbox height ratio `h[t]/h[t-1]`) from the start, alongside the multi-agent
displacement statistics — median-agent displacement alone is not assumed
sufficient.

`max_agents` padding cap: set to 20 in `configs/default.yaml` (train p95=7,
test p95=8, test max=18 — 20 gives headroom above the observed max without
being wasteful).

JAAD density has **not** been checked yet (JAAD data not on disk) — rerun
`scripts/02_density_check.py --dataset jaad` once available (plan Step 9)
and do not assume PIE's amber verdict transfers; JAAD's filming style
differs.

## LEMC v1 validation on PIE (Step 6) — weak, diagnosed

First trained LEMC checkpoint (seed 0, `configs/pie_lemc_on.yaml`) produced
**R² = 0.036** against PIE OBD speed (Pearson r = -0.19, n=16965). Per the
plan's gate criterion this does not clear the bar to trust the result.

Ruled out: undertrained checkpoint. Retrained with early_stop_patience raised
20 -> converged to the *identical* best checkpoint (epoch 1, val_loss=0.54879)
regardless of patience -- the prediction-loss optimum genuinely sits that
early; more epochs only overfit further and never improved it.

Likely real cause, from the qualitative figure
(`results/lemc_qualitative_test.png`): at mean speed=0.00 (car fully stopped,
zero true ego-motion), LEMC still applies a large late-window shift in some
windows but not others with a near-identical raw-track shape. This is
consistent with the AMBER density verdict above -- with median 2 agents/frame,
the "shared displacement across agents" statistic frequently has only ONE
other agent to draw from, so it has no robustness against that single
agent's own idiosyncratic real motion and can misattribute it as camera
motion.

Not yet tried: (a) requiring a minimum valid_agent_count before trusting/
using the displacement stats (falling back to scale-ratio-only or zero when
too few agents are present), (b) the detector-based density boost (Plan B
in the original spec) to raise agent density above the amber threshold.

**Update, same session:** implemented (a) as two fixes in
`lemc/models/lemc.py`: (1) exclude ego from the displacement-stat agent pool
entirely (previously ego's own motion could leak into its own "shared
motion" estimate -- a real circularity bug, fixed regardless of outcome
below); (2) require >=2 valid *other* agents per transition before trusting
the displacement stats, zero otherwise. Retrained + re-validated on PIE
test: **R² = 0.034** (was 0.036) -- essentially unchanged, and the same
qualitative pathology persists (large spurious shift on some zero-OBD-speed
windows, near-zero shift on other near-identical-speed windows).

**Conclusion: this is very likely the AMBER-density limitation itself, not
a remaining code bug.** With median 2 agents/frame, even after excluding
ego, MIN_OTHER_AGENTS=2 is satisfied by exactly 2 other agents in many
windows -- a median of 2 values has no real outlier-robustness (it's an
average), so a single other pedestrian's idiosyncratic motion can still
dominate the "shared" estimate. Fixing this for real likely requires
raising agent density, i.e. Plan B from the original spec (run a detector
over the extracted PIE frames to add non-pedestrian objects -- cars, poles,
signs -- as additional "other agents" for the shared-motion statistic).
Not yet attempted this session.

**Update: Plan B implemented and it worked as a density fix.** Ran YOLOv8n +
ByteTrack (classes: bicycle, car, motorcycle, bus, truck, traffic light, fire
hydrant, stop sign, parking meter -- deliberately excludes 'person', already
covered by PIE's ped_annotations) over all 293K extracted PIE frames on GPU
(AMD Radeon RX 7600 via ROCm 7.14 + PyTorch 2.12, ~65 min for the full
dataset). Merged as extra "other agent" pseudo-tracks
(`pie_track_store_augmented.pkl`) via `lemc/data/track_store.py:merge_detections_into_stores`
(pseudo-agents can never become an "ego" prediction target, only vote in
LEMC's shared-motion statistic).

Density check on the augmented store: **median 8 agents/frame (train), GREEN**
(was 2, AMBER) -- `configs/pie_lemc_on_augmented.yaml`, `max_agents: 32`
(observed max 29, up from 18).

Notable infra detour to get GPU working: torch/torchvision version mismatch
after installing ultralytics (fixed by reinstalling torchvision from the
matching CPU wheel index); a corrupt-on-read (but not actually corrupt --
PIL opens it fine) frame `set01/video_0002/06635.jpg` needed defensive
try/except handling in `detect_and_track_video`; CPU multiprocessing for the
detector initially oversubscribed cores (8 processes each pulling ~370% CPU
via un-restricted BLAS/OpenCV threading despite `torch.set_num_threads(1)`)
and was on pace for ~24h -- switched to the GPU path instead, which needed
relocating Docker's storage root from the nearly-full `/` partition (22GB
free) to `/home` (408GB free) before the ~20GB `rocm/pytorch` image would
fit, and swapping `opencv-python` for `opencv-python-headless` inside the
container (missing `libxcb.so.1`, a GUI dependency the minimal ROCm image
doesn't have). GPU speedup over CPU was modest (~13ms/frame vs ~20-22ms/frame)
since YOLO's `track()` mode is inherently sequential per-frame for tracking
continuity, not a batched-throughput workload.

Retraining LEMC on the augmented store and re-running OBD validation is the
immediate next step -- not done as of this note.

**Update: retrained + re-validated on the augmented (green-density) store.
R² got slightly WORSE (0.0245 vs 0.034 before), not better.** This
contradicts the density hypothesis. The qualitative figure
(`results/lemc_qualitative_pie_lemc_augmented_seed0_test.png`) makes the
mechanism visible: compensated tracks now form bizarre non-physical loops
(the raw track is smooth/straight, the LEMC-compensated one spirals back on
itself), where the pre-augmentation qualitative figure at least showed
plausible-looking (if occasionally spurious) shifts. Diagnosis: `shift =
cumsum(residual)` amplifies frame-to-frame residual noise into large
cumulative drift over the 15-frame window, and the detector adds real noise
sources the plain per-transition median doesn't filter -- unconfident/low-
quality boxes (no confidence threshold applied in `detect_and_track_video`),
ByteTrack ID switches (a track ID can silently jump to a different physical
object), and imprecise bounding boxes on small/distant/partially-occluded
objects. More agents didn't mean more *robust* votes, it meant more *noisy*
votes, and LEMC -- trained purely on downstream prediction loss with no
explicit constraint tying it to true geometric ego-motion -- has no pressure
to reject that noise.

**Revised conclusion:** two independent interventions (ego-exclusion bug fix,
density boost via detection) both failed to meaningfully improve R², and the
second one made the qualitative behavior visibly worse. This points away
from "not enough agents" and toward a more fundamental issue: LEMC's
training signal (downstream trajectory/intent loss alone) does not
constrain the residual to actually track camera ego-motion -- it only needs
to reduce prediction loss by whatever means. Two candidate fixes, neither
attempted yet: (a) confidence-threshold detections and/or drop the
noisiest-voting frames before feeding LEMC (data-quality fix), (b) add
auxiliary supervision -- a loss term correlating the residual against PIE's
OBD speed (or a classical geometric estimate) during training, not just at
eval time -- which is exactly what the original risk register flagged for
this scenario ("if R² is low, add auxiliary supervision on PIE as an
ablation"). (b) is the more principled fix since it directly targets the
actual gap (nothing currently tells LEMC what "correct" ego-motion looks
like), but is a real architecture/training-loop change, not a quick patch.

**Update: implemented (b), and it worked.** Added
`lemc/train/losses.py:obd_correlation_loss` -- an auxiliary, PIE-only,
training-time-only loss (`1 - Pearson r` between per-window mean residual
magnitude and mean OBD speed, computed per training batch; scale/shift
invariant by construction, directly optimizing the same statistic the
validation gate checks). Wired into `run_epoch` behind `loss.obd_weight`
(config `pie_lemc_on_augmented_auxobd.yaml`, weight 0.5). LEMC still takes
zero sensor input at inference on either dataset -- this only shapes what
it learns during PIE training; the JAAD-trained model simply won't have
this term available (`obd_weight` has no effect without OBD speed in the
batch).

Retrained (best checkpoint now epoch 4, not epoch 1-2 as before -- the aux
loss changes convergence dynamics; residual magnitude noticeably larger and
more stable, ~0.10 vs ~0.06-0.08). Re-validated on PIE test:
**R² = 0.297 (Pearson r = 0.545, positive sign -- residual magnitude now
correctly increases WITH speed, unlike the earlier negative-r results)**.
Scatter plot shows a genuine visible linear trend, not noise. Qualitative
figure shows LEMC-compensated tracks now smoothly tracking the raw tracks
across all speed regimes (the wild non-physical loops from the
density-only run are gone).

**This clears the plan's Step 6 trust gate.** Three findings worth carrying
into the paper: (1) density alone doesn't fix ego-motion correlation and can
even hurt it by adding noisy voters without supervision to filter them; (2)
an explicit auxiliary correlation loss against ground-truth ego-motion
during training is what actually closes the gap; (3) this is consistent
with -- and empirically supports -- the plan's own risk-register prediction.
Next: repeat with more seeds, then decide whether density-boost +
auxiliary-loss both stay in the final design or whether density boost alone
is droppable now that (2) is clearly the load-bearing fix.

## Path-to-paper update (2026-08-25)

Steps 1–2 on attempt-3: sign gate **FAIL** (0.75% standing flatten); LEMC-on
ADE **worse** (65.9 vs 43.9). OBD-magnitude aux is not valid compensation.

Adopted Zhang protocol (τ=16, TTE 30–60, 50% overlap) + ARB/FRB. Full-affine
ORB (not translation) is the geometric fix: standing flatten **93%**, ADE
**60.5±1.8** ≈ baseline **60.8±3.3**. Learned LEMC still fails to match ORB
under joint loss. GT speed helps intent (AUC 0.93) but hurts ADE (67.0).
JAAD baseline (3 seeds, test n=295): ADE **83.1±2.0**, FDE **156.3±4.3**, AUC **0.47±0.05**, F1 **0.90±0.00**. Fixed ORB on JAAD requires video clips → ORB extract (`scripts/14–15`, then `10_run_jaad_grid.sh` in Docker GPU).

## Conference framing freeze (2026-08-26)

Locked for ITSC/IV-class submission. Do not silently re-litigate.

**Primary method name:** Fixed ORB-affine compensation (sensor-free at inference).
**Learned LEMC:** failed ablation only (OBD-magnitude / learned ORB / no-aux).

**Primary metrics:** ADE/FDE (centres), ARB/FRB @ 30 frames, standing-pedestrian
flatten % (variance gate). AUC/F1 are secondary.

**Contribution claim (sell this):**
1. Full partial-affine ORB recovers camera motion without a speed sensor.
2. Standing variance gate exposes why magnitude–speed aux fails.
3. Fixed ORB is ADE-neutral on PIE vs bbox-only GRU, helps under acceleration,
   and transfers to JAAD (no OBD).
4. GT speed helps intent / hurts ADE — supports LIM; Zhang needs the sensor.

**Do not claim:** beating Zhang TIP ADE 17.41 (I3D + vision). Scope is a
bbox-only geometric compensation study with an honest backbone gap.

**Venue bar:** conference (ITSC or IV); JAAD required in main paper.

## JAAD density gate (2026-08-26)

Built `jaad_track_store.pkl` from JAAD 2.0 annotations (323 videos; train/val/test
177/29/117). Train-split agent density: **median 4.0 / frame** (mean 5.0, p95 11,
max 19). **Verdict: GREEN** — no detector density boost required on JAAD.
ORB/image experiments still need `JAAD_clips` → `images/` (download blocked on
this host; see `scripts/14_download_jaad_clips.sh`).
