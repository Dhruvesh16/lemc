# Submission checklist — ITSC / IV 2026

Target: **IEEE IV 2026** (6–8 pages, blind review). Adjust venue string in `paper/main.tex` if switching to ITSC.

## Content

- [x] Contribution claim frozen in `DECISIONS.md` (fixed ORB primary; learned LEMC ablation)
- [x] PIE main grid (3 seeds, ADE/FDE/ARB/FRB/AUC/F1)
- [x] PIE sign/variance gate (standing + moving)
- [x] JAAD baseline in main paper (3 seeds)
- [x] JAAD fixed ORB + sign gate (ADE 62.7±2.3 vs baseline 83.1±2.0; standing+moving flatten 96.9%)
- [x] Stratified ADE figure + multi-dataset story in paper
- [x] Honest limitations (bbox GRU vs Zhang I3D, depth confound, no on-car study)

## Format

- [x] LaTeX source in `paper/main.tex` + `paper/refs.bib`
- [ ] Compile blind PDF: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`
- [ ] Remove author block / acknowledgements for submission
- [ ] Page count ≤ 8 (excluding references if venue allows)

## Ethics & licenses

- [x] PIE / JAAD dataset citations in bib
- [x] JAAD clips CC-BY 4.0 noted in README

## Supplementary

- [x] `LEMC_path_to_paper_report.pdf` (internal figures + tables)
- [x] Sign-gate details + failed learned-LEMC logs in `results/eval_*.txt`

## Before submit

- [ ] Run `python scripts/16_aggregate_grid_metrics.py` after all seeds complete
- [ ] `pytest tests/ -q` green
- [ ] One outsider read: “why accept?” (sensor-free ORB + gate + two datasets)
- [ ] Upload code + configs + README_REPRO with paper

## Docker GPU quick check (AMD Radeon RX 7600)

```bash
bash docker/build.sh
docker/run.sh python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# JAAD fixed ORB (baseline already trained — skip retrain):
SKIP_JAAD_BASELINE=1 docker/run.sh bash scripts/10_run_jaad_grid.sh
```
