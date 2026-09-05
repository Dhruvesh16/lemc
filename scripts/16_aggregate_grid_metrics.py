"""Aggregate ADE/FDE/ARB/FRB/AUC/F1 from results/eval_*.txt (mean ± std over seeds)."""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

METRICS = ["ade", "fde", "arb", "frb", "auc", "f1", "frac_flatten_all", "frac_flatten_standing_moving"]


def parse_eval(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in open(path):
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="results/eval_*.txt")
    parser.add_argument("--out", default="results/grid_summary.txt")
    args = parser.parse_args()

    groups: dict[str, list[dict[str, float]]] = {}
    pat = re.compile(r"eval_(.+)_seed(\d+)\.txt$")
    for path in sorted(glob.glob(args.pattern)):
        m = pat.search(path)
        if not m:
            continue
        tag, seed = m.group(1), int(m.group(2))
        groups.setdefault(tag, []).append(parse_eval(path))

    lines = []
    for tag in sorted(groups):
        rows = groups[tag]
        lines.append(f"## {tag} (n={len(rows)} seeds)")
        for metric in METRICS:
            vals = [r[metric] for r in rows if metric in r]
            if not vals:
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=0))
            lines.append(f"  {metric}: {mean:.2f} ± {std:.2f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
