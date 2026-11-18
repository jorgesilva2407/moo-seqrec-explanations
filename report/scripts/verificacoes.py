"""Offline checks of three explanations the article gives (results/offline-checks.json).

1. Cost of seeding: share of the evaluated masks with at most 10 positions, NSGA-II vs seeded.
2. Drop of the N-S correlation in large masks: spread of N-hat and S-hat across masks of the same
   size bin (random + enumeration archives) against the Monte Carlo noise of an estimate,
   sigma / sqrt(M), with sigma the per-background margin spread measured by explain-bench.
3. PS ceiling: per instance, the smallest mask size at which any evaluated mask (NSGA-II and
   random archives) reaches PS >= 0.5 on the search backgrounds.

    uv run --group explain python report/scripts/verificacoes.py <out-dir>/explanations
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from seqrec.explain.scheduler import load_job

OUT = Path(__file__).resolve().parents[2] / "results" / "offline-checks.json"
BINS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LABELS = ["0-10", "10-25", "25-50", "50-75", "75-100"]


def jobs(root: Path, method: str):
    for path in sorted(root.glob(f"main/*/{method}/*.npz")):
        yield load_job(path)


def main() -> None:
    root = Path(sys.argv[1])
    out = {}

    share = {}
    for method in ("nsga2", "nsga2_seeded"):
        share[method] = float(np.mean([(job["objectives"][:, 2] <= 10).mean() for job in jobs(root, method)]))
    out["share_of_evaluated_masks_le10"] = share

    bench = json.load(open(next(root.glob("bench/*/*.json"))))
    m = 10
    noise = {k: float(v) / np.sqrt(m) for k, v in bench["margin_std"].items()}
    spread = defaultdict(lambda: defaultdict(list))
    for method in ("random", "enumeration"):
        for job in jobs(root, method):
            objectives, n = job["objectives"].astype(float), job["meta"]["n"]
            bins = np.searchsorted(BINS, objectives[:, 2] / n, side="left") - 1
            for b in range(len(LABELS)):
                rows = objectives[bins == b]
                if len(rows) >= 20:
                    spread[LABELS[b]]["necessity"].append(rows[:, 0].std())
                    spread[LABELS[b]]["sufficiency"].append(rows[:, 1].std())
    out["mc_noise_per_estimate"] = noise
    out["spread_by_size_bin"] = {
        label: {k: {"spread": float(np.mean(v)), "spread_over_noise": float(np.mean(v) / noise[k])} for k, v in d.items()}
        for label, d in spread.items()
    }

    smallest = defaultdict(lambda: np.inf)
    sizes = {}
    for method in ("nsga2", "random"):
        for job in jobs(root, method):
            key = (job["meta"]["user_index"], job["meta"]["target"])
            sizes[key] = job["meta"]["n"]
            reached = job["objectives"][job["ps"] >= 0.5, 2]
            if len(reached):
                smallest[key] = min(smallest[key], float(reached.min()))
    values = np.array([smallest[k] for k in sizes])
    relative = np.array([smallest[k] / sizes[k] for k in sizes])
    out["smallest_size_with_ps_ge_half"] = {
        "instances": len(values), "never": int(np.isinf(values).sum()),
        "median": float(np.median(values)), "quartiles": [float(np.quantile(values, q)) for q in (0.25, 0.75)],
        "share_le10": float((values <= 10).mean()), "median_relative": float(np.median(relative)),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
