"""Where do the searches spend their budget, by mask size? (results/search-size-profile.json)

Diagnostic for MOEA/D's weak results on small masks: for each method on the main instances, the
share of evaluated masks with at most 10 positions, the median size of evaluated masks, and the
share of front masks with at most 10 positions.

    uv run --group explain python report/scripts/diagnostico_moead.py <out-dir>/explanations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from seqrec.explain.scheduler import load_job

OUT = Path(__file__).resolve().parents[2] / "results" / "search-size-profile.json"
METHODS = ("nsga2", "smsemoa", "nsga3", "moead", "scalar_ga", "nsga2_seeded")


def main() -> None:
    root = Path(sys.argv[1])
    out = {}
    for method in METHODS:
        evaluated_small, evaluated_median, front_small, first_gen_median = [], [], [], []
        for path in sorted(root.glob(f"main/*/{method}/*.npz")):
            job = load_job(path)
            sizes = job["objectives"][:, 2]
            relative = sizes / job["meta"]["n"]
            evaluated_small.append((sizes <= 10).mean())
            evaluated_median.append(np.median(relative))
            front_small.append((sizes[job["front"]] <= 10).mean())
            first_gen_median.append(np.median(sizes[:21]))
        out[method] = {
            "jobs": len(evaluated_small),
            "share_evaluated_le10": float(np.mean(evaluated_small)),
            "median_relative_size_evaluated": float(np.mean(evaluated_median)),
            "share_front_le10": float(np.mean(front_small)),
            "median_size_first_21": float(np.mean(first_gen_median)),
        }
        print(method, {k: round(v, 3) for k, v in out[method].items()})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
