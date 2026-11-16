"""Statistics of the counterfactual GECE itself returns (results/gece-returned.json).

    uv run --group explain python report/scripts/gece_retornado.py <out-dir>/explanations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from seqrec.explain.scheduler import load_job

OUT = Path(__file__).resolve().parents[2] / "results" / "gece-returned.json"


def main() -> None:
    root = Path(sys.argv[1])
    out = {}
    for method in ("gece", "gece_restarts"):
        jobs = [load_job(path)["meta"] for path in sorted(root.glob(f"main/*/{method}/*.npz"))]
        valid = [job for job in jobs if job["gece"]["valid"]]
        scores = np.array([job["gece"]["chosen_eval"] for job in valid])  # N, S, size, PN, PS on the evaluation backgrounds
        out[method] = {
            "jobs": len(jobs), "valid": len(valid), "queries_mean": float(np.mean([job["queries"] for job in jobs])),
            "returned_size_mean": float(scores[:, 2].mean()),
            "returned_single_position_share": float((scores[:, 2] == 1).mean()),
            "returned_pn_mean": float(scores[:, 3].mean()), "returned_ps_mean": float(scores[:, 4].mean()),
            "returned_pn_above_half_share": float((scores[:, 3] > 0.5).mean()),
        }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
