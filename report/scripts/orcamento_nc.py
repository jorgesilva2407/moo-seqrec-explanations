"""What does NSGA-II gain by dropping S-hat? (results/nc-budget.json)

The NSGA-II that sees only (N, C) does not estimate S-hat, so the same 10^5 passes buy it 10,000
masks instead of 5,000. On the main instances this compares, on the search backgrounds (what each
search found on the backgrounds it optimised), the hypervolume of the (N, C) projection of masks
with at most 10 positions:

* three-objective NSGA-II (its 5,000 masks);
* the (N, C) NSGA-II after its first 5,000 masks (the same number of masks);
* the (N, C) NSGA-II after all 10,000 masks (the same number of passes);

and the (N, C) NSGA-II at 10,000 against itself at 5,000 masks (what the second half of the budget adds).

Every evaluated mask counts (the non-dominated set in (N, C) of the archive), scaled inside each
instance over the three and averaged over seeds; paired tests against NSGA-II over instances. The
fronts re-scored on independent backgrounds are compared in final-ml20m-method.json.

    uv run --group explain python report/scripts/orcamento_nc.py <out-dir>/explanations
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import wilcoxon

from seqrec.explain.analysis import REFERENCE, non_dominated, paired_statistics
from seqrec.explain.scheduler import load_job

OUT = Path(__file__).resolve().parents[2] / "results" / "nc-budget.json"
MAX_SIZE = 10
_HV2 = HV(ref_point=np.full(2, REFERENCE))


def nc_front(objectives: np.ndarray, masks: int | None = None) -> np.ndarray:
    """Non-dominated (N, C) points of the first ``masks`` evaluated masks with 1..MAX_SIZE positions."""
    points = objectives[:masks].astype(np.float64)
    points = points[(points[:, 2] > 0) & (points[:, 2] <= MAX_SIZE)][:, [0, 2]]
    return non_dominated(points)


def main() -> None:
    root = Path(sys.argv[1])
    values = defaultdict(lambda: defaultdict(list))  # variant -> instance -> one value per seed
    evaluations = {}
    for path in sorted(root.glob("main/*/nsga2/*.npz")):
        other = path.parent.parent / "nsga2_nc" / path.name
        full, nc = load_job(path), load_job(other)
        half = full["meta"]["evaluations"]
        evaluations = {"nsga2": half, "nsga2_nc": nc["meta"]["evaluations"]}
        fronts = {"nsga2": nc_front(full["objectives"]), "nsga2_nc_same_masks": nc_front(nc["objectives"], half),
                  "nsga2_nc": nc_front(nc["objectives"])}
        union = np.concatenate(list(fronts.values()))
        low, high = union.min(axis=0), union.max(axis=0)
        span = np.where(high > low, high - low, 1.0)
        instance = path.name.rsplit("_s", 1)[0]
        for name, front in fronts.items():
            values[name][instance].append(float(_HV2((front - low) / span)))
    means = {name: {i: float(np.mean(v)) for i, v in per.items()} for name, per in values.items()}
    instances = sorted(means["nsga2"])
    base = [means["nsga2"][i] for i in instances]
    out = {"instances": len(instances), "max_size": MAX_SIZE, "evaluations": evaluations, "hv_nc_search": {}}
    for name, per in means.items():
        x = [per[i] for i in instances]
        row = {"mean": float(np.mean(x))}
        if name != "nsga2":  # this variant against three-objective NSGA-II
            row.update(instances=len(instances), wins=sum(u > v for u, v in zip(x, base)), losses=sum(u < v for u, v in zip(x, base)),
                       p=float(wilcoxon(x, base).pvalue), **paired_statistics(x, base))
        out["hv_nc_search"][name] = row
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    # the second half of the budget: all 10,000 masks against the first 5,000 of the same runs
    x, y = ([means[name][i] for i in instances] for name in ("nsga2_nc", "nsga2_nc_same_masks"))
    out["doubling"] = {"instances": len(instances), "wins": sum(u > v for u, v in zip(x, y)), "losses": sum(u < v for u, v in zip(x, y)),
                       "p": float(wilcoxon(x, y).pvalue), **paired_statistics(x, y)}
    print("doubling", out["doubling"])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
