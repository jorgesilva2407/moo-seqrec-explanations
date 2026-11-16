"""Compare the background ablations with the main run (results/background-ablation.json).

Fronts of the stages main, abl-users and abl-padding are put on one hypervolume scale per instance
(all were re-scored on the main run's popularity backgrounds), with paired Wilcoxon tests of main
against each ablation and the overlap of the positions NSGA-II selects.

    uv run --group explain python report/scripts/ablacao.py <out-dir>/explanations
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from seqrec.explain import analysis

root = Path(sys.argv[1])
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "results" / "background-ablation.json"
stages = ["main", "abl-users", "abl-padding"]
methods = ["nsga2", "scalar_ga", "greedy", "ablation"]
missing = [stage for stage in stages if not (root / stage).is_dir()]
if missing:
    raise SystemExit(f"ERROR: {missing} not under {root}; download them first (just explain-report --results-repo ...)")
records = []
for stage in stages:
    for d in sorted((root / stage).glob("*/")):
        for r in analysis.load_jobs(d):
            if r["method"] in methods:
                r["stage"], r["label"] = stage, f"{stage}:{r['method']}"
                records.append(r)
analysis.add_hypervolume(records)
analysis.add_hypervolume(records, 10, "hv_le10")
summary = analysis.summarise(records, fields=("hv", "hv_le10"))
quality = analysis.explanation_quality(records, 10)
out = {"summary": summary, "quality": quality, "tests": {}, "overlap": {}}
print(f"{'':24s} inst jobs      HV  HV<=10  front   PN    PS  recency")
for m in methods:
    for s in stages:
        k = f"{s}:{m}"; r, q = summary[k], quality[k]
        print(f"{k:24s} {r['instances']:4d} {r['jobs']:4d} {r['hv_mean']:7.4f} {r['hv_le10_mean']:7.4f} {r['front_size']:6.1f} {q['best_pn']:5.2f} {q['best_ps']:5.2f} {q['recency']:7.2f}")
print("\npaired Wilcoxon, main against the other fill (same method, fronts judged on popularity backgrounds)")
for field in ("hv", "hv_le10"):
    means = analysis.per_instance_means(records, field=field)
    for m in methods:
        for s in stages[1:]:
            a, b = means[f"main:{m}"], means[f"{s}:{m}"]
            ids = sorted(set(a) & set(b)); x = [a[i] for i in ids]; y = [b[i] for i in ids]
            p = float(wilcoxon(x, y).pvalue)
            out["tests"][f"{field}|{m}|{s}"] = {"p": p, "main_better": sum(u > v for u, v in zip(x, y)), "n": len(ids)}
            print(f"  {field:8s} {m:10s} vs {s:12s} p = {p:.3g}, main better on {sum(u > v for u, v in zip(x, y))} of {len(ids)}")

# which positions are chosen: frequency of each position among the small front masks of NSGA-II
def freq(rs):
    masks = [r["front_masks"][r["front"][:, 2] <= 10] for r in rs]
    masks = np.concatenate([m for m in masks if len(m)]) if any(len(m) for m in masks) else None
    return None if masks is None else masks.mean(axis=0)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
groups = defaultdict(lambda: defaultdict(list))
for r in records:
    if r["method"] == "nsga2":
        groups[r["instance"]][r["stage"]].append(r)
sims = defaultdict(list)
for inst, by in groups.items():
    main = sorted(by["main"], key=lambda r: r["seed"])
    a, b = freq(main[:2]), freq(main[2:])
    if a is not None and b is not None: sims["main seeds 0-1 vs 2-4"].append(cos(a, b))
    for s in stages[1:]:
        a, b = freq(by["main"]), freq(by[s])
        if a is not None and b is not None: sims[f"main vs {s}"].append(cos(a, b))
    a, b = freq(by["abl-users"]), freq(by["abl-padding"])
    if a is not None and b is not None: sims["abl-users vs abl-padding"].append(cos(a, b))
print("\ncosine similarity of position-selection frequencies (NSGA-II, front masks <= 10 positions), mean over instances")
for k, v in sims.items():
    out["overlap"][k] = float(np.mean(v)); print(f"  {k:28s} {np.mean(v):.3f} (median {np.median(v):.3f}, {len(v)} instances)")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
