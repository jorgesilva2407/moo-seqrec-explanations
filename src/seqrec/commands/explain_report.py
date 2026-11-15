"""Compare stages or methods by the hypervolume of their fronts (re-scored on independent backgrounds)."""

from __future__ import annotations

import argparse
import json
from itertools import combinations

from scipy.stats import wilcoxon

from seqrec import paths
from seqrec.explain.analysis import add_hypervolume, load_jobs, per_instance_means, summarise


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser, processed=False)
    parser.add_argument("--stages", nargs="+", required=True, help="stage folders under <out-dir>/explanations")
    parser.add_argument("--by", choices=("stage", "method"), default="method",
                        help="stage: compare configurations (tuning); method: compare methods inside the stages")
    parser.add_argument("--methods", nargs="+", default=None, help="keep only these methods")
    parser.add_argument("--source", choices=("front_objectives", "objectives"), default="front_objectives")
    parser.add_argument("--results-repo", default=None, help="download the stages from this Hugging Face dataset first")
    parser.add_argument("--name", default=None, help="name of the saved report (default: the stages joined)")


def run(args: argparse.Namespace) -> int:
    root = paths.explanations_dir(paths.resolve_out_dir(args.out_dir))
    if args.results_repo:
        from huggingface_hub import snapshot_download

        snapshot_download(args.results_repo, repo_type="dataset", local_dir=root,
                          allow_patterns=[f"{stage}/*" for stage in args.stages])
    records = []
    for stage in args.stages:
        for checkpoint_dir in sorted((root / stage).glob("*/")):
            for record in load_jobs(checkpoint_dir, args.source):
                if args.methods and record["method"] not in args.methods:
                    continue
                record["label"] = stage if args.by == "stage" else record["method"]
                records.append(record)
    if not records:
        raise SystemExit(f"ERROR: no result files for {args.stages} under {root}")
    add_hypervolume(records)

    summary = summarise(records)
    order = sorted(summary, key=lambda name: -summary[name]["hv_mean"])
    print(f"{'':28s} {'inst':>5s} {'jobs':>5s} {'HV mean':>8s} {'HV med':>8s} {'front':>7s} {'evals':>8s}")
    for name in order:
        row = summary[name]
        print(f"{name:28s} {row['instances']:5d} {row['jobs']:5d} {row['hv_mean']:8.4f} {row['hv_median']:8.4f} "
              f"{row['front_size']:7.1f} {row['evaluations']:8.0f}")

    # paired Wilcoxon signed-rank tests on the per-instance means (two-sided, no correction)
    means = per_instance_means(records)
    tests = {}
    for a, b in combinations(order, 2):
        shared = sorted(set(means[a]) & set(means[b]))
        x, y = [means[a][i] for i in shared], [means[b][i] for i in shared]
        if len(shared) >= 6 and any(u != v for u, v in zip(x, y)):
            tests[f"{a} vs {b}"] = {"instances": len(shared), "p": float(wilcoxon(x, y).pvalue),
                                    "wins": sum(u > v for u, v in zip(x, y))}
    best = order[0]
    for name, test in tests.items():
        if name.startswith(f"{best} vs "):
            print(f"{name}: p = {test['p']:.4g}, {best} better on {test['wins']}/{test['instances']} instances")

    out_path = root / "reports" / f"{args.name or '+'.join(args.stages)}-{args.by}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stages": args.stages, "by": args.by, "source": args.source, "summary": summary, "wilcoxon": tests}, f, indent=2)
    print(f"saved to {out_path}")
    return 0
