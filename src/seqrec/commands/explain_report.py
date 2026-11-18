"""Compare stages or methods by the hypervolume of their fronts (re-scored on independent backgrounds)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from seqrec import paths
from seqrec.explain import analysis

CHECKPOINTS = (50, 100, 250, 500, 1000, 2500, 5000, 10000)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser, processed=False)
    parser.add_argument("--stages", nargs="+", required=True, help="stage folders under <out-dir>/explanations")
    parser.add_argument("--by", choices=("stage", "method"), default="method",
                        help="stage: compare configurations (tuning); method: compare methods inside the stages")
    parser.add_argument("--methods", nargs="+", default=None, help="keep only these methods")
    parser.add_argument("--tag-stages", action="store_true",
                        help="with --by method: label the jobs of every stage after the first as method@stage, "
                             "to compare variants of a method that ran in their own stages")
    parser.add_argument("--pairs", nargs="+", default=[], metavar="A,B",
                        help="extra paired tests between two labels (the default tests are the best label against each other)")
    parser.add_argument("--source", choices=("front_objectives", "objectives"), default="front_objectives")
    parser.add_argument("--max-size", type=int, default=10, help="size limit of the 'small explanations' columns")
    parser.add_argument("--full", action="store_true",
                        help="also read every archive: convergence curves and correlation between N and S")
    parser.add_argument("--results-repo", default=None, help="download the stages from this Hugging Face dataset first")
    parser.add_argument("--name", default=None, help="name of the saved report (default: the stages joined)")


def paired_tests(records, order, field) -> dict:
    """Two-sided Wilcoxon signed-rank test of the best label against each other one (Holm-adjusted)."""
    means = analysis.per_instance_means(records, field=field)
    best, tests = order[0], {}
    for other in order[1:]:
        shared = sorted(set(means[best]) & set(means[other]))
        x, y = [means[best][i] for i in shared], [means[other][i] for i in shared]
        if len(shared) >= 6 and any(u != v for u, v in zip(x, y)):
            tests[other] = {"instances": len(shared), "p": float(wilcoxon(x, y).pvalue),
                            "wins": sum(u > v for u, v in zip(x, y)), "losses": sum(u < v for u, v in zip(x, y)),
                            **analysis.paired_statistics(x, y)}
    for test, adjusted in zip(tests.values(), analysis.holm([t["p"] for t in tests.values()])):
        test["p_holm"] = adjusted
    return {"best": best, "against": tests}


def pair_test(records, first: str, second: str, field: str) -> dict:
    """Two-sided Wilcoxon signed-rank test between two labels, with effect sizes (p unadjusted)."""
    means = analysis.per_instance_means(records, field=field)
    shared = sorted(set(means[first]) & set(means[second]))
    x, y = [means[first][i] for i in shared], [means[second][i] for i in shared]
    return {"instances": len(shared), "p": float(wilcoxon(x, y).pvalue),
            "wins": sum(u > v for u, v in zip(x, y)), "losses": sum(u < v for u, v in zip(x, y)),
            **analysis.paired_statistics(x, y)}


def print_tests(title: str, tests: dict) -> None:
    print(f"\n{title}: {tests['best']} against")
    for other, t in tests["against"].items():
        print(f"  {other:26s} p = {t['p']:.4g} (Holm {t['p_holm']:.4g}), better on {t['wins']}, worse on {t['losses']} of {t['instances']}")


def save_curves(path, curves: dict, checkpoints, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in curves.items():
        ax.plot(checkpoints, values, marker="o", markersize=3, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("masks evaluated")
    ax.set_ylabel("hypervolume (scaled per instance)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    root = paths.explanations_dir(paths.resolve_out_dir(args.out_dir))
    if args.results_repo:
        from huggingface_hub import snapshot_download

        snapshot_download(args.results_repo, repo_type="dataset", local_dir=root,
                          allow_patterns=[f"{stage}/*" for stage in args.stages])
    records = []
    for stage in args.stages:
        for checkpoint_dir in sorted((root / stage).glob("*/")):
            for record in analysis.load_jobs(checkpoint_dir, args.source):
                if args.methods and record["method"] not in args.methods:
                    continue
                record["label"] = stage if args.by == "stage" else record["method"]
                if args.by == "method" and args.tag_stages and stage != args.stages[0]:
                    record["label"] = f"{record['method']}@{stage}"
                records.append(record)
    if not records:
        raise SystemExit(f"ERROR: no result files for {args.stages} under {root}")

    small = f"hv_le{args.max_size}"
    analysis.add_hypervolume(records)
    analysis.add_hypervolume(records, args.max_size, small)
    analysis.add_hypervolume(records, args.max_size, f"{small}_nc", columns=(0, 2))  # the (N, C) projection
    analysis.add_igd_plus(records)
    analysis.add_igd_plus(records, args.max_size, f"igd_plus_le{args.max_size}")
    summary = analysis.summarise(records, fields=("hv", small, f"{small}_nc", "igd_plus", f"igd_plus_le{args.max_size}"))
    quality = analysis.explanation_quality(records, args.max_size)
    order = sorted(summary, key=lambda name: -summary[name]["hv_mean"])
    small_order = sorted(summary, key=lambda name: -summary[name][f"{small}_mean"])

    print(f"{'':26s} {'inst':>4s} {'jobs':>5s} {'HV':>7s} {'HV<=' + str(args.max_size):>7s} {'front':>6s} {'passes':>8s} "
          f"{'valid':>6s} {'PN':>5s} {'PS':>5s} {'recency':>7s}")
    for name in order:
        row, q = summary[name], quality.get(name, {})
        print(f"{name:26s} {row['instances']:4d} {row['jobs']:5d} {row['hv_mean']:7.4f} {row[small + '_mean']:7.4f} "
              f"{row['front_size']:6.1f} {row['passes']:8.0f} {q.get('valid', float('nan')):6.2f} "
              f"{q.get('best_pn', float('nan')):5.2f} {q.get('best_ps', float('nan')):5.2f} {q.get('recency', float('nan')):7.2f}")
    print(f"(valid, PN, PS, recency: front masks with at most {args.max_size} positions)")

    report = {"stages": args.stages, "by": args.by, "source": args.source, "max_size": args.max_size,
              "summary": summary, "quality": quality,
              "wilcoxon_hv": paired_tests(records, order, "hv"),
              "wilcoxon_small": paired_tests(records, small_order, small)}
    print_tests("Hypervolume", report["wilcoxon_hv"])
    print_tests(f"Hypervolume, at most {args.max_size} positions", report["wilcoxon_small"])

    if args.pairs:
        # one Holm family per field over all the requested pairs
        report["pairs"] = {}
        for pair in args.pairs:
            first, second = pair.split(",")
            report["pairs"][pair] = {field: pair_test(records, first, second, field) for field in ("hv", small)}
        for field in ("hv", small):
            tests = [report["pairs"][pair][field] for pair in args.pairs]
            for test, adjusted in zip(tests, analysis.holm([t["p"] for t in tests])):
                test["p_holm"] = adjusted
        print("\nExtra paired tests (Holm over these pairs, per field)")
        for pair in args.pairs:
            first, second = pair.split(",")
            for field, t in report["pairs"][pair].items():
                print(f"  {field:8s} {first} against {second}: Holm p = {t['p_holm']:.4g}, better on {t['wins']}, "
                      f"worse on {t['losses']} of {t['instances']}, mean diff {t['mean_diff']:+.4f} "
                      f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}], r = {t['rank_biserial']:+.2f}")

    # hypervolume by popularity of the target (terciles over the instances)
    popularity = {r["instance"]: r["target_popularity"] for r in records if r["target_popularity"] is not None}
    if len(popularity) >= 6:
        cuts = np.quantile(list(popularity.values()), [1 / 3, 2 / 3])
        bands = defaultdict(lambda: defaultdict(list))
        for label, per_instance in analysis.per_instance_means(records, field=small).items():
            for instance, value in per_instance.items():
                if instance in popularity:
                    bands[label][int(np.searchsorted(cuts, popularity[instance], side="right"))].append(value)
        report["small_hv_by_target_popularity"] = {
            label: [float(np.mean(per[b])) if per[b] else None for b in range(3)] for label, per in bands.items()
        }
        print(f"\nHV<={args.max_size} by target popularity (low, mid, high tercile)")
        for name in small_order:
            values = report["small_hv_by_target_popularity"].get(name, [None] * 3)
            print(f"  {name:26s} " + " ".join("   -   " if v is None else f"{v:7.4f}" for v in values))

    name = args.name or "+".join(args.stages)
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        checkpoints = [c for c in CHECKPOINTS if c <= max(r["evaluations"] for r in records)]
        for limit, tag in ((None, "all"), (args.max_size, f"le{args.max_size}")):
            curves = analysis.convergence(records, checkpoints, limit)
            report[f"convergence_{tag}"] = {"checkpoints": checkpoints, "curves": curves}
            title = "all sizes" if limit is None else f"at most {limit} positions"
            save_curves(out_dir / f"{name}-{args.by}-convergence-{tag}.png", curves, checkpoints, title)
            print(f"\nHypervolume (search backgrounds, {title}) after " + ", ".join(str(c) for c in checkpoints) + " masks")
            for label in order:
                if label in curves:
                    print(f"  {label:26s} " + " ".join(f"{v:7.4f}" for v in curves[label]))
        unbiased = [r for r in records if r["method"] in ("random", "enumeration")]
        if unbiased:
            report["objective_correlation"] = analysis.objective_correlation(unbiased)
            print("\nSpearman correlation between N and S among masks of the same size (random + enumeration):")
            print(json.dumps(report["objective_correlation"], indent=2))

    out_path = out_dir / f"{name}-{args.by}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved to {out_path}")
    return 0
