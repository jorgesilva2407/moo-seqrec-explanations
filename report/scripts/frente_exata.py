"""Check NSGA-II against the exact Pareto front on short histories (results/exact-front.json).

Every ML-20M user has at least 19 items before the test item, so no history is short enough to
enumerate. The check therefore uses the last ``--length`` items of each sampled user as the
history (2^12 masks), with the same candidates, target rule and backgrounds as the main
experiments. NSGA-II and the random baseline get ``--budget`` mask evaluations; their fronts after
each prefix in ``--checkpoints`` are compared with the exact front on the search backgrounds
(the fitness is deterministic, so no re-scoring is needed).

    uv run --group explain python report/scripts/frente_exata.py --processed <data-dir>/processed [--device mps]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from seqrec.data import load_histories, load_popularity
from seqrec.explain import analysis
from seqrec.explain.backgrounds import draw_backgrounds
from seqrec.explain.instances import Instance
from seqrec.explain.methods import Context
from seqrec.explain.scheduler import _padded, _scores, job_specs, method_seed, run_jobs
from seqrec.explain.verifier import Verifier
from seqrec.models.loader import load_from_hub

RESULTS = Path(__file__).resolve().parents[2] / "results"


def short_instances(histories, scorer, popularity, length: int, count: int, k: int, seed: int) -> list[Instance]:
    rng = np.random.default_rng([seed, length])
    instances = []
    for user in rng.permutation(len(histories)):
        seen = histories.full(int(user))[:-1].astype(np.int64)
        history = seen[-length:]
        logits = scorer.logits(torch.from_numpy(history)[None]).clone()[0]
        logits[torch.from_numpy(seen) - 1] = float("-inf")
        target = int(logits.topk(k).indices[rng.integers(k)]) + 1
        instances.append(Instance(int(user), int(histories.user_ids[user]) if hasattr(histories, "user_ids") else int(user),
                                  history, seen, target, 0, int(popularity[target]), 0))
        if len(instances) == count:
            break
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--methods", nargs="+", default=["nsga2", "random", "greedy"])
    parser.add_argument("--results-repo", default=None, help="also upload the JSON to exact-front/ in this dataset")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=2500)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[100, 250, 500, 1000, 2500])
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=25)
    args = parser.parse_args()

    start = time.perf_counter()
    scorer = load_from_hub("AbyssWanderer/sasrec-ml20m", "fp32/best.pt", device=torch.device(args.device))
    histories, popularity = load_histories(args.processed), load_popularity(args.processed)
    # twice as many candidates: keep only those whose full mask removes the target (as in the main runs)
    candidates = short_instances(histories, scorer, popularity, args.length, 2 * args.instances, args.k, 0)
    fills = draw_backgrounds(candidates, popularity, scorer.item_embeddings(), args.m, 0)
    check = Verifier(scorer, candidates, fills, k=args.k)
    valid = (check.evaluate(torch.arange(len(candidates)), check.full_masks()).necessity < 0).cpu().numpy()
    keep = np.flatnonzero(valid)[: args.instances]
    instances = [candidates[i] for i in keep]
    verifier = Verifier(scorer, instances, [fills[i] for i in keep], k=args.k)
    n = args.length

    # every non-empty mask of every instance
    everything = ((np.arange(1, 2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    exact = []
    for row in range(len(instances)):
        index = torch.full((len(everything),), row)
        exact.append(_scores(verifier.evaluate(index, torch.from_numpy(_padded(everything, verifier.length))))[:, :3])
    print(f"{len(instances)} instances of {len(candidates)} candidates, {len(everything)} masks each, {time.perf_counter() - start:.0f} s")

    contexts = [Context(n, np.maximum(np.abs(e[-1]), 1e-6), np.zeros(n), args.budget, args.pop_size) for e in exact]
    jobs = []
    run_jobs(verifier, job_specs(range(len(instances)), args.methods, args.seeds), contexts,
             lambda spec: method_seed(spec, instances[spec.instance].user_index), jobs.append)

    hv = analysis.HV(ref_point=np.full(3, analysis.REFERENCE))
    rows = {method: {c: {"hv_ratio": [], "front_found": []} for c in args.checkpoints} for method in args.methods}
    for row, objectives in enumerate(exact):
        low, high = objectives.min(axis=0), objectives.max(axis=0)
        scale = lambda points: (points - low) / np.where(high > low, high - low, 1.0)  # noqa: E731
        front = analysis.non_dominated(objectives)
        sort = analysis.NonDominatedSorting()
        front_keys = {everything[i].tobytes() for i in sort.do(objectives, only_non_dominated_front=True)}
        full = float(hv(scale(front)))
        per = {method: {c: ([], []) for c in args.checkpoints} for method in rows}
        for job in jobs:
            if job.spec.instance != row:
                continue
            for c in args.checkpoints:
                found = analysis.non_dominated(job.objectives[:c])
                per[job.spec.method][c][0].append(float(hv(scale(found))) / full)
                mine = np.asarray(job.masks[:c])[sort.do(job.objectives[:c], only_non_dominated_front=True)]
                per[job.spec.method][c][1].append(len({m.tobytes() for m in mine} & front_keys) / len(front_keys))
        for method in rows:
            for c in args.checkpoints:
                rows[method][c]["hv_ratio"].append(float(np.mean(per[method][c][0])))
                rows[method][c]["front_found"].append(float(np.mean(per[method][c][1])))
    summary = {
        method: {str(c): {"hv_ratio_mean": float(np.mean(v["hv_ratio"])), "hv_ratio_min": float(np.min(v["hv_ratio"])),
                          "front_found_mean": float(np.mean(v["front_found"]))} for c, v in per_c.items()}
        for method, per_c in rows.items()
    }
    out = {"config": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "masks": len(everything),
                      "checkpoint_id": scorer.checkpoint_id},
           "instances": len(instances), "candidates": len(candidates),
           "exact_front_size_mean": float(np.mean([len(analysis.non_dominated(e)) for e in exact])), "summary": summary}
    out_path = RESULTS / f"exact-front-n{args.length}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if args.results_repo:
        from huggingface_hub import HfApi

        HfApi().upload_file(path_or_fileobj=str(out_path), path_in_repo=f"exact-front/{out_path.name}",
                            repo_id=args.results_repo, repo_type="dataset")
    print(json.dumps(out["summary"], indent=1), "front size", out["exact_front_size_mean"], f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
