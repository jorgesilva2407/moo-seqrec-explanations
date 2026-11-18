"""Run explanation searches (NSGA-II and baselines) under an equal budget of SASRec passes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from seqrec import paths
from seqrec.data import load_histories, load_meta, load_popularity
from seqrec.explain.backgrounds import EVAL_STREAM, FILLS, draw_backgrounds
from seqrec.explain.instances import Instance, build_instances
from seqrec.explain.methods import METHODS, Context, weight_lattice
from seqrec.explain.scheduler import Job, JobSpec, job_specs, method_seed, run_jobs, save_job
from seqrec.explain.verifier import DEFAULT_CHUNK, Verifier
from seqrec.models.loader import SASRecScorer, load, load_from_hub

DEFAULT_METHODS = ("nsga2", "scalar_ga", "random", "recency", "similarity", "ablation", "greedy", "enumeration")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser)
    parser.add_argument("--checkpoint", default=None, help="path to best.pt (default: download from the Hub)")
    parser.add_argument("--hub-repo", default="AbyssWanderer/sasrec-ml20m")
    parser.add_argument("--hub-file", default="fp32/best.pt")
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--stage", required=True, help="name of the output folder, e.g. main or tune-pop50")
    parser.add_argument("--instance-set", choices=("main", "tune", "redraw"), default="main",
                        help="tune = held-out users, disjoint from the main set; redraw = a second evaluation "
                             "set, disjoint from both main and tune")
    parser.add_argument("--per-stratum", type=int, default=33, help="instances per length stratum")
    parser.add_argument("--main-per-stratum", type=int, default=33,
                        help="size of the main set that the tune and redraw sets must avoid")
    parser.add_argument("--tune-per-stratum", type=int, default=10,
                        help="size of the tune set that the redraw set must avoid")
    parser.add_argument("--instance-seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--m", type=int, default=10, help="backgrounds per instance during the search")
    parser.add_argument("--m-eval", type=int, default=100,
                        help="independent backgrounds to re-score each front (0 = skip)")
    parser.add_argument("--fill", choices=FILLS, default="popularity", help="backgrounds used by the search")
    parser.add_argument("--eval-fill", choices=FILLS, default=None,
                        help="backgrounds that re-score the fronts (default: the same kind as --fill)")
    parser.add_argument("--instance-m", type=int, default=10,
                        help="popularity backgrounds of the check that selects instances (independent of --m and --fill)")
    parser.add_argument("--budget", type=int, default=100_000, help="SASRec passes per job")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--scalar-pop-size", type=int, default=20)
    parser.add_argument("--scalar-divisions", type=int, default=3,
                        help="weight vectors of the scalarised GA: simplex lattice divisions (2 -> 6, 3 -> 10, 4 -> 15)")
    parser.add_argument("--init", choices=("sparse", "uniform"), default="sparse", help="initial population of the GAs")
    parser.add_argument("--no-crossover", action="store_true", help="GAs use mutation only")
    parser.add_argument("--mutation-scale", type=float, default=1.0, help="per-bit flip rate is this over n")
    parser.add_argument("--partitions", type=int, default=5,
                        help="MOEA/D and NSGA-III: Das-Dennis partitions (3 objectives: 5 -> 21 directions, 6 -> 28)")
    parser.add_argument("--decomposition", choices=("tchebicheff", "pbi"), default="tchebicheff", help="MOEA/D")
    parser.add_argument("--neighbours", type=int, default=20, help="MOEA/D neighbourhood size")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--concurrency", type=int, default=64, help="jobs sharing one GPU batch")
    parser.add_argument("--shard", default="0/1", help="i/k: run the instances whose position is i modulo k")
    parser.add_argument("--results-repo", default=None,
                        help="Hugging Face dataset to resume from and upload to (needs HF_TOKEN)")
    parser.add_argument("--upload-every", type=float, default=600.0, help="seconds between uploads")


def job_path(stage_dir: Path, instance: Instance, spec: JobSpec) -> Path:
    return stage_dir / spec.method / f"u{instance.user_index}_t{instance.target}_s{spec.seed}.npz"


def select_instances(args, histories, scorer: SASRecScorer, popularity) -> tuple[list[Instance], dict]:
    """The main set (seed s), the tuning set (seed s + 1, other users) or the redrawn evaluation set
    (seed s + 2, users in neither). The earlier sets are rebuilt with their own sizes so the
    exclusions are exact."""
    build = lambda size, offset, exclude=(): build_instances(  # noqa: E731
        histories, scorer, popularity, size, args.instance_seed + offset, m=args.instance_m, k=args.k,
        exclude_users=exclude)
    main, stats = build(args.per_stratum if args.instance_set == "main" else args.main_per_stratum, 0)
    if args.instance_set == "main":
        return main, stats
    main_users = [inst.user_index for inst in main]
    tune_size = args.per_stratum if args.instance_set == "tune" else args.tune_per_stratum
    tune, tune_stats = build(tune_size, 1, main_users)
    if args.instance_set == "tune":
        return tune, tune_stats
    return build(args.per_stratum, 2, main_users + [inst.user_index for inst in tune])


def contexts_for(args, scorer: SASRecScorer, instances, verifier: Verifier) -> tuple[list[Context], np.ndarray]:
    """One context per instance, plus the objectives of the full mask (float [I, 3])."""
    full = verifier.evaluate(torch.arange(len(instances)), verifier.full_masks()).objectives.cpu().numpy()
    verifier.passes = 0  # shared reference values, not charged to any job
    embeddings = torch.nn.functional.normalize(scorer.item_embeddings(), dim=1).cpu()
    contexts = []
    for inst, row in zip(instances, full):
        scale = np.maximum(np.abs(row), 1e-6)
        similarity = (embeddings[torch.from_numpy(inst.history)] @ embeddings[inst.target]).numpy()
        contexts.append(
            Context(
                inst.n, scale, similarity, args.budget // (2 * args.m), args.pop_size, args.scalar_pop_size,
                weight_lattice(args.scalar_divisions), args.init, not args.no_crossover, args.mutation_scale,
                partitions=args.partitions, decomposition=args.decomposition, neighbours=args.neighbours,
            )
        )
    return contexts, full


def sync_down(repo: str, prefix: str, out_root: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{prefix}/*"], local_dir=out_root)


def sync_up(repo: str, prefix: str, stage_dir: Path) -> None:
    from huggingface_hub import HfApi

    try:
        HfApi().upload_folder(
            folder_path=stage_dir, repo_id=repo, repo_type="dataset", path_in_repo=prefix,
            ignore_patterns=["*.part"], commit_message=f"results: {prefix}",
        )
    except Exception as error:  # the files stay on disk; the next upload retries them
        print(f"WARNING: upload to {repo} failed: {error}")


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processed = paths.processed_dir(paths.resolve_data_dir(args.data_dir), args.processed_name)
    meta = load_meta(processed)
    scorer = load(args.checkpoint, device=device) if args.checkpoint else load_from_hub(args.hub_repo, args.hub_file, device=device)
    if scorer.num_items != meta["num_items"]:
        raise SystemExit(f"ERROR: checkpoint has {scorer.num_items} items but the processed data has {meta['num_items']}.")
    if scorer.data_hash and scorer.data_hash != meta["data_hash"]:
        print("WARNING: the processed data differs from the data this checkpoint was trained on.")
    shard, shards = (int(part) for part in args.shard.split("/"))

    out_root = paths.explanations_dir(paths.resolve_out_dir(args.out_dir))
    prefix = f"{args.stage}/{scorer.checkpoint_id}"
    stage_dir = out_root / prefix
    stage_dir.mkdir(parents=True, exist_ok=True)
    if args.results_repo:
        sync_down(args.results_repo, prefix, out_root)

    histories, popularity = load_histories(processed), load_popularity(processed)
    everyone, stats = select_instances(args, histories, scorer, popularity)
    instances = everyone[shard::shards]
    embeddings = scorer.item_embeddings()
    fills = draw_backgrounds(instances, popularity, embeddings, args.m, args.instance_seed,
                             fill=args.fill, histories=histories)
    verifier = Verifier(scorer, instances, fills, k=args.k, chunk=args.chunk)
    eval_verifier = None
    if args.m_eval:
        eval_fills = draw_backgrounds(instances, popularity, embeddings, args.m_eval, args.instance_seed, EVAL_STREAM,
                                      fill=args.eval_fill or args.fill, histories=histories)
        eval_verifier = Verifier(scorer, instances, eval_fills, k=args.k, chunk=args.chunk)
    contexts, full = contexts_for(args, scorer, instances, verifier)

    config = {key: value for key, value in vars(args).items() if key != "handler"}
    config.update(checkpoint_id=scorer.checkpoint_id, data_hash=meta["data_hash"], instance_stats=stats)
    described = [
        {"user_index": i.user_index, "user_id": i.user_id, "target": i.target, "n": i.n, "stratum": i.stratum,
         "target_rank": i.target_rank, "target_popularity": i.target_popularity,
         "full_mask_objectives": [float(x) for x in row]}
        for i, row in zip(instances, full)
    ]
    with open(stage_dir / f"shard-{shard}-of-{shards}.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "instances": described}, f, indent=2)

    specs = [
        spec for spec in job_specs(range(len(instances)), args.methods, args.seeds)
        if not job_path(stage_dir, instances[spec.instance], spec).exists()
    ]
    print(f"{len(instances)} instances, {len(specs)} jobs to run in {stage_dir}")
    start = time.perf_counter()
    last = {"upload": start, "print": start}

    def on_finish(job: Job) -> None:
        inst = instances[job.spec.instance]
        save_job(job_path(stage_dir, inst, job.spec), job, {
            "method": job.spec.method, "seed": job.spec.seed, "user_index": inst.user_index, "target": inst.target,
            "m": args.m, "m_eval": args.m_eval, "fill": args.fill, "eval_fill": args.eval_fill or args.fill, "k": args.k, "budget_passes": args.budget,
            "checkpoint_id": scorer.checkpoint_id,
        })

    def on_round(finished: int, total: int) -> None:
        now = time.perf_counter()
        if now - last["print"] >= 60 or finished == total:
            last["print"] = now
            print(f"{finished}/{total} jobs, {verifier.passes:,} passes, {verifier.passes / (now - start):,.0f} passes/s")
        if args.results_repo and now - last["upload"] >= args.upload_every:
            last["upload"] = now
            sync_up(args.results_repo, prefix, stage_dir)

    run_jobs(
        verifier, specs, contexts, lambda spec: method_seed(spec, instances[spec.instance].user_index),
        on_finish, args.concurrency, eval_verifier, on_round,
    )
    if args.results_repo:
        sync_up(args.results_repo, prefix, stage_dir)
    print(f"done in {time.perf_counter() - start:,.0f} s")
    return 0
