"""Run the GECE baseline (TRACE's implementation) on the instances of a stage."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from seqrec import paths
from seqrec.commands import explain_run
from seqrec.data import load_histories, load_meta, load_popularity
from seqrec.explain.backgrounds import EVAL_STREAM, draw_backgrounds
from seqrec.explain.gece import run_gece
from seqrec.explain.scheduler import Job, JobSpec, _padded, _rescore, _scores, save_job
from seqrec.explain.verifier import Verifier
from seqrec.models.loader import load, load_from_hub


def add_arguments(parser: argparse.ArgumentParser) -> None:
    explain_run.add_arguments(parser)  # same instances, backgrounds and output layout (search options are unused)
    parser.add_argument("--gece-population", type=int, default=100)
    parser.add_argument("--gece-generations", type=int, default=None,
                        help="default: budget // population (GECE stops at its first valid counterfactual)")
    parser.add_argument("--gece-max-edits", type=int, default=10, help="most positions a valid counterfactual may change")
    parser.add_argument("--gece-restarts", action="store_true",
                        help="start GECE again until the budget is spent (method name gece_restarts)")
    parser.add_argument("--gece-max-masks", type=int, default=5000, help="most position sets scored per job")


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processed = paths.processed_dir(paths.resolve_data_dir(args.data_dir), args.processed_name)
    meta = load_meta(processed)
    scorer = load(args.checkpoint, device=device) if args.checkpoint else load_from_hub(args.hub_repo, args.hub_file, device=device)
    if scorer.data_hash and scorer.data_hash != meta["data_hash"]:
        print("WARNING: the processed data differs from the data this checkpoint was trained on.")
    METHOD = "gece_restarts" if args.gece_restarts else "gece"
    shard, shards = (int(part) for part in args.shard.split("/"))

    out_root = paths.explanations_dir(paths.resolve_out_dir(args.out_dir))
    prefix = f"{args.stage}/{scorer.checkpoint_id}"
    stage_dir = out_root / prefix
    (stage_dir / METHOD).mkdir(parents=True, exist_ok=True)
    if args.results_repo:
        explain_run.sync_down(args.results_repo, f"{prefix}/{METHOD}", out_root)

    histories, popularity = load_histories(processed), load_popularity(processed)
    everyone, _ = explain_run.select_instances(args, histories, scorer, popularity)
    instances = everyone[shard::shards]
    embeddings = scorer.item_embeddings()
    draw = lambda m, stream, fill: draw_backgrounds(  # noqa: E731
        instances, popularity, embeddings, m, args.instance_seed, stream, fill=fill, histories=histories)
    verifier = Verifier(scorer, instances, draw(args.m, 0, args.fill), k=args.k, chunk=args.chunk)
    eval_verifier = Verifier(scorer, instances, draw(args.m_eval, EVAL_STREAM, args.eval_fill or args.fill), k=args.k, chunk=args.chunk)

    with open(stage_dir / METHOD / f"instances-{shard}-of-{shards}.json", "w", encoding="utf-8") as f:
        json.dump([{"user_index": i.user_index, "target": i.target, "n": i.n} for i in instances], f)
    start, done = time.perf_counter(), 0
    for row, inst in enumerate(instances):
        for seed in args.seeds:
            spec = JobSpec(row, METHOD, seed)
            path = explain_run.job_path(stage_dir, inst, spec)
            if path.exists():
                continue
            result = run_gece(scorer, inst, seed, args.k, args.budget, args.gece_population, args.gece_generations,
                              args.gece_max_edits, args.gece_max_masks, args.gece_restarts)
            job = Job(spec, None, inst.n, args.budget)
            chosen_eval = None
            if len(result.masks):
                index = torch.full((len(result.masks),), row)
                job.masks = list(result.masks)
                job.rows = list(_scores(verifier.evaluate(index, torch.from_numpy(_padded(result.masks, verifier.length)))))
                _rescore([job], eval_verifier)
                if result.chosen is not None:
                    mask = _padded(result.masks[result.chosen : result.chosen + 1], verifier.length)
                    chosen_eval = _scores(eval_verifier.evaluate(index[:1], torch.from_numpy(mask)))[0].tolist()
            save_job(path, job, {
                "method": METHOD, "seed": seed, "user_index": inst.user_index, "target": inst.target,
                "m": args.m, "m_eval": args.m_eval, "k": args.k, "budget_passes": args.budget,
                "checkpoint_id": scorer.checkpoint_id, "queries": result.queries,
                "gece": {"valid": result.chosen is not None, "chosen": result.chosen, "truncated": result.truncated,
                         "chosen_eval": chosen_eval, "population": args.gece_population, "max_edits": args.gece_max_edits},
            })
            done += 1
        print(f"{row + 1}/{len(instances)} instances, {done} jobs, {time.perf_counter() - start:,.0f} s", flush=True)
    if args.results_repo:
        explain_run.sync_up(args.results_repo, f"{prefix}/{METHOD}", stage_dir / METHOD)
    return 0
