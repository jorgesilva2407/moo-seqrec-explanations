"""Re-score more candidates of finished searches on the independent evaluation backgrounds.

A search returns its non-dominated masks, and only those are re-scored. A search that saw only
(N, C) keeps about one mask per size, far fewer than a three-objective front, so its re-scored front
has fewer chances to hold up on fresh backgrounds. This command takes, from each archive of
``--method`` in ``--source-stage``, the ``k`` masks with the lowest N-hat (search backgrounds) of
every size, re-scores them on the same evaluation backgrounds as every other front (outside the
budget), and writes one archive per variant in ``--stage``, laid out like ``explain-run`` output:

* ``<method>_le<max>_top<k>`` for each ``--top-k`` (sizes 1..``--max-size``);
* ``<method>_all_top<k>`` for each ``--top-k-all`` (every size the search evaluated).

The variant archive keeps the source run's meta (evaluations, passes: re-scoring is not charged),
its candidates as ``masks`` / ``objectives`` (search values, as stored) and all of them as the
``front``, with ``front_objectives`` / ``front_pn`` / ``front_ps`` from the evaluation backgrounds.

``--check`` re-scores the stored fronts of other archives (``method`` or ``method:seed``) and writes
the largest differences to the stored ``front_objectives`` in ``check.json``: the re-scoring must
reproduce what ``explain-run`` wrote.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from seqrec import paths
from seqrec.commands import explain_run
from seqrec.data import load_histories, load_meta, load_popularity
from seqrec.explain.backgrounds import EVAL_STREAM, draw_backgrounds
from seqrec.explain.scheduler import _padded, _scores, load_job
from seqrec.explain.verifier import Verifier
from seqrec.models.loader import load, load_from_hub

BATCH = 4096  # masks per evaluation call


def add_arguments(parser: argparse.ArgumentParser) -> None:
    explain_run.add_arguments(parser)  # same instances, evaluation backgrounds and output layout (search options unused)
    parser.add_argument("--source-stage", default="main", help="stage whose archives are re-scored")
    parser.add_argument("--method", default="nsga2_nc", help="method whose archives give the candidates")
    parser.add_argument("--max-size", type=int, default=10, help="largest size of the --top-k candidates")
    parser.add_argument("--top-k", type=int, nargs="*", default=[], help="masks per size 1..max-size")
    parser.add_argument("--top-k-all", type=int, nargs="*", default=[], help="masks per size, every size")
    parser.add_argument("--check", nargs="*", default=[], metavar="METHOD[:SEED]",
                        help="re-score the stored fronts of these archives and compare")


def top_per_size(objectives: np.ndarray, k: int, max_size: int | None) -> np.ndarray:
    """Rows of the ``k`` masks with the lowest N-hat of every size (1..max_size, or every size), in
    increasing size; ties keep the evaluation order."""
    sizes = objectives[:, 2].astype(int)
    rows = []
    for size in np.unique(sizes[sizes > 0]):
        if max_size is not None and size > max_size:
            break
        candidates = np.flatnonzero(sizes == size)
        rows.append(candidates[np.argsort(objectives[candidates, 0], kind="stable")[:k]])
    return np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)


def rescore(eval_verifier: Verifier, row: int, masks: np.ndarray) -> np.ndarray:
    """float64 [R, 5] (N, S, C, PN, PS) of ``masks`` (bool [R, n]) on the evaluation backgrounds."""
    out = []
    for start in range(0, len(masks), BATCH):
        part = masks[start : start + BATCH]
        index = torch.full((len(part),), row)
        out.append(_scores(eval_verifier.evaluate(index, torch.from_numpy(_padded(part, eval_verifier.length)))))
    return np.concatenate(out) if out else np.zeros((0, 5))


def write_variant(path: Path, source: dict, rows: np.ndarray, scores: np.ndarray, meta: dict) -> None:
    """An archive in the ``save_job`` layout whose front is every candidate."""
    arrays = {
        "masks": np.packbits(source["masks"][rows], axis=1),
        "objectives": source["objectives"][rows],
        "pn": source["pn"][rows],
        "ps": source["ps"][rows],
        "front": np.arange(len(rows), dtype=np.int64),
        "front_objectives": scores[:, :3].astype(np.float32),
        "front_pn": scores[:, 3].astype(np.float32),
        "front_ps": scores[:, 4].astype(np.float32),
        "meta": np.array(json.dumps(meta)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processed = paths.processed_dir(paths.resolve_data_dir(args.data_dir), args.processed_name)
    meta = load_meta(processed)
    scorer = load(args.checkpoint, device=device) if args.checkpoint else load_from_hub(args.hub_repo, args.hub_file, device=device)
    if scorer.data_hash and scorer.data_hash != meta["data_hash"]:
        print("WARNING: the processed data differs from the data this checkpoint was trained on.")

    out_root = paths.explanations_dir(paths.resolve_out_dir(args.out_dir))
    source_dir = out_root / args.source_stage / scorer.checkpoint_id
    prefix = f"{args.stage}/{scorer.checkpoint_id}"
    stage_dir = out_root / prefix
    stage_dir.mkdir(parents=True, exist_ok=True)
    checks = [(spec.split(":")[0], int(spec.split(":")[1]) if ":" in spec else None) for spec in args.check]
    if args.results_repo:
        from huggingface_hub import snapshot_download

        patterns = [f"{args.source_stage}/{scorer.checkpoint_id}/{args.method}/*"]
        patterns += [f"{args.source_stage}/{scorer.checkpoint_id}/{m}/*" + (f"_s{s}.npz" if s is not None else "")
                     for m, s in checks]
        snapshot_download(args.results_repo, repo_type="dataset", allow_patterns=patterns, local_dir=out_root)

    histories, popularity = load_histories(processed), load_popularity(processed)
    instances, _ = explain_run.select_instances(args, histories, scorer, popularity)
    fills = draw_backgrounds(instances, popularity, scorer.item_embeddings(), args.m_eval, args.instance_seed, EVAL_STREAM,
                             fill=args.eval_fill or args.fill, histories=histories)
    eval_verifier = Verifier(scorer, instances, fills, k=args.k, chunk=args.chunk)
    row_of = {(inst.user_index, inst.target): row for row, inst in enumerate(instances)}
    start = time.perf_counter()

    report = {}
    for method, seed in checks:
        worst = {"objectives": 0.0, "pn": 0.0, "ps": 0.0}
        paths_ = sorted((source_dir / method).glob(f"*_s{seed}.npz" if seed is not None else "*.npz"))
        for path in paths_:
            job = load_job(path)
            row = row_of[(job["meta"]["user_index"], job["meta"]["target"])]
            scores = rescore(eval_verifier, row, job["masks"][job["front"]])
            worst["objectives"] = max(worst["objectives"], float(np.abs(scores[:, :3] - job["front_objectives"]).max()))
            worst["pn"] = max(worst["pn"], float(np.abs(scores[:, 3] - job["front_pn"]).max()))
            worst["ps"] = max(worst["ps"], float(np.abs(scores[:, 4] - job["front_ps"]).max()))
        report[f"{method}" + (f":{seed}" if seed is not None else "")] = {"jobs": len(paths_), "max_abs_diff": worst}
        print(f"check {method} seed {seed}: {len(paths_)} jobs, largest differences {worst}", flush=True)
    if checks:
        with open(stage_dir / "check.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    variants = [(f"{args.method}_le{args.max_size}_top{k}", k, args.max_size) for k in args.top_k]
    variants += [(f"{args.method}_all_top{k}", k, None) for k in args.top_k_all]
    sources = sorted((source_dir / args.method).glob("*.npz"))
    for done, path in enumerate(sources, 1):
        targets = [stage_dir / name / path.name for name, _, _ in variants]
        if all(target.exists() for target in targets):
            continue
        job = load_job(path)
        row = row_of[(job["meta"]["user_index"], job["meta"]["target"])]
        chosen = {name: top_per_size(job["objectives"], k, limit) for name, k, limit in variants}
        union = np.unique(np.concatenate(list(chosen.values())))  # each mask is re-scored once
        scores = dict(zip(union.tolist(), rescore(eval_verifier, row, job["masks"][union])))
        for (name, k, limit), target in zip(variants, targets):
            rows = chosen[name]
            write_variant(target, job, rows, np.asarray([scores[r] for r in rows.tolist()]).reshape(-1, 5), {
                **job["meta"], "method": name, "source_method": args.method, "source_stage": args.source_stage,
                "top_k": k, "max_size": limit, "rescored": len(rows),
            })
        if done % 50 == 0 or done == len(sources):
            print(f"{done}/{len(sources)} archives, {eval_verifier.passes:,} passes, {time.perf_counter() - start:,.0f} s", flush=True)
    if args.results_repo:
        explain_run.sync_up(args.results_repo, prefix, stage_dir)
    print(f"done in {time.perf_counter() - start:,.0f} s")
    return 0
