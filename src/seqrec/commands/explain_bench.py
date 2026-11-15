"""Benchmark the explanation verifier: SASRec passes per second and how many backgrounds M are needed."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from seqrec import paths
from seqrec.data import load_histories, load_meta, load_popularity
from seqrec.explain.backgrounds import EVAL_STREAM, draw_backgrounds
from seqrec.explain.instances import build_instances
from seqrec.explain.verifier import Verifier
from seqrec.models.loader import load, load_from_hub


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser)
    parser.add_argument("--checkpoint", default=None, help="path to best.pt (default: download from the Hub)")
    parser.add_argument("--hub-repo", default="AbyssWanderer/sasrec-ml20m")
    parser.add_argument("--hub-file", default="fp32/best.pt")
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--per-stratum", type=int, default=4, help="instances per length stratum")
    parser.add_argument("--masks", type=int, default=200, help="random masks per instance")
    parser.add_argument("--ms", type=int, nargs="+", default=[5, 10, 20, 40], help="values of M to compare")
    parser.add_argument("--m-ref", type=int, default=200, help="independent backgrounds of the reference estimate")
    parser.add_argument("--chunks", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-repo", default=None, help="Hugging Face dataset to upload the result to (needs HF_TOKEN)")


def random_masks(verifier: Verifier, per_instance: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """``per_instance`` masks per instance with sizes uniform in 1..n/2 (index long [R], masks bool [R, L])."""
    valid = verifier.valid.cpu().numpy()
    index, masks = [], []
    for i, row in enumerate(valid):
        positions = np.flatnonzero(row)
        for size in rng.integers(1, max(len(positions) // 2, 1) + 1, size=per_instance):
            mask = np.zeros_like(row)
            mask[rng.choice(positions, size=size, replace=False)] = True
            index.append(i)
            masks.append(mask)
    return torch.tensor(index), torch.from_numpy(np.stack(masks))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a, ranks_b = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def mean_spearman(a: torch.Tensor, b: torch.Tensor, index: torch.Tensor) -> float:
    """Spearman correlation computed inside each instance, then averaged."""
    a, b, index = a.cpu().numpy(), b.cpu().numpy(), index.cpu().numpy()
    return float(np.mean([spearman(a[index == i], b[index == i]) for i in np.unique(index)]))


def throughput(verifier: Verifier, index: torch.Tensor, masks: torch.Tensor, chunk: int) -> dict:
    """Passes per second and peak GPU memory for one chunk size (``passes_per_second`` None on OOM)."""
    cuda = verifier.scorer.device.type == "cuda"
    verifier.chunk = chunk
    try:
        verifier.evaluate(index[:8], masks[:8])  # warm-up
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        before, start = verifier.passes, time.perf_counter()
        verifier.evaluate(index, masks)
        if cuda:
            torch.cuda.synchronize()
        speed = (verifier.passes - before) / (time.perf_counter() - start)
    except torch.OutOfMemoryError:
        speed = None
    peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda else None
    if cuda:
        torch.cuda.empty_cache()
    return {"passes_per_second": speed, "peak_gpu_gib": peak}


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processed = paths.processed_dir(paths.resolve_data_dir(args.data_dir), args.processed_name)
    meta = load_meta(processed)
    if args.checkpoint:
        scorer = load(args.checkpoint, device=device)
    else:
        scorer = load_from_hub(args.hub_repo, args.hub_file, device=device)
    if scorer.num_items != meta["num_items"]:
        raise SystemExit(
            f"ERROR: checkpoint has {scorer.num_items} items but the processed data has {meta['num_items']}."
        )
    if scorer.data_hash and scorer.data_hash != meta["data_hash"]:
        print("WARNING: the processed data differs from the data this checkpoint was trained on.")

    histories, popularity = load_histories(processed), load_popularity(processed)
    m_max = max(args.ms)
    instances, stats = build_instances(histories, scorer, popularity, args.per_stratum, args.seed, m=m_max, k=args.k)
    if not instances:
        raise SystemExit("ERROR: no valid instance found.")
    embeddings = scorer.item_embeddings()
    search = Verifier(scorer, instances, draw_backgrounds(instances, popularity, embeddings, m_max, args.seed), args.k)
    reference = Verifier(
        scorer, instances, draw_backgrounds(instances, popularity, embeddings, args.m_ref, args.seed, EVAL_STREAM), args.k
    )
    index, masks = random_masks(search, args.masks, np.random.default_rng(args.seed))

    result = {
        "checkpoint_id": scorer.checkpoint_id,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "instances": stats,
        "masks": int(masks.shape[0]),
        "throughput": {str(c): throughput(search, index, masks, c) for c in args.chunks},
    }
    speeds = {c: result["throughput"][str(c)]["passes_per_second"] for c in args.chunks}
    best = max((c for c in args.chunks if speeds[c]), key=lambda c: speeds[c], default=None)
    if best is None:
        raise SystemExit("ERROR: every chunk size ran out of GPU memory.")
    result["best_chunk"] = search.chunk = reference.chunk = best

    found, truth = search.evaluate(index, masks), reference.evaluate(index, masks)
    # spread of the margin over backgrounds: the standard error of N-hat / S-hat is std / sqrt(M)
    result["margin_std"] = {"necessity": float(truth.nec.std(dim=1).mean()), "sufficiency": float(truth.suf.std(dim=1).mean())}
    result["necessity_sufficiency_spearman"] = mean_spearman(truth.necessity, truth.sufficiency, index)
    result["agreement_with_reference"] = {}
    for m in sorted(args.ms):
        result["agreement_with_reference"][str(m)] = {
            "necessity_spearman": mean_spearman(found.nec[:, :m].mean(dim=1), truth.necessity, index),
            "sufficiency_spearman": mean_spearman(found.suf[:, :m].mean(dim=1), truth.suf.mean(dim=1), index),
            "pn_abs_error": float(((found.nec[:, :m] < 0).float().mean(dim=1) - truth.pn).abs().mean()),
            "ps_abs_error": float(((found.suf[:, :m] >= 0).float().mean(dim=1) - truth.ps).abs().mean()),
        }

    out_path = paths.explanations_dir(paths.resolve_out_dir(args.out_dir)) / "bench" / f"{scorer.checkpoint_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved to {out_path}")
    if args.results_repo:
        from huggingface_hub import HfApi

        target = f"bench/{result['device'].replace(' ', '_')}/{out_path.name}"
        HfApi().upload_file(path_or_fileobj=out_path, path_in_repo=target, repo_id=args.results_repo, repo_type="dataset")
        print(f"uploaded to {args.results_repo}/{target}")
    return 0
