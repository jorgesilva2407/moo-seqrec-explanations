"""Evaluate any checkpoint on the processed data (full ranking)."""

from __future__ import annotations

import argparse
import json

import torch

from seqrec import paths
from seqrec.data import load_histories, load_meta
from seqrec.data.tensors import SPLITS, build_padded
from seqrec.models.loader import load
from seqrec.training.metrics import DEFAULT_KS, evaluate_split


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser)
    parser.add_argument("--checkpoint", required=True, help="path to best.pt / last.pt")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = paths.resolve_data_dir(args.data_dir)
    out_dir = paths.resolve_out_dir(args.out_dir)
    processed = paths.processed_dir(data_dir, args.processed_name)
    meta = load_meta(processed)

    scorer = load(args.checkpoint, device=device, max_batch=args.batch_size)
    if scorer.num_items != meta["num_items"]:
        raise SystemExit(
            f"ERROR: checkpoint has {scorer.num_items} items but the processed data has {meta['num_items']}."
        )
    if scorer.data_hash and scorer.data_hash != meta["data_hash"]:
        print("WARNING: the processed data differs from the data this checkpoint was trained on.")

    data = build_padded(load_histories(processed), scorer.max_length, scorer.num_items)
    data = data.to_storage(torch.device("cpu"), device)
    result = {
        "checkpoint_id": scorer.checkpoint_id,
        "checkpoint": str(args.checkpoint),
        "data_hash": meta["data_hash"],
        "device": str(device),
    }
    for split in args.splits:
        result[split] = evaluate_split(scorer.logits, data, split, DEFAULT_KS, args.batch_size)

    out_path = paths.evaluations_dir(out_dir) / f"{scorer.checkpoint_id}_{args.processed_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved to {out_path}")
    return 0
