"""Has the explained SASRec learned sequential patterns? (results/model-checks.json)

On the test split (last item of each user, already-consumed items excluded, full ranking), compares
SASRec with two trivial baselines and with itself on shuffled histories:

* popularity: rank by training popularity;
* last-item transitions: rank by how often each item followed the user's last item in the training
  sequences (a first-order Markov chain; popularity breaks ties);
* SASRec on shuffled histories: same items, random order (the test of Klenitskiy et al.'s dataset
  analysis). A large drop means the model relies on the order of the history.

    uv run --group explain python report/scripts/sanidade_modelo.py --name zvuk --processed <dir> \
        --hub-repo AbyssWanderer/sasrec-zvuk --hub-file final/zvuk-h256-l3-d0.1/best.pt [--device mps]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from seqrec.data import load_histories, load_popularity
from seqrec.data.tensors import build_padded
from seqrec.models.loader import load_from_hub
from seqrec.training.metrics import evaluate_split

OUT = Path(__file__).resolve().parents[2] / "results" / "model-checks.json"


def transitions(histories, num_items: int) -> sparse.csr_matrix:
    """Counts of item b directly following item a in the training part of every history."""
    rows, cols = [], []
    for u in range(len(histories)):
        seq = histories.items[histories.offsets[u] : histories.offsets[u + 1]][:-2]
        rows.append(seq[:-1]), cols.append(seq[1:])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(num_items + 1, num_items + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--hub-repo", required=True)
    parser.add_argument("--hub-file", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--users", type=int, default=30000, help="random test users (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    scorer = load_from_hub(args.hub_repo, args.hub_file, device=device)
    histories, popularity = load_histories(args.processed), load_popularity(args.processed)
    data = build_padded(histories, scorer.max_length, scorer.num_items)
    rng = np.random.default_rng(args.seed)
    users = None if not args.users else torch.from_numpy(np.sort(rng.choice(len(histories), args.users, replace=False)))

    pop = torch.from_numpy(popularity[1:].astype(np.float32)).to(device)
    trans = transitions(histories, scorer.num_items)
    tie = (popularity[1:] / popularity.max() * 1e-3).astype(np.float32)

    def popularity_fn(inputs):
        return pop.expand(inputs.shape[0], -1).clone()

    def markov_fn(inputs):
        last = inputs[:, -1].cpu().numpy()
        return torch.from_numpy(trans[last][:, 1:].toarray() + tie).to(device)

    def sasrec_fn(inputs):
        return scorer.logits(inputs.to(device))

    generator = torch.Generator().manual_seed(args.seed)

    def shuffled_fn(inputs):
        inputs = inputs.cpu().clone()
        for row in inputs:
            real = torch.nonzero(row).squeeze(1)
            row[real] = row[real[torch.randperm(len(real), generator=generator)]]
        return scorer.logits(inputs.to(device))

    result = {}
    for label, fn in (("sasrec", sasrec_fn), ("sasrec_shuffled", shuffled_fn), ("popularity", popularity_fn),
                      ("last_item_transitions", markov_fn)):
        metrics = evaluate_split(lambda x, fn=fn: fn(x).float().cpu(), data, "test", ks=(10,), batch_size=256,
                                 user_index=users)  # the batches and the exclusion mask live on the CPU
        result[label] = {k: round(v, 4) for k, v in metrics.items()}
        print(args.name, label, result[label], flush=True)
    result["shuffle_drop_ndcg@10"] = round(1 - result["sasrec_shuffled"]["ndcg@10"] / result["sasrec"]["ndcg@10"], 4)
    result["users"] = args.users or len(histories)
    result["model"] = f"{args.hub_repo}/{args.hub_file}"
    out = json.load(open(OUT)) if OUT.exists() else {}
    out[args.name] = result
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
