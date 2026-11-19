"""Is the low N-S correlation of small masks real or Monte Carlo noise? (results/conflict-m100.json)

For every main instance, scores all single-position masks and a random sample of two-position masks
with M = 10 (the search backgrounds) and with M' = 100 (the independent evaluation backgrounds), and
reports the Spearman correlation between N-hat and S-hat among masks of each size under both. If the
correlation of small masks rises a lot with M', most of the apparent conflict was noise.

    uv run --group explain python report/scripts/conflito_m100.py --processed <data-dir>/processed [--device mps]
    (Zvuk: --processed <data-dir>/processed_zvuk --hub-repo AbyssWanderer/sasrec-zvuk
     --hub-file final/zvuk-h256-l3-d0.1/best.pt --per-stratum 15 --out conflict-m100-zvuk.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from seqrec.data import load_histories, load_popularity
from seqrec.explain.backgrounds import EVAL_STREAM, SEARCH_STREAM, draw_backgrounds
from seqrec.explain.instances import build_instances
from seqrec.explain.scheduler import _padded
from seqrec.explain.verifier import Verifier
from seqrec.models.loader import load_from_hub

RESULTS = Path(__file__).resolve().parents[2] / "results"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairs", type=int, default=300, help="two-position masks sampled per instance")
    parser.add_argument("--hub-repo", default="AbyssWanderer/sasrec-ml20m")
    parser.add_argument("--hub-file", default="fp32/best.pt")
    parser.add_argument("--per-stratum", type=int, default=33, help="size of the main set (Zvuk: 15)")
    parser.add_argument("--out", default="conflict-m100.json", help="file name under results/")
    args = parser.parse_args()

    scorer = load_from_hub(args.hub_repo, args.hub_file, device=torch.device(args.device))
    histories, popularity = load_histories(args.processed), load_popularity(args.processed)
    instances, _ = build_instances(histories, scorer, popularity, args.per_stratum, 0, m=10, k=10)  # the main set
    embeddings = scorer.item_embeddings()
    verifiers = {m: Verifier(scorer, instances, draw_backgrounds(instances, popularity, embeddings, m, 0, stream), k=10)
                 for m, stream in ((10, SEARCH_STREAM), (100, EVAL_STREAM))}
    rng = np.random.default_rng(0)
    result = {size: {m: [] for m in verifiers} for size in (1, 2)}
    for row, inst in enumerate(instances):
        n = inst.n
        singles = np.eye(n, dtype=bool)
        chosen = set()
        while len(chosen) < min(args.pairs, n * (n - 1) // 2):
            chosen.add(tuple(sorted(rng.choice(n, size=2, replace=False))))
        pairs = np.zeros((len(chosen), n), dtype=bool)
        for i, (a, b) in enumerate(sorted(chosen)):
            pairs[i, [a, b]] = True
        for size, masks in ((1, singles), (2, pairs)):
            index = torch.full((len(masks),), row)
            for m, verifier in verifiers.items():
                ev = verifier.evaluate(index, torch.from_numpy(_padded(masks, verifier.length)))
                result[size][m].append(spearman(ev.necessity.cpu().numpy(), ev.sufficiency.cpu().numpy()))
        print(f"{row + 1}/{len(instances)}", flush=True)
    out = {f"size_{size}": {f"M={m}": {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))} for m, v in per.items()}
           for size, per in result.items()}
    out["instances"] = len(instances)
    with open(RESULTS / args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
