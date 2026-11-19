"""Do the small sufficient masks survive independent backgrounds? (results/ps-rescored.json)

verificacoes.py finds, per instance, the smallest evaluated mask with PS >= 0.5 on the 10 search
backgrounds, among thousands of masks: some reach it by chance (selection bias). This script takes
that mask for every main instance and re-scores it on the 100 independent evaluation backgrounds.

    uv run --group explain python report/scripts/ps_reavaliado.py <out-dir>/explanations --processed <data-dir>/processed [--device mps]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from seqrec.data import load_histories, load_popularity
from seqrec.explain.backgrounds import EVAL_STREAM, draw_backgrounds
from seqrec.explain.instances import build_instances
from seqrec.explain.scheduler import _padded, load_job
from seqrec.explain.verifier import Verifier
from seqrec.models.loader import load_from_hub

OUT = Path(__file__).resolve().parents[2] / "results" / "ps-rescored.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("explanations", type=Path)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    best = {}  # (user, target) -> (size, -ps, mask)
    for method in ("nsga2", "random"):
        for path in sorted(args.explanations.glob(f"main/*/{method}/*.npz")):
            job = load_job(path)
            key = (job["meta"]["user_index"], job["meta"]["target"])
            rows = np.flatnonzero((job["ps"] >= 0.5) & (job["objectives"][:, 2] > 0))
            for r in rows:
                cand = (float(job["objectives"][r, 2]), -float(job["ps"][r]))
                if key not in best or cand < best[key][:2]:
                    best[key] = (*cand, job["masks"][r])

    scorer = load_from_hub("AbyssWanderer/sasrec-ml20m", "fp32/best.pt", device=torch.device(args.device))
    histories, popularity = load_histories(args.processed), load_popularity(args.processed)
    instances, _ = build_instances(histories, scorer, popularity, 33, 0, m=10, k=10)
    verifier = Verifier(scorer, instances, draw_backgrounds(instances, popularity, scorer.item_embeddings(), 100, 0,
                                                            EVAL_STREAM), k=10)
    rows, masks, sizes, search_ps = [], [], [], []
    for i, inst in enumerate(instances):
        hit = best.get((inst.user_index, inst.target))
        if hit is None:
            continue
        rows.append(i), masks.append(hit[2]), sizes.append(hit[0]), search_ps.append(-hit[1])
    ev = verifier.evaluate(torch.tensor(rows), torch.from_numpy(np.stack([_padded(m[None], verifier.length)[0] for m in masks])))
    ps = ev.ps.cpu().numpy()
    sizes = np.array(sizes)
    out = {
        "instances": len(rows), "median_size": float(np.median(sizes)),
        "search_ps_mean": float(np.mean(search_ps)), "eval_ps_mean": float(ps.mean()),
        "share_still_sufficient": float((ps >= 0.5).mean()),
        "share_still_sufficient_le10": float((ps[sizes <= 10] >= 0.5).mean()) if (sizes <= 10).any() else None,
        "instances_le10": int((sizes <= 10).sum()),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
