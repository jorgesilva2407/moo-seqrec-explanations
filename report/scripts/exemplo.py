"""Extract one worked example (history, target, an explanation, fronts of a few methods) into
report/dados/exemplo.json, so the article's example figure and table build from a committed file.

Needs the result files of the main stage, the processed data and the MovieLens zip (for titles):

    uv run --group explain python report/scripts/exemplo.py \
        --explanations <out-dir>/explanations --processed <data-dir>/processed --zip <data-dir>/raw/ml-20m.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from seqrec.data import load_histories
from seqrec.explain.scheduler import load_job

METHODS = ("nsga2", "scalar_ga", "greedy", "ablation", "gece_restarts")
OUT = Path(__file__).resolve().parents[1] / "dados" / "exemplo.json"


def titles(zip_path: Path) -> dict[int, str]:
    with zipfile.ZipFile(zip_path) as z, z.open("ml-20m/movies.csv") as f:
        return {int(row["movieId"]): row["title"] for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--explanations", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--max-n", type=int, default=40, help="only histories this short (readable table)")
    parser.add_argument("--max-size", type=int, default=4)
    args = parser.parse_args()
    stage = next((args.explanations / "main").glob("*/"))

    # the instance whose NSGA-II front (seed 0) has the small mask with the highest min(PN, PS)
    best = None
    for path in sorted((stage / "nsga2").glob("*_s0.npz")):
        job = load_job(path)
        if job["meta"]["n"] > args.max_n:
            continue
        size = job["front_objectives"][:, 2]
        score = np.where(size <= args.max_size, np.minimum(job["front_pn"], job["front_ps"]), -1.0)
        row = int(np.argmax(score - 1e-3 * size))
        if best is None or score[row] > best[0]:
            best = (float(score[row]), path, row)
    _, path, row = best
    job = load_job(path)
    meta = job["meta"]
    user, target, n = meta["user_index"], meta["target"], meta["n"]

    name = titles(args.zip)
    to_movie = json.load(open(args.processed / "item_map.json"))["model_to_orig"]
    title = lambda item: name[to_movie[int(item)]]  # noqa: E731
    history = load_histories(args.processed).full(user)[:-1][-128:]
    assert len(history) == n

    def describe(job: dict, row: int) -> dict:
        mask = job["masks"][job["front"][row]]
        n_hat, s_hat, size = (float(x) for x in job["front_objectives"][row])
        return {"positions": (np.flatnonzero(mask) + 1).tolist(), "N": n_hat, "S": s_hat, "size": int(size),
                "PN": float(job["front_pn"][row]), "PS": float(job["front_ps"][row])}

    single = np.flatnonzero(job["front_objectives"][:, 2] == 1)
    fronts = {}
    for method in METHODS:
        other = load_job(stage / method / path.name)
        fronts[method] = other["front_objectives"].astype(float).round(4).tolist()
    out = {
        "user_index": user, "n": n, "target": title(target), "seed": 0,
        "history": [title(item) for item in history],
        "explanation": describe(job, row),
        "best_single": describe(job, int(single[np.argmin(job["front_objectives"][single, 0])])) if len(single) else None,
        "fronts": fronts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"user {user}, n = {n}, target: {out['target']}")
    print("explanation:", out["explanation"], [out["history"][p - 1] for p in out["explanation"]["positions"]])
    print("best single position:", out["best_single"])


if __name__ == "__main__":
    main()
