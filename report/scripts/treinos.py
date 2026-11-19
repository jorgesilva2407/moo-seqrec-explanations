"""SASRec training results for the appendix (results/sasrec-training.json), read from the Hub.

* ML-20M: the explained run, `fp32/` of AbyssWanderer/sasrec-ml20m (config and final metrics);
* Zvuk: every run of the two-round search in AbyssWanderer/sasrec-zvuk (`search/<run>` = round 1,
  8 epochs; `final/<run>` = round 2, resumed to convergence), with validation NDCG@10 per epoch.

    uv run --group explain python report/scripts/treinos.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OUT = Path(__file__).resolve().parents[2] / "results" / "sasrec-training.json"


def run(repo: str, folder: str) -> dict:
    get = lambda name: hf_hub_download(repo, f"{folder}/{name}")  # noqa: E731
    metrics = json.load(open(get("final_metrics.json")))
    config = json.load(open(get("config.json")))
    rows = list(csv.DictReader(open(get("metrics.csv"))))
    model, train = config["model_config"], config["train_args"]
    return {
        "hidden_size": model["hidden_size"], "num_layers": model["num_layers"], "num_heads": model["num_heads"],
        "dropout": model["dropout"], "max_length": model["max_length"], "num_items": model["num_items"],
        "lr": train["lr"], "batch_size": train["batch_size"], "patience": train["patience"],
        "precision": train.get("precision"), "grad_clip": train.get("grad_clip"),
        "epochs_run": len(rows), "best_epoch": metrics["best_epoch"],
        "epoch_seconds": round(sum(float(r["epoch_seconds"]) for r in rows) / len(rows), 1),
        "val": metrics["val"], "test": metrics["test"],
        "val_ndcg10_by_epoch": [round(float(r["val_ndcg@10"]), 4) for r in rows],
    }


def main() -> None:
    out = {"ml-20m": run("AbyssWanderer/sasrec-ml20m", "fp32")}
    files = HfApi().list_repo_files("AbyssWanderer/sasrec-zvuk")
    for stage in ("search", "final"):
        names = sorted({f.split("/")[1] for f in files if f.startswith(stage + "/") and f.endswith("final_metrics.json")})
        out[f"zvuk-{stage}"] = {name: run("AbyssWanderer/sasrec-zvuk", f"{stage}/{name}") for name in names}
    best = max(out["zvuk-final"], key=lambda n: out["zvuk-final"][n]["val"]["ndcg@10"])
    out["zvuk-selected"] = best
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("selected:", best, {k: len(v) if isinstance(v, dict) else v for k, v in out.items()})


if __name__ == "__main__":
    main()
