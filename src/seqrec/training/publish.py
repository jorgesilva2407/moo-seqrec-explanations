"""Upload a finished run to the Hugging Face Hub.

The token is read only from the ``HF_TOKEN`` environment variable (never a CLI argument, never
written to disk). ``huggingface_hub`` comes from the optional ``upload`` dependency group.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from seqrec.models.loader import checkpoint_id

ENV_TOKEN = "HF_TOKEN"
RUN_FILES = ("best.pt", "config.json", "final_metrics.json", "metrics.csv", "train.log")


def require_token() -> str:
    token = os.environ.get(ENV_TOKEN)
    if not token:
        raise SystemExit(
            f"ERROR: {ENV_TOKEN} is not set. Export a Hugging Face token with write access "
            "(e.g. as an environment variable of the vast.ai instance) before using upload."
        )
    return token


def check_repo_id(repo_id: str) -> None:
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise SystemExit(f"ERROR: repo id must look like 'user-or-org/name', got {repo_id!r}")


def check_folder(folder: str) -> str:
    folder = folder.strip("/")
    if folder.startswith(".") or ".." in folder.split("/"):
        raise SystemExit(f"ERROR: repo folder must be a plain relative path like 'fp32', got {folder!r}")
    return folder


def build_model_card(run_dir: Path, repo_id: str, folder: str = "") -> str:
    metrics = {}
    metrics_path = run_dir / "final_metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return f"""---
tags:
- recommender-systems
- sequential-recommendation
- sasrec
- pytorch
---

# SASRec (full cross-entropy) trained on MovieLens-20M

Checkpoint id: `{checkpoint_id(run_dir / "best.pt")}`. Run: `{run_dir.name}`.

Weights and provenance only (no MovieLens data). `best.pt` holds the weights, the full model
config, training arguments, seed and data hash, so it can be rebuilt with the `seqrec` package:

```python
from huggingface_hub import hf_hub_download
from seqrec.models.loader import load

path = hf_hub_download("{repo_id}", "{(folder + "/" if folder else "") + "best.pt"}")
scorer = load(path, device="cpu")
```

Final metrics (full ranking, leave-one-out, history items excluded):

```json
{json.dumps(metrics, indent=2)}
```

Recipe: Klenitskiy & Vasilev, "Turning Dross Into Gold Loss: is BERT4Rec really better than
SASRec?" (RecSys 2023, arXiv 2309.07602); SASRec: Kang & McAuley (ICDM 2018). Dataset:
MovieLens-20M (Harper & Konstan, 2015). Max sequence length here is 128, not the paper's 200.
"""


def upload_run(run_dir: Path, repo_id: str, private: bool = True, folder: str = "", include_last: bool = False) -> str:
    """Upload the run's files in one commit; returns the repo URL.

    With ``folder`` (e.g. ``fp32``) the files go under that folder of the repo and the root README is
    left alone, so several runs can live side by side. Without it they go to the repo root with a model card.
    """
    check_repo_id(repo_id)
    folder = check_folder(folder)
    prefix = f"{folder}/" if folder else ""
    token = require_token()
    run_dir = Path(run_dir)
    if not (run_dir / "best.pt").is_file():
        raise SystemExit(f"ERROR: {run_dir / 'best.pt'} does not exist; nothing to upload.")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        raise SystemExit(
            "ERROR: huggingface_hub is not installed. Build the image with "
            "DEPENDENCY_GROUPS=upload (or run `uv sync --group upload` locally)."
        )
    operations = [
        CommitOperationAdd(path_in_repo=prefix + name, path_or_fileobj=str(run_dir / name))
        for name in RUN_FILES + (("last.pt",) if include_last else ())
        if (run_dir / name).is_file()
    ]
    if not folder:
        operations.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=build_model_card(run_dir, repo_id).encode("utf-8"))
        )
    api = HfApi(token=token)
    # exist_ok: an existing repo keeps its current visibility.
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Add run {run_dir.name}",
    )
    return f"https://huggingface.co/{repo_id}"
