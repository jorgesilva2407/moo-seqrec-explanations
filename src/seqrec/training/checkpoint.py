"""Checkpoint writing (atomic) and RNG state capture.

A checkpoint holds only tensors, numbers, strings, lists and dicts so that
``torch.load(..., weights_only=True)`` can read it, including ``last.pt``.
"""

from __future__ import annotations

import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from seqrec.models.loader import (
    KEY_CONFIG,
    KEY_DATA_HASH,
    KEY_SEED,
    KEY_STATE,
    KEY_TRAIN_ARGS,
    KEY_VERSIONS,
)
from seqrec.models.sasrec import SASRecConfig

FORMAT_VERSION = 1


def library_versions() -> dict:
    return {
        # plain str(): torch.__version__ is a str subclass that weights_only loading rejects
        "python": str(platform.python_version()),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda or "none"),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
    }


def build_checkpoint(
    model: torch.nn.Module,
    model_config: SASRecConfig,
    train_args: dict,
    seed: int,
    data_hash: str,
    extra: dict | None = None,
) -> dict:
    """Self-contained payload: weights + full model config + provenance."""
    payload = {
        "format_version": FORMAT_VERSION,
        KEY_STATE: {k: v.detach().cpu() for k, v in model.state_dict().items()},
        KEY_CONFIG: model_config.to_dict(),
        KEY_TRAIN_ARGS: dict(train_args),
        KEY_SEED: int(seed),
        KEY_DATA_HASH: data_hash,
        KEY_VERSIONS: library_versions(),
    }
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """Write to a temp file in the same directory, fsync, then rename over ``path``."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_training_state(path: str | Path) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` (CPU tensors)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def capture_rng_state() -> dict:
    py = random.getstate()
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state = {
        "py_version": py[0],
        "py_internal": list(py[1]),
        "py_gauss": py[2],
        "np_keys": torch.from_numpy(np_keys.astype(np.int64)),
        "np_pos": int(np_pos),
        "np_has_gauss": int(np_has_gauss),
        "np_cached": float(np_cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate((state["py_version"], tuple(state["py_internal"]), state["py_gauss"]))
    np.random.set_state(
        (
            "MT19937",
            state["np_keys"].numpy().astype(np.uint32),
            state["np_pos"],
            state["np_has_gauss"],
            state["np_cached"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
