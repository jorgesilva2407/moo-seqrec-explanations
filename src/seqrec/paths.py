"""The only place that resolves data/output paths and run directories.

Defaults come from the environment variables ``SASREC_DATA_DIR`` and
``SASREC_OUT_DIR`` and fall back to ``/workspace/data`` and ``/workspace/outputs``.
Nothing here touches the filesystem unless ``create=True`` is passed.

Layout::

    <data-dir>/raw/ml-20m.zip
    <data-dir>/<processed-name>/          (default name: "processed")
    <out-dir>/runs/<run-name>/
    <out-dir>/evaluations/
    <out-dir>/explanations/
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

ENV_DATA_DIR = "SASREC_DATA_DIR"
ENV_OUT_DIR = "SASREC_OUT_DIR"
DEFAULT_DATA_DIR = "/workspace/data"
DEFAULT_OUT_DIR = "/workspace/outputs"
DEFAULT_PROCESSED_NAME = "processed"
MOVIELENS_ZIP_NAME = "ml-20m.zip"


def default_data_dir() -> Path:
    return Path(os.environ.get(ENV_DATA_DIR) or DEFAULT_DATA_DIR)


def default_out_dir() -> Path:
    return Path(os.environ.get(ENV_OUT_DIR) or DEFAULT_OUT_DIR)


def resolve_data_dir(value: str | os.PathLike | None = None) -> Path:
    path = Path(value) if value else default_data_dir()
    return path.expanduser().resolve()


def resolve_out_dir(value: str | os.PathLike | None = None) -> Path:
    path = Path(value) if value else default_out_dir()
    return path.expanduser().resolve()


def raw_zip_path(data_dir: Path) -> Path:
    return data_dir / "raw" / MOVIELENS_ZIP_NAME


def processed_dir(data_dir: Path, name: str = DEFAULT_PROCESSED_NAME) -> Path:
    _check_name(name, "processed name")
    return data_dir / name


def runs_dir(out_dir: Path) -> Path:
    return out_dir / "runs"


def evaluations_dir(out_dir: Path) -> Path:
    return out_dir / "evaluations"


def explanations_dir(out_dir: Path) -> Path:
    return out_dir / "explanations"


def default_run_name(seed: int) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-seed{seed}"


def run_dir(out_dir: Path, run_name: str, create: bool = False) -> Path:
    _check_name(run_name, "run name")
    path = runs_dir(out_dir) / run_name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def add_path_arguments(parser: argparse.ArgumentParser, processed: bool = True) -> None:
    """Add the ``--data-dir`` / ``--out-dir`` (and ``--processed-name``) options."""
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"data directory (default: ${ENV_DATA_DIR} or {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"output directory (default: ${ENV_OUT_DIR} or {DEFAULT_OUT_DIR})",
    )
    if processed:
        parser.add_argument(
            "--processed-name",
            default=DEFAULT_PROCESSED_NAME,
            help="sub-directory of --data-dir holding the processed artifacts "
            f"(default: {DEFAULT_PROCESSED_NAME})",
        )


def _check_name(name: str, what: str) -> None:
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError(f"invalid {what}: {name!r} (must be a plain directory name)")
