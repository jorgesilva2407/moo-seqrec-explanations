"""Preprocess MovieLens-20M or Zvuk into padded-ready SASRec artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from seqrec import paths
from seqrec.data import preprocess

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
ZVUK_DATASET = "alexxl/zvuk-dataset"
ZVUK_FILE = "zvuk-interactions.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser)
    parser.add_argument("--dataset", choices=("ml-20m", "zvuk"), default="ml-20m")
    parser.add_argument("--zip-path", default=None, help="manually downloaded ml-20m.zip")
    parser.add_argument("--zvuk-path", default=None,
                        help="zvuk-interactions.parquet (default: <data-dir>/raw, downloaded with the Kaggle CLI "
                             "using KAGGLE_API_TOKEN)")
    parser.add_argument("--min-play-seconds", type=float, default=30.0, help="Zvuk: shortest play counted as a listen")
    parser.add_argument("--max-items", type=int, default=None,
                        help="keep only the N tracks with most listeners before the k-core (Zvuk)")
    parser.add_argument("--max-users", type=int, default=None, help="subsample users (smoke runs)")
    parser.add_argument("--seed", type=int, default=42, help="seed for user subsampling")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-user-count", type=int, default=5)
    parser.add_argument("--min-item-count", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="overwrite existing processed artifacts")


def _download(dest: Path, attempts: int = 5, timeout: float = 60.0) -> None:
    """Fetch the zip; a stalled or refused connection is retried instead of hanging forever."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"downloading {MOVIELENS_URL} -> {dest}")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(MOVIELENS_URL, timeout=timeout) as response, open(tmp, "wb") as f:
                shutil.copyfileobj(response, f, length=1 << 20)
            tmp.replace(dest)
            return
        except OSError as error:  # URLError, timeouts and connection resets
            print(f"download attempt {attempt}/{attempts} failed: {error}")
            if attempt == attempts:
                raise SystemExit(f"ERROR: could not download {MOVIELENS_URL}; pass --zip-path with a local copy.")
            time.sleep(30 * attempt)


def _download_zvuk(dest: Path) -> None:
    """Fetch the interactions file with the Kaggle CLI (credentials: KAGGLE_API_TOKEN)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {ZVUK_DATASET}/{ZVUK_FILE} -> {dest}")
    command = ["kaggle", "datasets", "download", ZVUK_DATASET, "-f", ZVUK_FILE, "-p", str(dest.parent), "--unzip"]
    if subprocess.run(command).returncode != 0 or not dest.is_file():
        zipped = dest.with_name(dest.name + ".zip")
        if zipped.is_file():  # some CLI versions ignore --unzip for single files
            shutil.unpack_archive(zipped, dest.parent)
            zipped.unlink()
    if not dest.is_file():
        raise SystemExit(f"ERROR: could not download {ZVUK_FILE}; set KAGGLE_API_TOKEN or pass --zvuk-path.")


def run(args: argparse.Namespace) -> int:
    data_dir = paths.resolve_data_dir(args.data_dir)
    out = paths.processed_dir(data_dir, args.processed_name)
    if (out / "data_meta.json").exists() and not args.force:
        raise SystemExit(f"ERROR: {out} already holds processed data; pass --force to overwrite.")

    if args.dataset == "zvuk":
        source = Path(args.zvuk_path).expanduser() if args.zvuk_path else data_dir / "raw" / ZVUK_FILE
        if not source.is_file():
            _download_zvuk(source)
        print(f"reading {source}")
        df = preprocess.read_zvuk(source, args.min_play_seconds, args.max_items)
    else:
        if args.zip_path:
            source = Path(args.zip_path).expanduser()
            if not source.is_file():
                raise SystemExit(f"ERROR: --zip-path {source} does not exist.")
        else:
            source = paths.raw_zip_path(data_dir)
            if not source.is_file():
                _download(source)
        print(f"reading {source}")
        df = preprocess.read_ratings(source)
    print(f"{len(df):,} raw interactions")
    art = preprocess.build_artifacts(
        df,
        min_user_count=args.min_user_count,
        min_item_count=args.min_item_count,
        max_users=args.max_users,
        seed=args.seed,
    )
    settings = {
        "min_user_count": args.min_user_count,
        "min_item_count": args.min_item_count,
        "kcore_iterative": True,
        "max_users": args.max_users,
        "seed": args.seed,
        "dataset": args.dataset,
        "source": source.name,
        **({"min_play_seconds": args.min_play_seconds, "max_items": args.max_items, "first_listen_only": True}
           if args.dataset == "zvuk" else {}),
    }
    meta = preprocess.write_artifacts(out, art, args.max_length, settings)
    print(
        f"wrote {out}: {meta['num_users']:,} users, {meta['num_items']:,} items, "
        f"{meta['num_interactions']:,} interactions, hash {meta['data_hash'][:12]}"
    )
    return 0
