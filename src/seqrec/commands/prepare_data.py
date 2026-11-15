"""Preprocess MovieLens-20M into padded-ready SASRec artifacts."""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
from pathlib import Path

from seqrec import paths
from seqrec.data import preprocess

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    paths.add_path_arguments(parser)
    parser.add_argument("--zip-path", default=None, help="manually downloaded ml-20m.zip")
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


def run(args: argparse.Namespace) -> int:
    data_dir = paths.resolve_data_dir(args.data_dir)
    out = paths.processed_dir(data_dir, args.processed_name)
    if (out / "data_meta.json").exists() and not args.force:
        raise SystemExit(f"ERROR: {out} already holds processed data; pass --force to overwrite.")

    if args.zip_path:
        zip_path = Path(args.zip_path).expanduser()
        if not zip_path.is_file():
            raise SystemExit(f"ERROR: --zip-path {zip_path} does not exist.")
    else:
        zip_path = paths.raw_zip_path(data_dir)
        if not zip_path.is_file():
            _download(zip_path)

    print(f"reading {zip_path}")
    df = preprocess.read_ratings(zip_path)
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
        "source_zip": zip_path.name,
    }
    meta = preprocess.write_artifacts(out, art, args.max_length, settings)
    print(
        f"wrote {out}: {meta['num_users']:,} users, {meta['num_items']:,} items, "
        f"{meta['num_interactions']:,} interactions, hash {meta['data_hash'][:12]}"
    )
    return 0
