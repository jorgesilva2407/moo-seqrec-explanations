"""MovieLens-20M preprocessing for SASRec with leave-one-out evaluation.

Reference: A. Klenitskiy and A. Vasilev, "Turning Dross Into Gold Loss: is BERT4Rec really
better than SASRec?", RecSys 2023 (arXiv 2309.07602). Their public repository
(github.com/antklen/sasrec-bert4rec-recsys23) was read for reference only; no code was copied.

Steps taken from the authors' public implementation:
  * interactions are implicit (only user, item, order are used, no rating value);
  * leave-one-out split per user: last item = test, second to last = validation, rest = train;
  * item ids start at 1 and 0 is padding;
  * user/item minimum-count filters are "distinct users per item" / "distinct items per user".

Things the authors' repository does NOT contain (it ships ML-20M already preprocessed by the
BERT4Rec repository) and that are therefore my own choices, marked TODO(verify):
  * TODO(verify): the thresholds (here 5 for users and items, as in BERT4Rec) and whether the
    original filter is iterative (here: repeated until stable, i.e. a k-core);
  * TODO(verify): tie-breaking among equal timestamps (here: by item id, deterministic);
  * item re-indexing (here: ascending original movieId -> 1..N).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seqrec.data.artifacts import (
    ITEM_MAP_FILE,
    META_FILE,
    POPULARITY_FILE,
    SEQUENCES_FILE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    compute_data_hash,
)

PADDING_ID = 0


def read_ratings(zip_path: Path) -> pd.DataFrame:
    """Read ``ratings.csv`` from the MovieLens-20M zip.

    Returns a DataFrame with int64 columns ``user_id``, ``item_id``, ``timestamp``.
    Ratings are ignored (all interactions are implicit feedback).
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("ratings.csv")]
        if len(names) != 1:
            raise ValueError(f"expected exactly one ratings.csv in {zip_path}, found {names}")
        with zf.open(names[0]) as f:
            df = pd.read_csv(
                f,
                usecols=["userId", "movieId", "timestamp"],
                dtype={"userId": np.int64, "movieId": np.int64, "timestamp": np.int64},
            )
    return df.rename(columns={"userId": "user_id", "movieId": "item_id"})


def read_zvuk(parquet_path: Path, min_play_seconds: float = 30.0, max_items: int | None = None) -> pd.DataFrame:
    """Read the Zvuk listening log (Shevchenko et al., KDD 2024; Kaggle ``alexxl/zvuk-dataset``,
    CC BY 4.0) as int64 ``user_id``, ``item_id``, ``timestamp`` (ms).

    * a play counts as a listen when it lasted at least ``min_play_seconds`` (about 40% of the
      plays are skips of a few seconds);
    * only the first listen of each (user, track) is kept, because the explanation task, like
      ML-20M, recommends items the user has not consumed yet;
    * with ``max_items``, only the tracks listened to by the most distinct users are kept, which
      keeps full cross-entropy training and every scoring pass affordable (both scale with the
      catalogue).
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    table = ds.dataset(str(parquet_path), format="parquet").to_table(
        columns=["user_id", "track_id", "datetime"], filter=pc.field("play_duration") >= min_play_seconds)
    df = pd.DataFrame({
        "user_id": table.column("user_id").to_numpy().astype(np.int64),
        "item_id": table.column("track_id").to_numpy().astype(np.int64),
        "timestamp": table.column("datetime").cast("int64").to_numpy(),
    })
    del table
    df = df.sort_values(["timestamp", "item_id"], kind="mergesort").drop_duplicates(["user_id", "item_id"], keep="first")
    if max_items is not None:
        listeners = df.groupby("item_id")["user_id"].size()  # one row per (user, item) now
        keep = listeners.sort_values(ascending=False, kind="mergesort").index[:max_items]
        df = df[df["item_id"].isin(keep)]
    return df.reset_index(drop=True)


def kcore_filter(df: pd.DataFrame, min_user_count: int, min_item_count: int) -> pd.DataFrame:
    """Drop items with < min_item_count distinct users and users with < min_user_count distinct
    items, repeating until nothing changes."""
    df = df.drop_duplicates(["user_id", "item_id"])
    while True:
        n_before = len(df)
        item_users = df.groupby("item_id")["user_id"].transform("nunique")
        df = df[item_users >= min_item_count]
        user_items = df.groupby("user_id")["item_id"].transform("nunique")
        df = df[user_items >= min_user_count]
        if len(df) == n_before:
            return df


def subsample_users(df: pd.DataFrame, max_users: int, seed: int) -> pd.DataFrame:
    """Keep a random subset of ``max_users`` users (deterministic given ``seed``)."""
    users = np.sort(df["user_id"].unique())
    if max_users >= len(users):
        return df
    rng = np.random.default_rng(seed)
    chosen = rng.choice(users, size=max_users, replace=False)
    return df[df["user_id"].isin(chosen)]


@dataclass
class Artifacts:
    """In-memory processed dataset; shapes and dtypes as in ``seqrec.data.artifacts``."""

    user_ids: np.ndarray
    offsets: np.ndarray
    items: np.ndarray
    split: np.ndarray
    popularity: np.ndarray
    model_to_orig: np.ndarray

    @property
    def num_users(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.model_to_orig.shape[0]) - 1


def build_artifacts(
    df: pd.DataFrame,
    min_user_count: int = 5,
    min_item_count: int = 5,
    max_users: int | None = None,
    seed: int = 42,
) -> Artifacts:
    """Filter, optionally subsample users, order by time, re-index items and split.

    ``df`` needs columns ``user_id``, ``item_id``, ``timestamp``.
    """
    if min_user_count < 3:
        raise ValueError("min_user_count must be >= 3 (train, validation and test item per user)")
    df = kcore_filter(df, min_user_count, min_item_count)
    if max_users is not None:
        df = subsample_users(df, max_users, seed)
    if len(df) == 0:
        raise ValueError("no interactions left after filtering")
    df = df.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")

    orig_items = np.sort(df["item_id"].unique())
    items = (np.searchsorted(orig_items, df["item_id"].to_numpy()) + 1).astype(np.int32)
    model_to_orig = np.concatenate([[PADDING_ID], orig_items]).astype(np.int64)

    user_ids, counts = np.unique(df["user_id"].to_numpy(), return_counts=True)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if counts.min() < 3:
        raise ValueError("every user must keep at least 3 interactions")

    split = np.full(items.shape[0], SPLIT_TRAIN, dtype=np.uint8)
    split[offsets[1:] - 2] = SPLIT_VALIDATION
    split[offsets[1:] - 1] = SPLIT_TEST

    popularity = np.bincount(items[split == SPLIT_TRAIN], minlength=len(model_to_orig)).astype(np.int64)
    return Artifacts(user_ids.astype(np.int64), offsets, items, split, popularity, model_to_orig)


def write_artifacts(
    processed_dir: Path,
    art: Artifacts,
    max_length: int,
    settings: dict,
) -> dict:
    """Write all artifacts and ``data_meta.json`` (last); return the metadata dict."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        processed_dir / SEQUENCES_FILE,
        user_ids=art.user_ids,
        offsets=art.offsets,
        items=art.items,
        split=art.split,
    )
    np.save(processed_dir / POPULARITY_FILE, art.popularity)
    item_map = {
        "model_to_orig": art.model_to_orig.tolist(),
        "orig_to_model": {str(int(o)): i for i, o in enumerate(art.model_to_orig) if i > 0},
    }
    with open(processed_dir / ITEM_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(item_map, f)

    meta = {
        "num_users": art.num_users,
        "num_items": art.num_items,
        "num_interactions": int(art.items.shape[0]),
        "max_length": int(max_length),
        "padding_id": PADDING_ID,
        "split": "leave-one-out (last=test, second-to-last=validation, rest=train)",
        "filters": settings,
        "data_hash": compute_data_hash(
            art.user_ids, art.offsets, art.items, art.split, art.popularity, art.model_to_orig
        ),
        "hash_algorithm": "sha256 over array contents",
    }
    with open(processed_dir / META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
