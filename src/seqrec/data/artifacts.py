"""Read-only access to the processed artifacts written by ``seqrec prepare-data``.

Every function takes the *processed directory* (``paths.processed_dir(data_dir)``).
Item ids are ``1..N`` and ``0`` is padding. Files:

* ``sequences.npz``  user_ids int64[U], offsets int64[U+1], items int32[T], split uint8[T]
* ``item_map.json``  ``model_to_orig`` (list, length N+1, entry 0 is 0) and ``orig_to_model``
* ``popularity.npy`` int64[N+1], interaction counts over the TRAIN portion only, entry 0 is 0
* ``data_meta.json`` counts, filter settings and ``data_hash``

Each user's items are stored chronologically in the flat ``items`` array, the user ``u``
owning ``items[offsets[u]:offsets[u+1]]``. ``split`` marks every flat entry with
0 = train, 1 = validation (second to last), 2 = test (last).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_FILE = "data_meta.json"
SEQUENCES_FILE = "sequences.npz"
ITEM_MAP_FILE = "item_map.json"
POPULARITY_FILE = "popularity.npy"

SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_TEST = 2


def compute_data_hash(
    user_ids: np.ndarray,
    offsets: np.ndarray,
    items: np.ndarray,
    split: np.ndarray,
    popularity: np.ndarray,
    model_to_orig: np.ndarray,
) -> str:
    """SHA-256 over the array contents (not the files, so it does not depend on zip timestamps)."""
    digest = hashlib.sha256()
    for array, dtype in (
        (user_ids, np.int64),
        (offsets, np.int64),
        (items, np.int32),
        (split, np.uint8),
        (popularity, np.int64),
        (model_to_orig, np.int64),
    ):
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Histories:
    """All users' chronological histories.

    Attributes:
        user_ids: int64[U], original MovieLens user ids.
        offsets: int64[U+1], user ``u`` owns ``items[offsets[u]:offsets[u+1]]``.
        items: int32[T], model item ids in 1..N.
        split: uint8[T], 0 train / 1 validation / 2 test.

    Every user has at least 3 interactions, so the train part is never empty.
    """

    user_ids: np.ndarray
    offsets: np.ndarray
    items: np.ndarray
    split: np.ndarray

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])

    def full(self, u: int) -> np.ndarray:
        """int64[n_u]: the whole history of user index ``u`` (train + validation + test)."""
        return self.items[self.offsets[u] : self.offsets[u + 1]].astype(np.int64)

    def train(self, u: int) -> np.ndarray:
        """int64[n_u - 2]: train items of user index ``u``."""
        return self.items[self.offsets[u] : self.offsets[u + 1] - 2].astype(np.int64)

    def validation(self, u: int) -> int:
        """The validation item (second to last) of user index ``u``."""
        return int(self.items[self.offsets[u + 1] - 2])

    def test(self, u: int) -> int:
        """The test item (last) of user index ``u``."""
        return int(self.items[self.offsets[u + 1] - 1])

    @property
    def validation_items(self) -> np.ndarray:
        """int64[U]: validation item of every user."""
        return self.items[self.offsets[1:] - 2].astype(np.int64)

    @property
    def test_items(self) -> np.ndarray:
        """int64[U]: test item of every user."""
        return self.items[self.offsets[1:] - 1].astype(np.int64)


@dataclass(frozen=True)
class ItemMap:
    """Item id mappings.

    Attributes:
        model_to_orig: int64[N+1], original MovieLens movieId for each model id (entry 0 is 0).
        orig_to_model: dict mapping original movieId to model id.
    """

    model_to_orig: np.ndarray
    orig_to_model: dict[int, int]

    def to_model(self, orig_id: int) -> int:
        return self.orig_to_model[int(orig_id)]

    def to_orig(self, model_id: int) -> int:
        return int(self.model_to_orig[model_id])


def load_meta(processed_dir: Path) -> dict:
    """The contents of ``data_meta.json`` (num_users, num_items, max_length, padding_id, ...)."""
    with open(Path(processed_dir) / META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_histories(processed_dir: Path) -> Histories:
    """Load the per-user sequences. See :class:`Histories` for shapes and dtypes."""
    with np.load(Path(processed_dir) / SEQUENCES_FILE) as z:
        histories = Histories(
            user_ids=z["user_ids"].astype(np.int64),
            offsets=z["offsets"].astype(np.int64),
            items=z["items"].astype(np.int32),
            split=z["split"].astype(np.uint8),
        )
    ends = histories.offsets[1:]
    if not (
        np.all(histories.split[ends - 1] == SPLIT_TEST)
        and np.all(histories.split[ends - 2] == SPLIT_VALIDATION)
    ):
        raise ValueError(f"corrupt split marks in {processed_dir}/{SEQUENCES_FILE}")
    return histories


def load_item_map(processed_dir: Path) -> ItemMap:
    """Load the item id mapping in both directions."""
    with open(Path(processed_dir) / ITEM_MAP_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ItemMap(
        model_to_orig=np.asarray(raw["model_to_orig"], dtype=np.int64),
        orig_to_model={int(k): int(v) for k, v in raw["orig_to_model"].items()},
    )


def load_popularity(processed_dir: Path) -> np.ndarray:
    """int64[N+1]: train-only interaction count per model item id (entry 0 is 0)."""
    return np.load(Path(processed_dir) / POPULARITY_FILE).astype(np.int64)


def recompute_data_hash(processed_dir: Path) -> str:
    """Recompute the data hash from the files, to compare with ``data_meta.json``."""
    histories = load_histories(processed_dir)
    return compute_data_hash(
        histories.user_ids,
        histories.offsets,
        histories.items,
        histories.split,
        load_popularity(processed_dir),
        load_item_map(processed_dir).model_to_orig,
    )
