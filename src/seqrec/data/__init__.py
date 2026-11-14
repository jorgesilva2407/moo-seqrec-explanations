"""Preprocessing and read-only access to the processed MovieLens-20M artifacts.

The public read API (numpy only) is re-exported here. Padded-tensor helpers live in
``seqrec.data.tensors`` and the preprocessing code in ``seqrec.data.preprocess``.
"""

from seqrec.data.artifacts import (
    Histories,
    ItemMap,
    load_histories,
    load_item_map,
    load_meta,
    load_popularity,
    recompute_data_hash,
)

__all__ = [
    "Histories",
    "ItemMap",
    "load_histories",
    "load_item_map",
    "load_meta",
    "load_popularity",
    "recompute_data_hash",
]
