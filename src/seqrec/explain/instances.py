"""Explanation instances: a test user's history plus one recommended item to explain.

The history ``s`` is the model's test input (the last ``max_length`` items before the test item).
The candidate set is the catalog without every item the user interacted with before the test item
(the same exclusion the evaluation protocol uses). The target ``v`` is drawn uniformly among the
top-K candidates. An instance is kept only if replacing the whole history takes ``v`` out of the
top-K on average (``N < 0`` for the full mask), i.e. at least one valid explanation exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from seqrec.data import Histories
from seqrec.models.loader import SASRecScorer

# inclusive bounds on n = min(history length, max_length)
STRATA = ((30, 49), (50, 99), (100, 128))


@dataclass(frozen=True)
class Instance:
    """One (history, target) pair.

    Attributes:
        user_index: row of the user in ``Histories``; ``user_id`` is the MovieLens id.
        history: int64 [n], the model input, chronological.
        seen: int64, every item before the test item (never a candidate).
        target: the explained item ``v`` in 1..N; ``target_rank`` is its 0-based rank among the
            candidates and ``target_popularity`` its train interaction count.
        stratum: index into the strata the instance was sampled from.
    """

    user_index: int
    user_id: int
    history: np.ndarray
    seen: np.ndarray
    target: int
    target_rank: int
    target_popularity: int
    stratum: int

    @property
    def n(self) -> int:
        return int(self.history.shape[0])


def eligible_users(
    histories: Histories, max_length: int, strata: Sequence[tuple[int, int]] = STRATA
) -> list[np.ndarray]:
    """User indices per stratum, by n = min(items before the test item, max_length)."""
    n = np.minimum(np.diff(histories.offsets) - 1, max_length)
    return [np.flatnonzero((n >= low) & (n <= high)) for low, high in strata]


def propose(
    histories: Histories,
    scorer: SASRecScorer,
    popularity: np.ndarray,
    users: Iterable[int],
    stratum: int,
    k: int,
    rng: np.random.Generator,
) -> list[Instance]:
    """Draw a target among the top-``k`` candidates of each user (no validity check)."""
    users = [int(u) for u in users]
    if not users:
        return []
    length = scorer.max_length
    seqs = torch.zeros(len(users), length, dtype=torch.long)
    seen = []
    for row, u in enumerate(users):
        before_test = histories.full(u)[:-1]
        seen.append(before_test)
        tail = before_test[-length:]
        seqs[row, length - len(tail) :] = torch.from_numpy(tail)
    logits = scorer.logits(seqs).cpu().clone()
    for row, items in enumerate(seen):
        logits[row, torch.from_numpy(items) - 1] = float("-inf")
    top = logits.topk(k, dim=1).indices.numpy() + 1
    ranks = rng.integers(k, size=len(users))
    out = []
    for row, u in enumerate(users):
        target = int(top[row, ranks[row]])
        out.append(
            Instance(
                user_index=u,
                user_id=int(histories.user_ids[u]),
                history=seen[row][-length:],
                seen=seen[row],
                target=target,
                target_rank=int(ranks[row]),
                target_popularity=int(popularity[target]),
                stratum=stratum,
            )
        )
    return out


def build_instances(
    histories: Histories,
    scorer: SASRecScorer,
    popularity: np.ndarray,
    per_stratum: int,
    seed: int,
    m: int,
    k: int = 10,
    strata: Sequence[tuple[int, int]] = STRATA,
    neighbours: int = 50,
    exclude_users: Iterable[int] = (),
) -> tuple[list[Instance], dict]:
    """Sample up to ``per_stratum`` valid instances per stratum.

    Validity (``N < 0`` for the full mask) is checked with the ``m`` search backgrounds of
    ``seed``, so callers that redraw them with the same ``seed`` and ``m`` see the same values.
    ``exclude_users`` (user indices) keeps tuning and evaluation users disjoint.
    Returns the instances and, per stratum, how many were proposed and discarded.
    """
    # imported here: backgrounds and verifier import Instance from this module
    from seqrec.explain.backgrounds import draw_backgrounds
    from seqrec.explain.verifier import Verifier

    rng = np.random.default_rng(seed)
    embeddings = scorer.item_embeddings()
    skip = np.fromiter(exclude_users, dtype=np.int64)
    kept: list[Instance] = []
    stats = []
    for stratum, users in enumerate(eligible_users(histories, scorer.max_length, strata)):
        users = rng.permutation(np.setdiff1d(users, skip))
        found: list[Instance] = []
        proposed = discarded = 0
        while len(found) < per_stratum and proposed < len(users):
            block = users[proposed : proposed + 2 * (per_stratum - len(found))]
            proposed += len(block)
            candidates = propose(histories, scorer, popularity, block, stratum, k, rng)
            fills = draw_backgrounds(candidates, popularity, embeddings, m, seed, neighbours=neighbours)
            verifier = Verifier(scorer, candidates, fills, k=k)
            necessity = verifier.evaluate(torch.arange(len(candidates)), verifier.full_masks()).necessity
            valid = [inst for inst, value in zip(candidates, necessity.tolist()) if value < 0]
            discarded += len(candidates) - len(valid)
            found += valid
        found = found[:per_stratum]
        kept += found
        stats.append(
            {
                "bounds": list(strata[stratum]),
                "eligible": len(users),
                "proposed": proposed,
                "discarded": discarded,
                "kept": len(found),
            }
        )
    return kept, {"strata": stats, "seed": seed, "m": m, "k": k}
