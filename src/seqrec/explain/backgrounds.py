"""Background histories: the "neutral" items that replace positions of a history.

Every position of a background is an item drawn independently in proportion to its train
popularity. The target, its nearest neighbours in embedding space and the user's own items are
never drawn, so a fill cannot put evidence for the target back.

Two other fills exist for the ablations (``FILLS``): ``users`` takes a contiguous snippet of another
user's history for each background (forbidden items are redrawn by popularity), so the fill is a
plausible sequence instead of independent items; ``padding`` uses the padding id, which the model
drops, i.e. the positions are deleted and the rest of the sequence shifts.

Backgrounds are a pure function of ``(seed, stream, user, target)``: drawn once per instance and
reused for every mask and every method (common random numbers), which makes the fitness of a mask
deterministic. ``EVAL_STREAM`` gives a set independent of the one used during the search.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from seqrec.data import Histories
from seqrec.explain.instances import Instance

SEARCH_STREAM = 0
EVAL_STREAM = 1
DEFAULT_NEIGHBOURS = 50
FILLS = ("popularity", "users", "padding")


def nearest_neighbours(embeddings: torch.Tensor, targets: torch.Tensor, count: int) -> torch.Tensor:
    """long [B, count]: the items closest to each target by cosine similarity (target excluded).

    ``embeddings`` is float [N+1, H] with row 0 = padding; ``targets`` long [B] in 1..N.
    ``count`` is clamped to N - 1.
    """
    items = torch.nn.functional.normalize(embeddings[1:].float(), dim=1)
    count = min(int(count), items.shape[0] - 1)
    targets = torch.as_tensor(targets, device=items.device, dtype=torch.long)
    if count <= 0:
        return torch.zeros(targets.shape[0], 0, dtype=torch.long, device=items.device)
    sims = items[targets - 1] @ items.T
    sims.scatter_(1, (targets - 1).unsqueeze(1), float("-inf"))
    return sims.topk(count, dim=1).indices + 1


def fill_distribution(instance: Instance, popularity: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """float64 [N+1]: sampling probability of each item id for this instance's fills."""
    weights = popularity.astype(np.float64).copy()
    weights[0] = 0.0
    weights[instance.target] = 0.0
    weights[neighbours] = 0.0
    weights[instance.seen] = 0.0
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"no item left to draw backgrounds for user index {instance.user_index}")
    return weights / total


def draw_backgrounds(
    instances: Sequence[Instance],
    popularity: np.ndarray,
    embeddings: torch.Tensor,
    m: int,
    seed: int,
    stream: int = SEARCH_STREAM,
    neighbours: int = DEFAULT_NEIGHBOURS,
    fill: str = "popularity",
    histories: Histories | None = None,
) -> list[np.ndarray]:
    """One int64 [m, n] array per instance (n = length of that instance's history).

    ``fill`` is one of ``FILLS``; ``users`` needs ``histories``.
    """
    if fill not in FILLS:
        raise ValueError(f"unknown fill {fill!r}, expected one of {FILLS}")
    if not instances:
        return []
    if fill == "padding":
        return [np.zeros((m, inst.n), dtype=np.int64) for inst in instances]
    if fill == "users" and histories is None:
        raise ValueError("the users fill needs the histories")
    targets = torch.tensor([inst.target for inst in instances])
    near = nearest_neighbours(embeddings, targets, neighbours).cpu().numpy()
    out = []
    for inst, near_items in zip(instances, near):
        rng = np.random.default_rng([seed, stream, inst.user_index, inst.target])
        probs = fill_distribution(inst, popularity, near_items)
        if fill == "users":
            out.append(_user_snippets(inst, histories, probs, m, rng))
        else:
            out.append(rng.choice(len(probs), size=(m, inst.n), p=probs).astype(np.int64))
    return out


def _user_snippets(inst: Instance, histories: Histories, probs: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """int64 [m, n]: each row is n consecutive items of another user (everything before their test item).

    Items with zero probability in ``probs`` (target, its neighbours, the user's own items) are
    replaced by popularity draws.
    """
    lengths = np.diff(histories.offsets) - 1
    donors = np.flatnonzero(lengths >= inst.n)
    donors = donors[donors != inst.user_index]
    if len(donors) == 0:
        raise ValueError(f"no other user has {inst.n} items to fill user index {inst.user_index}")
    rows = np.empty((m, inst.n), dtype=np.int64)
    for row, donor in zip(rows, rng.choice(donors, size=m)):
        start = rng.integers(0, lengths[donor] - inst.n + 1)
        row[:] = histories.full(int(donor))[start : start + inst.n]
    forbidden = probs[rows] == 0
    rows[forbidden] = rng.choice(len(probs), size=int(forbidden.sum()), p=probs)
    return rows
