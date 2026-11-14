"""Full-ranking metrics (no sampling) over all N items.

Protocol (mirrors the authors' evaluation): items already in the user's history are removed from
the ranking (validation: train items; test: train + validation items); one target per user.
A target that is itself in the history can therefore never be retrieved (counts as a miss).
Ties are counted against the target (pessimistic).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from seqrec.data.tensors import PaddedData

DEFAULT_KS = (1, 5, 10, 20)


def target_ranks(
    logits: torch.Tensor, targets0: torch.Tensor, exclude: torch.Tensor | None = None
) -> torch.Tensor:
    """0-based rank of each target. logits float [B, N], targets0 long [B] (0-based column),
    exclude bool [B, N] or None. Returns long [B]."""
    n = logits.shape[1]
    if exclude is not None:
        logits = logits.masked_fill(exclude, float("-inf"))
    target_logit = logits.gather(1, targets0.unsqueeze(1))
    ranks = (logits >= target_logit).sum(dim=1) - 1  # minus the target itself
    if exclude is not None:
        target_hidden = exclude.gather(1, targets0.unsqueeze(1)).squeeze(1)
        ranks = torch.where(target_hidden, torch.full_like(ranks, n), ranks)
    return ranks


def metrics_from_ranks(ranks: torch.Tensor, ks: Sequence[int]) -> dict[str, torch.Tensor]:
    """Per-user HR@k and NDCG@k tensors (float [B])."""
    out = {}
    gain = 1.0 / torch.log2(ranks.float() + 2.0)
    for k in ks:
        hit = ranks < k
        out[f"hr@{k}"] = hit.float()
        out[f"ndcg@{k}"] = torch.where(hit, gain, torch.zeros_like(gain))
    return out


@torch.no_grad()
def evaluate_split(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    data: PaddedData,
    split: str,
    ks: Sequence[int] = DEFAULT_KS,
    batch_size: int = 256,
    user_index: torch.Tensor | None = None,
) -> dict[str, float]:
    """Average HR@k / NDCG@k over users.

    ``logits_fn`` maps long [B, L] left-padded inputs to float [B, N] logits (column j = item j+1).
    ``user_index`` optionally restricts evaluation to a subset of users (long [M]).
    """
    users = torch.arange(data.num_users) if user_index is None else user_index.cpu()
    sums: dict[str, torch.Tensor] = {}
    for start in range(0, len(users), batch_size):
        idx = users[start : start + batch_size]
        inputs, targets, exclude = data.eval_batch(split, idx)
        ranks = target_ranks(logits_fn(inputs), targets - 1, exclude)
        for name, values in metrics_from_ranks(ranks, ks).items():
            sums[name] = sums.get(name, torch.zeros((), device=values.device)) + values.sum()
    count = max(len(users), 1)
    return {name: float(total.item()) / count for name, total in sums.items()}
