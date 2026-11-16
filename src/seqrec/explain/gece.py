"""GECE baseline, run through TRACE's implementation (``third_party/trace``).

GECE edits the sequence itself (replace, add, delete) with a scalar fitness and returns one
counterfactual. To compare it with the mask methods, every sequence it queries is mapped to the
positions where it differs from the history, and those position sets are scored by our verifier
like any other mask. Its own cost is the number of sequences it queried.

The validity condition given to GECE is the one of our instances: the target leaves the top-k of
the candidates (TRACE's ``push_out`` rank constraint; the user's other items never count).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch

from seqrec.explain.instances import Instance
from seqrec.explain.third_party.trace.base import RankConstraint
from seqrec.explain.third_party.trace.ga_explainer import GAExplainer
from seqrec.models.loader import SASRecScorer


class _Model:
    """What TRACE's explainer sees: ``predict(seqs) -> scores [B, N + 1]`` indexed by item id."""

    def __init__(self, scorer: SASRecScorer, instance: Instance) -> None:
        self.scorer = scorer
        self.blocked = torch.as_tensor(np.setdiff1d(instance.seen, [instance.target]), device=scorer.device)
        self.queried: list[np.ndarray] = []

    def predict(self, seqs: torch.Tensor) -> torch.Tensor:
        self.queried.append(seqs.detach().cpu().numpy())
        with torch.autocast(device_type="cuda", enabled=False):  # TRACE turns fp16 on; we explain fp32
            logits = self.scorer.logits(seqs)
        scores = torch.cat([torch.full_like(logits[:, :1], float("-inf")), logits], dim=1)
        scores[:, self.blocked] = float("-inf")
        return scores


@dataclass
class GeceResult:
    masks: np.ndarray  # bool [E, n]: distinct edited-position sets, in query order (never empty sets)
    chosen: int | None  # row of ``masks`` of the returned counterfactual (None: no valid one found)
    queries: int
    truncated: bool  # more than ``max_masks`` distinct sets were queried
    counterfactual: list[int] | None


def run_gece(
    scorer: SASRecScorer,
    instance: Instance,
    seed: int,
    k: int = 10,
    budget: int = 100_000,
    population: int = 100,
    generations: int | None = None,
    max_edits: int = 10,
    max_masks: int = 5000,
    restarts: bool = False,
    **ga_options,
) -> GeceResult:
    """GECE on one instance. TRACE's GA draws from numpy's global generator, seeded here.

    TRACE's GECE stops at its first valid counterfactual, usually after a few hundred queries.
    With ``restarts`` it is started again until ``budget`` queries are spent, so that it uses the
    same number of model queries as the other methods; the returned counterfactual is then the
    valid one with the fewest edits over all starts.
    """
    np.random.seed(zlib.crc32(f"gece-{seed}-{instance.user_index}-{instance.target}".encode()))
    length, n = scorer.max_length, instance.n
    original = np.zeros(length, dtype=np.int64)
    original[length - n :] = instance.history
    model = _Model(scorer, instance)
    forbidden = set(instance.seen.tolist()) | {instance.target}

    def positions(seqs: np.ndarray) -> np.ndarray:
        return (seqs != original)[:, length - n :]

    counterfactual, queries = None, 0
    while queries < budget:
        explainer = GAExplainer(
            model, scorer.num_items, population_size=population,
            generations=generations if generations is not None else max(1, budget // population),
            max_edits=max_edits, query_budget=budget - queries, device=str(scorer.device), **ga_options,
        )
        found, info = explainer.explain(
            original.tolist(), instance.target, scorer.num_items, forbidden,
            rank_constraint=RankConstraint.push_out(instance.target, top_k=k),
        )
        queries += int(info["queries_used"])
        edits = lambda seq: int(positions(np.asarray([seq])).sum())  # noqa: E731
        if found is not None and (counterfactual is None or edits(found) < edits(counterfactual)):
            counterfactual = found
        if not restarts or not info["queries_used"]:
            break

    rows, seen = [], {}
    for mask in positions(np.concatenate(model.queried)) if model.queried else []:
        key = mask.tobytes()
        if mask.any() and key not in seen:
            seen[key] = len(rows)
            rows.append(mask)
    chosen = None
    if counterfactual is not None:
        chosen = seen[positions(np.asarray([counterfactual]))[0].tobytes()]
    truncated = len(rows) > max_masks
    if truncated:  # keep the first ones, and the returned counterfactual
        keep = list(range(max_masks))
        if chosen is not None and chosen >= max_masks:
            keep[-1], chosen = chosen, max_masks - 1
        rows = [rows[i] for i in keep]
    masks = np.asarray(rows, dtype=bool).reshape(-1, n)
    return GeceResult(masks, chosen, queries, truncated, counterfactual)
