"""Monte Carlo verifier: margins of the target under necessity and sufficiency interventions.

For an instance with history ``s`` (n items), target ``v`` and backgrounds ``r^(1..M)``, a mask
``z`` selects the positions ``X``. For every draw ``m``:

* necessity:   ``s`` with the positions in ``X`` replaced by ``r^(m)`` (erase the explanation);
* sufficiency: ``r^(m)`` with the positions in ``X`` replaced by ``s`` (keep only the explanation).

Both are substitutions at the same positions: nothing is deleted or shifted. Each intervened
sequence costs one SASRec pass and gives the margin ``f_v - f_(K)``, where ``f_(K)`` is the K-th
highest score among the other candidates (the catalog without the user's items and without ``v``),
so ``margin >= 0`` iff ``v`` is in the top-K.

Masks are bool [R, L] in the model's left-padded layout (L = max_length): the instance's n
positions are the last n columns; bits on padding are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from seqrec.explain.instances import Instance
from seqrec.models.loader import SASRecScorer

DEFAULT_K = 10
DEFAULT_CHUNK = 8192


@dataclass(frozen=True)
class Evaluation:
    """Per-draw margins of R masks. ``nec`` / ``suf`` float [R, M], ``size`` long [R] (= |X|)."""

    nec: torch.Tensor
    suf: torch.Tensor
    size: torch.Tensor

    @property
    def necessity(self) -> torch.Tensor:
        """N-hat [R]: mean margin after erasing X (lower = more necessary)."""
        return self.nec.mean(dim=1)

    @property
    def sufficiency(self) -> torch.Tensor:
        """S-hat [R]: minus the mean margin keeping only X (lower = more sufficient)."""
        return -self.suf.mean(dim=1)

    @property
    def pn(self) -> torch.Tensor:
        """Fraction of draws where erasing X takes v out of the top-K."""
        return (self.nec < 0).float().mean(dim=1)

    @property
    def ps(self) -> torch.Tensor:
        """Fraction of draws where X alone keeps v in the top-K."""
        return (self.suf >= 0).float().mean(dim=1)

    @property
    def objectives(self) -> torch.Tensor:
        """float [R, 3]: (N-hat, S-hat, |X|), all minimised."""
        return torch.stack([self.necessity, self.sufficiency, self.size.to(self.nec.dtype)], dim=1)


class Verifier:
    """Scores masks of several instances in shared batches and counts SASRec passes."""

    def __init__(
        self,
        scorer: SASRecScorer,
        instances: Sequence[Instance],
        backgrounds: Sequence[np.ndarray],
        k: int = DEFAULT_K,
        chunk: int = DEFAULT_CHUNK,
    ) -> None:
        if len(instances) != len(backgrounds):
            raise ValueError("one background array per instance is required")
        draws = {b.shape[0] for b in backgrounds}
        if len(draws) > 1:
            raise ValueError(f"all instances need the same number of backgrounds, got {sorted(draws)}")
        self.scorer = scorer
        self.k = int(k)
        self.chunk = int(chunk)
        self.m = draws.pop() if draws else 0
        self.passes = 0
        length, device = scorer.max_length, scorer.device
        self.length = length

        hist = torch.zeros(len(instances), length, dtype=torch.long)
        bg = torch.zeros(len(instances), self.m, length, dtype=torch.long)
        exclude = torch.zeros(len(instances), scorer.num_items, dtype=torch.bool)
        for i, (inst, fills) in enumerate(zip(instances, backgrounds)):
            if fills.shape != (self.m, inst.n):
                raise ValueError(f"backgrounds of instance {i} have shape {fills.shape}, expected {(self.m, inst.n)}")
            hist[i, length - inst.n :] = torch.from_numpy(inst.history)
            bg[i, :, length - inst.n :] = torch.from_numpy(fills)
            exclude[i, torch.from_numpy(inst.seen) - 1] = True
            exclude[i, inst.target - 1] = True
        if len(instances) and int((~exclude).sum(dim=1).min()) < self.k:
            raise ValueError(f"an instance has fewer than k={self.k} candidates")
        self.hist = hist.to(device)
        self.bg = bg.to(device)
        self.exclude = exclude.to(device)
        self.valid = self.hist != scorer.padding_id
        self.target0 = torch.tensor([inst.target - 1 for inst in instances], dtype=torch.long, device=device)

    def __len__(self) -> int:
        return int(self.hist.shape[0])

    def full_masks(self) -> torch.Tensor:
        """bool [I, L]: the mask selecting every position of each instance."""
        return self.valid.clone()

    @torch.inference_mode()
    def evaluate(self, index: torch.Tensor, masks: torch.Tensor) -> Evaluation:
        """Score ``masks`` bool [R, L]; row r belongs to instance ``index[r]``. Costs 2*M*R passes."""
        device = self.scorer.device
        index = torch.as_tensor(index, device=device, dtype=torch.long)
        masks = torch.as_tensor(masks, device=device, dtype=torch.bool)
        if masks.shape != (index.shape[0], self.length):
            raise ValueError(f"masks must be [{index.shape[0]}, {self.length}], got {tuple(masks.shape)}")
        masks = masks & self.valid[index]
        per_mask = 2 * self.m
        rows = max(1, self.chunk // max(per_mask, 1))
        margins = []
        for idx, z in zip(index.split(rows), masks.split(rows)):
            z = z.unsqueeze(1)
            hist, fills = self.hist[idx].unsqueeze(1), self.bg[idx]
            seqs = torch.cat([torch.where(z, fills, hist), torch.where(z, hist, fills)], dim=1)
            logits = self.scorer.logits(seqs.reshape(-1, self.length), max_batch=self.chunk)
            target = self.target0[idx].repeat_interleave(per_mask).unsqueeze(1)
            score = logits.gather(1, target).squeeze(1)
            others = logits.masked_fill(self.exclude[idx].repeat_interleave(per_mask, dim=0), float("-inf"))
            kth = others.topk(self.k, dim=1).values[:, -1]
            margins.append((score - kth).view(-1, per_mask))
        delta = torch.cat(margins) if margins else torch.zeros(0, per_mask, device=device)
        self.passes += delta.numel()
        return Evaluation(nec=delta[:, : self.m], suf=delta[:, self.m :], size=masks.sum(dim=1))
