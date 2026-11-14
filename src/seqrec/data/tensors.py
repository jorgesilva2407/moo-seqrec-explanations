"""Padded tensors for training and full-ranking evaluation.

Everything is left-padded with 0. Shapes (U users, L = max_length, N items):

* ``train``       int32[U, L+1]  last L+1 train items; input = [:, :-1], next-item labels = [:, 1:]
* ``val_input``   int32[U, L]    last L train items            -> target ``val_target``
* ``test_input``  int32[U, L]    last L train+validation items -> target ``test_target``
* ``*_target``    int32[U]       model item id in 1..N
* ``flat_items``  int32[T], ``offsets`` int64[U+1]: the full histories, used to exclude seen items
  from the ranking (validation excludes train items, test excludes train + validation items).

Item ``i`` (1..N) is column ``i - 1`` of every [*, N] logits / mask tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from seqrec.data.artifacts import Histories

SPLITS = ("val", "test")


def _left_pad_last(seq: np.ndarray, length: int, out: np.ndarray) -> None:
    out[:] = 0
    tail = seq[-length:] if len(seq) else seq
    if len(tail):
        out[length - len(tail) :] = tail


@dataclass
class PaddedData:
    max_length: int
    num_items: int
    train: torch.Tensor
    val_input: torch.Tensor
    val_target: torch.Tensor
    test_input: torch.Tensor
    test_target: torch.Tensor
    flat_items: torch.Tensor
    offsets: torch.Tensor
    compute_device: torch.device = torch.device("cpu")
    pin_memory: bool = False

    @property
    def num_users(self) -> int:
        return int(self.val_target.shape[0])

    def _tensors(self) -> dict[str, torch.Tensor]:
        names = ("train", "val_input", "val_target", "test_input", "test_target", "flat_items", "offsets")
        return {n: getattr(self, n) for n in names}

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._tensors().values())

    def to_storage(self, storage_device: torch.device, compute_device: torch.device) -> "PaddedData":
        """Return a copy stored on ``storage_device``; batches are moved to ``compute_device``.

        CPU storage with a CUDA compute device uses pinned memory and async copies.
        """
        pin = storage_device.type == "cpu" and compute_device.type == "cuda"
        moved = {}
        for name, tensor in self._tensors().items():
            tensor = tensor.to(storage_device)
            moved[name] = tensor.pin_memory() if pin else tensor
        return replace(self, compute_device=compute_device, pin_memory=pin, **moved)

    # ------------------------------------------------------------------ batches
    def _fetch(self, tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch = tensor.index_select(0, idx.to(tensor.device))
        if batch.device.type != self.compute_device.type:
            if self.pin_memory:
                batch = batch.pin_memory()
            batch = batch.to(self.compute_device, non_blocking=self.pin_memory)
        return batch.long()

    def train_batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(inputs long[B, L], labels long[B, L]). Label 0 = ignore (padded input position)."""
        row = self._fetch(self.train, idx)
        inputs, nxt = row[:, :-1], row[:, 1:]
        labels = torch.where(inputs != 0, nxt, torch.zeros_like(nxt))
        return inputs, labels

    def eval_batch(
        self, split: str, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(inputs long[B, L], targets long[B] in 1..N, exclude bool[B, N] of already seen items)."""
        if split == "val":
            inputs = self._fetch(self.val_input, idx)
            targets = self._fetch(self.val_target, idx)
            drop = 2
        elif split == "test":
            inputs = self._fetch(self.test_input, idx)
            targets = self._fetch(self.test_target, idx)
            drop = 1
        else:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        exclude = self._history_mask(idx.to(self.offsets.device), drop).to(self.compute_device)
        return inputs, targets, exclude

    def _history_mask(self, idx: torch.Tensor, drop: int) -> torch.Tensor:
        """bool[B, N]: items in each user's history without its last ``drop`` entries."""
        dev = self.offsets.device
        starts = self.offsets[idx]
        lens = self.offsets[idx + 1] - starts - drop
        total = int(lens.sum().item())
        mask = torch.zeros(idx.shape[0], self.num_items, dtype=torch.bool, device=dev)
        if total == 0:
            return mask
        rows = torch.repeat_interleave(torch.arange(idx.shape[0], device=dev), lens)
        within = torch.arange(total, device=dev) - torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        pos = torch.repeat_interleave(starts, lens) + within
        cols = self.flat_items[pos].long() - 1
        mask[rows, cols] = True
        return mask


def build_padded(histories: Histories, max_length: int, num_items: int) -> PaddedData:
    """Build CPU tensors from the histories (users with fewer than 3 items are not expected)."""
    n = len(histories)
    train = np.zeros((n, max_length + 1), dtype=np.int32)
    val_input = np.zeros((n, max_length), dtype=np.int32)
    test_input = np.zeros((n, max_length), dtype=np.int32)
    val_target = np.zeros(n, dtype=np.int32)
    test_target = np.zeros(n, dtype=np.int32)
    for u in range(n):
        seq = histories.items[histories.offsets[u] : histories.offsets[u + 1]]
        _left_pad_last(seq[:-2], max_length + 1, train[u])
        _left_pad_last(seq[:-2], max_length, val_input[u])
        _left_pad_last(seq[:-1], max_length, test_input[u])
        val_target[u] = seq[-2]
        test_target[u] = seq[-1]
    return PaddedData(
        max_length=max_length,
        num_items=num_items,
        train=torch.from_numpy(train),
        val_input=torch.from_numpy(val_input),
        val_target=torch.from_numpy(val_target),
        test_input=torch.from_numpy(test_input),
        test_target=torch.from_numpy(test_target),
        flat_items=torch.from_numpy(histories.items.astype(np.int32)),
        offsets=torch.from_numpy(histories.offsets.astype(np.int64)),
    )


def plan_storage(nbytes: int, compute_device: torch.device, reserve_gib: float = 6.0) -> tuple[torch.device, str]:
    """Pick GPU storage if the data fits with ``reserve_gib`` left for training, else CPU.

    Returns (storage_device, human-readable reason) for logging.
    """
    if compute_device.type != "cuda":
        return torch.device("cpu"), "compute device is not CUDA: data stays on CPU"
    free, total = torch.cuda.mem_get_info(compute_device)
    need = nbytes + int(reserve_gib * 1024**3)
    gib = 1024**3
    if need <= free:
        return compute_device, (
            f"dataset {nbytes / gib:.2f} GiB + {reserve_gib:.1f} GiB reserve fits in free GPU memory "
            f"({free / gib:.2f} of {total / gib:.2f} GiB): stored on GPU"
        )
    return torch.device("cpu"), (
        f"dataset {nbytes / gib:.2f} GiB + {reserve_gib:.1f} GiB reserve exceeds free GPU memory "
        f"({free / gib:.2f} GiB): stored in pinned CPU memory with async transfers"
    )
