"""Rebuild a trained SASRec from a checkpoint file alone and expose it as a black-box scorer.

Imports only the model; nothing training-related. Usage::

    scorer = load("runs/<run>/best.pt", device="cpu")
    logits = scorer.logits(seqs)   # LongTensor[B, L] -> float32 Tensor[B, N]; column j = item j+1
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from seqrec.models.sasrec import SASRec, SASRecConfig

# Checkpoint keys shared with the code that writes checkpoints (seqrec.training.checkpoint).
KEY_STATE = "model_state"
KEY_CONFIG = "model_config"
KEY_DATA_HASH = "data_hash"
KEY_SEED = "seed"
KEY_TRAIN_ARGS = "train_args"
KEY_VERSIONS = "versions"


def checkpoint_id(path: str | Path) -> str:
    """First 12 hex chars of the SHA-256 of the checkpoint file bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


class SASRecScorer:
    """Eval-mode SASRec returning next-item logits. Holds no global state."""

    def __init__(
        self,
        model: SASRec,
        checkpoint_id: str,
        device: torch.device,
        max_batch: int,
        info: dict,
    ) -> None:
        self._model = model
        self._checkpoint_id = checkpoint_id
        self._device = device
        self._max_batch = int(max_batch)
        self._info = info

    @property
    def num_items(self) -> int:
        return self._model.cfg.num_items

    @property
    def max_length(self) -> int:
        return self._model.cfg.max_length

    @property
    def padding_id(self) -> int:
        return self._model.cfg.padding_id

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def data_hash(self) -> str | None:
        """Hash of the processed data the model was trained on (None if not recorded)."""
        return self._info.get(KEY_DATA_HASH)

    def item_embeddings(self) -> torch.Tensor:
        """float32 [N + 1, H] copy of the item embedding matrix; row ``i`` is item ``i``, row 0 padding."""
        return self._model.item_emb.weight.detach().float().clone()

    def logits(self, seqs: torch.Tensor, max_batch: int | None = None) -> torch.Tensor:
        """Next-item logits for the last position.

        Args:
            seqs: integer tensor [B, L], 1 <= L <= max_length. Item ids in 1..N and the padding
                id 0 may appear anywhere, repeats are fine. Padding is removed and the real items
                are left-padded to ``max_length`` (order preserved); an all-padding row gives
                all-zero logits.
            max_batch: rows per forward pass (default: the value given to ``load``); the result
                does not depend on it beyond float rounding.

        Returns:
            float32 [B, N] on the scorer's device; column ``j`` is item ``j + 1``. The tensor is
            created under ``torch.inference_mode()`` (an inference tensor).
        """
        seqs = torch.as_tensor(seqs)
        if seqs.dim() != 2:
            raise ValueError(f"seqs must be 2-D [B, L], got shape {tuple(seqs.shape)}")
        if seqs.dtype.is_floating_point or seqs.dtype == torch.bool:
            raise ValueError(f"seqs must be an integer tensor, got {seqs.dtype}")
        batch, length = seqs.shape
        if not 1 <= length <= self.max_length:
            raise ValueError(f"sequence length {length} not in [1, {self.max_length}]")
        chunk_size = int(max_batch or self._max_batch)
        if chunk_size < 1:
            raise ValueError("max_batch must be >= 1")
        seqs = seqs.to(self._device, dtype=torch.long)
        if batch == 0:
            return torch.zeros(0, self.num_items, device=self._device)
        if int(seqs.min()) < 0 or int(seqs.max()) > self.num_items:
            raise ValueError(f"item ids must lie in [0, {self.num_items}]")
        with torch.inference_mode():
            outputs = [self._logits_chunk(chunk) for chunk in seqs.split(chunk_size, dim=0)]
            return torch.cat(outputs, dim=0)

    def _logits_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        pad = self.padding_id
        # Stable sort with pads first: real items keep their relative order and end up right-aligned.
        order = torch.sort((chunk == pad).to(torch.int8), dim=1, descending=True, stable=True).indices
        chunk = chunk.gather(1, order)
        missing = self.max_length - chunk.shape[1]
        if missing > 0:
            chunk = F.pad(chunk, (missing, 0), value=pad)
        return self._model.last_logits(chunk).float()


def load(checkpoint_path: str | Path, device: str | torch.device = "cuda", max_batch: int = 1024) -> SASRecScorer:
    """Rebuild the model from the checkpoint alone and return an eval-mode scorer."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = SASRecConfig.from_dict(payload[KEY_CONFIG])
    model = SASRec(config)
    model.load_state_dict(payload[KEY_STATE])
    device = torch.device(device)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    info = {
        KEY_DATA_HASH: payload.get(KEY_DATA_HASH),
        KEY_SEED: payload.get(KEY_SEED),
        KEY_TRAIN_ARGS: payload.get(KEY_TRAIN_ARGS),
        KEY_VERSIONS: payload.get(KEY_VERSIONS),
    }
    return SASRecScorer(model, checkpoint_id(path), device, max_batch, info)


def load_from_hub(
    repo_id: str, filename: str = "fp32/best.pt", device: str | torch.device = "cuda", max_batch: int = 1024
) -> SASRecScorer:
    """Download a checkpoint from the Hugging Face Hub (cached) and ``load`` it.

    Needs ``huggingface_hub`` (the ``explain`` or ``upload`` dependency group); a private repo
    needs the ``HF_TOKEN`` environment variable.
    """
    from huggingface_hub import hf_hub_download

    return load(hf_hub_download(repo_id, filename), device=device, max_batch=max_batch)
