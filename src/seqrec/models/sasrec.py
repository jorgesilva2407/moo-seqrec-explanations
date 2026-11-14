"""SASRec: self-attentive sequential recommendation with tied item embeddings.

References:
  * W.-C. Kang and J. McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
  * A. Klenitskiy and A. Vasilev, "Turning Dross Into Gold Loss: is BERT4Rec really better than
    SASRec?", RecSys 2023 (arXiv 2309.07602) -- the "SASRec+" recipe (full cross-entropy over
    all items).

Details taken from the authors' public implementation (github.com/antklen/sasrec-bert4rec-recsys23,
read for reference only, no code copied): 2 blocks, 1 head, dropout 0.1 (embeddings, attention
probabilities and feed-forward), feed-forward width equal to the hidden size with ReLU, learned
positional embeddings, item embeddings scaled by sqrt(hidden), pre-layer-norm blocks with a final
layer norm (eps 1e-8), N(0, 0.02) initialisation, output logits tied to the item embedding matrix
and the full cross-entropy loss over all items.

Deviations from the reference, on purpose:
  * the residual stream is the plain pre-LN one (x + f(LN(x))); the reference adds the *normalised*
    activations back for the attention sub-layer;
  * causality and padding go through one boolean mask (``is_causal`` cannot be combined with an
    ``attn_mask`` in ``F.scaled_dot_product_attention``).

Conventions: item ids are 1..N, 0 is padding. Logits have N columns; column ``j`` is item
``j + 1``. Sequences are left-padded; positional embeddings are right-aligned so that a shorter
input gives the same result as the same input left-padded to ``max_length``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SASRecConfig:
    num_items: int
    max_length: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 1
    dropout: float = 0.1
    ffn_size: int = 256
    padding_id: int = 0
    layer_norm_eps: float = 1e-8
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.padding_id != 0:
            raise ValueError("padding_id must be 0")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_items < 1 or self.max_length < 1:
            raise ValueError("num_items and max_length must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SASRecConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**data)


class SelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x [B, L, H]; attn_mask bool [B, 1, L, L], True = may attend
        b, length, hidden = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, heads, L, head_dim]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(b, length, hidden))


class Block(nn.Module):
    def __init__(self, cfg: SASRecConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = SelfAttention(cfg.hidden_size, cfg.num_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.ffn_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_size, cfg.hidden_size),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x * valid  # keep padded positions at exactly zero


class SASRec(nn.Module):
    def __init__(self, cfg: SASRecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.num_items + 1, cfg.hidden_size, padding_idx=cfg.padding_id)
        self.pos_emb = nn.Embedding(cfg.max_length, cfg.hidden_size)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _attention_mask(valid: torch.Tensor) -> torch.Tensor:
        """bool [B, 1, L, L]: causal AND key not padding, plus the diagonal.

        The diagonal guarantees every row has at least one allowed key, so fully padded rows
        (padded queries) attend to themselves instead of producing NaN from an all -inf softmax.
        Real queries are unaffected: their own position is already allowed.
        """
        length = valid.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
        eye = torch.eye(length, dtype=torch.bool, device=valid.device)
        mask = (causal[None, None] & valid[:, None, None, :]) | eye[None, None]
        return mask

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """seqs long[B, L] with L <= max_length -> hidden [B, L, H]; zero at padded positions."""
        _, length = seqs.shape
        if length > self.cfg.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length {self.cfg.max_length}")
        valid = seqs != self.cfg.padding_id
        valid_f = valid.unsqueeze(-1)
        x = self.item_emb(seqs) * math.sqrt(self.cfg.hidden_size)
        positions = torch.arange(self.cfg.max_length - length, self.cfg.max_length, device=seqs.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        x = self.emb_dropout(x) * valid_f
        attn_mask = self._attention_mask(valid)
        valid_x = valid_f.to(x.dtype)
        for block in self.blocks:
            x = block(x, attn_mask, valid_x)
        return self.final_ln(x) * valid_f.to(x.dtype)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.encode(seqs)

    def item_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [..., H] -> logits [..., N] against the tied item embeddings (padding excluded)."""
        return F.linear(hidden, self.item_emb.weight[1:])

    def last_logits(self, seqs: torch.Tensor) -> torch.Tensor:
        """Logits [B, N] for the next item after the last position of left-padded ``seqs``."""
        return self.item_logits(self.encode(seqs)[:, -1])
