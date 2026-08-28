"""Minimal Transformer with L0-sparsity-ready linear layers.

All Linear layers are replaced with L0Linear. When sparsity is disabled
(use_mask=False) the module is equivalent to a dense baseline.

Design decisions:
- Pre-norm with RMSNorm (cleaner signal for mechanistic interpretability).
- No dropout (we want deterministic circuit extraction).
- Sinusoidal positional encoding, fixed (not learned) — simpler to interpret.
- Tied input/output vocab optional (default: untied).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .l0 import L0Linear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    # For odd d_model, pe[:, 1::2] has one fewer column than div — slice to match.
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe  # (seq_len, d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_mask: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = L0Linear(d_model, d_model, use_mask=use_mask)
        self.k = L0Linear(d_model, d_model, use_mask=use_mask)
        self.v = L0Linear(d_model, d_model, use_mask=use_mask)
        self.o = L0Linear(d_model, d_model, use_mask=use_mask)
        # GPT-2-style scaled init for the residual-branch output projection:
        # std = 0.02 / sqrt(2 * n_layers). For this project n_layers is fixed
        # at the time we build this module, so we hard-code a default that
        # matches "not too small, not PyTorch's overconfident kaiming".
        nn.init.normal_(self.o.weight, std=0.1)
        if self.o.bias is not None:
            nn.init.zeros_(self.o.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                                      # (B, H, T, d_h)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_mask: bool):
        super().__init__()
        self.fc1 = L0Linear(d_model, d_ff, use_mask=use_mask)
        self.fc2 = L0Linear(d_ff, d_model, use_mask=use_mask)
        # Same scaled init as attention output projection (see MHA).
        nn.init.normal_(self.fc2.weight, std=0.1)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_mask: bool):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, use_mask)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, use_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        use_mask: bool = True,
        out_dim: int | None = None,
        per_position_out: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "pos_embed", sinusoidal_positional_encoding(seq_len, d_model)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, use_mask)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.per_position_out = per_position_out
        self.out_dim = out_dim if out_dim is not None else vocab_size
        self.head = L0Linear(d_model, self.out_dim, use_mask=use_mask)

    def forward(
        self, tokens: torch.Tensor, cls_pos: int | None = None
    ) -> torch.Tensor:
        # tokens: (B, T) int64
        x = self.tok(tokens) + self.pos_embed[: tokens.size(1)]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        if self.per_position_out:
            return self.head(x)                      # (B, T, out_dim)
        if cls_pos is None:
            cls_pos = -1
        return self.head(x[:, cls_pos])              # (B, out_dim)
