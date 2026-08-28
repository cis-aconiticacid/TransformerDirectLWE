"""LWE Mini data generation.

Regev-style LWE encryption of a single bit, at toy parameters.

Parameters (Mini default):
    n   = 16       (lattice dimension)
    q   = 17       (modulus, prime)
    ham = 4        (Hamming weight of binary secret s)
    e_bound = 1    (noise e ~ uniform {-1, 0, +1})

Encryption of bit m ∈ {0, 1}:
    a ~ Uniform(Z_q^n), e ~ Uniform({-e_bound,..,+e_bound})
    b = (<a, s> + e + m * (q // 2)) mod q

Decryption (target for the network):
    Given (a, b), recover m.
    Correct iff |e| <= (q - 2) / 4, which for q=17, e_bound=1 is always true.

Tokenization for Mode A (generic decoder):
    Sequence layout:
        [s_0, ..., s_{n-1}, SEP, a_0, ..., a_{n-1}, SEP, b, CLS]
    Total length = n + 1 + n + 1 + 1 + 1 = 2n + 4 = 36
    Vocab = q + 3 special tokens (PAD, SEP, CLS) = 20
        token 0..q-1    -> value modulo q (also covers s∈{0,1})
        token q         -> PAD (unused in Mode A, reserved)
        token q+1       -> SEP
        token q+2       -> CLS
"""

from __future__ import annotations

import numpy as np
import torch


class LWEMiniConfig:
    """
    mode:
      'A'  → generic decoder: s randomized per sample and prepended as tokens.
             Seq: [s, SEP, a, SEP, b, CLS], seq_len = 2n + 4.
             Model must learn the algorithm, not a fixed key.
      'B'  → fixed-secret: s* chosen once (seeded by `fixed_s_seed`) and
             implicit in model weights. Seq: [a, SEP, b, CLS], seq_len = n + 3.
             Strictly easier — good as a pipeline sanity.
    """

    def __init__(
        self,
        n: int = 16,
        q: int = 17,
        hamming: int = 4,
        e_bound: int = 1,
        mode: str = "A",
        fixed_s_seed: int = 7,
    ):
        assert q >= 2 and q % 2 == 1, "q should be an odd prime for toy Regev"
        assert 0 < hamming <= n
        assert mode in ("A", "B")
        self.n = n
        self.q = q
        self.hamming = hamming
        self.e_bound = e_bound
        self.mode = mode
        self.vocab_size = q + 3
        self.PAD = q
        self.SEP = q + 1
        self.CLS = q + 2
        if mode == "A":
            self.seq_len = 2 * n + 4
            self.fixed_s = None
        else:
            self.seq_len = n + 3
            rng = np.random.default_rng(fixed_s_seed)
            self.fixed_s = sample_secret(n, hamming, rng)

    def tokenize(self, s: np.ndarray, a: np.ndarray, b: int) -> np.ndarray:
        if self.mode == "A":
            seq = np.concatenate([
                s.astype(np.int64),
                np.array([self.SEP], dtype=np.int64),
                a.astype(np.int64),
                np.array([self.SEP], dtype=np.int64),
                np.array([int(b)], dtype=np.int64),
                np.array([self.CLS], dtype=np.int64),
            ])
        else:
            seq = np.concatenate([
                a.astype(np.int64),
                np.array([self.SEP], dtype=np.int64),
                np.array([int(b)], dtype=np.int64),
                np.array([self.CLS], dtype=np.int64),
            ])
        assert seq.shape[0] == self.seq_len
        return seq


def sample_secret(n: int, hamming: int, rng: np.random.Generator) -> np.ndarray:
    s = np.zeros(n, dtype=np.int64)
    idx = rng.choice(n, size=hamming, replace=False)
    s[idx] = 1
    return s


def encrypt_bit(
    s: np.ndarray,
    m: int,
    cfg: LWEMiniConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    a = rng.integers(0, cfg.q, size=cfg.n, dtype=np.int64)
    e = int(rng.integers(-cfg.e_bound, cfg.e_bound + 1))
    b = int((int(np.dot(a, s)) + e + m * (cfg.q // 2)) % cfg.q)
    return a, b


def make_batch(
    batch_size: int,
    cfg: LWEMiniConfig,
    rng: np.random.Generator,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of LWE Mini samples (Mode A or B)."""
    tokens_batch = np.empty((batch_size, cfg.seq_len), dtype=np.int64)
    labels = np.empty(batch_size, dtype=np.int64)
    for i in range(batch_size):
        s = sample_secret(cfg.n, cfg.hamming, rng) if cfg.mode == "A" else cfg.fixed_s
        m = int(rng.integers(0, 2))
        a, b = encrypt_bit(s, m, cfg, rng)
        tokens_batch[i] = cfg.tokenize(s, a, b)
        labels[i] = m
    return (
        torch.from_numpy(tokens_batch).to(device),
        torch.from_numpy(labels).to(device),
    )


def make_fixed_eval(
    num_samples: int,
    cfg: LWEMiniConfig,
    seed: int = 123,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    return make_batch(num_samples, cfg, rng, device)
