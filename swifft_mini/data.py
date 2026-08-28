"""SWIFFT Mini data generation.

SWIFFT hashes m binary polynomials x_1, ..., x_m (each of degree < n in Z_p[X])
into a single polynomial h(x) = sum_i a_i * x_i in the ring Z_p[X] / (X^n + 1).

Multiplication in Z_p[X] / (X^n + 1) is anti-cyclic (negacyclic) convolution.

Mini parameters (vs full SWIFFT m=16, n=64, p=257):
    m = 4
    n = 16
    p = 17

The multipliers a_1, ..., a_m are sampled once at dataset-construction time
(seed 17 by convention) and kept fixed across all samples. The network learns
to compute the SWIFFT output given only the input polynomials — the constants
are "baked in" in the sense that they're fixed from the model's perspective.

Tokenization:
    Input  = [x_1_0, ..., x_1_{n-1}, ..., x_m_0, ..., x_m_{n-1}, SEP]   (length m*n + 1)
    Output = (n,) coefficients in {0, ..., p-1}
    Model: per-position classifier on the last n positions of a decoder-style
    output head. To keep Mini simple, we use an encoder-only model and train a
    per-output-slot head that attends over the whole sequence.

    Vocab = p + 3 (PAD, SEP, CLS)
        token 0..p-1    -> binary input bits are in {0, 1} (⊂ Z_p)
        token p         -> PAD
        token p+1       -> SEP
        token p+2       -> CLS  (n of them, one per output coefficient)

    Sequence layout:
        [x_1_0, ..., x_m_{n-1}, SEP, CLS, CLS, ..., CLS]  (length m*n + 1 + n)

    The n CLS tokens at positions (m*n + 1, m*n + 2, ..., m*n + n) act as
    query slots. The head reads the final residual at each CLS position and
    emits a categorical prediction over Z_p.
"""

from __future__ import annotations

import numpy as np
import torch


class SWIFFTMiniConfig:
    def __init__(self, m: int = 4, n: int = 16, p: int = 17, mult_seed: int = 17):
        self.m = m
        self.n = n
        self.p = p
        self.vocab_size = p + 3
        self.PAD = p
        self.SEP = p + 1
        self.CLS = p + 2
        self.seq_len = m * n + 1 + n
        self.multipliers = self._sample_multipliers(mult_seed)

    def _sample_multipliers(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.p, size=(self.m, self.n), dtype=np.int64)


def negacyclic_mul(a: np.ndarray, x: np.ndarray, p: int) -> np.ndarray:
    """Compute (a * x) mod (X^n + 1, p) where a, x are length-n coef vectors."""
    n = a.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            sign = 1 if (i + j) < n else -1
            out[k] = (out[k] + sign * int(a[i]) * int(x[j])) % p
    return out


def swifft(xs: np.ndarray, multipliers: np.ndarray, p: int) -> np.ndarray:
    """xs: (m, n) binary. Returns (n,) polynomial coefficients in Z_p."""
    m, n = xs.shape
    h = np.zeros(n, dtype=np.int64)
    for i in range(m):
        h = (h + negacyclic_mul(multipliers[i], xs[i], p)) % p
    return h


def tokenize(xs: np.ndarray, cfg: SWIFFTMiniConfig) -> np.ndarray:
    flat = xs.reshape(-1).astype(np.int64)                   # (m*n,)
    seq = np.concatenate([
        flat,
        np.array([cfg.SEP], dtype=np.int64),
        np.full(cfg.n, cfg.CLS, dtype=np.int64),
    ])
    assert seq.shape[0] == cfg.seq_len
    return seq


def make_batch(
    batch_size: int, cfg: SWIFFTMiniConfig, rng: np.random.Generator,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_batch = np.empty((batch_size, cfg.seq_len), dtype=np.int64)
    targets = np.empty((batch_size, cfg.n), dtype=np.int64)
    for i in range(batch_size):
        xs = rng.integers(0, 2, size=(cfg.m, cfg.n), dtype=np.int64)
        tokens_batch[i] = tokenize(xs, cfg)
        targets[i] = swifft(xs, cfg.multipliers, cfg.p)
    return (
        torch.from_numpy(tokens_batch).to(device),
        torch.from_numpy(targets).to(device),
    )


def make_fixed_eval(
    num_samples: int, cfg: SWIFFTMiniConfig, seed: int = 321, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    return make_batch(num_samples, cfg, rng, device)
