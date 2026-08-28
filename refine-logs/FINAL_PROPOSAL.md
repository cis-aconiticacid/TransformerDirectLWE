# FINAL_PROPOSAL — LWE Decryption as an Interpretable Neural Subgraph (Mini Scope)

**Status:** Mini-scope slice of the full research plan described in `lwe in transformer.docx`. This file only covers Phase 1 (LWE Mini) and Phase 4 (SWIFFT Mini), which are the two experiments that can be run on a single RTX 4060 (8 GB).

## Research question (Mini scope)

> Can a weight-sparse Transformer, trained with L0 annealing in the style of Gao et al. 2025, learn the Regev LWE decryption circuit — **and** the SWIFFT-style lattice hash — in a form where the underlying arithmetic (inner product mod q, ⌊q/4⌋ rounding, NTT butterflies) is mechanistically readable from the sparse weight graph?

## Contribution we are targeting

1. **C1 (LWE-Mini):** Empirical demonstration that a sparse 2-layer Transformer can decrypt Regev LWE at toy parameters with ≥99 % accuracy, and that the extracted L0-sparse circuit exhibits identifiable nodes for (a) the inner product ⟨a,s⟩ mod q, (b) the subtraction b−⟨a,s⟩, and (c) the ⌊q/4⌋ decision boundary.
2. **C4 (SWIFFT-Mini):** Empirical demonstration that a sparse 2–3 layer Transformer can compute a toy SWIFFT compression (m=4, n=16, p=17) and that the learned sparse subgraph aligns with the NTT butterfly structure that SWIFFT requires.

## Method (Mini scope)

### LWE Mini task

- Parameters: `n=16, q=17, binary s ∈ {0,1}^16 with Hamming weight exactly 4`, Regev CPA-style encryption of a single bit.
- Encryption: `b = (⟨a,s⟩ + e + m·⌊q/2⌋) mod q` where `e ∈ {-1,0,+1}` uniform, `m ∈ {0,1}`.
- Input to Transformer: sequence `[a_0, a_1, …, a_{n-1}, b]` with each element drawn from `{0, …, q−1}` ∪ `{SEP, CLS}`. Total seq_len = 18 (16 coords of `a` + `b` + `CLS`). Vocab = q + 2 = 19.
- Output: single binary prediction head on the `CLS` token's final residual. Loss = binary cross-entropy.
- Two training modes:
  - **Mode A (generic decoder):** `s` randomized every sample, additionally prepended as tokens `[s_0,…,s_{n-1}]`. Seq_len = 34, vocab = 19 (we treat binary `s` as values in {0,1}, which already fit in q+2). The model learns **the algorithm**, not a specific key.
  - **Mode B (secret-as-weight):** Fix one `s*` globally, never shown as input. The model must memorize `s*` in its weights. Seq_len = 18.
- Mini scope runs **Mode A only** to validate feasibility; Mode B is deferred.

### SWIFFT Mini task

- Parameters: `m=4, n=16, p=17` (real SWIFFT is `m=16, n=64, p=257`). Fixed random multipliers `a_1,…,a_m ∈ (ℤ_p[X]/(X^n+1))`.
- Input: `m` binary polynomials `x_1,…,x_m`, each a length-`n` binary vector → flattened to `m·n = 64` binary tokens + SEP.
- Output: `n = 16` polynomial coefficients in `{0,…,p−1}`, produced via `h(x) = Σ_i a_i · x_i` in the ring.
- Seq-to-seq prediction with per-output-coefficient cross-entropy over the p-valued vocab.
- No secret; this phase only tests whether NTT butterflies lift into sparse Transformer weights.

### Architecture (shared)

- 2-layer encoder-only Transformer (LWE Mini) or 3-layer encoder-decoder (SWIFFT Mini).
- `d_model = 128`, `d_ff = 512`, `heads = 4`, `dropout = 0`.
- Token embeddings untied, sinusoidal positional encoding (fixed, not learned — simpler to interpret).
- Layernorm replaced by **RMSNorm** (cleaner for circuit extraction).
- Output head: single linear layer on `CLS` (LWE) or per-position linear (SWIFFT).

### L0 weight-sparsity training (Gao 2025-style)

- Every weight matrix `W` is reparameterized as `W = W_dense ⊙ M`, where `M` is a differentiable binary mask via hard-concrete / Gumbel-sigmoid.
- Loss: `L_task + λ(t) · ||M||_0`.
- `λ(t)` follows an annealing schedule: warmup 5 000 steps at `λ=0`, then linearly ramp `λ` from `0 → λ_max` over 45 000 steps, then hold at `λ_max` for 20 000 steps. `λ_max` tuned so the final L0 density is ~1/1000 of dense parameter count (same ratio Gao et al. report).
- Mean ablation check: for every edge below mask threshold, replace its forward activation by its batch mean; verify task accuracy unchanged.

### Mechanistic interpretability probes

- TransformerLens instrumentation on all residual streams, attention patterns, MLP neuron activations.
- **LWE probes:**
  - Inner-product probe: train a 1-layer linear probe on each residual stream to predict `⟨a,s⟩ mod q`. Look for the layer/position where probe AUC jumps to ~1.0.
  - Decision-boundary probe: similar probe for `(b − ⟨a,s⟩) mod q` and for the binary decryption output.
  - Head attribution: ablate one head at a time; expect a small subset of heads to carry ≥95 % of the decryption accuracy.
- **SWIFFT probes:**
  - Butterfly probe: for each pair `(k, k+n/2)` of polynomial coefficients, regress the residual stream onto the DFT basis `ω^k`. Expect Fourier-basis feature directions to light up in specific attention heads.

## Success criteria (Mini)

| Criterion | LWE Mini | SWIFFT Mini |
|---|---|---|
| Dense baseline accuracy | ≥ 99.5 % | ≥ 99.0 % per-coeff |
| Sparse (L0 ≈ 1/1000) accuracy | ≥ 99.0 % | ≥ 98.0 % per-coeff |
| Mean-ablation drop on pruned edges | ≤ 0.5 % abs | ≤ 1.0 % abs |
| Interpretable circuit node count | ≤ 50 neurons involved end-to-end | ≤ 80 neurons |
| Probe AUC for ⟨a,s⟩ / butterfly | ≥ 0.95 | ≥ 0.90 |

If the sparse run fails to meet its bar but dense baseline meets its bar, we report a negative result on mechanistic transparency — this is still a useful data point for the full proposal.

## Out of scope for Mini

- Mode B (secret-as-weight) — defer to Toy scope.
- Full SWIFFT parameters `m=16, n=64, p=257`.
- CPA reduction / attack taxonomy — theoretical, not this round.
- Comparison against Shamir 2025/288 implementation of crypto-as-DNN.

## Compute budget (Mini)

One RTX 4060 (8 GB). Per run, expected wall-clock:

| Run | Approx step count | Batch | Wall-clock on 4060 |
|---|---|---|---|
| LWE Mini dense | 20 000 | 512 | ~45 min |
| LWE Mini sparse | 70 000 | 512 | ~3 h |
| SWIFFT Mini dense | 20 000 | 256 | ~1 h |
| SWIFFT Mini sparse | 70 000 | 256 | ~4 h |

Total ≈ 9 GPU-hours single-GPU serial. Acceptable for Mini.
