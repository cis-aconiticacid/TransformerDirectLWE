# EXPERIMENT_PLAN — Mini scope only

Generated from `FINAL_PROPOSAL.md` on 2026-04-18. Only LWE-Mini and SWIFFT-Mini runs are included; Toy/Full scopes are deferred.

## Milestone order

1. **M0 – Sanity** (LWE Mini, tiny config, CPU-feasible) — 2 min
2. **M1 – Baseline** (LWE Mini dense, SWIFFT Mini dense)
3. **M2 – Main method** (LWE Mini sparse, SWIFFT Mini sparse)
4. **M3 – Ablation** (inside Mini: drop sparsity schedule / drop RMSNorm / vary L0 target)
5. **M4 – Polish** (circuit visualization + probe metrics)

Must-run: M0, M1, M2. Nice-to-have: M3, M4.

## M0: Sanity

**Goal:** Verify training loop, data generation, loss goes down, eval works. No sparsity, no interpretability.

⚠️ **Note:** The original `n=16, q=17` task is a grokking-style problem — at this scale on CPU the model stays at ~50 % accuracy for many thousands of steps before a sudden phase transition. The real sanity uses a scaled-down LWE (`n=4, q=7, Hamming(s)=2`) which exhibits the same dynamics but completes in ~35 s on CPU.

- Dataset: Mode B LWE Mini (`n=4, q=7, Hamming(s)=2, e ∈ {-1,0,1}`), on-the-fly.
- Architecture: 2-layer, d_model=64, d_ff=128, heads=4, RMSNorm.
- Training: 3 000 steps, batch 128, AdamW lr 3e-3, no L0, no sparsity.
- Metric: decryption accuracy on fixed 2 000-sample eval set.
- **Success criterion:** eval accuracy ≥ 99 % by step 3000.
- **Runtime target:** ≤ 60 s on CPU, ≤ 5 s on 4060. Expect classic grokking: loss plateaus at log(2) ≈ 0.693 for 800–1200 steps, then drops sharply.

Command:
```bash
python -m lwe_mini.train --config configs/sanity_small.yaml
```

**Pipeline sanity (CPU) result logged on 2026-04-18:** 50 % → 66 % → 91 % → 99 % → 100 % between steps 900 → 2700. Confirms code correctness.

## M1: Dense baselines

### M1.a — LWE Mini dense
- Dataset: LWE Mini, 500 000 train (on-the-fly generation), 10 000 eval.
- Architecture: 2-layer, d_model=128, d_ff=512, heads=4, RMSNorm.
- Training: 20 000 steps, batch 512, AdamW lr 3e-4 cosine-decay, seed 42.
- Mode: Mode A (generic decoder, `s` in input).
- **Success criterion:** eval accuracy ≥ 99.5 %.
- **Runtime target:** ~45 min on 4060. **Watch for grokking phase transition** — accuracy may sit at 50 % for 5–15k steps before jumping. If still at 50 % by step 18k, try `lr=1e-3`, extend to 50k steps, or fall back to Mode B to isolate issues.

### M1.b — SWIFFT Mini dense
- Dataset: SWIFFT Mini (`m=4, n=16, p=17`), 300 000 train, 10 000 eval. Fixed random multipliers `a_1..a_4` (seed 17).
- Architecture: 3-layer encoder-decoder, d_model=128, d_ff=512, heads=4, RMSNorm.
- Training: 20 000 steps, batch 256, Adam lr 3e-4 cosine-decay, seed 42.
- **Success criterion:** per-coefficient eval accuracy ≥ 99.0 %.
- **Runtime target:** ~1 h on 4060.

## M2: Sparse main method (L0 annealing)

### M2.a — LWE Mini sparse
- Same data/arch as M1.a, but every linear weight wrapped in hard-concrete mask.
- Training: 70 000 steps. `λ` schedule: warmup 5 000 → ramp 5 000→50 000 → hold 50 000→70 000. `λ_max` tuned to hit L0 density ≈ 1/1000.
- **Success criteria:**
  - Eval accuracy ≥ 99.0 %
  - Final L0 density ≤ 2/1000
  - Mean-ablation drop on pruned edges ≤ 0.5 % abs
- **Runtime target:** ~3 h on 4060.

### M2.b — SWIFFT Mini sparse
- Same as M1.b with masks.
- Training: 70 000 steps, same λ schedule.
- **Success criteria:**
  - Per-coefficient eval accuracy ≥ 98.0 %
  - L0 density ≤ 2/1000
- **Runtime target:** ~4 h on 4060.

## M3: Mini ablations (NICE-TO-HAVE)

| Run | Change from M2.a | Question |
|---|---|---|
| A1 | Drop RMSNorm (use LayerNorm) | Does Norm choice affect circuit extractability? |
| A2 | Drop λ annealing (constant λ_max from step 0) | Does annealing matter? |
| A3 | L0 target 1/500 vs 1/2000 | Sparsity sweet spot. |
| A4 | n=8 instead of 16 | Circuit size scaling. |

Run only if M2.a converged and budget allows.

## M4: Interpretability polish (NICE-TO-HAVE)

Not a training run — post-hoc analysis of the M2.a / M2.b checkpoints.

- Probe training (linear probes for ⟨a,s⟩, b−⟨a,s⟩, butterfly features).
- Head ablation grid → circuit contribution table.
- Neuron-level activation patching for ⌊q/4⌋ boundary.
- Sparse-graph visualization (Graphviz dot file).

## Overall Mini budget

- Must-run total: ~8.5 GPU-hours on a single 4060.
- With ablations: ~18 GPU-hours.

## Seeds

Fixed: data seed 17 (SWIFFT multipliers), training seed 42. For multi-seed ablation, use seeds [42, 200, 201] — defer unless M2 succeeds.
