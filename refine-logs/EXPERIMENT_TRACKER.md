# EXPERIMENT_TRACKER (Mini)

Status legend: `PENDING` / `RUNNING` / `DONE` / `FAILED` / `SKIPPED`

| Run ID | Milestone | Description | Config file | Status | Key metric | Notes |
|---|---|---|---|---|---|---|
| R000 | M0 | LWE small-n sanity (n=4,q=7 Mode B) | `configs/sanity_small.yaml` | DONE | 100% eval acc @ step 2700 | CPU-verified 2026-04-18. Grokking at ~step 1200 |
| R000h | M0 | LWE n=4 Mode B (GPU, std=0.1 init) | `configs/sanity_small.yaml` | DONE | 100% eval acc @ step 2600 | GPU 4060, reconfirms init change is safe |
| R000b | M0 | SWIFFT sanity (m=2,n=4,p=17) | `configs/swifft_sanity.yaml` | DONE | 100% coef + full acc @ step 1800 | CPU-verified. |
| R000f | M0 | LWE sparse sanity (n=4, L0 anneal) | `configs/sparse_sanity_n4.yaml` | DONE | soft=hard=1.000, density_hard=0.917 | Probe shows circuit at L1 |
| R004a | M2b | SWIFFT sparse toy (m=2, n=4) | `configs/swifft_sparse_toy.yaml` | DONE | soft=0.991, hard=0.964, density_hard=0.987 | GPU 4060, ~2 min |
| R001 | M1.a | LWE Mode A n=16 | `configs/lwe_dense.yaml` | FAILED | Loss stuck log(2) @ 28k | Scaling barrier |
| R001b | M1.a' | LWE Mode B n=16 | `configs/lwe_dense_modeB_n16.yaml` | FAILED | Loss stuck @ 13k | Scaling barrier |
| R001c | M1.a' v2 | Same, n=4 hyperparams | `configs/lwe_dense_modeB_n16_v2.yaml` | FAILED | Plateau | Confirms arch limit |
| R001d | M1.a'' | LWE Mode B n=8, q=11, h=3 | `configs/lwe_dense_modeB_n8.yaml` | FAILED | Plateau | Scaling barrier n=8 |
| R001e | M1.a'' | LWE Mode B n=8, q=7, h=2 | `configs/lwe_modeB_n8_q7.yaml` | PARTIAL | 0.53 @ 8k | Slow learning — extrapolate 30-50k+ for grok |
| R002a | M1.b' | SWIFFT m=4, n=8, p=17 | `configs/swifft_m4n8.yaml` | FAILED | Plateau log(17) | Same arch barrier |
| R000x | M0 | Original n=16 sanity (deprecated) | `configs/sanity.yaml` | SKIPPED | — | Too hard for 500 steps |
| R001 | M1.a | LWE Mini dense baseline | `configs/lwe_dense.yaml` | PENDING | — | seed 42 |
| R002 | M1.b | SWIFFT Mini dense baseline | `configs/swifft_dense.yaml` | PENDING | — | seed 42, mult seed 17 |
| R003 | M2.a | LWE Mini sparse (L0 anneal) | `configs/lwe_sparse.yaml` | PENDING | — | seed 42, λ anneal |
| R004 | M2.b | SWIFFT Mini sparse (L0 anneal) | `configs/swifft_sparse.yaml` | PENDING | — | seed 42, λ anneal |
| R005 | M3.A1 | LWE sparse, LayerNorm | `configs/ablate_ln.yaml` | PENDING | — | nice-to-have |
| R006 | M3.A2 | LWE sparse, constant λ | `configs/ablate_constant_lambda.yaml` | PENDING | — | nice-to-have |
| R007 | M3.A3 | LWE sparse, L0 target 1/500 | `configs/ablate_dense_l0.yaml` | PENDING | — | nice-to-have |
| R008 | M3.A4 | LWE sparse, n=8 | `configs/ablate_n8.yaml` | PENDING | — | nice-to-have |
| R009 | M4 | Interp polish (post-hoc) | `analysis/interp.py` | PENDING | — | runs on M2 checkpoints |
