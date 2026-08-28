# TransformerDirectLWE

An archived research snapshot exploring whether a small Transformer can learn
toy Learning With Errors (LWE) decryption, with optional L0 weight sparsity and
post-hoc mechanistic probes.

This is an unfinished research project. It is preserved as an experimental
starting point for anyone who wants to continue it; the results should not be
read as a cryptographic security claim.

## What is here

The main line is **LWE Mini**:

- Regev-style encryption of one bit at toy parameters.
- A small encoder-only Transformer with RMSNorm and fixed sinusoidal positions.
- Optional hard-concrete/L0 gates on linear weights.
- Linear probes for the inner product, residual term, and decrypted bit.

There is also a secondary `swifft_mini/` implementation and matching configs.
It is retained for completeness, but it is not the main focus of this
snapshot.

### LWE task

For a binary secret `s`, random vector `a`, noise `e`, modulus `q`, and bit
`m`, the generator uses

```text
b = (<a, s> + e + m * floor(q / 2)) mod q
```

The model predicts `m` from tokenized values.

- **Mode A:** a fresh secret is provided as part of every input. This tests
  whether the model can learn the algorithm.
- **Mode B:** one fixed secret is never provided as input. This is easier and
  tests a secret-as-weight toy setup; it is not evidence that the secret is
  cryptographically hidden.

The default Mini parameters are `n=16`, `q=17`, secret Hamming weight `4`,
and `e ∈ {-1, 0, 1}`. The most successful archived runs use the smaller
`n=4` or `n=8`, `q=7`, Mode B task.

## Archived result summary

These are the strongest results whose artifacts are still present locally:

| Run | Task | Final result | Interpretation |
|---|---|---:|---|
| `R000h_n4_init0.1` | Dense LWE, Mode B, `n=4` | `1.000` accuracy | Toy pipeline succeeds |
| `R000f_lwe_sparse_n4` | Sparse LWE, Mode B, `n=4` | soft `1.000`, hard `1.000`, hard density `0.917` | L0 and hard-prune path work |
| `R015_lwe_sparse_n8` | Sparse LWE, Mode B, `n=8` | soft `1.000`, hard `0.991`, hard density `0.570` | Larger toy task still works, with a pruning gap |
| `R017_lwe_dense_n8` | Dense LWE, Mode B, `n=8` | `0.9065` at step `15,000` | Incomplete/partial convergence |
| `R013_modeB_n16_patience` | Dense LWE, Mode B, `n=16` | `0.505` at step `80,000` | No observed grokking |
| `R000g_lwe_modeA_n4` | Dense LWE, Mode A, `n=4` | `0.525` at step `6,000` | Algorithm-learning mode did not converge |

The historical notes in `refine-logs/` contain additional claims, including
a successful `R012` result, but the current directory does not contain that
run's metrics or checkpoint. Treat those claims as unverified from this
snapshot.

## Quick start

Run commands from the repository root. PyTorch is required even for data
generation because the data helpers return tensors.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Use a separate output directory so archived runs are not overwritten.
python -m lwe_mini.train \
  --config configs/sanity_small.yaml \
  --override out_dir=runs/local_sanity

python -m lwe_mini.eval \
  --checkpoint runs/local_sanity/checkpoint.pt \
  --n_eval 1000 \
  --out runs/local_sanity/eval_report.json
```

For CUDA, install the PyTorch build appropriate for the target driver before
installing the remaining requirements. The original experiments were run in
a CUDA-enabled Docker environment; no Dockerfile is included in this project
directory.

Useful analysis commands:

```bash
python analysis/plot_metrics.py runs/R000f_lwe_sparse_n4
python analysis/plot_probes.py runs/R000f_lwe_sparse_n4/eval_report.json
python analysis/plot_scaling.py
```

## Layout

```text
common/       Transformer and hard-concrete/L0 implementation
lwe_mini/     LWE data generation, training, evaluation, probes
swifft_mini/  Secondary SWIFFT toy implementation
configs/      Historical YAML experiment configurations
analysis/     Plotting utilities for saved metrics and probes
runs/         Saved metrics, plots, configs, and selected checkpoints
refine-logs/  Research plan, tracker, results, and decision notes
```

Training data is generated on the fly; there is no separate dataset download.
`checkpoint.pt` files are local generated artifacts and are intentionally
ignored by Git. Metrics, resolved configs, and plots remain useful for
inspection and can be selected for publication separately.

## Scope and limitations

- This is a toy ML experiment, not a cryptosystem and not a security proof.
- Positive results are primarily for fixed-secret Mode B at `n=4` and `n=8`.
- Mode A did not converge in the archived attempts, so the algorithm-learning
  claim remains open.
- Full LWE parameters, formal CPA reductions, attack analysis, and robust
  multi-seed ablations were not completed.
- Some old plan entries refer to missing configs or analysis scripts. The
  repository has not been made artificially consistent by deleting those
  historical notes.

## License

No license has been selected for this snapshot. Choose and add an explicit
open-source license before publishing it for reuse.
