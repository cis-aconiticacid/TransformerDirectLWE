# TransformerDirectLWE

Archived research code and evidence for toy experiments on learning LWE
decryption and SWIFFT-style arithmetic with sparse Transformers.

## Status

This direction was stopped after the mini-scope evaluation. The strongest
observations were limited to toy settings:

- fixed-secret LWE (Mode B) reached 100% accuracy for small `n=4` and `n=8`,
  but failed to scale to `n=16` in the attempted budgets;
- algorithmic LWE (Mode A) did not grok in the attempted runs;
- toy SWIFFT succeeded at `m=2, n=4`, but larger settings stayed near chance;
- the proposed security reduction and attack analysis were not completed.

The candid stop/continue assessment is in
[`refine-logs/DECISION.md`](refine-logs/DECISION.md). The original mini-scope
proposal, experiment plan, and measured results remain under `refine-logs/`.

## Repository layout

- `lwe_mini/`, `swifft_mini/`: task generation, training, and evaluation code.
- `common/`: Transformer and L0-sparsity components.
- `configs/`: experiment configurations.
- `analysis/`: plotting and probe analysis.
- `runs/`: small logs, metrics, and figures retained as evidence.
- `refine-logs/`: proposal, novelty review, tracker, results, and final decision.

Model checkpoints, caches, virtual environments, and downloaded data are not
stored in Git. They are runtime artifacts rather than source evidence.

## Archival note

The repository is provided as an auditable research snapshot, not as a claim
that the original five-part proposal was achieved. See the limitations in
`DECISION.md` and `EXPERIMENT_RESULTS.md` before reusing conclusions.
