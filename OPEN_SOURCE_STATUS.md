# Open-source preparation status

Checked on 2026-08-28. This file records what is present in the archived
project snapshot and what still needs attention before a public release.

## Present

- LWE implementation: `common/`, `lwe_mini/`
- Secondary SWIFFT implementation: `swifft_mini/`
- 20 YAML experiment configurations: 14 LWE/sanity configurations and 6
  SWIFFT configurations
- Three analysis scripts for metrics, probes, and scaling plots
- Six research notes in `refine-logs/`
- 30 run directories, including metrics, resolved configurations, plots, and
  18 PyTorch checkpoints
- On-the-fly data generation; no external dataset is required

## Main LWE evidence still on disk

- `runs/R000f_lwe_sparse_n4/`: complete sparse n=4 run, evaluation report,
  curves, and probe plot
- `runs/R015_lwe_sparse_n8/`: complete sparse n=8 run, evaluation report,
  curves, and probe plot
- `runs/R017_lwe_dense_n8/`: partial dense n=8 run and curve
- `runs/R013_modeB_n16_patience/`: 80k-step n=16 patience run, still at
  chance-level accuracy
- `runs/R000g_lwe_modeA_n4/`: unsuccessful generic Mode A attempt

The binary checkpoints are present locally, but should be treated as generated
artifacts rather than source. They are excluded by `.gitignore` so a future
`git add .` does not accidentally create a large binary repository.

## Reconciliation issues

The historical notes are valuable, but they are not a complete machine-
verifiable ledger:

1. `refine-logs/EXPERIMENT_RESULTS.md` describes an `R012` success, while the
   current `runs/R012_modeB_n8_q7_long/` directory contains only a resolved
   config.
2. The tracker lists `configs/ablate_*.yaml` and `analysis/interp.py`, but
   those files are not present.
3. `configs/sanity_small.yaml` defaults to `runs/R000_sanity`, which is also
   the output directory of an older, different sanity configuration. Public
   reproduction should override `out_dir`, as shown in `README.md`.
4. The original proposal document referenced by the notes is not part of this
   project directory.

These are documented rather than silently corrected so that the archived
research history remains intact.

## Before publishing

- Choose and add a license.
- Decide whether to publish raw metrics/plots, or only the source and a short
  curated results table.
- If checkpoints are published, document their exact provenance and expected
  PyTorch version.
- Optionally add a small automated test suite and pin a tested PyTorch/
  NumPy/PyYAML environment.
- Review research-note citations and replace any local-only references with
  public links if the notes are going to be included.
