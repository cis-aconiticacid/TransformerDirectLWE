# Experiment artifacts

This directory contains historical outputs from the LWE and secondary SWIFFT
experiments. A run may contain:

- `resolved_config.yaml`: configuration saved at the start of training;
- `metrics.json`: periodic training/evaluation metrics;
- `eval_report.json`: post-training accuracy and probe results;
- `*.png`: generated curves or probe plots;
- `checkpoint.pt`: a generated PyTorch checkpoint.

Not every directory is complete. Directories with only a resolved config are
aborted or otherwise incomplete runs. The historical research notes in
`../refine-logs/` explain the intended experiment sequence, while
`../OPEN_SOURCE_STATUS.md` reconciles it with the artifacts currently present.
