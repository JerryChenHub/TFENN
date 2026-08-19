# Experiment archive layout

Completed studies use one named subfolder under `experiments`.

Each completed experiment folder contains exactly three top level items:

1. `PROJECT_INFO.md` records the data identity, concise purpose, protocol, run time, environment, and history archive checksum.
2. `TRAINING_RESULTS.md` or `TRAINING_RESULTS.xlsx` records the model designs and measured training, validation, and test results.
3. `HISTORY.tar.gz` stores the complete cloud history, including per model configuration, epoch history, summary, status, Comet identity, logs, best checkpoint, final checkpoint, and study level manifests.

`benzene_pair` remains the shared executable source directory because the study runners and tests use its module paths. Future completed studies should be archived as another sibling folder following the same three item contract.

Historical labels such as `raw bypass`, `residual`, or `GroupConv` do not override the current flow, raw-access, invariant-context, and same-TypeKey bypass definitions in `src/TFENN`.
