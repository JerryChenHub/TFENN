# G Series Project Information

## Scope

The G series tests E311 mechanisms with fourteen variants. This curated record adopts CPU results only and contains the first three complete paired seeds for every variant.

## Protocol

1. The shared E311 split contains 320,000 training, 40,000 validation, and 40,000 test samples.

2. The split indices SHA256 is `50e6bf0e32c1bb9b0bddb689097a4a38a5d74a5bcf12b0fc8471f6b1f4cf50b1`.

3. Each run uses 500 epochs and batch size 1,000 under the CPU protocol.

4. `TRAINING_RESULTS.csv` contains 42 complete rows, comprising fourteen variants and three seeds per variant.

5. `HISTORY.tar.gz` preserves the CPU histories, checkpoints, summaries, manifests, logs, and Gate audit artifacts. Its SHA256 is `9e567b594f5dab7e98d2c18afdd7e266f9588fc889c1cc8948804491d76a33fb`.

## Interpretation

The concise supported conclusions are recorded in `../../EXPERIMENT_RECORD.md`. The archived Gate audit artifacts remain available for a future invariant analysis, but that analysis is outside the present conclusion set.
