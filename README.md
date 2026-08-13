# TFENN

TFENN predicts the force on the first molecule of a benzene pair with an
information preserving invariant gate pipeline.  The model respects the
independent proper D6 gauge of both benzene molecules and SO3 covariance in
world coordinates.

## Project structure

1. `src/TFENN/tensor_math` contains the mathematical compiler and runtime.

2. `src/TFENN/models/invariant_gate_pipeline_v2.py` contains the current
   configurable network.

3. `src/TFENN/data` contains the long format dataset loader and OPLS data
   generator.

4. `experiments/benzene_pair` contains one training entry point, one current
   configuration, and the data validator.

5. `tests` contains the mathematical, model, data, symmetry, and workflow
   checks.

6. `Archive` is ignored and contains historical experiment artifacts.  The
   active code does not read it.

## Current data

The active training dataset is revision three:

```text
data/benzene_pair/benzene_pair_opls_2_0_0_v3.csv
data/benzene_pair/benzene_pair_opls_2_0_0_v3.json
data/benzene_pair/benzene_pair_opls_2_0_0_v3.validation.json
```

It contains 5000 two benzene configurations with closer centers, explicit
force and moment health limits, complete provenance, and a numerical audit.

`Benzene_10000_5.0_10.0_3.0_gamma1.csv` is retained only as a frozen fixture
for the unchanged tensor_math regression tests.  It is not current training
data.

## Information preserving pipeline

The two reserved geometric inputs are `x`, the displacement in the first
benzene frame, and `r`, the relative pose encoded into primitive B blocks.
Every stage declares its A or B stream, readable sources, channel count,
invariant trunk width, skip sources, and enabled covariant path families.

Typed state is cumulative.  Raw geometry and earlier hidden states remain
available to later stages.  Each compiled path remains a distinct channel
branch until concatenation and a representation preserving channel
projection.  Mixed displacement and pose contractions, complete low degree C
bases, and manifest driven STF shortcuts provide pose sensitive scalar and
covariant features without discarding source identity.

The default acyclic graph is configured in
`experiments/benzene_pair/config_v2.json`.  Its final stage is one A channel,
which represents the predicted force vector in the root benzene frame.

## Offline and runtime boundary

`build_invariant_gate_pipeline_v2` receives the group generators and a plain
configuration.  During model construction it compiles anchors, primitive B
components, type catalogs, lifts, and C bases.  These fixed objects are runtime
buffers and are never trained.

Forward evaluation performs only geometry conversion, pose encoding, scalar
evaluation, fixed contractions, invariant trunks, coefficient heads, and typed
channel projections.  It never calls the compiler.  Checkpoints contain only
the configuration and learned parameters, so loading recompiles all fixed
mathematical objects instead of reading a stored orbit or basis artifact.

## Training workflow

Run the current workflow from the repository root:

```text
python -m experiments.benzene_pair.train
```

The default configuration records dataset revision, split and model seeds,
optimizer, scheduler, pipeline structure, epoch count, validation selection,
and the required final loss ratio.  Each run writes `history.csv`,
`summary.json`, `best.pt`, and `final.pt` under its configured run directory.

History records training and validation loss from epoch zero onward.  Summary
records physical regression metrics, relative RMSE, R squared, split identity,
data hashes, the complete pipeline configuration, candidate path audit,
runtime versions, gradient checks, and symmetry residuals.  The best checkpoint
is selected only by validation normalized MSE.

Run the focused verification with:

```text
python -m pytest tests/models/test_invariant_gate_pipeline_v2.py tests/data/test_benzene_cluster.py tests/experiments/test_benzene_pair_training.py
```
