# TFENN

TFENN uses complete symmetry constrained tensor maps as neural network paths.
The current experiment predicts the force on the first molecule in a benzene
pair while preserving independent proper D6 gauge symmetry for both molecules
and ordinary SO3 covariance in world coordinates.

## Project structure

1. `src/TFENN/tensor_math` contains the mathematical compiler and runtime.
2. `src/TFENN/models` contains the configurable invariant gate pipeline.
3. `src/TFENN/data` contains the long format dataset loader and OPLS generator.
4. `data/benzene_pair` contains the current OPLS 2.0.0 pair dataset and one
   explicitly versioned legacy dataset.
5. `experiments/benzene_pair` contains one complete training workflow and its
   editable JSON configuration.
6. `tests` contains tensor, model, data, symmetry, and workflow checks.

Historical studies and superseded outputs are isolated in the ignored
`.legacy` directory.  They are not copied into Docker or committed by default.

## Configurable pipeline

The pipeline has two reserved geometric inputs.  `x` is the displacement in
the first benzene frame and belongs to A space.  `r` is the relative pose after
offline anchor compilation and belongs to the discovered primitive B space.

The default information preserving network is defined in
`experiments/benzene_pair/config_v2.json`.  Each stage declares:

1. Its unique name and A or B output stream.
2. Any earlier stages or geometric inputs that feed it.
3. Its learned channel count.
4. Which raw and earlier hidden sources remain visible.
5. Whether to compile unary, symmetric degree two, raw mixed, and manifest
   driven STF paths.
6. The width and activation of one shared invariant trunk.
7. Explicit limits for invariant channels, coefficient heads, and compiler
   allocation.

The ordered stage list is an acyclic graph.  Raw displacement and every
primitive pose block remain in cumulative typed state.  Every path output is
kept as a separate channel branch, concatenated with configured skips, and
then mapped by a representation preserving channel projection.  The final A
readout can therefore use raw and hidden sources directly.  The configured
final stage must be one A channel because the current target is one force
vector.

## Offline and runtime boundary

`build_invariant_gate_pipeline_v2` receives only the group generators and the
plain configuration.  It compiles anchors, discovers primitive B components,
builds the type catalog, and compiles complete low degree scalar and covariant
C bases once.  Manifest ranks also create generic STF scalar and vector
shortcuts.  Forward performs geometry conversion, pose encoding, normalized
scalar evaluation, fixed C contractions, shared invariant trunks, path heads,
and typed channel projections.

No orbit table, anchor file, C basis file, or artifact cache is stored in the
repository.  Rebuilding a model recompiles these fixed tensors from the two
generators.  Forward never invokes a compiler.

## Training workflow

The complete default run is defined in
`experiments/benzene_pair/config_v2.json`.  It uses the closer revision three
pair data and contains data provenance paths,
split seeds, model seed, optimizer, scheduler, all pipeline settings, 500
epochs, validation settings, and the required final loss ratio.

`experiments/benzene_pair/config.json` remains the explicit V1 example and old
V1 checkpoints still restore through the version aware training loader.

Run it from the repository root with:

```text
python -m experiments.benzene_pair.train
```

Each run writes `history.csv`, `summary.json`, `best.pt`, and `final.pt` in its
configured output directory.  History records training and validation loss at
epoch zero and after every epoch.  Summary records physical MSE, RMSE, MAE,
relative RMSE, R squared, split membership, data hashes, the full network
configuration, gate basis dimensions, runtime versions, gradient checks, and
symmetry residuals.

Checkpoint files contain learned parameters and configuration only.  Loading a
checkpoint calls the same builder to recompile fixed anchors, lifts, and C bases
from the two generators.  Fixed tensor artifacts and group orbits are never
serialized into a checkpoint.

Run the focused validation with:

```text
python -m pytest tests/models/test_invariant_gate_pipeline_v2.py tests/experiments/test_benzene_pair_training.py
```

## Pair hyperparameter study

The fixed study definition is split across three files.  The catalog generator
in `experiments/benzene_pair/hyper_catalog.py` defines ten network topologies
and ten coupled complexity profiles.  Their Cartesian product gives exactly one
hundred unique designs.  `catalog_v1.json` stores the offline preflight result
for every design, including parameter count, gate count, basis dimensions, and
the final simple to complex order.  `hyper_config.json` stores the common data,
split, seed, runtime, and 1500 epoch protocol.

Invariant Gate width 64 means that every scalar coefficient MLP has one hidden
layer with 64 units.  Covariant tensor channels remain a searched quantity from
two through six, which keeps the largest design below two million learned
parameters.  Profiles jointly change ranks, lifts, channels, scalar features,
and optimizer settings, so the study is a broad coupled search rather than a
single factor ablation.

The current 5000 sample input and its numerical audit are:

```text
data/benzene_pair/benzene_pair_opls_2_0_0_v2.csv
data/benzene_pair/benzene_pair_opls_2_0_0_v2.json
data/benzene_pair/benzene_pair_opls_2_0_0_v2.validation.json
```

The metadata identifies the OPLS 2.0.0 source as a local candidate and records
its source tree digest.  It must not be represented as a clean public release.

Run a short isolated check with:

```text
python -m experiments.benzene_pair.hyper_search smoke --device cpu --epochs 1 --sample_limit 96
```

Start or resume the complete study with:

```text
python -m experiments.benzene_pair.hyper_search run
```

Each design has its own configuration, status, history, error report, summary,
best model, and final model.  Resume state is written every 25 epochs and stores
only learned parameters, optimizer state, scheduler state, and random number
states.  Fixed tensor bases and group orbits are never serialized.  A failed
design is recorded and the parent process continues with the next design.

The study writes `master_results.csv` in catalog complexity order and
`validation_ranking.csv` in validation loss order.  Test metrics are recorded
only after selecting the best validation checkpoint and never affect ranking.
