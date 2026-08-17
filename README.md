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

The compact revision three dataset supports local development and focused
tests:

```text
data/benzene_pair/benzene_pair_opls_2_0_0_v3.csv
data/benzene_pair/benzene_pair_opls_2_0_0_v3.json
data/benzene_pair/benzene_pair_opls_2_0_0_v3.validation.json
```

It contains 5000 two benzene configurations with closer centers, explicit
force and moment health limits, complete provenance, and a numerical audit.

The formal comparison uses four independently generated revision three shards
under `data/benzene_pair/v3_100k_shards`.  Each shard contains 100000 samples,
so the shared study dataset contains 400000 samples in total.  Every shard has
its own OPLS 2.0.0 provenance and validation records.  The study validates all
four shards before constructing one shared split.

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
the configuration, learned parameters, and running RMS buffers.  Loading
recompiles all fixed mathematical objects instead of reading a stored orbit or
basis artifact.

Each invariant schema uses a running RMS fitted from the training partition.
The statistic is accumulated across samples and schema channels, rather than
within each individual sample.  A scalar schema therefore keeps its amplitude
instead of collapsing to its sign.  The RMS values and counts are fixed buffers
and add no trainable parameters.

## Training workflow

Run the current workflow from the repository root:

```text
python -m experiments.benzene_pair.train
```

Select a GPU without changing the shared experiment configuration:

```text
python -m experiments.benzene_pair.train --device cuda
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

## Thirty one model comparison

The formal study always trains the model level GroupConv MLP baseline `G00`
first.  This baseline has 20160 trainable parameters and applies the complete
model level D6 by D6 group average.  It is followed, in fixed order, by the
Invariant Gate V2 models `C01` through `C30`.  Their catalog is defined in
`experiments/benzene_pair/invariant_gate_v2_20k_sweep.py` and covers direct,
parallel, serial, alternating, path ablation, small capacity, and full capacity
designs.

The shared protocol is stored in
`experiments/benzene_pair/sweep30_config.json`.  All 31 models use the same
400000 samples, deterministic duplicate aware split, optimizer settings, and
random seeds.  The split contains 320000 training samples, 40000 validation
samples, and 40000 test samples.  Every model trains for 500 epochs with an
effective batch size and physical micro batch size of 10000.  Validation runs
after every epoch, the best checkpoint is chosen only by validation normalized
MSE, and the test partition is evaluated once after checkpoint selection.

### Comet recording

Install the optional experiment dependency without changing the base runtime:

```text
python -m pip install -e ".[experiment]"
```

The comparison is one Comet project named
`tfenn_pair_benzene_model_comparison_400k`.  Each of the 31 models is one Comet
experiment within that project.  Online recording is required for the formal
run, so `COMET_API_KEY` must exist in the launch environment.  The key is read
only from the environment and must not be added to the configuration or the
repository.

Every epoch records training loss, validation loss, learning rate, and timing.
The selected checkpoint records physical metrics for the training, validation,
and test partitions.  A deterministic sample from each partition also records
the relative difference in force norm:

```text
abs(norm(F prediction) minus norm(F target)) divided by max(norm(F target), epsilon)
```

The final record includes the minimum, maximum, median, mean, p90, p95, and p99
of this quantity together with the sample count and near zero target count.
Configurations, histories, summaries, and compact checkpoints are attached to
the corresponding Comet experiment.

### Output layout

The local study directory is
`experiments/benzene_pair/runs/sweep31_400k_v2`.  Its shared split manifest and
indices define the comparison population.  Model artifacts are isolated under
`models/G00` and `models/C01` through `models/C30`.  Each model directory
contains `config.json`, `status.json`, `history.csv`, `best.pt`, `final.pt`,
`summary.json`, `stdout.log`, and `stderr.log`.  `resume.pt` exists only while a
trial is resumable, and `error.json` exists only after a failed trial.  The
study refreshes `results.csv` and `comparison.json` after every model.

### GPU launch in tmux

From the repository root on the GPU host, activate the configured environment,
install the optional dependency, enter the Comet key without echoing it, and
pass it to the new tmux session:

```text
conda activate torch_env
python -m pip install -e ".[experiment]"
read -s COMET_API_KEY
export COMET_API_KEY
tmux set-environment -g COMET_API_KEY "$COMET_API_KEY"
tmux new-session -d -s tfenn_sweep31 "conda run --no-capture-output -n torch_env python -m experiments.benzene_pair.sweep30 run --device cuda"
tmux set-environment -gu COMET_API_KEY
unset COMET_API_KEY
tmux attach-session -t tfenn_sweep31
```

Detaching from tmux leaves the GPU process running.  The formal runner refuses
to start when online Comet recording is enabled but the API key is unavailable.
