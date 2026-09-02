# Y13 to Y15 E311 controls

These experiments isolate the point at which the historical E311 accuracy is
lost. They do not use the newer receiver-local multibody MessageBlock and do
not add paths, Gate width, hidden node state, or trainable graph parameters.

| ID | Model and data | Controlled purpose | Formal budget |
|---|---|---|---|
| Y13 | E311-Exact-400K; historical two-benzene shards | Reproduce the non-GNN E311 reference without reimplementing it | 500 epochs, 10k batch, 16k updates, 160M train-pair exposures |
| Y14 | E311-OddGraph-400K; same data and split as Y13 | Measure two-direction evaluation, endpoint OddPair, and signed scatter | Same optimizer, split, epochs, effective batch, updates, and train-pair exposures as Y13 |
| Y15 | E311-OddGraph-5B100K; molecular force supervision on five nodes | Test the graph output forces on five benzene without pair force labels | Current repeat uses 1000 epochs, 512 graphs per batch, 157k updates |

All models have exactly 14,926 trainable parameters. Y14 and Y15 evaluate the
same E311 kernel on both endpoint orders in one vectorized call:

\[
q_{i\leftarrow j}=\operatorname{E311}(i,j),\qquad
q_{j\leftarrow i}=\operatorname{E311}(j,i),\qquad
f_{ij}=\tfrac12(q_{i\leftarrow j}-q_{j\leftarrow i}).
\]

They then add \(f_{ij}\) to node \(i\) and \(-f_{ij}\) to node \(j\).
The two calls are required: a single canonical-index E311 call is not
structurally endpoint-odd and would make a multi-node graph depend on node
numbering.

The budget table matches labeled unordered-pair exposure and optimizer
updates. Y14/Y15 nevertheless perform two ordered E311 evaluations per
unordered pair: 320M directed evaluations versus Y13's 160M. This is the
unavoidable compute cost of enforcing endpoint oddness without changing the
historical kernel.

## Locked E311 definition

- C17 sequential typed pipeline
- A(1) -> B(2) -> B(1) -> A(1)
- four width-8 SiLU invariant Gates
- signed identity coefficient heads with dense channel mixing
- legacy same-TypeKey carrier
- generic covariant raw-mixed paths disabled
- invariant raw-mixed descriptors and STF shortcuts retained
- 14,926 trainable parameters

E311's own world-to-root-local-to-world calculation is retained. “No local
frame MessageBlock” means that the newer
src/TFENN/models/e311_multibody_message_block_v1.py is not used.

## Data contract

Y13 and Y14 use the four exact historical revision-3, OPLS 2.0.0 shards and
the shared E-series split.

Y15 requires:

- the validated 100k five-benzene CSV;
- 100k independently generated configurations with five complete molecule rows
  per sample.

The runner reads no pair force artifact. It assigns one independent group to
each CSV configuration and splits complete configurations, preventing molecule
leakage between train, validation, and test.

Y15 trains on normalized molecular force MSE over all five nodes and all three
force components. Validation uses the same molecular force loss and selects the
best checkpoint. Final test MAE and SAE are computed only from the five
molecular output forces in physical units. The graph still computes internal
edge messages before signed aggregation, but they are neither labels nor
reported performance metrics.

## Commands

Set COMET_API_KEY for all three formal runs. They default to the existing Y
series project `tfenn_e311_gnn_y12_diagnostic_v2`. Each run records only train
loss, validation loss, epoch duration, final test MAE, final test SAE, and the
locked hyperparameters.

    python -m experiments.gnn.e311_y13_y15_pair_control_runner_v1 y13 \
      --study-root experiments/gnn/runs/e311_y13_y15_pair_controls_v1/Y13_exact_e311_400k \
      --device cuda

    python -m experiments.gnn.e311_y13_y15_pair_control_runner_v1 y14 \
      --e-study-root experiments/gnn/runs/e311_y13_y15_pair_controls_v1/Y13_exact_e311_400k \
      --output-directory experiments/gnn/runs/e311_y13_y15_pair_controls_v1/Y14_e311_odd_graph_400k \
      --device cuda

    python -m experiments.gnn.e311_y13_y15_pair_control_runner_v1 y15 \
      --csv PATH/TO/five_benzene_100k.csv \
      --output-directory experiments/gnn/runs/e311_y13_y15_pair_controls_v1/Y15_e311_odd_graph_5b100k_node_force \
      --epochs 1000 \
      --batch-size 512 \
      --device cuda

Y13 uses the historical E series preparation, config, enriched E311 spec,
model builder, and common trainer directly. It asserts that preflight compiled
E311 with exactly 14,926 parameters. Y14 uses the historical common trainer
and best validation checkpoint rule. Y15 also loads
the best molecular force validation checkpoint before its one-time test evaluation. The Y15
selected-model audit then checks global rotation/translation covariance,
independent molecular D6 gauges, node permutation, and zero
total force with TF32 disabled.

Y14 materializes 20,000 directed pair inputs in its formal training batch.
The current Y15 repeat materializes 10,240 internal directed edge messages per
batch from 512 graphs and 20 directions per graph.
Run a one-batch CUDA memory smoke test before committing a long formal run.

## Interpretation boundary

Y14 jointly changes the orientation population seen by RunningRMS and applies
OddPair; its selected-model audit reports raw-forward MAE, raw-reverse MAE,
OddPair MAE, and even leakage so these effects remain observable. Y14/Y15 are
strictly pairwise graph controls, not multilevel or many-body GNNs. The core
runner and strict Y series Comet route require fresh output paths. Preserve a
failed attempt and retrain in a new path instead of weakening provenance
checks or changing the model mathematics.
