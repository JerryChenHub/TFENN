# Y13--Y15 exact-E311 pair controls

These experiments isolate the point at which the historical E311 accuracy is
lost. They do not use the newer receiver-local multibody MessageBlock and do
not add paths, Gate width, hidden node state, or trainable graph parameters.

| ID | Model and data | Controlled purpose | Formal budget |
|---|---|---|---|
| Y13 | E311-Exact-400K; historical two-benzene shards | Reproduce the non-GNN E311 reference without reimplementing it | 500 epochs, 10k batch, 16k updates, 160M train-pair exposures |
| Y14 | E311-OddGraph-400K; same data and split as Y13 | Measure two-direction evaluation, endpoint OddPair, and signed scatter | Same optimizer, split, epochs, effective batch, updates, and train-pair exposures as Y13 |
| Y15 | E311-OddGraph-5B100K; 10 complete-graph edges per configuration | Test the same strictly pairwise graph strategy on five benzene | 200 epochs, 1k graphs per batch, 16k updates, 160M train-pair exposures |

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
- a sibling validated pair-force NPZ with sample_id, canonical pair_index of
  shape (10, 2), and pair_force_kcal_mol_A of shape (100000, 10, 3);
- 100k independently generated configurations. Formal Y15 rejects trajectory
  grouping and requires group_id to equal sample_id when group_id is present.

The runner rejects the data unless signed aggregation of the ten pair labels
reconstructs every CSV node force within \(10^{-9}\). It splits configurations
before exposing their edges, preventing edge leakage between train,
validation, and test.

Y15 trains on pair-force MSE. Its pair MAE uses the same metric and units as
E311, but cross-dataset numerical comparison is descriptive unless the pair
geometry and target distributions match. Summed node MAE is secondary and is
not directly comparable to the single-pair 0.0018 result.

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
      --pair-npz PATH/TO/five_benzene_100k_pair_forces.npz \
      --output-directory experiments/gnn/runs/e311_y13_y15_pair_controls_v1/Y15_e311_odd_graph_5b100k \
      --device cuda

Y13 uses the historical E series preparation, config, enriched E311 spec,
model builder, and common trainer directly. It asserts that preflight compiled
E311 with exactly 14,926 parameters. Y14 uses the historical common trainer
and best validation checkpoint rule. Y15 also loads
the best-validation checkpoint before its one-time test evaluation. The Y15
selected-model audit then checks global rotation/translation covariance,
independent molecular D6 gauges, node permutation, OddPair identity, and zero
total force with TF32 disabled.

Y14 and Y15 materialize 20,000 directed pair inputs in a formal training batch
(10k two-benzene examples for Y14; 1k graphs times 20 directions for Y15).
Run a one-batch CUDA memory smoke test before committing a long formal run.

## Interpretation boundary

Y14 jointly changes the orientation population seen by RunningRMS and applies
OddPair; its selected-model audit reports raw-forward MAE, raw-reverse MAE,
OddPair MAE, and even leakage so these effects remain observable. Y14/Y15 are
strictly pairwise graph controls, not multilevel or many-body GNNs. The core
runner and strict Y series Comet route require fresh output paths. Preserve a
failed attempt and retrain in a new path instead of weakening provenance
checks or changing the model mathematics.
