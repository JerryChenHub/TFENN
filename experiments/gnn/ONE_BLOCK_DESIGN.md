# Five Benzene One Block GNN

## Completion criterion

The experiment uses exactly one Message Block. A second Message Block is not used because the first block already reduces validation loss by more than twenty percent.

The final accepted run uses one thousand complete five molecule samples. The split unit is a complete `sample_id` graph. The deterministic split contains eight hundred training graphs, one hundred validation graphs, and one hundred test graphs.

## Graph

Each graph contains five molecular nodes. Every ordered pair of distinct nodes is connected, giving twenty directed edges. For the edge from sender (j) to receiver (i), the A type source is

\[
d_{ij}=\frac{X_j-X_i}{8}.
\]

There is no distance cutoff and no graph library dependency.

Incoming messages are aggregated independently for every TypeKey by ordinary summation. The final A message has one channel, so its incoming sum is the normalized molecular force prediction. The returned B messages are also summed by TypeKey for audit, but there is no later node update because this experiment ends after one Message Block.

## Typed inputs

The rotation matrix of every molecule enters `PoseEncoder`. The primitive benzene manifest is discovered from the supplied generators and contains the rank two and rank six B blocks. The implementation never hardcodes their representation dimensions or Hom basis dimensions.

The initial hidden values use

\[
e=d,\qquad h_i^{B}=B_i,\qquad h_j^{B}=B_j.
\]

All typed tensors use channel as the penultimate axis and representation coordinate as the final axis.

## Message Block

Every covariant path is unary and uses a complete registered Hom basis. All coefficient heads have signed linear output. All channel projections act only on channel axes and use no typed bias.

### A middle stage

The target is A with one channel. Direct carriers are `d` and `e`. Registered paths are A from `d`, A from receiver B zero, A from sender B zero, A from receiver B one, and A from sender B one.

### B wide stage

Each B TypeKey has two output channels. Direct carriers are the sender raw B and sender hidden B of the same TypeKey. Registered paths are B from `d` and B from `a1`.

### B output stage

Each B TypeKey is compressed to one output channel. Direct carriers are sender raw B, sender hidden B, and the matching two channel wide B value. Registered paths are B from `d` and B from `a1`.

### A output stage

The target is A with one channel. Direct carriers are `d`, `e`, and `a1`. Registered paths are A from refined B zero and A from refined B one.

The full model has five hundred fifty trainable parameters.

## Gate invariants

Every stage has its own `Linear` then `SiLU` Gate trunk with width eight. Each trunk reads four raw invariants.

1. Constant one

2. Scaled edge distance

3. Receiver and sender B zero inner product

4. Receiver and sender B one inner product

The two pose contractions only combine values with the same TypeKey. Construction verifies that the registered B actions are orthogonal in the current normalized STF coordinates before using the identity metric.

## Target scaling

Only force is trained in this first experiment. Torque is not a target. All three Cartesian force components share one scalar training scale

\[
f_{\mathrm{scale}}=\sqrt{\operatorname{mean}_{\mathrm{train}} F^2}
=0.707347040705358.
\]

There is no component specific centering or scaling. The final A projection is initialized to zero, so epoch zero is exactly the zero force baseline.

## Accepted result

The run seed is `20260824`. Adam uses learning rate `0.01`, batch size one hundred, and twenty epochs on CPU. The best validation checkpoint is epoch three.

Initial validation physical MSE is `0.47745959144238703`.

Best validation physical MSE is `0.29741078527872233`.

Validation loss reduction is `37.70974746150647` percent.

Test physical MSE at the selected checkpoint is `0.3141443845372594`.

Test component MAE is `0.40348110050952346` kcal per mol per angstrom.

The required reduction is twenty percent, so one Message Block passes and a second block is not introduced.

## Reproduction

From the repository root, use the existing `ml_torch` environment and run

```powershell
D:\miniconda3\envs\ml_torch\python.exe -m experiments.gnn.train_one_block
```

The default output directory is ignored by Git and contains `history.csv`, `summary.json`, and `best_checkpoint.pt`.

The checkpoint is the selected validation model for evaluation and later warm starting. It does not promise an exact replay of the stochastic data order after the selected epoch.

## Current boundary

This experiment verifies learnability, typed shape discipline, generator covariance, graph permutation equivariance, translation invariance, finite gradients, and the requested validation loss reduction. It does not yet impose a conservative energy model, pairwise Newton symmetry, or torque prediction.
