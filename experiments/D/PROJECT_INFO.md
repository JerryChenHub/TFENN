# D Series Experiment

## Data

The experiment used the same 400,000 two benzene configurations generated with OPLS 2.0.0 dataset revision 3, stored as four shards of 100,000 samples each:

| Shard | CSV SHA256 |
|---|---|
| `benzene_pair_opls_2_0_0_v3_100k_shard_01.csv` | `bf2968a01fe8740fdd650b52dc244d68369d1afb93cc20848b76f8c1d3a588eb` |
| `benzene_pair_opls_2_0_0_v3_100k_shard_02.csv` | `c34a1ed1629983ca4a30c7fc8f2f405258f4c412785c6769fe4d8b063010e17e` |
| `benzene_pair_opls_2_0_0_v3_100k_shard_03.csv` | `3f3143294c6b56f3748bed16d3acc022b7c888e50aee80bfe2f44642251d244e` |
| `benzene_pair_opls_2_0_0_v3_100k_shard_04.csv` | `5fecab0466a2c234a9d0f08fb75e01cce71100cfad8b837380ff39af783ccf3e` |

The group aware split contained 320,000 Train, 40,000 Validation, and 40,000 Test samples. Its manifest hash was `f09a6123912277c103eaff90b19cd188aa18f4b3203f461ef664f34654aa95c0`.

## Purpose

Experiment 1 tested source visibility, raw-covariant reread, same-TypeKey concat-project bypass, and separately configured additive skip variants. Experiment 2 tested A and B ordering, channel placement, path families, and polynomial degree. Experiment 3 tested Invariant Gate coefficient functions, head parameterization, descriptor transforms, trunk width/depth, and explicit additive skip variants.

## Protocol and time

The study contained 75 models in three concurrent groups of 25. Every model used 500 epochs, batch size 10,000, AdamW, initial learning rate 0.003, weight decay 0.0001, StepLR every 125 epochs with gamma 0.5, float32, and TF32. Model selection used minimum Validation normalized MSE. The three Comet projects were `tfenn_d_series_experiment_1_dense_bypass`, `tfenn_d_series_experiment_2_architecture_paths`, and `tfenn_d_series_experiment_3_invariant_gate`.

The cloud run covered 2026 08 15 08:25:16 UTC through 2026 08 16 16:29:55 UTC. The runtime was Python 3.11.14, PyTorch 2.7.0 with CUDA 12.8.

## History archive

`HISTORY.tar.gz` is a fresh download of the complete cloud result directory `d_series_400k_v1`. It contains 75 complete per model histories, summaries, configurations, status files, Comet identities, stdout and stderr logs, best checkpoints, final checkpoints, the shared split, and all three study manifests, comparisons, and results tables.

Archive size: 17,197,745 bytes
Archive entries: 769
SHA256: `0be840131a0b3d2c32ae5077f9401144a827ce90bd71e8e9655d3b086fc09ef8`
