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

Experiment 1 tested source visibility, raw bypass, and typed residual policies. Experiment 2 tested A and B ordering, channel placement, path families, and polynomial degree. Experiment 3 tested Invariant Gate coefficient functions, head parameterization, descriptor transforms, trunk width, depth, and residual structure.

## Protocol and time

The study contained 75 models in three concurrent groups of 25. Every model used 500 epochs, batch size 10,000, AdamW, initial learning rate 0.003, weight decay 0.0001, StepLR every 125 epochs with gamma 0.5, float32, and TF32. Model selection used minimum Validation normalized MSE. The three Comet projects were `tfenn_d_series_experiment_1_dense_bypass`, `tfenn_d_series_experiment_2_architecture_paths`, and `tfenn_d_series_experiment_3_invariant_gate`. The first retained model completed at 2026 08 15 02:34:02 PDT, the final model completed at 2026 08 16 09:29:55 PDT, and the verified export was created at 2026 08 17 16:56:35 UTC.

