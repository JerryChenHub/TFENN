# C Series Experiment

## Data

The experiment used 400,000 two benzene configurations generated with OPLS 2.0.0 dataset revision 3, stored as four shards of 100,000 samples each:

| Shard | CSV SHA256 |
|---|---|
| `benzene_pair_opls_2_0_0_v3_100k_shard_01.csv` | `bf2968a01fe8740fdd650b52dc244d68369d1afb93cc20848b76f8c1d3a588eb` |
| `benzene_pair_opls_2_0_0_v3_100k_shard_02.csv` | `c34a1ed1629983ca4a30c7fc8f2f405258f4c412785c6769fe4d8b063010e17e` |
| `benzene_pair_opls_2_0_0_v3_100k_shard_03.csv` | `3f3143294c6b56f3748bed16d3acc022b7c888e50aee80bfe2f44642251d244e` |
| `benzene_pair_opls_2_0_0_v3_100k_shard_04.csv` | `5fecab0466a2c234a9d0f08fb75e01cce71100cfad8b837380ff39af783ccf3e` |

The fixed split was 320,000 Train, 40,000 Validation, and 40,000 Test samples with split seed `20260821`.

## Purpose

Compare a 20k parameter model level GroupConv MLP with 30 Invariant Gate V2 designs while testing A and B routes, interleaving, channel allocation, path ablations, invariant context, and lower and upper capacity controls.

## Protocol and time

The study contained 31 models, each trained for 500 epochs with batch size 10,000, AdamW, initial learning rate 0.003, weight decay 0.0001, StepLR every 125 epochs with gamma 0.5, float32, and TF32. Model selection used minimum Validation normalized MSE. The Comet project was `tfenn_pair_benzene_model_comparison_400k`. The locally retained result document was finalized on 2026 08 14 at 22:47 PDT; exact cloud start and finish timestamps were not retained in that document.

