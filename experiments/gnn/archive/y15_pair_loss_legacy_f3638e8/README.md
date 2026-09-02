# Archived Y15 pair loss implementation

Status: invalid for the five molecule force objective and retained only for provenance.

The previous implementation is archived by Git commit `f3638e83d7de68b3c5e8f5680717a1ca5d37c99b`. No copied or alternate implementation is kept in the current source tree. This note is the entry point for identifying the invalid loss and retrieving the exact historical source when provenance work requires it.

## Error being archived

The archived runner loaded `pair_force_kcal_mol_A` as the supervised target. It then:

1. computed target RMS from pair force components;
2. optimized MSE between the graph edge output and pair force labels;
3. evaluated validation loss on pair force labels;
4. selected the best checkpoint using pair force validation MSE;
5. sent pair force MAE and SAE to Comet as the final Y15 result.

That objective does not match the corrected Y15 question. Y15 is evaluated as a graph that outputs the forces on five molecules. Pair messages are internal model quantities for this experiment. They are not the declared supervision target and are not uniquely determined by the five molecular force labels.

For a complete graph with five nodes, ten stored edge vectors contain thirty components. The five observable molecular forces have fifteen components and satisfy the three component total force constraint. Internal cycle contributions can therefore change a pair decomposition without changing any molecular output force. Treating one such decomposition as the required label changes the scientific task.

The old implementation and its completed experiment must not be resumed, merged with, or compared numerically as though they used the corrected loss definition.

The affected records are:

1. `target_scale`;
2. `train_normalized_mse`;
3. `validation_normalized_mse`;
4. `best_validation_normalized_mse`;
5. Comet per epoch `train_loss` and `validation_loss`;
6. `protocol.target_scale_component_rms`;
7. checkpoint schema `tfenn_y15_selected_checkpoint` version 1;
8. summary schema `tfenn_y15_e311_odd_graph_result` version 1;
9. `selected_checkpoint.rule`;
10. `selected_test.pair_force`;
11. `selected_test.node_force_after_sum.normalized_mse_using_pair_component_rms`;
12. Comet `final_test_mae` and `final_test_sae`.

The old `best.pt` and `final.pt` weights were trained and selected under pair loss. They cannot be restored into the corrected run. The physical node force MAE stored as an old post selection diagnostic describes that old model only. It was not selected by node force validation and is not a corrected Y15 result.

Y13 and Y14 do not use this Y15 loss path and are not invalidated by this issue.

## Correct replacement

The corrected implementation begins at local commit `7ef423b90081c8d17c2ca12bc86e0afcccb81cf2`. The equivalent deployment short commit on the training server is `4bc60fa`.

The replacement reads only the five molecular force vectors from the validated CSV. Training loss, validation loss, checkpoint selection, final test MAE, final test SAE, and the public graph forward output all use molecular node force.

## Historical experiment identity

Old experiment key: `97c335f4d108461aa2bfc461dd2db407`

Old output directory: `/root/autodl-tmp/TFENN_y13_y15_repeat_bs512_e1000_f3638e8/Y15_E311_OddGraph_5B100K_BS512_E1000`

These artifacts are preserved as evidence of the superseded pair supervised run, not as a valid Y15 molecular force result.

## Exact source retrieval

Inspect the archived runner without changing the working tree:

```text
git show f3638e83d7de68b3c5e8f5680717a1ca5d37c99b:experiments/gnn/e311_y13_y15_pair_control_runner_v1.py
```

For a full historical checkout, use a detached worktree at that commit. Do not copy the archived implementation back over the corrected runner.
