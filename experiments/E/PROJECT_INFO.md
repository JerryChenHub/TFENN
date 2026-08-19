# E Series Experiment Archive

## 1. Data identity

The study used 400000 two benzene configurations generated with OPLS 2.0.0 dataset revision 3. The data were stored as four shards of 100000 samples. The shared group aware split contained 320000 training samples, 40000 validation samples, and 40000 test samples.

The center distance range was 4.5 to 7.0 angstrom and the minimum interatomic distance was 3.2 angstrom.

CSV SHA256 values in shard order were:

1. `bf2968a01fe8740fdd650b52dc244d68369d1afb93cc20848b76f8c1d3a588eb`
2. `c34a1ed1629983ca4a30c7fc8f2f405258f4c412785c6769fe4d8b063010e17e`
3. `3f3143294c6b56f3748bed16d3acc022b7c888e50aee80bfe2f44642251d244e`
4. `5fecab0466a2c234a9d0f08fb75e01cce71100cfad8b837380ff39af783ccf3e`

The split manifest hash was `3a64eb6ac96805aad4fe41ef1fd44a0cdc2417d193f8c0852244780a3563ec98`. The full 108 model preflight passed with hash `eaf9de06750ee07e9d605222370f556789d1122882c0dc1806a5bb508225ba58`.

## 2. Concise purpose

1. E0 compared ordinary MLP, model level group averaging, and selected earlier reference networks.
2. E1 tested raw-covariant reread, source visibility, and same-TypeKey bypass behavior.
3. E2 tested synchronous dual stream A and B exchange patterns.
4. E3 tested covariant path families, gate width, and path ablations.
5. E4 tested compact structured mechanisms near an 8000 parameter budget.

The plan title mentioned 125 models, while its explicit model tables defined 108 models. The executed catalog followed the 108 explicit definitions.

## 3. Training protocol

Every planned model used 500 epochs, physical and effective batch size 10000, AdamW with learning rate 0.003 and weight decay 0.0001, and StepLR with step size 125 and gamma 0.5. Training used float32 with TF32 enabled. The selected checkpoint minimized validation normalized MSE. Final train, validation, and test metrics were then evaluated from that selected checkpoint. Test was evaluated once after model selection.

The five Comet projects were:

1. `tfenn_e_series_e0_controls`
2. `tfenn_e_series_e1_raw_reuse_bypass`
3. `tfenn_e_series_e2_dual_stream_exchange`
4. `tfenn_e_series_e3_path_gate_width`
5. `tfenn_e_series_e4_compact_8k`

## 4. Run status and time

The formal run began at `2026-08-17T22:41:28.950876Z`. At the user requested pause on `2026-08-19T04:06:02.058055Z`, 97 models were complete, E219 was interrupted at epoch 319, E422 was interrupted at epoch 22, and 9 models had not started. Both interrupted models retained history, best checkpoint, and resume checkpoint. No E Series training process remained after the pause.

The completed model counts were E0 8 of 8, E1 25 of 25, E2 18 of 25, E3 25 of 25, and E4 21 of 25.

The runtime environment was Python 3.11.14, PyTorch 2.7.0 with CUDA 12.8, and an NVIDIA GeForce RTX 5090.

## 5. Results and archive provenance

`TRAINING_RESULTS.xlsx` contains exactly five columns and all 108 planned model IDs. The 97 completed models contain final values. The 11 unfinished models retain parameter counts and have blank final metric cells.

Final Test MAE is the full test partition mean absolute force component error in kcal per mol per angstrom. Final Validation Loss and Final Train Loss are full partition normalized MSE values from the checkpoint with minimum validation normalized MSE.

`HISTORY.tar.gz` contains the stopped cloud study snapshot, shared split records, preflight manifest, configurations, histories, summaries, statuses, Comet identities, logs, checkpoints, resume state for interrupted models, experiment definitions, model source files, archive audit, and a per file SHA256 manifest.

Archive size was 26279418 bytes. Archive SHA256 was `6531c52af83a307f7feca15d5ed8564acbf8819cc1a6b8c92c28b65ce8788a99`. The archive contained 983 files. `FILE_SHA256SUMS.txt` covered the other 982 files, and local verification found zero missing or mismatched files.
