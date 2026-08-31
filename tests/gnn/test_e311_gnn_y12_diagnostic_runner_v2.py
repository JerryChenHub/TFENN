from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from experiments.gnn.e311_gnn_12_experiment_core_v1 import SupervisionV1
from experiments.gnn import e311_gnn_y12_diagnostic_runner_v2 as runner


def _assert_nested_equal(first: Any, second: Any, location: str = "root") -> None:
    if isinstance(first, Tensor):
        assert isinstance(second, Tensor), location
        assert torch.equal(first, second), location
    elif isinstance(first, Mapping):
        assert isinstance(second, Mapping), location
        assert tuple(first) == tuple(second), location
        for key in first:
            _assert_nested_equal(first[key], second[key], f"{location}.{key}")
    elif isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        assert isinstance(second, Sequence), location
        assert len(first) == len(second), location
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            _assert_nested_equal(left, right, f"{location}.{index}")
    else:
        assert first == second, location


def _tensor_prefix(value: runner.TensorBucketV2, count: int) -> runner.TensorBucketV2:
    return runner.TensorBucketV2(
        centers=value.centers[:count],
        frames=value.frames[:count],
        normalized_target=value.normalized_target[:count],
        physical_node_target=value.physical_node_target[:count],
        pair_index=value.pair_index,
    )


def _legacy_evaluate_loss(
    model: Any,
    tensor_data: runner.TensorBucketV2,
    supervision: SupervisionV1,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    squared_sum = 0.0
    component_count = 0
    pair_index = tensor_data.pair_index.to(device)
    with torch.inference_mode():
        for start in range(0, len(tensor_data.centers), batch_size):
            end = min(start + batch_size, len(tensor_data.centers))
            difference = runner._normalized_prediction(
                model,
                tensor_data.centers[start:end].to(device),
                tensor_data.frames[start:end].to(device),
                pair_index,
                supervision,
            ) - tensor_data.normalized_target[start:end].to(device)
            squared_sum += float(difference.double().square().sum().cpu())
            component_count += difference.numel()
    return squared_sum / component_count


def _legacy_train_step(
    model: Any,
    tensor_data: runner.TensorBucketV2,
    selection: Tensor,
    supervision: SupervisionV1,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    difference = runner._normalized_prediction(
        model,
        tensor_data.centers[selection].to(device),
        tensor_data.frames[selection].to(device),
        tensor_data.pair_index.to(device),
        supervision,
    ) - tensor_data.normalized_target[selection].to(device)
    loss = difference.square().mean()
    assert bool(torch.isfinite(loss))
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())
    optimizer.step()
    return loss.detach()


class FakeCometBackend:
    disabled = False
    url = "https://example.invalid/experiment"

    def __init__(self) -> None:
        self.parameters: dict[str, Any] | None = None
        self.metric_calls: list[tuple[dict[str, float], dict[str, Any]]] = []
        self.end_count = 0

    def get_key(self) -> str:
        return "experiment_key_123"

    def log_parameters(self, values: Mapping[str, Any]) -> None:
        self.parameters = dict(values)

    def log_metrics(self, values: Mapping[str, float], **kwargs: Any) -> None:
        self.metric_calls.append((dict(values), dict(kwargs)))

    def end(self) -> None:
        self.end_count += 1


def test_fixed_protocols_and_supervision_contract() -> None:
    p0 = runner.P0_CURRENT_X01_PROTOCOL
    p1 = runner.P1_LEGACY_5K_V3_PROTOCOL
    assert asdict(p0) == {
        "name": "current_x01_protocol",
        "optimizer": "AdamW",
        "learning_rate": 0.002,
        "weight_decay": 0.000001,
        "batch_size": 100,
        "scheduler": "ReduceLROnPlateau",
        "evaluation_cadence_steps": 16,
        "scheduler_step_updates": None,
        "scheduler_gamma": None,
        "scheduler_factor": 0.5,
        "scheduler_patience": 20,
        "minimum_learning_rate": 0.00001,
    }
    assert asdict(p1) == {
        "name": "legacy_5k_v3_protocol",
        "optimizer": "AdamW",
        "learning_rate": 0.005,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "scheduler": "StepLR",
        "evaluation_cadence_steps": 63,
        "scheduler_step_updates": 6300,
        "scheduler_gamma": 0.5,
        "scheduler_factor": None,
        "scheduler_patience": None,
        "minimum_learning_rate": None,
    }
    assert runner.target_steps_v2("Y01") == 8000
    assert all(runner.target_steps_v2(f"Y{index:02d}") == 31500 for index in range(2, 13))
    assert all(
        runner.y_supervision_v2(f"Y{index:02d}") is SupervisionV1.PAIR_FORCE
        for index in range(1, 6)
    )
    assert all(
        runner.y_supervision_v2(f"Y{index:02d}") is SupervisionV1.NODE_FORCE
        for index in range(6, 13)
    )
    assert runner.RUNNER_VARIANT == "fast_v2"
    assert runner.P0_CURRENT_X01_PROTOCOL.evaluation_cadence_steps == 16
    assert runner.P1_LEGACY_5K_V3_PROTOCOL.evaluation_cadence_steps == 63


def test_fast_v2_finite_record_and_tail_weighting_cadences() -> None:
    assert [
        step
        for step in range(1, 205)
        if runner.should_check_gradients_v2(step)
    ] == [1, 2, 3, 4, 100, 200]
    assert runner.should_check_gradients_v2(8001) is False
    assert runner.should_check_gradients_v2(8100) is True
    assert [
        index
        for index in range(1, 13)
        if runner.should_persist_evaluation_v2(index, index, 12)
    ] == [1, 10, 12]
    accumulator = runner.ProtocolEpochLossAccumulatorV2(torch.device("cpu"))
    accumulator.add(
        runner.TrainStepResultV2(
            loss=torch.tensor(1.0),
            component_count=4,
            loss_finite=torch.tensor(True),
        )
    )
    accumulator.add(
        runner.TrainStepResultV2(
            loss=torch.tensor(9.0),
            component_count=1,
            loss_finite=torch.tensor(True),
        )
    )
    assert accumulator.finish() == pytest.approx(2.6, abs=0.0)
    nonfinite = runner.ProtocolEpochLossAccumulatorV2(torch.device("cpu"))
    nonfinite.add(
        runner.TrainStepResultV2(
            loss=torch.tensor(float("nan")),
            component_count=3,
            loss_finite=torch.tensor(False),
        )
    )
    with pytest.raises(RuntimeError, match="training loss became nonfinite"):
        nonfinite.finish()


def test_dataset_routing_counts_and_shared_five_benzene_split() -> None:
    current = runner.prepare_y_experiment_data_v2("Y01")
    legacy = runner.prepare_y_experiment_data_v2("Y04")
    five_first = runner.prepare_y_experiment_data_v2("Y06")
    five_last = runner.prepare_y_experiment_data_v2("Y12")
    assert current.dataset_id == "pair_2k_current"
    assert current.supervision is SupervisionV1.PAIR_FORCE
    assert (current.train.sample_count, current.validation.sample_count, current.test.sample_count) == (
        1600,
        200,
        200,
    )
    assert legacy.dataset_id == "pair_5k_legacy_v3"
    assert legacy.supervision is SupervisionV1.PAIR_FORCE
    assert (legacy.train.sample_count, legacy.validation.sample_count, legacy.test.sample_count) == (
        4000,
        500,
        500,
    )
    assert five_first.supervision is SupervisionV1.NODE_FORCE
    assert five_last.supervision is SupervisionV1.NODE_FORCE
    assert (five_first.train.sample_count, five_first.validation.sample_count, five_first.test.sample_count) == (
        800,
        100,
        100,
    )
    assert np.array_equal(five_first.split.train, five_last.split.train)
    assert np.array_equal(five_first.train.centers_world, five_last.train.centers_world)
    assert five_first.force_scale == five_last.force_scale


@pytest.mark.parametrize(
    ("experiment_id", "protocol"),
    (
        ("Y01", runner.P0_CURRENT_X01_PROTOCOL),
        ("Y12", runner.P1_LEGACY_5K_V3_PROTOCOL),
    ),
)
def test_fast_v2_matches_legacy_tensor_trajectory_and_float64_evaluator(
    experiment_id: str,
    protocol: runner.OptimizerProtocolV2,
) -> None:
    device = torch.device("cpu")
    data = runner.prepare_y_experiment_data_v2(experiment_id)
    train_tensor = _tensor_prefix(runner._tensor_bucket(data.train, data), 3)
    validation_tensor = _tensor_prefix(runner._tensor_bucket(data.validation, data), 3)
    legacy_model = runner._build_model(experiment_id, data.force_scale, 77, device)
    fast_model = runner._build_model(experiment_id, data.force_scale, 77, device)
    legacy_optimizer, legacy_scheduler = runner._build_optimizer_and_scheduler(
        legacy_model,
        protocol,
    )
    fast_optimizer, fast_scheduler = runner._build_optimizer_and_scheduler(
        fast_model,
        protocol,
    )
    legacy_sampler = runner.StatefulBatchSamplerV2(3, 2, 91)
    fast_sampler = runner.StatefulBatchSamplerV2(3, 2, 91)
    for global_step in (1, 2):
        legacy_selection = legacy_sampler.next_indices()
        fast_selection = fast_sampler.next_indices()
        assert torch.equal(legacy_selection, fast_selection)
        legacy_loss = _legacy_train_step(
            legacy_model,
            train_tensor,
            legacy_selection,
            data.supervision,
            legacy_optimizer,
            device,
        )
        fast_result = runner._train_step(
            fast_model,
            train_tensor,
            fast_selection,
            data.supervision,
            fast_optimizer,
            device,
            global_step,
        )
        assert torch.equal(legacy_loss, fast_result.loss)
        if protocol.scheduler == "StepLR":
            legacy_scheduler.step()
            fast_scheduler.step()
        _legacy_evaluate_loss(
            legacy_model,
            train_tensor,
            data.supervision,
            2,
            device,
        )
        legacy_validation = _legacy_evaluate_loss(
            legacy_model,
            validation_tensor,
            data.supervision,
            2,
            device,
        )
        fast_validation = runner._evaluate_loss(
            fast_model,
            validation_tensor,
            data.supervision,
            2,
            device,
        )
        assert fast_validation == legacy_validation
        if protocol.scheduler == "ReduceLROnPlateau":
            legacy_scheduler.step(legacy_validation)
            fast_scheduler.step(fast_validation)
        _assert_nested_equal(legacy_model.state_dict(), fast_model.state_dict(), "model")
        _assert_nested_equal(
            legacy_optimizer.state_dict(),
            fast_optimizer.state_dict(),
            "optimizer",
        )
        _assert_nested_equal(
            legacy_scheduler.state_dict(),
            fast_scheduler.state_dict(),
            "scheduler",
        )
        _assert_nested_equal(
            legacy_sampler.state_dict(),
            fast_sampler.state_dict(),
            "sampler",
        )


def test_stateful_sampler_matches_three_dataloader_epochs_and_resumes_midcycle() -> None:
    sample_count = 17
    batch_size = 5
    seed = 31
    loader = DataLoader(
        TensorDataset(torch.arange(sample_count)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    expected_epochs = [torch.cat([batch[0] for batch in loader]) for _ in range(3)]
    sampler = runner.StatefulBatchSamplerV2(sample_count, batch_size, seed)
    actual_epochs = []
    batch_count = (sample_count + batch_size - 1) // batch_size
    for _ in range(3):
        actual_epochs.append(torch.cat([sampler.next_indices() for _ in range(batch_count)]))
    for expected, actual in zip(expected_epochs, actual_epochs, strict=True):
        assert torch.equal(expected, actual)

    sampler = runner.StatefulBatchSamplerV2(sample_count, batch_size, seed)
    first = sampler.next_indices()
    second = sampler.next_indices()
    state = sampler.state_dict()
    expected_tail = [sampler.next_indices() for _ in range(batch_count * 2 + 2)]
    restored = runner.StatefulBatchSamplerV2(sample_count, batch_size, seed)
    restored.load_state_dict(state)
    actual_tail = [restored.next_indices() for _ in range(batch_count * 2 + 2)]
    assert torch.equal(torch.cat((first, second)), expected_epochs[0][:10])
    for expected, actual in zip(expected_tail, actual_tail, strict=True):
        assert torch.equal(expected, actual)


def test_stateful_sampler_rejects_corrupt_state() -> None:
    sampler = runner.StatefulBatchSamplerV2(17, 5, 31)
    sampler.next_indices()
    valid = sampler.state_dict()
    corruptions = []

    wrong_schema = copy.deepcopy(valid)
    wrong_schema["schema_name"] = "wrong"
    corruptions.append(wrong_schema)

    short_permutation = copy.deepcopy(valid)
    short_permutation["permutation"] = short_permutation["permutation"][:-1]
    corruptions.append(short_permutation)

    out_of_range = copy.deepcopy(valid)
    out_of_range["permutation"][0] = 17
    corruptions.append(out_of_range)

    duplicate = copy.deepcopy(valid)
    duplicate["permutation"][0] = duplicate["permutation"][1]
    corruptions.append(duplicate)

    wrong_position = copy.deepcopy(valid)
    wrong_position["position"] = 1
    corruptions.append(wrong_position)

    wrong_cycle = copy.deepcopy(valid)
    wrong_cycle["cycle"] = -1
    corruptions.append(wrong_cycle)

    wrong_count = copy.deepcopy(valid)
    wrong_count["consumed_batch_count"] = 2
    corruptions.append(wrong_count)

    for state in corruptions:
        restored = runner.StatefulBatchSamplerV2(17, 5, 31)
        with pytest.raises((TypeError, ValueError)):
            restored.load_state_dict(state)


def test_y02_checkpoint_restore_preserves_model_optimizer_scheduler_and_sampler(
    tmp_path: Path,
) -> None:
    seed = runner.DEFAULT_MODEL_SEED
    device = torch.device("cpu")
    source_model = runner._build_model("Y01", 1.0, seed, device)
    target_model = runner._build_model("Y02", 1.0, seed, device)
    source_optimizer, source_scheduler = runner._build_optimizer_and_scheduler(
        source_model,
        runner.P0_CURRENT_X01_PROTOCOL,
    )
    target_optimizer, target_scheduler = runner._build_optimizer_and_scheduler(
        target_model,
        runner.P0_CURRENT_X01_PROTOCOL,
    )
    source_sampler = runner.StatefulBatchSamplerV2(1600, 100, runner.DEFAULT_SHUFFLE_SEED)
    target_sampler = runner.StatefulBatchSamplerV2(1600, 100, runner.DEFAULT_SHUFFLE_SEED)
    for _ in range(8000):
        source_sampler.next_indices()
    assert source_sampler.cycle == 500
    assert source_sampler.position == 1600
    assert source_sampler.consumed_batch_count() == 8000
    for parameter in source_model.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    source_optimizer.step()
    source_optimizer.zero_grad(set_to_none=True)
    source_scheduler.step(0.75)
    payload = runner._checkpoint_payload(
        experiment_id="Y01",
        model_seed=seed,
        global_step=8000,
        target_steps=8000,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        sampler=source_sampler,
        protocol=runner.P0_CURRENT_X01_PROTOCOL,
        data_records={"dataset_id": "pair_2k_current", "supervision": "pair_force", "split": {}},
        best_step=7984,
        best_validation_loss=0.25,
        train_loss=0.2,
        train_loss_source="final_full_train_evaluation",
        validation_loss=0.3,
        evaluation_index=500,
    )
    checkpoint = tmp_path / "Y01_final_checkpoint.pt"
    runner._save_checkpoint_atomic(checkpoint, payload)
    loaded = runner._load_checkpoint(
        checkpoint,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        sampler=target_sampler,
        expected_protocol=runner.P0_CURRENT_X01_PROTOCOL,
        expected_model_seed=seed,
    )
    assert loaded["experiment_id"] == "Y01"
    assert loaded["global_step"] == 8000
    _assert_nested_equal(source_model.state_dict(), target_model.state_dict(), "model")
    _assert_nested_equal(source_optimizer.state_dict(), target_optimizer.state_dict(), "optimizer")
    _assert_nested_equal(source_scheduler.state_dict(), target_scheduler.state_dict(), "scheduler")
    _assert_nested_equal(source_sampler.state_dict(), target_sampler.state_dict(), "sampler")
    continuation_data = runner.prepare_y_experiment_data_v2("Y01")
    continuation_tensor = runner._tensor_bucket(continuation_data.train, continuation_data)
    source_selection = source_sampler.next_indices()
    target_selection = target_sampler.next_indices()
    assert torch.equal(source_selection, target_selection)
    source_result = runner._train_step(
        source_model,
        continuation_tensor,
        source_selection,
        continuation_data.supervision,
        source_optimizer,
        device,
        8001,
    )
    target_result = runner._train_step(
        target_model,
        continuation_tensor,
        target_selection,
        continuation_data.supervision,
        target_optimizer,
        device,
        8001,
    )
    assert torch.equal(source_result.loss, target_result.loss)
    _assert_nested_equal(source_model.state_dict(), target_model.state_dict(), "continued_model")
    _assert_nested_equal(
        source_optimizer.state_dict(),
        target_optimizer.state_dict(),
        "continued_optimizer",
    )
    _assert_nested_equal(source_sampler.state_dict(), target_sampler.state_dict(), "continued_sampler")
    for _ in range(12):
        assert torch.equal(source_sampler.next_indices(), target_sampler.next_indices())
    mismatched = dict(payload)
    mismatched["global_step"] = 7999
    mismatch_checkpoint = tmp_path / "Y01_mismatch_checkpoint.pt"
    runner._save_checkpoint_atomic(mismatch_checkpoint, mismatched)
    with pytest.raises(ValueError, match="global step does not match sampler progress"):
        runner._load_checkpoint(
            mismatch_checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            sampler=target_sampler,
            expected_protocol=runner.P0_CURRENT_X01_PROTOCOL,
            expected_model_seed=seed,
        )


def test_checkpoint_data_match_covers_hashes_and_ignores_artifact_location() -> None:
    payload = {
        "data_records": {
            "dataset_id": "pair_2k_current",
            "supervision": "pair_force",
            "split_seed": 11,
            "pair_2k_current": {"csv_sha256": "abc"},
            "split": {"train_indices_sha256": "def"},
            "force_scale": 0.5,
            "split_artifact": {"path": "first", "sha256": "one"},
        }
    }
    current = {
        "dataset_id": "pair_2k_current",
        "supervision": "pair_force",
        "split_seed": 11,
        "pair_2k_current": {"csv_sha256": "abc"},
        "split": {"train_indices_sha256": "def"},
        "force_scale": 0.5,
        "split_artifact": {"path": "second", "sha256": "two"},
    }
    runner._checkpoint_matches_data(payload, current)
    current["pair_2k_current"] = {"csv_sha256": "changed"}
    with pytest.raises(ValueError, match="data provenance changed"):
        runner._checkpoint_matches_data(payload, current)


def _write_candidate_summary(
    root: Path,
    experiment_id: str,
    seed: int,
    protocol: runner.OptimizerProtocolV2,
    validation_loss: float,
) -> Path:
    path = root / experiment_id / f"seed_{seed}" / "summary.json"
    path.parent.mkdir(parents=True)
    architecture = {"core": "frozen", "layer_count": 1}
    path.write_text(
        json.dumps(
            {
                "schema_name": runner.RESULT_SCHEMA_NAME,
                "runner_variant": runner.RUNNER_VARIANT,
                "status": "complete",
                "experiment": {
                    "experiment_id": experiment_id,
                    "dataset": "pair_2k_current",
                    "optimizer_steps": 31500,
                },
                "model_seed": seed,
                "split_seed": 17,
                "shuffle_seed": 19,
                "training": {
                    "global_steps_completed": 31500,
                    "target_steps": 31500,
                    "resolved_protocol": asdict(protocol),
                    "final_validation_loss": validation_loss,
                },
                "data": {
                    "dataset_id": "pair_2k_current",
                    "supervision": "pair_force",
                    "split_seed": 17,
                    "force_scale": 0.5,
                    "pair_2k_current": {
                        "csv_sha256": "4" * 64,
                        "metadata_sha256": "5" * 64,
                        "validation_sha256": "6" * 64,
                    },
                    "split": {
                        "train_indices_sha256": "1" * 64,
                        "validation_indices_sha256": "2" * 64,
                        "test_indices_sha256": "3" * 64,
                    },
                },
                "architecture": architecture,
                "architecture_fingerprint_sha256": runner._json_sha256(architecture),
            }
        ),
        encoding="utf_8",
    )
    return path


def test_pstar_selection_is_explicit_hash_bound_and_loadable(tmp_path: Path) -> None:
    seed = 91
    config = runner.YRunnerConfigV2(
        output_root=tmp_path,
        device="cpu",
        comet_enabled=False,
    )
    _write_candidate_summary(
        tmp_path,
        "Y02",
        seed,
        runner.P0_CURRENT_X01_PROTOCOL,
        0.2,
    )
    y03_path = _write_candidate_summary(
        tmp_path,
        "Y03",
        seed,
        runner.P1_LEGACY_5K_V3_PROTOCOL,
        0.1,
    )
    record = runner.select_pstar_protocol_v2(config, seed)
    selection_path = tmp_path / f"pstar_seed_{seed}.json"
    assert selection_path.is_file()
    assert record["selected_protocol"] == "legacy_5k_v3_protocol"
    assert record["model_seed"] == seed
    assert record["common_evidence"] == {
        "model_seed": seed,
        "target_steps": 31500,
        "global_steps_completed": 31500,
        "dataset_id": "pair_2k_current",
        "supervision": "pair_force",
        "split_seed": 17,
        "shuffle_seed": 19,
        "split_hashes": {
            "train_indices_sha256": "1" * 64,
            "validation_indices_sha256": "2" * 64,
            "test_indices_sha256": "3" * 64,
        },
        "dataset_hashes": {
            "csv_sha256": "4" * 64,
            "metadata_sha256": "5" * 64,
            "validation_sha256": "6" * 64,
        },
        "force_scale": 0.5,
        "architecture_fingerprint_sha256": runner._json_sha256(
            {"core": "frozen", "layer_count": 1}
        ),
    }
    protocol, loaded = runner.load_pstar_protocol_v2(selection_path)
    assert protocol is runner.P1_LEGACY_5K_V3_PROTOCOL
    assert loaded == record
    resolved, resolved_record = runner.resolve_protocol_v2("Y12", config, seed)
    assert resolved is runner.P1_LEGACY_5K_V3_PROTOCOL
    assert resolved_record == record
    with pytest.raises(ValueError, match="model seed does not match"):
        runner.load_pstar_protocol_v2(
            selection_path,
            expected_model_seed=seed + 1,
        )
    y03_path.write_text("{}", encoding="utf_8")
    with pytest.raises(ValueError, match="changed after selection"):
        runner.load_pstar_protocol_v2(selection_path)


def test_pstar_rejects_candidate_common_fact_mismatch(tmp_path: Path) -> None:
    seed = 92
    config = runner.YRunnerConfigV2(
        output_root=tmp_path,
        device="cpu",
        comet_enabled=False,
    )
    _write_candidate_summary(
        tmp_path,
        "Y02",
        seed,
        runner.P0_CURRENT_X01_PROTOCOL,
        0.2,
    )
    y03_path = _write_candidate_summary(
        tmp_path,
        "Y03",
        seed,
        runner.P1_LEGACY_5K_V3_PROTOCOL,
        0.1,
    )
    y03 = json.loads(y03_path.read_text(encoding="utf_8"))
    y03["data"]["split"]["train_indices_sha256"] = "9" * 64
    y03_path.write_text(json.dumps(y03), encoding="utf_8")
    with pytest.raises(ValueError, match="common fact changed: split_hashes"):
        runner.select_pstar_protocol_v2(config, seed)


def test_finite_optimizer_smoke_covers_two_layer_ema_path() -> None:
    data = runner.prepare_y_experiment_data_v2("Y12")
    report = runner._finite_optimizer_smoke_v2("Y12", data, torch.device("cpu"))
    assert report["layer_count"] == 2
    assert tuple(report["ema_layers"]) == (1,)
    assert tuple(report["running_rms_policy_by_layer"]) == ("cumulative", "ema")
    assert report["forward_finite"] is True
    assert report["backward_finite"] is True
    assert report["optimizer_state_finite"] is True
    assert report["optimizer_step_minimum"] == 1.0
    assert report["optimizer_step_maximum"] == 1.0


def test_low_step_full_run_writes_complete_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = runner.prepare_y_experiment_data_v2("Y01")
    train_indices = np.arange(4, dtype=np.int64)
    validation_indices = np.arange(2, dtype=np.int64)
    test_indices = np.arange(2, dtype=np.int64)
    split = runner.SplitIndicesV1(train_indices, validation_indices, test_indices)
    records = copy.deepcopy(dict(full.records))
    records["split"] = {
        "train_count": 4,
        "validation_count": 2,
        "test_count": 2,
        "train_indices_sha256": runner._array_sha256(train_indices),
        "validation_indices_sha256": runner._array_sha256(validation_indices),
        "test_indices_sha256": runner._array_sha256(test_indices),
    }
    tiny_data = runner.PreparedYDataV2(
        experiment_id="Y01",
        dataset_id=full.dataset_id,
        supervision=full.supervision,
        train=full.train.subset(train_indices),
        validation=full.validation.subset(validation_indices),
        test=full.test.subset(test_indices),
        force_scale=full.force_scale,
        split=split,
        records=records,
    )
    original_get_spec = runner.get_y_experiment_spec_v2

    def tiny_spec(experiment_id: str) -> Any:
        spec = original_get_spec(experiment_id)
        if spec.experiment_id == "Y01":
            return replace(spec, optimizer_steps=2)
        return spec

    tiny_protocol = replace(
        runner.P0_CURRENT_X01_PROTOCOL,
        evaluation_cadence_steps=1,
    )
    monkeypatch.setattr(runner, "get_y_experiment_spec_v2", tiny_spec)
    monkeypatch.setattr(runner, "target_steps_v2", lambda experiment_id: 2)
    monkeypatch.setattr(runner, "P0_CURRENT_X01_PROTOCOL", tiny_protocol)
    monkeypatch.setattr(
        runner,
        "prepare_y_experiment_data_v2",
        lambda experiment_id, split_seed: tiny_data,
    )
    config = runner.YRunnerConfigV2(
        output_root=tmp_path,
        device="cpu",
        comet_enabled=False,
    )
    summary = runner.run_y_experiment_v2("Y01", 123, config, resume=False)
    output = tmp_path / "Y01" / "seed_123"
    expected_files = (
        "summary.json",
        "status.json",
        "history.csv",
        "comet.json",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "resume_checkpoint.pt",
        "split_indices.npz",
    )
    assert all((output / name).is_file() for name in expected_files)
    assert summary["status"] == "complete"
    assert summary["continuation"] is None
    assert summary["training"]["global_steps_completed"] == 2
    assert summary["training"]["target_steps"] == 2
    status = json.loads((output / "status.json").read_text(encoding="utf_8"))
    assert status["status"] == "complete"
    assert status["global_step"] == 2
    comet = json.loads((output / "comet.json").read_text(encoding="utf_8"))
    assert comet["enabled"] is False
    with (output / "history.csv").open(newline="", encoding="utf_8") as stream:
        history = list(csv.DictReader(stream))
    assert [int(row["global_step"]) for row in history] == [0, 1, 2]
    checkpoints = {
        name: torch.load(output / name, map_location="cpu", weights_only=False)
        for name in (
            "best_checkpoint.pt",
            "final_checkpoint.pt",
            "resume_checkpoint.pt",
        )
    }
    assert all(
        payload["schema_name"] == runner.CHECKPOINT_SCHEMA_NAME
        for payload in checkpoints.values()
    )
    assert checkpoints["final_checkpoint.pt"]["global_step"] == 2
    assert checkpoints["resume_checkpoint.pt"]["global_step"] == 2
    assert (
        checkpoints["resume_checkpoint.pt"]["sampler_state"]["consumed_batch_count"]
        == 2
    )
    assert history[-1]["train_loss_source"] == "protocol_epoch_optimization_loss"
    assert checkpoints["final_checkpoint.pt"]["train_loss_source"] == (
        "final_full_train_evaluation"
    )


def test_fast_v2_validation_scheduler_recording_best_and_full_history_cadences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = runner.prepare_y_experiment_data_v2("Y01")
    train_indices = np.arange(4, dtype=np.int64)
    validation_indices = np.arange(2, dtype=np.int64)
    test_indices = np.arange(2, dtype=np.int64)
    tiny_data = runner.PreparedYDataV2(
        experiment_id="Y01",
        dataset_id=full.dataset_id,
        supervision=full.supervision,
        train=full.train.subset(train_indices),
        validation=full.validation.subset(validation_indices),
        test=full.test.subset(test_indices),
        force_scale=full.force_scale,
        split=runner.SplitIndicesV1(train_indices, validation_indices, test_indices),
        records=copy.deepcopy(dict(full.records)),
    )

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def architecture_record(self) -> Mapping[str, Any]:
            return {"kind": "tiny_fast_v2", "layer_count": 1}

    scheduler_calls: list[float] = []

    class CountingScheduler:
        def step(self, value: float | None = None) -> None:
            scheduler_calls.append(float(value) if value is not None else float("nan"))

        def state_dict(self) -> dict[str, Any]:
            return {"call_count": len(scheduler_calls)}

        def load_state_dict(self, state: Mapping[str, Any]) -> None:
            del state

    train_steps: list[int] = []
    full_train_calls: list[int] = []
    validation_calls: list[int] = []

    def fake_train_step(
        model: Any,
        tensor_data: runner.TensorBucketV2,
        selection: Tensor,
        supervision: SupervisionV1,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        global_step: int,
    ) -> runner.TrainStepResultV2:
        del model, tensor_data, supervision, optimizer
        train_steps.append(global_step)
        return runner.TrainStepResultV2(
            loss=torch.tensor(float(global_step), device=device),
            component_count=int(selection.numel()) * 3,
            loss_finite=torch.tensor(True, device=device),
        )

    def fake_evaluate(
        model: Any,
        tensor_data: runner.TensorBucketV2,
        supervision: SupervisionV1,
        batch_size: int,
        device: torch.device,
    ) -> float:
        del model, supervision, batch_size, device
        if len(tensor_data.centers) == 4:
            full_train_calls.append(len(full_train_calls) + 1)
            return 50.0 + len(full_train_calls)
        validation_calls.append(len(validation_calls) + 1)
        return 100.0 - len(validation_calls)

    original_get_spec = runner.get_y_experiment_spec_v2

    def tiny_spec(experiment_id: str) -> Any:
        spec = original_get_spec(experiment_id)
        return replace(spec, optimizer_steps=12) if spec.experiment_id == "Y01" else spec

    protocol = replace(runner.P0_CURRENT_X01_PROTOCOL, evaluation_cadence_steps=1)
    save_calls: list[tuple[str, int, int]] = []
    original_save = runner._save_checkpoint_atomic

    def tracking_save(path: Path, payload: Mapping[str, Any]) -> None:
        save_calls.append(
            (
                path.name,
                int(payload["global_step"]),
                int(payload["evaluation_index"]),
            )
        )
        original_save(path, payload)

    backend = FakeCometBackend()
    monkeypatch.setenv("COMET_API_KEY", "fast_v2_test_credential")
    monkeypatch.setattr(runner, "get_y_experiment_spec_v2", tiny_spec)
    monkeypatch.setattr(runner, "target_steps_v2", lambda experiment_id: 12)
    monkeypatch.setattr(runner, "P0_CURRENT_X01_PROTOCOL", protocol)
    monkeypatch.setattr(
        runner,
        "prepare_y_experiment_data_v2",
        lambda experiment_id, split_seed: tiny_data,
    )
    monkeypatch.setattr(runner, "_build_model", lambda *args: TinyModel())
    monkeypatch.setattr(
        runner,
        "_build_optimizer_and_scheduler",
        lambda model, selected_protocol: (
            torch.optim.AdamW(model.parameters(), lr=0.001),
            CountingScheduler(),
        ),
    )
    monkeypatch.setattr(runner, "_train_step", fake_train_step)
    monkeypatch.setattr(runner, "_evaluate_loss", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "_physical_metrics",
        lambda *args: runner.PhysicalMetricsV2(0.1, 0.2, 0.3, 0.4, 6),
    )
    monkeypatch.setattr(runner, "_save_checkpoint_atomic", tracking_save)
    config = runner.YRunnerConfigV2(
        output_root=tmp_path,
        device="cpu",
        comet_enabled=True,
    )
    summary = runner.run_y_experiment_v2(
        "Y01",
        321,
        config,
        resume=False,
        comet_backend_factory=lambda **kwargs: backend,
    )
    output = tmp_path / "Y01" / "seed_321"
    with (output / "history.csv").open(newline="", encoding="utf_8") as stream:
        history = list(csv.DictReader(stream))
    assert train_steps == list(range(1, 13))
    assert len(validation_calls) == 13
    assert len(scheduler_calls) == 12
    assert full_train_calls == [1, 2]
    assert [int(row["evaluation_index"]) for row in history] == list(range(13))
    assert history[-1]["train_loss"] == "12.0"
    assert history[-1]["train_loss_source"] == "protocol_epoch_optimization_loss"
    evaluation_metric_calls = [
        call
        for call in backend.metric_calls
        if set(call[0]) == {
            "train_loss",
            "validation_loss",
            "epoch_duration_seconds",
        }
    ]
    assert [call[1]["step"] for call in evaluation_metric_calls] == [1, 10, 12]
    assert ("best_checkpoint.pt", 2, 2) in save_calls
    assert [
        step
        for name, step, index in save_calls
        if name == "resume_checkpoint.pt"
    ] == [0, 1, 10, 12, 12]
    final_checkpoint = torch.load(
        output / "final_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert final_checkpoint["train_loss"] == 52.0
    assert final_checkpoint["train_loss_source"] == "final_full_train_evaluation"
    assert summary["training"]["final_train_loss"] == 52.0
    assert summary["training"]["dataset_steps_per_pass"] == 1
    assert summary["training"]["protocol_epoch_steps"] == 1
    assert summary["training"]["initial_train_loss_source"] == (
        "initial_full_train_evaluation"
    )


def test_comet_identity_is_written_immediately_and_metric_registry_is_restricted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = "credential_for_test_only"
    monkeypatch.setenv("COMET_API_KEY", credential)
    backend = FakeCometBackend()
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeCometBackend:
        captured.update(kwargs)
        return backend

    config = runner.YRunnerConfigV2(
        output_root=tmp_path,
        device="cpu",
        comet_project="project_name",
        comet_enabled=True,
    )
    logger = runner._create_and_record_comet(tmp_path, config, "Y02", 71, factory)
    identity = json.loads((tmp_path / "comet.json").read_text(encoding="utf_8"))
    assert identity == {
        "enabled": True,
        "experiment_key": "experiment_key_123",
        "experiment_name": "Y02_seed_71_fast_v2",
        "project": "project_name",
        "url": "https://example.invalid/experiment",
        "workspace": None,
    }
    assert credential not in (tmp_path / "comet.json").read_text(encoding="utf_8")
    assert captured["api_key"] == credential
    assert "fast_v2" in captured["config"].tags
    logger.log_parameters(
        {
            "optimizer": {"name": "AdamW", "learning_rate": 0.002},
            "target_steps": 31500,
        }
    )
    logger.log_evaluation(
        global_step=8016,
        train_loss=0.3,
        validation_loss=0.4,
        epoch_duration_seconds=1.5,
    )
    logger.log_final(global_step=31500, final_test_mae=0.01, final_test_sae=2.0)
    assert backend.parameters == {
        "optimizer.name": "AdamW",
        "optimizer.learning_rate": 0.002,
        "target_steps": 31500,
    }
    assert backend.metric_calls[0] == (
        {
            "train_loss": 0.3,
            "validation_loss": 0.4,
            "epoch_duration_seconds": 1.5,
        },
        {"step": 8016},
    )
    assert backend.metric_calls[1] == (
        {"final_test_mae": 0.01, "final_test_sae": 2.0},
        {"step": 31500},
    )
    with pytest.raises(ValueError, match="sensitive parameter"):
        logger.log_parameters({"api_key": credential})
    logger.finish()
    assert backend.end_count == 1


def test_disabled_comet_writes_a_nonsecret_identity_without_calling_backend(
    tmp_path: Path,
) -> None:
    config = runner.YRunnerConfigV2(
        output_root=tmp_path,
        device="cpu",
        comet_enabled=False,
    )

    def forbidden_factory(**kwargs: Any) -> Any:
        raise AssertionError(kwargs)

    logger = runner._create_and_record_comet(tmp_path, config, "Y01", 1, forbidden_factory)
    assert logger.enabled is False
    assert json.loads((tmp_path / "comet.json").read_text(encoding="utf_8")) == {
        "enabled": False,
        "experiment_key": None,
        "experiment_name": None,
        "project": None,
        "url": None,
    }


def test_status_and_cli_expose_global_steps_resume_and_pstar() -> None:
    status = runner._status_record(
        status="running",
        experiment_id="Y02",
        model_seed=5,
        global_step=8016,
        target_steps=31500,
        protocol=runner.P0_CURRENT_X01_PROTOCOL,
    )
    assert status["global_step"] == 8016
    assert status["target_steps"] == 31500
    assert status["updated_at_utc"]
    parser = runner.build_argument_parser_v2()
    run = parser.parse_args(("run", "Y02", "--resume", "--device", "cpu", "--disable-comet"))
    assert run.experiment_id == "Y02"
    assert run.resume is True
    assert run.device == "cpu"
    assert run.disable_comet is True
    select = parser.parse_args(("select-pstar", "--seed", "9", "--overwrite"))
    assert select.seed == 9
    assert select.overwrite is True
    launcher = Path(runner.__file__).with_name("launch_e311_gnn_y12_v2.sh")
    launcher_text = launcher.read_text(encoding="utf_8")
    assert "/root/autodl-tmp/TFENN_y12_fast_v2_runs" in launcher_text
    assert 'SESSION_A="${Y_SESSION_A:-tfenn_y_a}"' in launcher_text
    assert 'SESSION_B="${Y_SESSION_B:-tfenn_y_b}"' in launcher_text
    assert "validate_complete_summary" in launcher_text
    assert "resume_arguments=(--resume)" in launcher_text
    assert "has artifacts but no complete summary or resume checkpoint" in launcher_text
    assert "select_or_validate_pstar" in launcher_text
    assert "load_pstar_protocol_v2" in launcher_text
    assert "preserve_comet_identity" in launcher_text
    assert 'resume_audit/${LAUNCH_ID}' in launcher_text
    assert "comet_before_resume.json" in launcher_text
    assert 'preflight_${LAUNCH_ID}.json' in launcher_text
    assert '> "${preflight_path}"' in launcher_text
    assert 'launcher_logs/preflight.json' not in launcher_text
