from __future__ import annotations

import json
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.gnn.e311_y13_y15_pair_control_core_v1 import (
    complete_pair_index_v1,
)
from experiments.gnn.e311_y13_y15_pair_control_runner_v1 import (
    DEFAULT_COMET_PROJECT,
    Y15NodeArraysV2,
    _StrictYSeriesCometAdapter,
    _deterministic_group_split,
    _evaluate_y15_node_force_loss_v2,
    _formal_comet_config,
    _resolve_training_protocol,
    _resolved_experiment_name,
    _restore_y15_node_force_checkpoint_v2,
    _save_y15_node_force_checkpoint_v2,
    _selected_y15_node_force_metrics_v2,
    _update_strict_comet_epoch_record,
    build_argument_parser_v1,
)


def test_y15_node_arrays_have_no_pair_force_label() -> None:
    names = {field.name for field in fields(Y15NodeArraysV2)}
    assert "node_force_world" in names
    assert "pair_force_world" not in names


def test_y15_split_occurs_at_configuration_level() -> None:
    group_id = np.repeat(np.arange(100, dtype=np.int64), 3)
    split = _deterministic_group_split(group_id, 20260821)
    assert split.counts() == {"train": 240, "validation": 30, "test": 30}
    train_groups = set(group_id[split.train].tolist())
    validation_groups = set(group_id[split.validation].tolist())
    test_groups = set(group_id[split.test].tolist())
    assert not train_groups & validation_groups
    assert not train_groups & test_groups
    assert not validation_groups & test_groups


def test_cli_has_three_separate_experiment_commands(tmp_path) -> None:
    parser = build_argument_parser_v1()
    y13 = parser.parse_args(
        (
            "y13",
            "--study-root",
            str(tmp_path / "e"),
            "--output-directory",
            str(tmp_path / "y13_repeat"),
            "--epochs",
            "1000",
            "--batch-size",
            "512",
            "--device",
            "cpu",
        )
    )
    y14 = parser.parse_args(
        (
            "y14",
            "--e-study-root",
            str(tmp_path / "e"),
            "--output-directory",
            str(tmp_path / "y14"),
            "--device",
            "cpu",
        )
    )
    y15 = parser.parse_args(
        (
            "y15",
            "--csv",
            str(tmp_path / "five.csv"),
            "--output-directory",
            str(tmp_path / "y15"),
            "--device",
            "cpu",
        )
    )
    assert (y13.command, y14.command, y15.command) == ("y13", "y14", "y15")
    assert y13.comet_project == DEFAULT_COMET_PROJECT
    assert y14.comet_project == DEFAULT_COMET_PROJECT
    assert y15.comet_project == DEFAULT_COMET_PROJECT
    assert y13.output_directory == tmp_path / "y13_repeat"
    assert y13.epochs == 1000
    assert y13.batch_size == 512
    assert y14.epochs is None
    assert y15.batch_size is None
    assert not hasattr(y15, "pair_npz")


def test_repeat_protocol_changes_only_epochs_and_batch_size() -> None:
    protocols = (("Y13", 125), ("Y14", 125), ("Y15", 50))
    for experiment_id, scheduler_step_size in protocols:
        spec, epochs, batch_size = _resolve_training_protocol(
            experiment_id,
            1000,
            512,
        )
        assert epochs == 1000
        assert batch_size == 512
        assert spec.scheduler_step_size == scheduler_step_size
        assert _resolved_experiment_name(
            f"{experiment_id}_base",
            spec,
            epochs,
            batch_size,
        ) == f"{experiment_id}_base_BS512_E1000"


def test_formal_comet_config_targets_y_series_without_checkpoints() -> None:
    config = _formal_comet_config(DEFAULT_COMET_PROJECT, None, "Y15")
    assert config.enabled
    assert config.required_online
    assert config.project_name == "tfenn_e311_gnn_y12_diagnostic_v2"
    assert not config.upload_checkpoints
    assert config.tags[-1] == "Y15"


def test_strict_comet_adapter_forwards_only_locked_metrics_and_parameters() -> None:
    class FakeLogger:
        identity = {
            "enabled": True,
            "project": DEFAULT_COMET_PROJECT,
            "experiment_key": "abc123",
        }

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def log_parameters(self, parameters: object) -> None:
            self.calls.append(("parameters", parameters))

        def log_evaluation(self, **values: object) -> None:
            self.calls.append(("evaluation", values))

        def log_final(self, **values: object) -> None:
            self.calls.append(("final", values))

        def finish(self) -> None:
            self.calls.append(("finish", None))

    fake = FakeLogger()
    adapter = _StrictYSeriesCometAdapter(fake)
    adapter.log_config(
        study_config={"series": "Y"},
        trial_config={"experiment_id": "Y15"},
        parameters={"epochs": 200},
    )
    adapter.log_epoch(
        epoch=7,
        train_loss=0.3,
        validation_loss=0.2,
        learning_rate=0.003,
        extra_metrics={
            "epoch_duration_seconds": 4.5,
            "must_not_be_forwarded": 99.0,
        },
    )
    adapter.log_final(
        metrics={"test": {"mae": 0.1, "sae": 30.0, "rmse": 0.2}},
        relative_force_norm_stats={"must_not_be_forwarded": 1.0},
        summary={"must_not_be_forwarded": True},
    )
    adapter.log_asset("history.csv")
    adapter.log_checkpoint_reference("best", "best.pt", sha256="0" * 64)
    adapter.finish("complete")

    assert [name for name, _value in fake.calls] == [
        "parameters",
        "evaluation",
        "final",
        "finish",
    ]
    assert fake.calls[1][1] == {
        "global_step": 7,
        "train_loss": 0.3,
        "validation_loss": 0.2,
        "epoch_duration_seconds": 4.5,
    }
    assert fake.calls[2][1] == {
        "global_step": 7,
        "final_test_mae": 0.1,
        "final_test_sae": 30.0,
    }


def test_strict_comet_epoch_record_advances_atomically(tmp_path) -> None:
    path = tmp_path / "comet.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "tfenn_sweep31_comet_trial",
                "last_logged_epoch": -1,
            }
        ),
        encoding="utf_8",
    )
    _update_strict_comet_epoch_record(path, 1)
    assert json.loads(path.read_text(encoding="utf_8"))["last_logged_epoch"] == 1


def test_y15_validation_uses_only_molecular_force_output() -> None:
    class ZeroModel:
        training = True

        def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
            del args, kwargs
            raise AssertionError("Y15 validation must not call pair output forward")

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

        def core_output(
            self,
            centers: torch.Tensor,
            frames: torch.Tensor,
            pair_index: torch.Tensor,
        ) -> SimpleNamespace:
            del frames
            pair_shape = (centers.shape[0], pair_index.numel() // 2, 3)
            return SimpleNamespace(
                normalized_pair_force_world=torch.full(pair_shape, torch.nan),
                normalized_node_force_world=torch.zeros_like(centers),
                raw_forward_world=torch.full(pair_shape, torch.nan),
                raw_reverse_world=torch.full(pair_shape, torch.nan),
            )

    sample_count = 2
    centers = torch.zeros((sample_count, 5, 3))
    frames = torch.eye(3).expand(sample_count, 5, 3, 3).clone()
    node_target = torch.ones((sample_count, 5, 3))
    loader = DataLoader(
        TensorDataset(centers, frames, node_target),
        batch_size=sample_count,
    )
    loss = _evaluate_y15_node_force_loss_v2(
        ZeroModel(),
        loader,
        complete_pair_index_v1(5),
        torch.device("cpu"),
    )
    assert loss == 1.0


def test_y15_molecular_force_sae_matches_mae_times_component_count() -> None:
    class ZeroModel:
        training = True

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool = True) -> None:
            self.training = mode

        def core_output(
            self,
            centers: torch.Tensor,
            frames: torch.Tensor,
            pair_index: torch.Tensor,
        ) -> SimpleNamespace:
            del frames
            pair_shape = (centers.shape[0], pair_index.numel() // 2, 3)
            return SimpleNamespace(
                normalized_pair_force_world=torch.full(pair_shape, torch.nan),
                normalized_node_force_world=torch.zeros_like(centers),
                raw_forward_world=torch.full(pair_shape, torch.nan),
                raw_reverse_world=torch.full(pair_shape, torch.nan),
            )

    sample_count = 2
    centers = torch.zeros((sample_count, 5, 3))
    frames = torch.eye(3).expand(sample_count, 5, 3, 3).clone()
    node_target = torch.ones((sample_count, 5, 3))
    loader = DataLoader(
        TensorDataset(centers, frames, node_target),
        batch_size=sample_count,
    )
    metrics = _selected_y15_node_force_metrics_v2(
        ZeroModel(),
        loader,
        complete_pair_index_v1(5),
        torch.device("cpu"),
        target_scale=2.0,
    )
    molecular = metrics["molecular_force"]
    assert molecular["component_count"] == 30
    assert molecular["mae"] == 2.0
    assert molecular["sae"] == molecular["mae"] * molecular["component_count"]
    assert "pair_force" not in metrics


def test_y15_checkpoint_schema_locks_node_force_selection(tmp_path) -> None:
    path = tmp_path / "best.pt"
    model = torch.nn.Linear(2, 1)
    _save_y15_node_force_checkpoint_v2(
        path,
        model,
        epoch=7,
        validation_loss=0.125,
        target_scale=0.75,
        source_sha256="1" * 64,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_name"] == "tfenn_y15_node_force_selected_checkpoint"
    assert payload["schema_version"] == 2
    assert payload["validation_normalized_node_force_mse"] == 0.125
    assert payload["node_force_target_scale_component_rms"] == 0.75
    assert "validation_normalized_mse" not in payload


def test_y15_restore_rejects_legacy_pair_checkpoint(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema_name": "tfenn_y15_selected_checkpoint",
            "schema_version": 1,
            "source_sha256": "1" * 64,
        },
        path,
    )
    with pytest.raises(RuntimeError, match="node force supervision"):
        _restore_y15_node_force_checkpoint_v2(
            path,
            torch.nn.Linear(2, 1),
            source_sha256="1" * 64,
        )
