from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.gnn.e311_y13_y15_pair_control_core_v1 import (
    complete_pair_index_v1,
)
from experiments.gnn.e311_y13_y15_pair_control_runner_v1 import (
    DEFAULT_COMET_PROJECT,
    _StrictYSeriesCometAdapter,
    _deterministic_group_split,
    _formal_comet_config,
    _numpy_signed_scatter,
    _selected_y15_metrics,
    _update_strict_comet_epoch_record,
    build_argument_parser_v1,
)


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


def test_pair_target_contract_reaggregates_with_zero_total_force() -> None:
    pair_index = complete_pair_index_v1(5).cpu().numpy()
    pair_force = np.arange(60, dtype=np.float64).reshape(2, 10, 3)
    node_force = _numpy_signed_scatter(pair_force, pair_index, 5)
    np.testing.assert_array_equal(node_force.sum(axis=1), np.zeros((2, 3)))


def test_cli_has_three_separate_experiment_commands(tmp_path) -> None:
    parser = build_argument_parser_v1()
    y13 = parser.parse_args(
        ("y13", "--study-root", str(tmp_path / "e"), "--device", "cpu")
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
            "--pair-npz",
            str(tmp_path / "five_pair.npz"),
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


def test_y15_pair_sae_matches_mae_times_component_count() -> None:
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
                normalized_pair_force_world=torch.zeros(pair_shape),
                normalized_node_force_world=torch.zeros_like(centers),
                raw_forward_world=torch.zeros(pair_shape),
                raw_reverse_world=torch.zeros(pair_shape),
            )

    sample_count = 2
    centers = torch.zeros((sample_count, 5, 3))
    frames = torch.eye(3).expand(sample_count, 5, 3, 3).clone()
    pair_target = torch.ones((sample_count, 10, 3))
    node_target = torch.zeros((sample_count, 5, 3))
    loader = DataLoader(
        TensorDataset(centers, frames, pair_target, node_target),
        batch_size=sample_count,
    )
    metrics = _selected_y15_metrics(
        ZeroModel(),
        loader,
        complete_pair_index_v1(5),
        torch.device("cpu"),
        target_scale=2.0,
    )
    pair = metrics["pair_force"]
    assert pair["component_count"] == 60
    assert pair["mae"] == 2.0
    assert pair["sae"] == pair["mae"] * pair["component_count"]
