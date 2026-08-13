"""Validate the matched network level group convolution experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from experiments.benzene_pair.group_conv_baseline import (
    COMPONENT_LABELS,
    DEFAULT_CONFIG_PATH,
    HISTORY_FIELDS,
    BaselineConfig,
    PairTrainingData,
    _build_model,
    _component_errors,
    load_trained_group_conv,
    run_baseline,
)


class StoredPredictionModel(nn.Module):
    def forward(self, centers: Tensor, _frames: Tensor) -> Tensor:
        return centers[:, 0]


def test_formal_config_matches_trial_029_training_protocol() -> None:
    config = BaselineConfig.from_path(DEFAULT_CONFIG_PATH)
    assert config.hidden_widths == (96, 96, 96)
    assert config.expected_parameter_count == 20160
    assert config.epochs == 1500
    assert config.split_fractions == (0.8, 0.1, 0.1)
    assert (config.split_seed, config.model_seed, config.shuffle_seed) == (
        20260813,
        20260814,
        20260815,
    )
    assert config.optimizer == "adamw"
    assert config.learning_rate == 0.002
    assert config.weight_decay == 0.00001
    assert config.batch_size == 128
    assert config.scheduler_step_size == 400
    assert config.scheduler_gamma == 0.5
    assert config.expected_sample_count == 5000
    assert config.expected_dataset_revision == 2
    assert config.expected_opls_version == "2.0.0"

    model = _build_model(config, device="cpu", dtype="float64")
    assert sum(parameter.numel() for parameter in model.parameters()) == 20160
    centers = torch.zeros((2, 2, 3), dtype=torch.float64)
    centers[:, 1, 0] = 5.0
    frames = torch.eye(3, dtype=torch.float64).expand(2, 2, 3, 3).clone()
    torch.testing.assert_close(
        model(centers, frames),
        torch.zeros((2, 3), dtype=torch.float64),
    )


def test_component_percentage_error_definitions(tmp_path: Path) -> None:
    target = torch.tensor(
        ((1.0, 2.0, 4.0), (2.0, 4.0, 8.0)),
        dtype=torch.float64,
    )
    prediction = target * torch.tensor((1.1, 1.2, 1.2), dtype=torch.float64)
    centers = torch.zeros((2, 2, 3), dtype=torch.float64)
    centers[:, 0] = prediction
    frames = torch.eye(3, dtype=torch.float64).expand(2, 2, 3, 3).clone()
    data = PairTrainingData(
        centers=centers,
        frames=frames,
        root_force=target * 2.0,
        metadata={},
        csv_path=tmp_path / "data.csv",
        metadata_path=tmp_path / "data.json",
        csv_sha256="0" * 64,
        metadata_sha256="1" * 64,
    )
    metrics = _component_errors(
        StoredPredictionModel(),
        data,
        torch.tensor((0, 1)),
        2,
        torch.tensor(2.0, dtype=torch.float64),
        zero_threshold=0.0,
    )

    assert metrics["component_labels"] == list(COMPONENT_LABELS)
    assert metrics["component_relative_rmse_percent"] == pytest.approx(
        (10.0, 20.0, 20.0)
    )
    assert metrics["component_mape_percent"] == pytest.approx(
        (10.0, 20.0, 20.0)
    )
    assert metrics["component_median_ape_percent"] == pytest.approx(
        (10.0, 20.0, 20.0)
    )
    assert metrics["mape_valid_count"] == [2, 2, 2]
    assert metrics["mape_excluded_count"] == [0, 0, 0]


def test_one_epoch_training_smoke_writes_complete_comparison_record(
    tmp_path: Path,
) -> None:
    config = BaselineConfig.from_path(DEFAULT_CONFIG_PATH)
    summary = run_baseline(
        config,
        epochs_override=1,
        sample_limit=30,
        output_directory=tmp_path,
        device_override="cpu",
    )

    assert summary["status"] == "complete"
    assert summary["model"]["parameter_count"] == 20160
    assert summary["model"]["trainable_parameter_count"] == 20160
    assert summary["model"]["reynolds_action_count"] == 144
    assert summary["comparison_reference"] == {
        "trial_id": "trial_029",
        "candidate_id": "pair_hp_t06_p03",
        "matched_training_protocol": True,
    }
    assert summary["history"]["row_count"] == 2
    assert summary["selection"]["criterion"] == (
        "minimum validation normalized MSE"
    )
    assert summary["selection"]["symmetry"]["passed"] is True
    assert summary["checkpoints"]["fixed_tensor_artifacts_stored"] is False

    for partition in ("train", "validation", "test"):
        overall = summary["selection"]["selected_metrics"][partition]
        assert math.isfinite(overall["normalized_mse"])
        assert overall["relative_rmse_percent"] == pytest.approx(
            overall["relative_rmse"] * 100.0
        )
        component = summary["selection"]["percentage_errors"][partition]
        assert component["component_labels"] == list(COMPONENT_LABELS)
        for name in (
            "component_relative_rmse_percent",
            "component_mape_percent",
            "component_median_ape_percent",
        ):
            assert len(component[name]) == 3
            assert all(math.isfinite(value) for value in component[name])

    with (tmp_path / "history.csv").open(newline="", encoding="utf_8") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == HISTORY_FIELDS
    assert [int(row["epoch"]) for row in rows] == [0, 1]

    model, target_scale, checkpoint = load_trained_group_conv(
        tmp_path / "best.pt",
        device="cpu",
        dtype="float64",
    )
    assert target_scale > 0.0
    assert checkpoint["fixed_tensor_artifacts_stored"] is False
    assert sum(parameter.numel() for parameter in model.parameters()) == 20160
