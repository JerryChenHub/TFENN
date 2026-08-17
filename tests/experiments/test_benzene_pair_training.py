from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.benzene_pair.train import (
    HISTORY_FIELDS,
    TrainingConfig,
    _build_model,
    _save_checkpoint,
    append_history_row,
    build_parser,
    create_split,
    make_history_row,
    load_trained_model,
    regression_metrics,
    write_history_csv,
    write_summary_json,
)
from TFENN.models import InvariantGatePipelineV2Config, InvariantGateStageV2Config


def test_create_split_is_deterministic_disjoint_and_complete() -> None:
    first = create_split(1000, seed=17)
    second = create_split(1000, seed=17)
    assert first.train.numel() == 800
    assert first.validation.numel() == 100
    assert first.test.numel() == 100
    assert torch.equal(first.train, second.train)
    assert torch.equal(first.validation, second.validation)
    assert torch.equal(first.test, second.test)

    combined = torch.cat((first.train, first.validation, first.test))
    assert torch.unique(combined).numel() == 1000
    torch.testing.assert_close(torch.sort(combined).values, torch.arange(1000))

    proportional = create_split(20, seed=19, fractions=(0.5, 0.25, 0.25))
    assert tuple(
        partition.numel()
        for partition in (
            proportional.train,
            proportional.validation,
            proportional.test,
        )
    ) == (10, 5, 5)


def test_default_training_config_round_trip() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "benzene_pair"
        / "config_v2.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf_8"))
    config = TrainingConfig.from_dict(payload)
    assert config.epochs == 500
    assert config.csv_path.name == "benzene_pair_opls_2_0_0_v3.csv"
    assert config.dataset_revision == 3
    assert config.pipeline_version == "v2"
    assert isinstance(config.pipeline, InvariantGatePipelineV2Config)
    assert tuple(stage.name for stage in config.pipeline.stages) == (
        "a1",
        "a2",
        "b2",
        "a3",
    )
    assert config.pipeline.stages[2].source_names == ("x", "r", "a1", "a2")
    assert config.pipeline.stages[0].trunk_width == 32
    serialized = config.as_json()
    assert serialized["pipeline_version"] == "v2"
    assert serialized["dataset_revision"] == 3
    assert build_parser().parse_args([]).config.name == "config_v2.json"
    assert build_parser().parse_args([]).device is None
    assert build_parser().parse_args(("--device", "cuda")).device == "cuda"


def test_regression_metrics_report_physical_and_relative_accuracy() -> None:
    target = torch.tensor(((0.0, 1.0), (2.0, 3.0)), dtype=torch.float64)
    prediction = target + torch.tensor(
        ((1.0, -1.0), (0.0, 2.0)),
        dtype=torch.float64,
    )
    metrics = regression_metrics(prediction, target, target_scale=2.0)

    assert metrics["normalized_mse"] == pytest.approx(1.5)
    assert metrics["mse"] == pytest.approx(6.0)
    assert metrics["rmse"] == pytest.approx(math.sqrt(6.0))
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["relative_rmse"] == pytest.approx(math.sqrt(1.5 / 3.5))
    assert metrics["r2"] == pytest.approx(-0.5)

    perfect = regression_metrics(target, target, target_scale=2.0)
    assert perfect["normalized_mse"] == 0.0
    assert perfect["relative_rmse"] == 0.0
    assert perfect["r2"] == 1.0


def test_history_and_summary_round_trip(tmp_path) -> None:
    first_metrics = {
        "normalized_mse": 1.0,
        "mse": 4.0,
        "rmse": 2.0,
        "mae": 1.5,
        "relative_rmse": 1.0,
        "r2": 0.0,
    }
    second_metrics = {
        "normalized_mse": 0.25,
        "mse": 1.0,
        "rmse": 1.0,
        "mae": 0.75,
        "relative_rmse": 0.5,
        "r2": 0.75,
    }
    rows = (
        make_history_row(0, 0.005, first_metrics, first_metrics),
        make_history_row(1, 0.005, second_metrics, second_metrics),
    )
    history_path = write_history_csv(tmp_path / "history.csv", rows)
    append_history_row(
        history_path,
        make_history_row(2, 0.0025, second_metrics, second_metrics),
    )
    with history_path.open(newline="", encoding="utf_8") as stream:
        loaded_rows = list(csv.DictReader(stream))
    assert tuple(loaded_rows[0]) == HISTORY_FIELDS
    assert [int(row["epoch"]) for row in loaded_rows] == [0, 1, 2]
    assert float(loaded_rows[1]["validation_r2"]) == pytest.approx(0.75)

    summary = {
        "schema_name": "tfenn_benzene_pair_training_run",
        "schema_version": 1,
        "status": "complete",
        "target": {
            "definition": "force on molecule_id 0 in the root coordinate frame",
            "uses_moment": False,
        },
        "history": {"row_count": 2},
    }
    summary_path = write_summary_json(tmp_path / "summary.json", summary)
    loaded_summary = json.loads(summary_path.read_text(encoding="utf_8"))
    assert loaded_summary == summary


def test_checkpoint_stores_only_learned_parameters(tmp_path) -> None:
    pipeline = InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config(
                "readout",
                "A",
                ("x", "r"),
                1,
                trunk_width=8,
                include_symmetric_unary=False,
                include_raw_mixed_pairs=True,
                include_stf_shortcuts=True,
            ),
        ),
        output_stage="readout",
        anchor_ranks=(1, 2),
        max_constraint_entries=2_000_000,
        max_gate_coefficients=100_000,
        max_invariant_channels=10_000,
    )
    config = TrainingConfig(
        pipeline=pipeline,
        dtype="float64",
        zero_output_heads=False,
    )
    model, _count = _build_model(config)
    centers = torch.tensor(
        (((0.0, 0.0, 0.0), (6.0, 1.0, -0.5)),),
        dtype=torch.float64,
    )
    frames = torch.eye(3, dtype=torch.float64).expand(1, 2, 3, 3).clone()
    expected = model(centers, frames)
    expected_normalization = model.normalization_state_dict()
    checkpoint = _save_checkpoint(
        tmp_path / "model.pt",
        model,
        epoch=0,
        target_scale=torch.tensor(2.0),
        config=config,
        metrics={"normalized_mse": 1.0},
    )
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["fixed_tensor_artifacts_stored"] is False
    assert "model_state_dict" not in payload
    assert set(payload["parameter_state_dict"]) == {
        name for name, _parameter in model.named_parameters()
    }
    assert set(payload["normalization_state_dict"]) == set(expected_normalization)
    for name, value in expected_normalization.items():
        torch.testing.assert_close(payload["normalization_state_dict"][name], value)
    restored, target_scale, _payload = load_trained_model(
        checkpoint,
        dtype="float64",
    )
    restored.eval()
    model.eval()
    torch.testing.assert_close(restored(centers, frames), expected)
    restored_normalization = restored.normalization_state_dict()
    for name, value in expected_normalization.items():
        torch.testing.assert_close(restored_normalization[name], value)
    assert target_scale == 2.0
