from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.gnn.e311_one_block_gnn import (
    E311OneBlockGNN,
    E311OneBlockGNNConfig,
)
from experiments.gnn.evaluate_two_benzene_rotations import (
    DEFAULT_DATA,
    DEFAULT_SPLIT_SEED,
    _frobenius_residual,
    _resolve_model_family,
    _sha256,
    _split_indices,
    evaluate,
)


def test_frobenius_residual_reports_absolute_relative_and_percent() -> None:
    expected = np.asarray(((3.0, 4.0, 0.0),), dtype=np.float64)
    actual = np.asarray(((3.0, 0.0, 0.0),), dtype=np.float64)
    residual = _frobenius_residual(actual, expected)
    assert residual["absolute_frobenius_residual"] == 4.0
    assert residual["reference_frobenius_norm"] == 5.0
    assert residual["relative_frobenius_residual"] == 0.8
    assert residual["relative_frobenius_residual_percent"] == 80.0
    assert residual["maximum_component_absolute_residual"] == 4.0


def test_e311_evaluation_loads_family_and_freezes_running_rms(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "e311_two_benzene_run"
    run_path.mkdir()
    train, validation, test = _split_indices(2000, DEFAULT_SPLIT_SEED)
    np.savez(
        run_path / "split_indices.npz",
        train_sample_id=train,
        validation_sample_id=validation,
        test_sample_id=test,
    )

    torch.manual_seed(20260826)
    config = E311OneBlockGNNConfig()
    model = E311OneBlockGNN(1.0, config, dtype=torch.float32)
    provenance = {"specified_design_source": {"verified": True}}
    checkpoint = {
        "epoch": 300,
        "purpose": "final_epoch",
        "model_family": "e311_multibody_one_block_v1",
        "model_state": model.state_dict(),
        "force_scale": 1.0,
        "model_config": asdict(config),
        "dtype": "float32",
        "seed": DEFAULT_SPLIT_SEED,
        "data_sha256": _sha256(DEFAULT_DATA),
        "source_provenance": provenance,
    }
    torch.save(checkpoint, run_path / "final_checkpoint.pt")

    result = evaluate(
        run_path,
        DEFAULT_DATA,
        recompute_opls=False,
    )

    assert result["passed"] is True
    assert result["checkpoint"]["model_family"] == (
        "e311_multibody_one_block_v1"
    )
    assert result["checkpoint"]["source_provenance"] == provenance
    assert result["sample"]["sample_id"] == int(test[0]) == 425
    assert result["sample"]["selection"] == "fixed_test_split_first_sample"
    assert result["sample"]["is_in_fixed_test_split"] is True
    assert result["checkpoint"]["name"] == "final_checkpoint.pt"
    assert result["checks"]["angle_60_d6_world_force_invariance"] is True
    assert result["model"]["angle_45"]["relative_output_change"] > 0.0
    assert result["model"]["angle_45"][
        "naive_partial_covariance_residual"
    ] > 0.0
    assert result["model"]["running_rms_eval_state"]["checked"] is True
    assert result["model"]["running_rms_eval_state"]["unchanged"] is True
    assert result["model"]["running_rms_eval_state"]["buffer_count"] == 424
    assert result["opls_ground_truth"]["status"] == "skipped"

    report_path = run_path / "rotation_evaluation.json"
    stored = json.loads(report_path.read_text(encoding="utf_8"))
    assert stored["sample"]["sample_id"] == 425
    assert stored["model"]["angle_60"][
        "relative_frobenius_residual"
    ] == result["model"]["angle_60"]["relative_frobenius_residual"]


def test_rotation_evaluation_rejects_non_e311_model_family() -> None:
    with pytest.raises(ValueError, match="must identify the E311 model"):
        _resolve_model_family({})
    with pytest.raises(ValueError, match="unsupported checkpoint model_family"):
        _resolve_model_family({"model_family": "bidirectional_one_block_gnn"})
