from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.benzene_pair import hyper_catalog
from experiments.benzene_pair.hyper_search import (
    CATALOG_HASH_KEYS,
    HISTORY_FIELDS,
    StudyConfig,
    TrialPaths,
    _canonical_sha256,
    _load_catalog,
    _load_json,
    _read_history,
    _restore_resume,
    _save_resume,
    _selected_designs,
    _write_history,
    _write_status,
)
from TFENN.data import BENZENE_CLUSTER_COLUMNS
from TFENN.models import PairPipelineConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "benzene_pair"
CATALOG_PATH = EXPERIMENT_ROOT / "catalog_v1.json"
STUDY_CONFIG_PATH = EXPERIMENT_ROOT / "hyper_config.json"


def catalog_sort_key(design: dict[str, object]) -> tuple[object, ...]:
    preflight = design["preflight"]
    assert isinstance(preflight, dict)
    primitive_b_ranks = preflight["primitive_b_ranks"]
    assert isinstance(primitive_b_ranks, list)
    return (
        preflight["parameter_count"],
        preflight["gate_count"],
        preflight["basis_dimension_sum"],
        preflight["stage_count"],
        len(primitive_b_ranks),
        preflight["maximum_lift_order"],
        preflight["active_mix_flag_count"],
        design["topology_code"],
        design["profile_code"],
        design["candidate_id"],
    )


def history_row(epoch: int, value: float) -> dict[str, object]:
    return dict.fromkeys(HISTORY_FIELDS, value) | {"epoch": epoch}


def test_catalog_has_one_hundred_unique_round_trippable_designs(
    tmp_path: Path,
) -> None:
    catalog, designs = _load_catalog(CATALOG_PATH)
    specifications = hyper_catalog.build_trial_specs()

    assert catalog["design_count"] == 100
    assert catalog["gate_mlp_hidden_width"] == 64
    assert len(designs) == len(specifications) == 100
    assert {design["candidate_id"] for design in designs} == {
        specification.candidate_id for specification in specifications
    }
    assert len({design["candidate_id"] for design in designs}) == 100
    assert len({design["config_hash"] for design in designs}) == 100
    assert len({design["functional_hash"] for design in designs}) == 100
    assert [design["trial_id"] for design in designs] == [
        f"trial_{index:03d}" for index in range(1, 101)
    ]

    for design in designs:
        pipeline_value = dict(design["pipeline"])
        pipeline_value["architecture_id"] = design["candidate_id"]
        pipeline = PairPipelineConfig.from_dict(pipeline_value)
        assert pipeline.as_dict() == pipeline_value
        assert all(stage.mlp.hidden_widths == (64,) for stage in pipeline.stages)
        config_value = {key: design[key] for key in CATALOG_HASH_KEYS}
        assert _canonical_sha256(config_value) == design["config_hash"]
        functional_pipeline = dict(design["pipeline"])
        functional_pipeline.pop("architecture_id")
        functional_value = {
            "pipeline": functional_pipeline,
            "learning_rate": design["learning_rate"],
            "weight_decay": design["weight_decay"],
            "batch_size": design["batch_size"],
            "scheduler_step_size": design["scheduler_step_size"],
            "scheduler_gamma": design["scheduler_gamma"],
        }
        assert (
            hyper_catalog._canonical_sha256(functional_value)
            == design["functional_hash"]
        )

    assert catalog["catalog_sha256"] == _canonical_sha256(list(designs))
    copy_path = tmp_path / "catalog.json"
    copy_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    copied_catalog, copied_designs = _load_catalog(copy_path)
    assert copied_catalog == catalog
    assert copied_designs == designs


def test_catalog_respects_budgets_and_is_ordered_by_complexity() -> None:
    catalog, designs = _load_catalog(CATALOG_PATH)
    budgets = catalog["budgets"]

    for design in designs:
        preflight = design["preflight"]
        pipeline = design["pipeline"]
        assert preflight["parameter_count"] <= budgets["maximum_parameter_count"]
        assert preflight["gate_count"] <= budgets["maximum_gate_count"]
        assert (
            preflight["maximum_invariant_channels"]
            <= budgets["maximum_invariant_channels"]
        )
        assert (
            preflight["maximum_gate_coefficient_count"]
            <= budgets["maximum_gate_coefficients"]
        )
        assert preflight["estimated_two_model_files_bytes_float32"] == (
            preflight["parameter_count"] * 8
        )
        assert all(
            stage["channels"] <= budgets["maximum_stage_channels"]
            and len(stage["inputs"]) <= budgets["maximum_stage_inputs"]
            and stage["mlp"]["hidden_widths"] == [64]
            for stage in pipeline["stages"]
        )

    keys = [catalog_sort_key(design) for design in designs]
    assert keys == sorted(keys)
    assert keys[0] < keys[-1]

    assert [item["trial_id"] for item in _selected_designs(designs, (), 2, 3)] == [
        "trial_002",
        "trial_003",
        "trial_004",
    ]
    assert [
        item["trial_id"]
        for item in _selected_designs(
            designs,
            ("trial_100", "trial_001"),
            1,
            None,
        )
    ] == ["trial_100", "trial_001"]
    with pytest.raises(ValueError, match="start_index"):
        _selected_designs(designs, (), 0, None)


def write_small_pair_dataset(path: Path) -> None:
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    rows = (
        (0, 0, 0.0, 0.0, 0.0, *identity, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0, 1, 8.0, 0.0, 0.0, *identity, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1, 0, 0.0, 0.0, 0.0, *identity, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (1, 1, 0.0, 8.0, 0.0, *identity, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    )
    with path.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(BENZENE_CLUSTER_COLUMNS)
        writer.writerows(rows)
    csv_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "schema_name": "tfenn_rigid_system",
        "schema_version": 2,
        "dataset": "benzene_pair",
        "dataset_revision": 2,
        "sample_count": 2,
        "molecule_count": 2,
        "rows_per_sample": 2,
        "row_count": 4,
        "columns": list(BENZENE_CLUSTER_COLUMNS),
        "sampling": {
            "seed": 17,
            "distance_range_A": [5.0, 10.0],
            "min_interatomic_distance_A": 3.0,
        },
        "opls": {
            "runtime_version": importlib.metadata.version("opls2020-static"),
        },
        "csv_sha256": csv_sha256,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf_8",
    )


def test_static_data_validator_reports_small_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "small_pair.csv"
    write_small_pair_dataset(csv_path)
    script = (
        "import json,sys; "
        "from experiments.benzene_pair.validate_data import validate_pair_dataset; "
        "report=validate_pair_dataset(sys.argv[1], expected_sample_count=2, "
        "recompute_count=0, numerical_tolerance=1e-10); "
        "print(json.dumps(report, allow_nan=False))"
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (sys.executable, "-c", script, str(csv_path)),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf_8",
    )
    report = json.loads(
        tuple(line for line in completed.stdout.splitlines() if line)[-1]
    )

    assert report["passed"] is True
    assert report["statistics"]["sample_count"] == 2
    assert report["statistics"]["unique_pose_count"] == 2
    assert report["recomputed"] == {
        "sample_count": 0,
        "maximum_force_error": 0.0,
        "maximum_moment_error": 0.0,
        "records": [],
    }
    assert report["checks"]
    assert all(report["checks"].values())


def test_study_config_status_history_and_resume_round_trip(tmp_path: Path) -> None:
    study = StudyConfig.from_path(STUDY_CONFIG_PATH)
    assert study.epochs == 1500
    assert study.resume_every == 25
    assert study.expected_sample_count == 5000
    assert study.expected_dataset_revision == 2
    assert study.protocol_dict()["split_fractions"] == [0.8, 0.1, 0.1]

    paths = TrialPaths.from_directory(tmp_path / "trial")
    rows = [history_row(0, 1.0), history_row(1, 0.5)]
    _write_history(paths.history, rows)
    assert _read_history(paths.history) == rows
    _write_status(
        paths,
        trial_id="trial_001",
        candidate_id="pair_hp_t01_p01",
        trial_hash="trial_hash",
        status="running",
        epoch=1,
        best_epoch=1,
        train_loss=0.5,
        validation_loss=0.4,
    )
    status = _load_json(paths.status)
    assert status["status"] == "running"
    assert status["epoch"] == 1
    assert status["best_epoch"] == 1

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    inputs = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
    model(inputs).square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    generator = torch.Generator().manual_seed(23)
    original_rng_state = torch.get_rng_state()
    try:
        _save_resume(
            paths.resume,
            model,
            optimizer,
            scheduler,
            generator,
            epoch=25,
            target_scale=torch.tensor(2.0),
            trial_hash="trial_hash",
            best_epoch=19,
            best_validation={"normalized_mse": 0.25},
            best_parameter_state=expected_parameters,
            initial_train_loss=1.0,
        )
        expected_random = torch.rand(4, generator=generator)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        generator.manual_seed(99)

        payload = _restore_resume(
            paths.resume,
            model,
            optimizer,
            scheduler,
            generator,
            trial_hash="trial_hash",
        )
        assert payload["epoch"] == 25
        assert payload["best_epoch"] == 19
        assert payload["fixed_tensor_artifacts_stored"] is False
        assert "model_state_dict" not in payload
        assert set(payload["best_parameter_state_dict"]) == set(expected_parameters)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, expected_parameters[name])
        torch.testing.assert_close(torch.rand(4, generator=generator), expected_random)
    finally:
        torch.set_rng_state(original_rng_state)
