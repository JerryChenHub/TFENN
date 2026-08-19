from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.benzene_pair.comet_logging import CometConfig, NullCometTrialLogger
from experiments.benzene_pair.sweep30 import (
    GROUP_CONV_SPEC,
    STUDY_SPECS,
    SweepConfig,
    SplitIndices,
    TrainingData,
    TrialPaths,
    _build_model,
    _checkpoint_payload,
    _log_comet_epoch,
    _restore_model_state,
    create_group_aware_split,
    get_study_spec,
    run_trial,
)
from experiments.benzene_pair.invariant_gate_v2_20k_sweep import get_model_spec


def test_formal_config_fixes_common_protocol() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "benzene_pair"
        / "sweep30_config.json"
    )
    config = SweepConfig.from_path(path)
    assert config.epochs == 500
    assert config.effective_batch_size == 10000
    assert config.micro_batch_size == 10000
    assert config.scheduler_step_size == 125
    assert config.scheduler_gamma == 0.5
    assert config.expected_sample_count == 400000
    assert config.enable_tf32 is True
    assert config.validation_every == 1
    assert config.relative_force_norm_sample_count == 10000
    assert config.comet.enabled is True
    assert config.comet.required_online is True
    assert config.comet.project_name == "tfenn_pair_benzene_model_comparison_400k"
    assert len(config.shard_paths) == 4
    assert json.loads(path.read_text(encoding="utf_8"))["device"] == "auto"


def test_group_aware_split_keeps_exact_duplicates_together() -> None:
    centers = torch.arange(72, dtype=torch.float32).reshape(12, 2, 3)
    frames = torch.eye(3).expand(12, 2, 3, 3).clone()
    centers[10] = centers[2]
    centers[11] = centers[2]
    split, report = create_group_aware_split(
        centers,
        frames,
        seed=19,
        fractions=(0.5, 0.25, 0.25),
    )
    membership = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for index in indices.tolist():
            membership[index] = name
    assert membership[2] == membership[10] == membership[11]
    assert sum(split.counts().values()) == 12
    assert report["duplicate_group_count"] == 1
    assert report["duplicate_extra_sample_count"] == 2
    assert report["duplicate_groups_cross_partitions"] == 0


def test_checkpoint_keeps_learned_and_normalization_state_only() -> None:
    spec = get_model_spec("C29")

    model = _build_model(spec, "cpu")
    payload = _checkpoint_payload(
        model,
        spec=spec,
        epoch=3,
        target_scale=torch.tensor(2.0),
        metrics={"normalized_mse": 0.5},
        trial_hash="example",
    )
    assert payload["fixed_tensor_artifacts_stored"] is False
    assert set(payload["parameter_state_dict"]) == {
        name for name, _value in model.named_parameters()
    }
    assert payload["normalization_state_dict"]
    restored = _build_model(spec, "cpu")
    _restore_model_state(
        restored,
        payload["parameter_state_dict"],
        payload["normalization_state_dict"],
    )
    for first, second in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)


def test_study_order_places_group_conv_before_thirty_v2_models() -> None:
    expected = ("G00", *(f"C{index:02d}" for index in range(1, 31)))
    assert tuple(spec.model_id for spec in STUDY_SPECS) == expected
    assert get_study_spec("g00") is GROUP_CONV_SPEC
    assert get_study_spec("c01") is STUDY_SPECS[1]


def test_group_conv_checkpoint_has_no_normalization_state() -> None:
    model = _build_model(GROUP_CONV_SPEC, "cpu")
    payload = _checkpoint_payload(
        model,
        spec=GROUP_CONV_SPEC,
        epoch=5,
        target_scale=torch.tensor(3.0),
        metrics={"normalized_mse": 0.25},
        trial_hash="group_conv_example",
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_160
    assert payload["normalization_state_dict"] == {}
    restored = _build_model(GROUP_CONV_SPEC, "cpu")
    _restore_model_state(
        restored,
        payload["parameter_state_dict"],
        payload["normalization_state_dict"],
    )
    for first, second in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)


def test_group_conv_trial_records_norm_statistics_and_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generator = torch.Generator().manual_seed(37)
    centers = torch.zeros(12, 2, 3)
    centers[:, 1] = torch.randn(12, 3, generator=generator)
    frames = torch.eye(3).expand(12, 2, 3, 3).clone()
    force = torch.randn(12, 3, generator=generator)
    data = TrainingData(
        centers=centers,
        frames=frames,
        root_force=force,
        provenance={"source": "synthetic"},
    )
    monkeypatch.setattr(
        "experiments.benzene_pair.sweep30.load_data",
        lambda *args, **kwargs: data,
    )
    config = SweepConfig(
        shard_paths=tuple(tmp_path / f"shard_{index}.csv" for index in range(4)),
        study_directory=tmp_path / "study",
        epochs=500,
        effective_batch_size=10000,
        micro_batch_size=10000,
        learning_rate=0.003,
        weight_decay=0.0001,
        scheduler_step_size=125,
        scheduler_gamma=0.5,
        validation_every=1,
        split_seed=1,
        model_seed=2,
        shuffle_seed=3,
        split_fractions=(0.5, 0.25, 0.25),
        device="cpu",
        dtype="float32",
        threads=1,
        symmetry_tolerance=1.0e-4,
        symmetry_probe_count=2,
        expected_sample_count=400000,
        expected_dataset_revision=3,
        expected_opls_version="2.0.0",
        enable_tf32=True,
        relative_force_norm_sample_count=2,
        relative_force_norm_seed=5,
        comet=CometConfig.from_mapping(None),
    )
    split = SplitIndices(
        train=torch.arange(0, 8),
        validation=torch.arange(8, 10),
        test=torch.arange(10, 12),
    )
    paths = TrialPaths.create(tmp_path / "trial")
    summary = run_trial(
        config,
        GROUP_CONV_SPEC,
        paths,
        split,
        {
            "data_provenance": data.provenance,
            "manifest_hash": "synthetic_manifest",
        },
        NullCometTrialLogger(),
        device="cpu",
        epochs=1,
        sample_limit=12,
        study_metadata={
            "concurrent_run": True,
            "shared_gpu_process_count": 3,
        },
    )
    assert summary["status"] == "complete"
    assert summary["model"]["parameter_count"] == 20_160
    assert summary["concurrent_run"] is True
    assert summary["shared_gpu_process_count"] == 3
    assert "selected_model_audit" not in summary
    assert set(summary["relative_force_norm_difference"]) == {
        "train",
        "validation",
        "test",
    }
    for partition in summary["relative_force_norm_difference"].values():
        assert partition["count"] == 2
        assert partition["min"] <= partition["median"] <= partition["max"]
    assert paths.history.is_file()
    assert paths.best.is_file()
    assert paths.final.is_file()
    assert paths.summary.is_file()
    assert not paths.resume.exists()


def test_comet_epoch_record_prevents_duplicate_resume_logging(tmp_path: Path) -> None:
    paths = TrialPaths.create(tmp_path / "trial")
    paths.directory.mkdir(parents=True)
    paths.comet.write_text(
        json.dumps(
            {
                "schema_name": "tfenn_sweep31_comet_trial",
                "schema_version": 1,
                "last_logged_epoch": 0,
            }
        ),
        encoding="utf_8",
    )

    class Logger:
        enabled = True

        def __init__(self) -> None:
            self.epochs: list[int] = []

        def log_epoch(self, **values) -> None:
            self.epochs.append(int(values["epoch"]))

    logger = Logger()
    initial = {
        "epoch": 0,
        "learning_rate": 0.003,
        "train_normalized_mse": 1.0,
        "validation_normalized_mse": 1.1,
        "validation_relative_rmse_percent": 100.0,
        "normalization_minimum_count": 0,
        "normalization_maximum_count": 0,
        "epoch_duration_seconds": 0.0,
    }
    next_epoch = {**initial, "epoch": 1, "train_normalized_mse": 0.9}
    _log_comet_epoch(logger, paths, initial)
    _log_comet_epoch(logger, paths, next_epoch)
    _log_comet_epoch(logger, paths, next_epoch)
    assert logger.epochs == [1]
    assert json.loads(paths.comet.read_text(encoding="utf_8"))[
        "last_logged_epoch"
    ] == 1
