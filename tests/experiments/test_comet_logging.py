from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.benzene_pair.comet_logging import (
    CometConfig,
    NullCometTrialLogger,
    create_comet_trial_logger,
)


class FakeExperiment:
    disabled = False
    url = "https://example.invalid/experiment/example"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, method: str, *values: Any, **options: Any) -> None:
        self.calls.append((method, values, options))

    def get_key(self) -> str:
        return "example_key"

    def log_other(self, *values: Any, **options: Any) -> None:
        self._record("log_other", *values, **options)

    def log_others(self, *values: Any, **options: Any) -> None:
        self._record("log_others", *values, **options)

    def log_parameters(self, *values: Any, **options: Any) -> None:
        self._record("log_parameters", *values, **options)

    def log_metrics(self, *values: Any, **options: Any) -> None:
        self._record("log_metrics", *values, **options)

    def log_asset_data(self, *values: Any, **options: Any) -> None:
        self._record("log_asset_data", *values, **options)

    def log_asset(self, *values: Any, **options: Any) -> None:
        self._record("log_asset", *values, **options)

    def log_model(self, *values: Any, **options: Any) -> None:
        self._record("log_model", *values, **options)

    def end(self) -> None:
        self._record("end")


def _enabled_config(*, upload_checkpoints: bool = True) -> CometConfig:
    return CometConfig.from_mapping(
        {
            "enabled": True,
            "required_online": True,
            "project_name": "tfenn_pair_comparison",
            "workspace": None,
            "upload_checkpoints": upload_checkpoints,
            "tags": ["benzene_pair", "gpu"],
        }
    )


def test_disabled_config_returns_null_logger_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    called = False

    def factory(**values: Any) -> FakeExperiment:
        del values
        nonlocal called
        called = True
        return FakeExperiment()

    logger = create_comet_trial_logger(
        CometConfig.from_mapping(None),
        experiment_name="C01",
        study_name="study",
        backend_factory=factory,
    )
    assert isinstance(logger, NullCometTrialLogger)
    assert logger.identity == {"enabled": False}
    assert called is False


def test_enabled_config_requires_environment_key(monkeypatch) -> None:
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="COMET_API_KEY"):
        create_comet_trial_logger(
            _enabled_config(),
            experiment_name="C01",
            study_name="study",
            backend_factory=lambda **values: FakeExperiment(),
        )


def test_online_trial_records_config_epochs_results_and_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    secret = "private_example_key"
    monkeypatch.setenv("COMET_API_KEY", secret)
    backend = FakeExperiment()
    factory_arguments: dict[str, Any] = {}

    def factory(**values: Any) -> FakeExperiment:
        factory_arguments.update(values)
        return backend

    logger = create_comet_trial_logger(
        _enabled_config(),
        experiment_name="C15",
        study_name="sweep31_400k",
        tags=("primary",),
        backend_factory=factory,
    )
    assert factory_arguments["api_key"] == secret
    assert factory_arguments["tags"] == ("benzene_pair", "gpu", "primary")
    assert logger.identity == {
        "enabled": True,
        "project_name": "tfenn_pair_comparison",
        "workspace": None,
        "experiment_name": "C15",
        "experiment_key": "example_key",
        "url": "https://example.invalid/experiment/example",
    }
    logger.log_config(
        study_config={
            "epochs": 500,
            "COMET_API_KEY": secret,
            "concurrent_run": True,
            "shared_gpu_process_count": 3,
        },
        trial_config={"model_id": "C15", "route": ["A2", "A2", "B1"]},
        parameters={"parameter_count": 20005},
    )
    logger.log_epoch(
        epoch=3,
        train_loss=0.4,
        validation_loss=0.5,
        learning_rate=0.003,
        extra_metrics={"epoch_duration_seconds": 1.2},
    )
    logger.log_final(
        metrics={"test": {"relative_rmse_percent": 4.2}},
        relative_force_norm_stats={
            "test": {
                "minimum_percent": 0.1,
                "median_percent": 3.0,
                "maximum_percent": 22.0,
            }
        },
        summary={"status": "complete"},
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"example")
    logger.log_checkpoint_reference(
        "best",
        checkpoint,
        sha256="abc",
        metadata={"epoch": 3},
    )
    history = tmp_path / "history.csv"
    history.write_text("epoch,loss\n3,0.4\n", encoding="utf_8")
    logger.log_asset(history, name="history.csv")
    logger.finish()
    logger.finish()

    config_call = next(
        call
        for call in backend.calls
        if call[0] == "log_asset_data"
        and call[2].get("name") == "experiment_config.json"
    )
    assert secret not in config_call[1][0]
    assert json.loads(config_call[1][0])["study"]["COMET_API_KEY"] == "[redacted]"
    parameter_call = next(call for call in backend.calls if call[0] == "log_parameters")
    assert parameter_call[1][0]["study_concurrent_run"] is True
    assert parameter_call[1][0]["study_shared_gpu_process_count"] == 3
    epoch_call = next(
        call
        for call in backend.calls
        if call[0] == "log_metrics" and call[2].get("epoch") == 3
    )
    assert epoch_call[1][0]["train_normalized_mse"] == 0.4
    assert epoch_call[1][0]["validation_normalized_mse"] == 0.5
    assert epoch_call[1][0]["train_loss"] == 0.4
    assert epoch_call[1][0]["validation_loss"] == 0.5
    final_call = next(
        call for call in backend.calls if call[0] == "log_metrics" and not call[2]
    )
    assert final_call[1][0]["relative_norm_force_diff_test_median_percent"] == 3.0
    assert sum(call[0] == "log_model" for call in backend.calls) == 1
    assert sum(call[0] == "end" for call in backend.calls) == 1


def test_error_is_redacted_and_checkpoint_upload_can_be_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    secret = "secret_value"
    monkeypatch.setenv("COMET_API_KEY", secret)
    backend = FakeExperiment()
    logger = create_comet_trial_logger(
        _enabled_config(upload_checkpoints=False),
        experiment_name="C29",
        study_name="study",
        backend_factory=lambda **values: backend,
    )
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"example")
    logger.log_checkpoint_reference("final", checkpoint, sha256="def")
    logger.log_error(
        RuntimeError(f"failed with {secret}"),
        stage="evaluation",
    )
    logger.finish("error")
    assert all(call[0] != "log_model" for call in backend.calls)
    error_call = next(
        call
        for call in backend.calls
        if call[0] == "log_asset_data" and call[2].get("name") == "error.json"
    )
    assert secret not in error_call[1][0]
    assert "[redacted]" in error_call[1][0]


def test_nonfinite_metrics_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("COMET_API_KEY", "example")
    logger = create_comet_trial_logger(
        _enabled_config(),
        experiment_name="C30",
        study_name="study",
        backend_factory=lambda **values: FakeExperiment(),
    )
    with pytest.raises(ValueError, match="must be finite"):
        logger.log_epoch(
            epoch=1,
            train_loss=float("nan"),
            validation_loss=1.0,
            learning_rate=0.003,
        )
