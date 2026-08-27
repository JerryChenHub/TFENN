from __future__ import annotations

import sys
from typing import Any

import pytest

from experiments.gnn.e311_gnn_comet_v1 import (
    EPOCH_METRIC_NAMES_V1,
    FINAL_METRIC_NAMES_V1,
    E311GNNCometConfigV1,
    create_e311_gnn_comet_logger_v1,
)


class _FakeBackend:
    disabled = False
    url = "https://example.invalid/experiment"

    def __init__(self) -> None:
        self.parameter_calls: list[dict[str, Any]] = []
        self.metric_calls: list[tuple[dict[str, float], dict[str, int]]] = []
        self.end_count = 0

    def get_key(self) -> str:
        return "fake_experiment_key"

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        self.parameter_calls.append(dict(parameters))

    def log_metrics(self, metrics: dict[str, float], **context: int) -> None:
        self.metric_calls.append((dict(metrics), dict(context)))

    def end(self) -> None:
        self.end_count += 1


def test_default_backend_disables_all_automatic_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    captured_config: dict[str, Any] = {}
    captured_start: dict[str, Any] = {}

    class FakeExperimentConfig:
        def __init__(self, **values: Any) -> None:
            captured_config.update(values)

    class FakeCometModule:
        ExperimentConfig = FakeExperimentConfig

        @staticmethod
        def start(**values: Any) -> _FakeBackend:
            captured_start.update(values)
            return backend

    monkeypatch.setitem(sys.modules, "comet_ml", FakeCometModule())
    monkeypatch.setenv("COMET_API_KEY", "environment_only_key")
    config = E311GNNCometConfigV1(
        project_name="tfenn_e311_gnn_12_v1",
        workspace="workspace",
        tags=("tfenn", "e311"),
    )

    logger = create_e311_gnn_comet_logger_v1(
        config,
        experiment_name="X03",
    )

    assert captured_start["api_key"] == "environment_only_key"
    assert captured_start["project_name"] == config.project_name
    assert captured_start["workspace"] == config.workspace
    assert captured_start["online"] is True
    assert captured_start["mode"] == "create"
    assert captured_config["name"] == "X03"
    assert captured_config["tags"] == ["tfenn", "e311"]
    disabled_options = {
        "log_code",
        "log_graph",
        "parse_args",
        "log_git_metadata",
        "log_git_patch",
        "log_env_details",
        "log_env_gpu",
        "log_env_host",
        "log_env_cpu",
        "log_env_network",
        "log_env_disk",
        "auto_output_logging",
        "auto_param_logging",
        "auto_metric_logging",
        "auto_log_co2",
        "auto_histogram_gradient_logging",
        "auto_histogram_activation_logging",
        "auto_histogram_tensorboard_logging",
        "auto_histogram_weight_logging",
    }
    assert all(captured_config[name] is False for name in disabled_options)
    assert captured_config["auto_metric_step_rate"] == 0
    assert captured_config["auto_histogram_epoch_rate"] == 0
    assert captured_config["display_summary_level"] == 0
    assert logger.identity == {
        "enabled": True,
        "project_name": "tfenn_e311_gnn_12_v1",
        "workspace": "workspace",
        "experiment_name": "X03",
        "experiment_key": "fake_experiment_key",
        "url": "https://example.invalid/experiment",
    }


def test_logger_sends_only_registered_metric_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    captured_factory: dict[str, Any] = {}

    def factory(**values: Any) -> _FakeBackend:
        captured_factory.update(values)
        return backend

    monkeypatch.setenv("COMET_API_KEY", "runtime_key")
    logger = create_e311_gnn_comet_logger_v1(
        E311GNNCometConfigV1(project_name="project"),
        experiment_name="X06",
        backend_factory=factory,
    )
    logger.log_parameters(
        parameters={
            "experiment": {"id": "X06", "layer_count": 2},
            "seeds": (20260821, 20260822),
        }
    )
    logger.log_epoch(
        epoch=7,
        train_loss=0.25,
        validation_loss=0.5,
        epoch_duration_seconds=1.75,
    )
    logger.log_final(
        final_test_mae=0.125,
        final_normal_f_difference=2.5,
    )

    assert captured_factory["api_key"] == "runtime_key"
    assert backend.parameter_calls == [
        {
            "experiment.id": "X06",
            "experiment.layer_count": 2,
            "seeds": "[20260821,20260822]",
        }
    ]
    assert len(backend.metric_calls) == 2
    epoch_metrics, epoch_context = backend.metric_calls[0]
    final_metrics, final_context = backend.metric_calls[1]
    assert set(epoch_metrics) == EPOCH_METRIC_NAMES_V1
    assert epoch_metrics == {
        "train_loss": 0.25,
        "validation_loss": 0.5,
        "epoch_duration_seconds": 1.75,
    }
    assert epoch_context == {"step": 7, "epoch": 7}
    assert set(final_metrics) == FINAL_METRIC_NAMES_V1
    assert final_metrics == {
        "final_test_mae": 0.125,
        "final_normal_f_difference": 2.5,
    }
    assert final_context == {}


def test_logger_rejects_extra_or_invalid_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    monkeypatch.setenv("COMET_API_KEY", "runtime_key")
    logger = create_e311_gnn_comet_logger_v1(
        E311GNNCometConfigV1(project_name="project"),
        experiment_name="X09",
        backend_factory=lambda **values: backend,
    )

    with pytest.raises(TypeError):
        getattr(logger, "log_epoch")(
            epoch=1,
            train_loss=0.1,
            validation_loss=0.2,
            epoch_duration_seconds=0.3,
            learning_rate=0.01,
        )
    with pytest.raises(ValueError, match="finite"):
        logger.log_epoch(
            epoch=1,
            train_loss=float("nan"),
            validation_loss=0.2,
            epoch_duration_seconds=0.3,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        logger.log_final(
            final_test_mae=0.1,
            final_normal_f_difference=-1.0,
        )
    assert backend.metric_calls == []


def test_logger_uses_no_configured_credential_and_finishes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    config = E311GNNCometConfigV1(project_name="project")
    with pytest.raises(RuntimeError, match="COMET_API_KEY"):
        create_e311_gnn_comet_logger_v1(
            config,
            experiment_name="X12",
            backend_factory=lambda **values: backend,
        )

    monkeypatch.setenv("COMET_API_KEY", "runtime_key")
    logger = create_e311_gnn_comet_logger_v1(
        config,
        experiment_name="X12",
        backend_factory=lambda **values: backend,
    )
    with pytest.raises(ValueError, match="sensitive"):
        logger.log_parameters(parameters={"api_key": "forbidden"})
    with pytest.raises(ValueError, match="COMET_API_KEY"):
        logger.log_parameters(parameters={"note": "contains runtime_key"})
    logger.finish("complete")
    logger.finish("complete")
    assert backend.end_count == 1
    with pytest.raises(RuntimeError, match="already finished"):
        logger.log_epoch(
            epoch=1,
            train_loss=0.1,
            validation_loss=0.2,
            epoch_duration_seconds=0.3,
        )


def test_disabled_logger_does_not_require_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    logger = create_e311_gnn_comet_logger_v1(
        E311GNNCometConfigV1(project_name="", enabled=False),
        experiment_name="smoke",
    )
    assert logger.identity == {"enabled": False}
    logger.log_parameters(parameters={})
    logger.log_epoch(
        epoch=1,
        train_loss=0.1,
        validation_loss=0.2,
        epoch_duration_seconds=0.3,
    )
    logger.log_final(
        final_test_mae=0.1,
        final_normal_f_difference=1.0,
    )
    logger.finish("complete")
