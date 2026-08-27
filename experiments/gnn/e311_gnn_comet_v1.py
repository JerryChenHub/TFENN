"""Minimal Comet interface for the E311 GNN experiment study."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


COMET_API_KEY_ENVIRONMENT_VARIABLE_V1 = "COMET_API_KEY"
EPOCH_METRIC_NAMES_V1 = frozenset(
    {
        "train_loss",
        "validation_loss",
        "epoch_duration_seconds",
    }
)
FINAL_METRIC_NAMES_V1 = frozenset(
    {
        "final_test_mae",
        "final_normal_f_difference",
    }
)
_SENSITIVE_PARAMETER_FRAGMENTS_V1 = (
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
)
_FINISH_STATUSES_V1 = frozenset({"complete", "error", "interrupted"})

BackendFactoryV1 = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class E311GNNCometConfigV1:
    """Describe one online Comet destination without containing credentials."""

    project_name: str
    workspace: str | None = None
    tags: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        if not isinstance(self.project_name, str):
            raise TypeError("project_name must be str")
        project_name = self.project_name.strip()
        if self.enabled and not project_name:
            raise ValueError("an enabled logger requires a project name")
        object.__setattr__(self, "project_name", project_name)

        if self.workspace is not None and not isinstance(self.workspace, str):
            raise TypeError("workspace must be str or None")
        workspace = None if self.workspace is None else self.workspace.strip() or None
        object.__setattr__(self, "workspace", workspace)

        if isinstance(self.tags, (str, bytes)) or not isinstance(self.tags, Sequence):
            raise TypeError("tags must be a sequence of strings")
        tags = tuple(str(tag).strip() for tag in self.tags)
        if any(not tag for tag in tags):
            raise ValueError("tags must not contain empty values")
        if len(set(tags)) != len(tags):
            raise ValueError("tags must be unique")
        object.__setattr__(self, "tags", tags)


def _default_backend_factory_v1(
    *,
    config: E311GNNCometConfigV1,
    experiment_name: str,
    api_key: str,
) -> Any:
    try:
        import comet_ml
    except ImportError as error:
        raise RuntimeError(
            "Comet logging is enabled but comet_ml is unavailable"
        ) from error

    experiment_config = comet_ml.ExperimentConfig(
        disabled=False,
        name=experiment_name,
        tags=list(config.tags),
        log_code=False,
        log_graph=False,
        parse_args=False,
        display_summary_level=0,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        log_env_gpu=False,
        log_env_host=False,
        log_env_cpu=False,
        log_env_network=False,
        log_env_disk=False,
        auto_output_logging=False,
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_log_co2=False,
        auto_metric_step_rate=0,
        auto_histogram_epoch_rate=0,
        auto_histogram_gradient_logging=False,
        auto_histogram_activation_logging=False,
        auto_histogram_tensorboard_logging=False,
        auto_histogram_weight_logging=False,
    )
    return comet_ml.start(
        api_key=api_key,
        project_name=config.project_name,
        workspace=config.workspace,
        online=True,
        mode="create",
        experiment_config=experiment_config,
    )


def _is_sensitive_parameter_name_v1(name: object) -> bool:
    normalized = str(name).strip().lower().replace(" ", "_")
    return any(fragment in normalized for fragment in _SENSITIVE_PARAMETER_FRAGMENTS_V1)


def _parameter_value_v1(value: Any, *, api_key: str) -> Any:
    if isinstance(value, Enum):
        return _parameter_value_v1(value.value, api_key=api_key)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Comet parameters must be finite")
        return value
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        if api_key and api_key in value:
            raise ValueError("Comet parameters must not contain COMET_API_KEY")
        return value
    if isinstance(value, Mapping):
        safe_mapping = {}
        for key, item in value.items():
            if _is_sensitive_parameter_name_v1(key):
                raise ValueError("sensitive parameter names are not allowed")
            safe_mapping[str(key)] = _parameter_value_v1(item, api_key=api_key)
        return safe_mapping
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_parameter_value_v1(item, api_key=api_key) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _parameter_value_v1(item_method(), api_key=api_key)
        except (TypeError, ValueError, RuntimeError):
            pass
    text = str(value)
    if api_key and api_key in text:
        raise ValueError("Comet parameters must not contain COMET_API_KEY")
    return text


def _flatten_parameters_v1(
    values: Mapping[str, Any],
    *,
    api_key: str,
    prefix: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, raw_value in values.items():
        if _is_sensitive_parameter_name_v1(raw_key):
            raise ValueError("sensitive parameter names are not allowed")
        key = str(raw_key).strip()
        if not key:
            raise ValueError("parameter names must not be empty")
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(raw_value, Mapping):
            nested = _flatten_parameters_v1(
                raw_value,
                api_key=api_key,
                prefix=name,
            )
            for nested_name, nested_value in nested.items():
                if nested_name in result:
                    raise ValueError("flattened parameter names must be unique")
                result[nested_name] = nested_value
            continue
        value = _parameter_value_v1(raw_value, api_key=api_key)
        if isinstance(value, (list, dict)):
            value = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        if name in result:
            raise ValueError("flattened parameter names must be unique")
        result[name] = value
    return result


def _finite_metric_v1(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


class NullE311GNNCometLoggerV1:
    """Provide the same interface when Comet is explicitly disabled."""

    enabled = False

    @property
    def identity(self) -> dict[str, Any]:
        return {"enabled": False}

    def log_parameters(self, *, parameters: Mapping[str, Any]) -> None:
        del parameters

    def log_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        epoch_duration_seconds: float,
    ) -> None:
        del epoch, train_loss, validation_loss, epoch_duration_seconds

    def log_final(
        self,
        *,
        final_test_mae: float,
        final_normal_f_difference: float,
    ) -> None:
        del final_test_mae, final_normal_f_difference

    def finish(self, status: str = "complete") -> None:
        if status not in _FINISH_STATUSES_V1:
            raise ValueError("unknown Comet finish status")


class E311GNNCometLoggerV1:
    """Log only the explicitly registered E311 study values."""

    enabled = True

    def __init__(
        self,
        backend: Any,
        config: E311GNNCometConfigV1,
        experiment_name: str,
        api_key: str,
    ) -> None:
        self._backend = backend
        self._config = config
        self._experiment_name = experiment_name
        self._api_key = api_key
        self._finished = False

    @property
    def identity(self) -> dict[str, Any]:
        key_method = getattr(self._backend, "get_key", None)
        experiment_key = key_method() if callable(key_method) else None
        return {
            "enabled": True,
            "project_name": self._config.project_name,
            "workspace": self._config.workspace,
            "experiment_name": self._experiment_name,
            "experiment_key": experiment_key,
            "url": getattr(self._backend, "url", None),
        }

    def _require_active(self) -> None:
        if self._finished:
            raise RuntimeError("the Comet logger is already finished")

    def log_parameters(self, *, parameters: Mapping[str, Any]) -> None:
        self._require_active()
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        flattened = _flatten_parameters_v1(parameters, api_key=self._api_key)
        if not flattened:
            raise ValueError("parameters must not be empty")
        self._backend.log_parameters(flattened)

    def log_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        epoch_duration_seconds: float,
    ) -> None:
        self._require_active()
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        metrics = {
            "train_loss": _finite_metric_v1("train_loss", train_loss),
            "validation_loss": _finite_metric_v1(
                "validation_loss", validation_loss
            ),
            "epoch_duration_seconds": _finite_metric_v1(
                "epoch_duration_seconds", epoch_duration_seconds
            ),
        }
        if set(metrics) != EPOCH_METRIC_NAMES_V1:
            raise RuntimeError("the epoch metric registry changed unexpectedly")
        self._backend.log_metrics(metrics, step=epoch, epoch=epoch)

    def log_final(
        self,
        *,
        final_test_mae: float,
        final_normal_f_difference: float,
    ) -> None:
        self._require_active()
        metrics = {
            "final_test_mae": _finite_metric_v1(
                "final_test_mae", final_test_mae
            ),
            "final_normal_f_difference": _finite_metric_v1(
                "final_normal_f_difference", final_normal_f_difference
            ),
        }
        if set(metrics) != FINAL_METRIC_NAMES_V1:
            raise RuntimeError("the final metric registry changed unexpectedly")
        self._backend.log_metrics(metrics)

    def finish(self, status: str = "complete") -> None:
        if status not in _FINISH_STATUSES_V1:
            raise ValueError("unknown Comet finish status")
        if self._finished:
            return
        self._backend.end()
        self._finished = True


def create_e311_gnn_comet_logger_v1(
    config: E311GNNCometConfigV1,
    *,
    experiment_name: str,
    backend_factory: BackendFactoryV1 | None = None,
) -> E311GNNCometLoggerV1 | NullE311GNNCometLoggerV1:
    """Create an online logger or an explicit null logger."""

    if not isinstance(config, E311GNNCometConfigV1):
        raise TypeError("config must be E311GNNCometConfigV1")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name must not be empty")
    experiment_name = experiment_name.strip()
    if not config.enabled:
        return NullE311GNNCometLoggerV1()

    api_key = os.environ.get(COMET_API_KEY_ENVIRONMENT_VARIABLE_V1, "").strip()
    if not api_key:
        raise RuntimeError("COMET_API_KEY must be set for online Comet logging")
    factory = _default_backend_factory_v1 if backend_factory is None else backend_factory
    backend = factory(
        config=config,
        experiment_name=experiment_name,
        api_key=api_key,
    )
    if backend is None:
        raise RuntimeError("Comet backend creation returned no experiment")
    if bool(getattr(backend, "disabled", False)):
        raise RuntimeError("Comet returned a disabled experiment")
    return E311GNNCometLoggerV1(
        backend,
        config,
        experiment_name,
        api_key,
    )


__all__ = [
    "COMET_API_KEY_ENVIRONMENT_VARIABLE_V1",
    "E311GNNCometConfigV1",
    "E311GNNCometLoggerV1",
    "EPOCH_METRIC_NAMES_V1",
    "FINAL_METRIC_NAMES_V1",
    "NullE311GNNCometLoggerV1",
    "create_e311_gnn_comet_logger_v1",
]
