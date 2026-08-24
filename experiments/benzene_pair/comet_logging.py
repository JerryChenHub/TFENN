"""Comet recording for the benzene pair comparison study."""

from __future__ import annotations

import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


BackendFactory = Callable[..., Any]
_REDACTED = "[redacted]"
_SENSITIVE_KEYS = ("api_key", "apikey", "password", "secret", "token")
FULL_METRIC_PROFILE = "full"
LOSS_TIME_TEST_ERROR_PROFILE = "loss_time_test_error"
METRIC_PROFILES = (FULL_METRIC_PROFILE, LOSS_TIME_TEST_ERROR_PROFILE)


@dataclass(frozen=True, slots=True)
class CometConfig:
    """Describe online Comet recording without containing credentials."""

    enabled: bool
    required_online: bool
    project_name: str
    workspace: str | None
    upload_checkpoints: bool
    tags: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> CometConfig:
        if value is None:
            return cls(False, False, "", None, False, ())
        allowed = {
            "enabled",
            "required_online",
            "project_name",
            "workspace",
            "upload_checkpoints",
            "tags",
        }
        unknown = set(value) - allowed
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ValueError(f"unknown Comet configuration fields: {names}")
        workspace_value = value.get("workspace")
        result = cls(
            enabled=bool(value.get("enabled", False)),
            required_online=bool(value.get("required_online", True)),
            project_name=str(value.get("project_name", "")).strip(),
            workspace=None
            if workspace_value is None
            else str(workspace_value).strip() or None,
            upload_checkpoints=bool(value.get("upload_checkpoints", True)),
            tags=tuple(str(item).strip() for item in value.get("tags", ())),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.required_online:
            raise ValueError("enabled Comet recording must require online delivery")
        if not self.project_name:
            raise ValueError("Comet project name is required")
        if any(not item for item in self.tags):
            raise ValueError("Comet tags must not be empty")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("Comet tags must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "required_online": self.required_online,
            "project_name": self.project_name,
            "workspace": self.workspace,
            "upload_checkpoints": self.upload_checkpoints,
            "tags": list(self.tags),
        }


def _is_sensitive_key(value: object) -> bool:
    normalized = str(value).strip().lower().replace(" ", "_")
    return any(item in normalized for item in _SENSITIVE_KEYS)


def _redact_text(value: str) -> str:
    secret = os.environ.get("COMET_API_KEY", "").strip()
    return value.replace(secret, _REDACTED) if secret else value


def _safe_value(value: Any, *, sensitive: bool = False) -> Any:
    if sensitive:
        return _REDACTED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Path):
        return _redact_text(str(value))
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item, sensitive=_is_sensitive_key(key))
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_safe_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe_value(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return _redact_text(str(value))


def _flatten_scalars(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten_scalars(item, prefix=name))
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            result[name] = json.dumps(_safe_value(item), separators=(",", ":"))
        else:
            result[name] = _safe_value(item, sensitive=_is_sensitive_key(key))
    return result


def _numeric_metrics(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_numeric_metrics(item, prefix=name))
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError(f"Comet metric {name} must be finite")
        result[name] = numeric
    return result


def _default_backend_factory(
    *,
    config: CometConfig,
    experiment_name: str,
    api_key: str,
    tags: Sequence[str],
) -> Any:
    try:
        import comet_ml
    except ImportError as error:
        raise RuntimeError(
            "Comet recording is enabled but the comet_ml package is unavailable"
        ) from error
    experiment_config = comet_ml.ExperimentConfig(
        name=experiment_name,
        tags=list(tags),
    )
    return comet_ml.start(
        api_key=api_key,
        project_name=config.project_name,
        workspace=config.workspace,
        online=True,
        mode="create",
        experiment_config=experiment_config,
    )


class NullCometTrialLogger:
    """Accept logging calls when Comet is explicitly disabled."""

    enabled = False

    @property
    def identity(self) -> dict[str, Any]:
        return {"enabled": False}

    def log_config(self, **values: Any) -> None:
        del values

    def log_epoch(self, **values: Any) -> None:
        del values

    def log_final(self, **values: Any) -> None:
        del values

    def log_asset(self, *values: Any, **options: Any) -> None:
        del values, options

    def log_checkpoint_reference(self, *values: Any, **options: Any) -> None:
        del values, options

    def log_error(self, *values: Any, **options: Any) -> None:
        del values, options

    def finish(self, status: str = "complete") -> None:
        del status


class CometTrialLogger:
    """Record one model trial as one online Comet experiment."""

    enabled = True

    def __init__(
        self,
        backend: Any,
        config: CometConfig,
        experiment_name: str,
        study_name: str,
        metric_profile: str = FULL_METRIC_PROFILE,
    ) -> None:
        if metric_profile not in METRIC_PROFILES:
            raise ValueError(f"unknown Comet metric profile: {metric_profile}")
        self._backend = backend
        self._config = config
        self._experiment_name = experiment_name
        self._study_name = study_name
        self._metric_profile = metric_profile
        self._finished = False
        self._backend.log_other("study_name", study_name)
        self._backend.log_other("trial_name", experiment_name)
        self._backend.log_other("metric_profile", metric_profile)
        self._backend.log_other("status", "running")

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

    def log_config(
        self,
        *,
        study_config: Mapping[str, Any],
        trial_config: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> None:
        safe_document = {
            "study": _safe_value(study_config),
            "trial": _safe_value(trial_config),
            "parameters": _safe_value(parameters),
        }
        flattened = {
            **_flatten_scalars(study_config, prefix="study"),
            **_flatten_scalars(trial_config, prefix="trial"),
            **_flatten_scalars(parameters, prefix="model"),
        }
        self._backend.log_parameters(flattened)
        self._backend.log_asset_data(
            json.dumps(
                safe_document,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            name="experiment_config.json",
            overwrite=True,
        )

    def log_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        learning_rate: float,
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if self._metric_profile == LOSS_TIME_TEST_ERROR_PROFILE:
            if int(epoch) <= 0:
                return
            if extra_metrics is None or "epoch_duration_seconds" not in extra_metrics:
                raise ValueError("epoch duration is required by the Comet metric profile")
            metrics = _numeric_metrics(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "epoch_duration_seconds": extra_metrics["epoch_duration_seconds"],
                }
            )
            if set(metrics) != {
                "train_loss",
                "validation_loss",
                "epoch_duration_seconds",
            }:
                raise ValueError("train loss, validation loss, and epoch duration are required")
            self._backend.log_metrics(metrics, step=int(epoch), epoch=int(epoch))
            return
        values: dict[str, Any] = {
            "train_normalized_mse": train_loss,
            "validation_normalized_mse": validation_loss,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
        }
        if extra_metrics is not None:
            values.update(extra_metrics)
        metrics = _numeric_metrics(values)
        self._backend.log_metrics(metrics, step=int(epoch), epoch=int(epoch))

    def log_final(
        self,
        *,
        metrics: Mapping[str, Any],
        relative_force_norm_stats: Mapping[str, Any],
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        if self._metric_profile == LOSS_TIME_TEST_ERROR_PROFILE:
            test_metrics = metrics.get("test")
            if not isinstance(test_metrics, Mapping):
                raise ValueError("final test metrics are required by the Comet metric profile")
            final_metrics = _numeric_metrics(
                {
                    "test": {
                        "mae": test_metrics.get("mae"),
                        "sae": test_metrics.get("sae"),
                    }
                },
                prefix="final",
            )
            if set(final_metrics) != {"final_test_mae", "final_test_sae"}:
                raise ValueError("final test MAE and SAE are required")
        else:
            final_metrics = {
                **_numeric_metrics(metrics, prefix="final"),
                **_numeric_metrics(
                    relative_force_norm_stats,
                    prefix="relative_norm_force_diff",
                ),
            }
        self._backend.log_metrics(final_metrics)
        if self._metric_profile == LOSS_TIME_TEST_ERROR_PROFILE:
            final_document: dict[str, Any] = {
                "metrics": {
                    "test": {
                        "mae": final_metrics["final_test_mae"],
                        "sae": final_metrics["final_test_sae"],
                    }
                }
            }
        else:
            final_document = {
                "metrics": _safe_value(metrics),
                "relative_norm_force_diff": _safe_value(
                    relative_force_norm_stats
                ),
            }
            if summary is not None:
                final_document["summary"] = _safe_value(summary)
        self._backend.log_asset_data(
            json.dumps(
                final_document,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            name="final_metrics.json",
            overwrite=True,
        )

    def log_asset(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        asset = Path(path).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        if (
            self._metric_profile == LOSS_TIME_TEST_ERROR_PROFILE
            and asset.name == "history.csv"
        ):
            return
        self._backend.log_asset(
            str(asset),
            file_name=asset.name if name is None else name,
            metadata=_safe_value(metadata or {}),
        )

    def log_checkpoint_reference(
        self,
        name: str,
        path: str | Path,
        *,
        sha256: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        checkpoint = Path(path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        reference = {
            "name": name,
            "path": str(checkpoint),
            "sha256": sha256,
            "uploaded": self._config.upload_checkpoints,
            "metadata": _safe_value(metadata or {}),
        }
        self._backend.log_asset_data(
            json.dumps(reference, indent=2, sort_keys=True, allow_nan=False),
            name=f"checkpoint_{name}_reference.json",
            overwrite=True,
        )
        if self._config.upload_checkpoints:
            self._backend.log_model(
                name,
                str(checkpoint),
                file_name=checkpoint.name,
                metadata={
                    "sha256": sha256,
                    **_safe_value(metadata or {}),
                },
            )

    def log_error(
        self,
        error: BaseException,
        *,
        stage: str,
        traceback_text: str | None = None,
    ) -> None:
        document = {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": _redact_text(str(error)),
            "traceback": _redact_text(
                traceback_text
                if traceback_text is not None
                else "".join(traceback.format_exception(error))
            ),
        }
        self._backend.log_others(
            {
                "status": "error",
                "error_stage": document["stage"],
                "error_type": document["exception_type"],
                "error_message": document["message"],
            }
        )
        self._backend.log_asset_data(
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False),
            name="error.json",
            overwrite=True,
        )

    def finish(self, status: str = "complete") -> None:
        if self._finished:
            return
        self._backend.log_other("status", status)
        self._backend.end()
        self._finished = True


def create_comet_trial_logger(
    config: CometConfig,
    *,
    experiment_name: str,
    study_name: str,
    tags: Sequence[str] = (),
    metric_profile: str = FULL_METRIC_PROFILE,
    backend_factory: BackendFactory | None = None,
) -> CometTrialLogger | NullCometTrialLogger:
    """Create a new online experiment or an explicit null logger."""

    config.validate()
    if not config.enabled:
        return NullCometTrialLogger()
    api_key = os.environ.get("COMET_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "COMET_API_KEY must be set when online Comet recording is enabled"
        )
    combined_tags = tuple(dict.fromkeys((*config.tags, *map(str, tags))))
    factory = _default_backend_factory if backend_factory is None else backend_factory
    backend = factory(
        config=config,
        experiment_name=experiment_name,
        api_key=api_key,
        tags=combined_tags,
    )
    if backend is None:
        raise RuntimeError("Comet backend creation returned no experiment")
    if bool(getattr(backend, "disabled", False)):
        raise RuntimeError("Comet returned a disabled experiment")
    return CometTrialLogger(
        backend,
        config,
        experiment_name,
        study_name,
        metric_profile,
    )
