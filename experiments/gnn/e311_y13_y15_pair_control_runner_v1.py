"""Core runners for the Y13 through Y15 E311 controls.

Y13 delegates to the unchanged E series runner. Y14 reuses the E series data,
split, optimizer, scheduler, normalization, checkpoint selection, and metrics,
changing only the model to the parameter free OddGraph wrapper. Y15 trains that
graph model using only five molecule force supervision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.benzene_pair import sweep30 as e_common
from experiments.benzene_pair.comet_logging import (
    LOSS_TIME_TEST_ERROR_PROFILE,
    CometConfig,
    NullCometTrialLogger,
)
from experiments.benzene_pair.data.benzene_cluster import (
    load_benzene_cluster_csv,
)
from experiments.benzene_pair.e_series import runner as e_runner
from experiments.gnn.e311_gnn_y12_diagnostic_runner_v2 import (
    YCometConfigV2,
    create_y_comet_logger_v2,
)
from experiments.gnn.e311_y13_y15_pair_control_core_v1 import (
    E311OddGraphCoreV1,
    E311TwoNodeOddControlV1,
    HISTORICAL_E311_MODEL_ID,
    HISTORICAL_E311_PARAMETER_COUNT,
    HISTORICAL_MODEL_SEED,
    HISTORICAL_SHUFFLE_SEED,
    HISTORICAL_SPLIT_SEED,
    assert_historical_e311_definition_v1,
    build_e311_odd_graph_core_v1,
    build_y14_two_node_control_v1,
    complete_pair_index_v1,
    get_y_pair_control_spec_v1,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "gnn" / "runs" / "e311_y13_y15_pair_controls_v1"
)
DEFAULT_COMET_PROJECT = "tfenn_e311_gnn_y12_diagnostic_v2"
COMET_SERIES_TAG = "e311_gnn_y13_y15_pair_controls_v1"
Y15_LEARNING_RATE = 0.003
Y15_WEIGHT_DECAY = 1.0e-4
Y15_SCHEDULER_GAMMA = 0.5
Y15_TF32 = True
Y15_OPLS_VERSION = "2.0.0"
Y15_OPLS_COMMIT = "a5f874ed00152b156cd2525c961bd81030237e31"


def _resolve_training_protocol(
    experiment_id: str,
    epochs: int | None,
    batch_size: int | None,
) -> tuple[Any, int, int]:
    spec = get_y_pair_control_spec_v1(experiment_id)
    resolved_epochs = spec.epochs if epochs is None else int(epochs)
    resolved_batch_size = (
        spec.graph_batch_size if batch_size is None else int(batch_size)
    )
    if resolved_epochs < 1:
        raise ValueError("epochs must be positive")
    if resolved_batch_size < 1:
        raise ValueError("batch size must be positive")
    return spec, resolved_epochs, resolved_batch_size


def _resolved_experiment_name(
    base_name: str,
    spec: Any,
    epochs: int,
    batch_size: int,
) -> str:
    if epochs == spec.epochs and batch_size == spec.graph_batch_size:
        return base_name
    return f"{base_name}_BS{batch_size}_E{epochs}"


def _formal_comet_config(
    project: str,
    workspace: str | None,
    experiment_id: str,
) -> CometConfig:
    resolved_project = project.strip()
    if not resolved_project:
        raise ValueError("formal Y13 to Y15 runs require a Comet project")
    return CometConfig(
        enabled=True,
        required_online=True,
        project_name=resolved_project,
        workspace=workspace,
        upload_checkpoints=False,
        tags=("e311_gnn_y_series", COMET_SERIES_TAG, experiment_id),
    )


class _StrictYSeriesCometAdapter:
    """Adapt the strict Y logger to the historical trainer interface."""

    enabled = True

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._last_epoch = 0

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._logger.identity)

    def log_config(
        self,
        *,
        study_config: Mapping[str, Any],
        trial_config: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> None:
        self._logger.log_parameters(
            {
                "study": dict(study_config),
                "trial": dict(trial_config),
                "parameters": dict(parameters),
            }
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
        del learning_rate
        if int(epoch) <= 0:
            return
        if extra_metrics is None or "epoch_duration_seconds" not in extra_metrics:
            raise ValueError("strict Y Comet logging requires epoch duration")
        self._last_epoch = int(epoch)
        self._logger.log_evaluation(
            global_step=self._last_epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            epoch_duration_seconds=float(extra_metrics["epoch_duration_seconds"]),
        )

    def log_final(
        self,
        *,
        metrics: Mapping[str, Any],
        relative_force_norm_stats: Mapping[str, Any],
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        del relative_force_norm_stats, summary
        test_metrics = metrics.get("test")
        if not isinstance(test_metrics, Mapping):
            raise ValueError("strict Y Comet logging requires final test metrics")
        self._logger.log_final(
            global_step=self._last_epoch,
            final_test_mae=float(test_metrics["mae"]),
            final_test_sae=float(test_metrics["sae"]),
        )

    def log_asset(self, *values: Any, **options: Any) -> None:
        del values, options

    def log_checkpoint_reference(self, *values: Any, **options: Any) -> None:
        del values, options

    def log_error(self, *values: Any, **options: Any) -> None:
        del values, options

    def finish(self, status: str = "complete") -> None:
        del status
        self._logger.finish()


def _create_strict_y_series_comet_logger(
    *,
    comet_path: Path,
    model_id: str,
    experiment_name: str,
    project: str,
    workspace: str | None,
) -> _StrictYSeriesCometAdapter:
    if comet_path.exists():
        raise FileExistsError(
            "formal Y13 to Y15 strict Comet runs require a fresh output path"
        )
    logger = create_y_comet_logger_v2(
        YCometConfigV2(
            project=project.strip(),
            workspace=workspace,
            tags=("e311_gnn_y_series", COMET_SERIES_TAG, model_id),
            enabled=True,
        ),
        experiment_name=experiment_name,
    )
    adapted = _StrictYSeriesCometAdapter(logger)
    identity = adapted.identity
    experiment_key = str(identity.get("experiment_key", ""))
    if not experiment_key:
        raise RuntimeError("Comet did not provide an experiment key")
    _write_json_atomic(
        comet_path,
        {
            "schema_name": "tfenn_sweep31_comet_trial",
            "schema_version": 1,
            "model_id": model_id,
            "experiment_name": experiment_name,
            "project_name": project.strip(),
            "metric_profile": LOSS_TIME_TEST_ERROR_PROFILE,
            "experiment_key": experiment_key,
            "identity": identity,
            "last_logged_epoch": -1,
            "updated_at_utc": _utc_now(),
        },
    )
    return adapted


def _update_strict_comet_epoch_record(comet_path: Path, epoch: int) -> None:
    record = json.loads(comet_path.read_text(encoding="utf_8"))
    if record.get("schema_name") != "tfenn_sweep31_comet_trial":
        raise ValueError("unexpected strict Comet record schema")
    if int(epoch) <= int(record.get("last_logged_epoch", -1)):
        raise ValueError("strict Comet epoch record did not advance")
    record["last_logged_epoch"] = int(epoch)
    record["updated_at_utc"] = _utc_now()
    _write_json_atomic(comet_path, record)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("e311_y13_y15_pair_control_core_v1.py"),
        Path(__file__).with_name("e311_gnn_y12_diagnostic_runner_v2.py"),
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "e_series" / "catalog.py",
        REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "e_series"
        / "model_factory.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "e_series" / "runner.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "sweep30.py",
        REPOSITORY_ROOT / "src" / "TFENN" / "models" / "invariant_gate_pipeline_v2.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf_8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf_8",
    )
    os.replace(partial, path)


def _write_history_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("history cannot be empty")
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ("git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _assert_y13_preflight(study_root: Path) -> Mapping[str, Any]:
    preflight = e_runner._require_preflight(study_root)
    record = e_runner._preflight_record(preflight, HISTORICAL_E311_MODEL_ID)
    if (
        record.get("status") != "passed"
        or record.get("planned_parameter_count")
        != HISTORICAL_E311_PARAMETER_COUNT
        or record.get("actual_parameter_count")
        != HISTORICAL_E311_PARAMETER_COUNT
    ):
        raise RuntimeError(f"Y13 E311 preflight mismatch: {record}")
    return preflight


def run_y13_exact_reproduction_v1(
    study_root: Path,
    device: str,
    comet_project: str = DEFAULT_COMET_PROJECT,
    comet_workspace: str | None = None,
    output_directory: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> Path:
    """Run the historical non GNN E311 path with controlled protocol values."""

    assert_historical_e311_definition_v1()
    control_spec, resolved_epochs, resolved_batch_size = _resolve_training_protocol(
        "Y13",
        epochs,
        batch_size,
    )
    runtime_control_spec = replace(
        control_spec,
        epochs=resolved_epochs,
        graph_batch_size=resolved_batch_size,
    )
    study_root = study_root.resolve()
    resolved_output_directory = (
        study_root
        / "e3_path_gate_width"
        / "models"
        / HISTORICAL_E311_MODEL_ID
        if output_directory is None
        else output_directory.resolve()
    )
    completed = resolved_output_directory / "summary.json"
    if completed.exists():
        raise FileExistsError(
            "Y13 requires a fresh output directory; an E311 summary already exists"
        )
    if not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError("formal exact Y13 requires COMET_API_KEY")
    prepare_status = e_runner.main(("prepare", "--study_root", str(study_root)))
    if prepare_status != 0:
        raise RuntimeError("Y13 E-series preparation failed")
    preflight = _assert_y13_preflight(study_root)
    spec = e_runner._enriched_spec(
        e_runner.get_model_spec(HISTORICAL_E311_MODEL_ID),
        preflight,
    )
    config = e_runner.make_config(3, study_root=study_root)
    runtime_study_directory = (
        config.study_directory
        if output_directory is None
        else resolved_output_directory.parent
    )
    config = replace(
        config,
        study_directory=runtime_study_directory,
        epochs=resolved_epochs,
        effective_batch_size=resolved_batch_size,
        micro_batch_size=resolved_batch_size,
        comet=_formal_comet_config(
            comet_project,
            comet_workspace,
            "Y13",
        ),
    )
    split_directory = e_runner._shared_split_directory(study_root)
    split, split_manifest = e_common._load_split(split_directory)
    paths = e_common.TrialPaths.create(resolved_output_directory)
    experiment_name = _resolved_experiment_name(
        "Y13_E311_Exact_400K",
        control_spec,
        resolved_epochs,
        resolved_batch_size,
    )
    alias = {
        "schema_name": "tfenn_y13_exact_e311_alias",
        "schema_version": 1,
        "experiment": runtime_control_spec.as_dict(),
        "delegated_module": "experiments.benzene_pair.sweep30.run_trial",
        "delegated_command": {
            "experiment": 3,
            "model": HISTORICAL_E311_MODEL_ID,
            "device": device,
            "epochs": resolved_epochs,
            "batch_size": resolved_batch_size,
        },
        "experiment_name": experiment_name,
        "output_directory": str(resolved_output_directory),
        "comet_project": config.comet.project_name,
        "preflight_hash": preflight["preflight_hash"],
        "source_sha256": _source_sha256(),
        "git_commit": _git_commit(),
        "created_at_utc": _utc_now(),
    }
    alias_path = (
        study_root / "y13_exact_e311_alias.json"
        if output_directory is None
        else resolved_output_directory / "y13_exact_e311_alias.json"
    )
    _write_json_atomic(alias_path, alias)
    logger: Any = NullCometTrialLogger()
    try:
        logger = _create_strict_y_series_comet_logger(
            comet_path=paths.comet,
            model_id=HISTORICAL_E311_MODEL_ID,
            experiment_name=experiment_name,
            project=config.comet.project_name,
            workspace=config.comet.workspace,
        )
        e_common.run_trial(
            config,
            spec,
            paths,
            split,
            split_manifest,
            logger,
            device=str(_resolve_device(device)),
            epochs=resolved_epochs,
            model_builder=e_runner._build_model,
            source_sha256=e_runner._source_sha256(),
            study_metadata={
                **e_runner._study_metadata(preflight),
                "series": "Y13_Y15_pair_controls_v1",
                "experiment_id": "Y13",
                "comet_project": config.comet.project_name,
                "epochs": resolved_epochs,
                "batch_size": resolved_batch_size,
            },
        )
    except KeyboardInterrupt:
        logger.finish("interrupted")
        raise
    except BaseException as error:
        e_common._record_error(paths, spec, error)
        try:
            logger.log_error(error, stage="Y13")
        finally:
            logger.finish("error")
        raise
    if not paths.summary.is_file():
        raise RuntimeError("the delegated E311 run did not produce a summary")
    return paths.summary


def _build_y14_model(_spec: object, device: str) -> E311TwoNodeOddControlV1:
    result = build_y14_two_node_control_v1(
        dtype=torch.float32,
        device=device,
        seed=HISTORICAL_MODEL_SEED,
    )
    if _parameter_count(result) != HISTORICAL_E311_PARAMETER_COUNT:
        raise RuntimeError("Y14 parameter count changed")
    return result


def _sum_abs(difference: Tensor) -> tuple[float, int]:
    return float(difference.abs().sum().detach().cpu()), difference.numel()


def _y14_selected_audit(
    *,
    model: nn.Module,
    data: e_common.TrainingData,
    split: e_common.SplitIndices,
    config: e_common.SweepConfig,
    device: str,
    target_scale: Tensor,
    **_unused: Any,
) -> dict[str, Any]:
    if not isinstance(model, E311TwoNodeOddControlV1):
        raise TypeError("Y14 audit received the wrong model")
    scale = float(target_scale.detach().cpu())
    totals = {
        "raw_forward": 0.0,
        "raw_reverse": 0.0,
        "odd_pair": 0.0,
        "even_leakage": 0.0,
        "direct_parity": 0.0,
    }
    count = 0
    max_net_force = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, int(split.test.numel()), config.micro_batch_size):
            selection = split.test[start : start + config.micro_batch_size]
            centers = data.centers[selection].to(device)
            frames = data.frames[selection].to(device)
            target = data.root_force[selection].to(device) / target_scale
            output = model.core_output(centers, frames)
            forward = output.raw_forward_world[..., 0, :]
            reverse = output.raw_reverse_world[..., 0, :]
            odd = output.normalized_pair_force_world[..., 0, :]
            direct = model.pair_kernel(centers, frames)
            pieces = {
                "raw_forward": forward - target,
                "raw_reverse": reverse + target,
                "odd_pair": odd - target,
                "even_leakage": 0.5 * (forward + reverse),
                "direct_parity": forward - direct,
            }
            for name, difference in pieces.items():
                value, piece_count = _sum_abs(difference)
                totals[name] += value
                if name == "odd_pair":
                    count += piece_count
            net = output.normalized_node_force_world.sum(dim=-2)
            max_net_force = max(
                max_net_force,
                float(net.abs().max().detach().cpu()),
            )
    model.train(was_training)
    return {
        "metric_unit": "kcal_per_mol_per_angstrom",
        "raw_forward_mae": totals["raw_forward"] * scale / count,
        "raw_reverse_mae": totals["raw_reverse"] * scale / count,
        "odd_pair_mae": totals["odd_pair"] * scale / count,
        "even_leakage_mae": totals["even_leakage"] * scale / count,
        "raw_forward_direct_call_mae": totals["direct_parity"] * scale / count,
        "maximum_normalized_net_force_component": max_net_force,
        "test_component_count": count,
        "two_directions_share_one_kernel_call": True,
        "running_rms_population": "both_endpoint_orientations",
    }


def run_y14_odd_graph_400k_v1(
    e_study_root: Path,
    output_directory: Path,
    device: str,
    comet_project: str = DEFAULT_COMET_PROJECT,
    comet_workspace: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> Path:
    """Run Y14 on the exact Y13 data and split using the common E runner."""

    control_spec, resolved_epochs, resolved_batch_size = _resolve_training_protocol(
        "Y14",
        epochs,
        batch_size,
    )
    e_study_root = e_study_root.resolve()
    split_directory = e_runner._shared_split_directory(e_study_root)
    split, split_manifest = e_common._load_split(split_directory)
    config = e_runner.make_config(3, study_root=e_study_root)
    if (
        config.expected_sample_count != control_spec.sample_count
        or config.epochs != control_spec.epochs
        or config.effective_batch_size != control_spec.graph_batch_size
        or config.micro_batch_size != control_spec.graph_batch_size
        or config.learning_rate != 0.003
        or config.weight_decay != 1.0e-4
        or config.scheduler_step_size != control_spec.scheduler_step_size
        or config.scheduler_gamma != 0.5
        or config.validation_every != 1
        or config.split_seed != HISTORICAL_SPLIT_SEED
        or config.model_seed != HISTORICAL_MODEL_SEED
        or config.shuffle_seed != HISTORICAL_SHUFFLE_SEED
        or config.split_fractions != (0.8, 0.1, 0.1)
        or config.dtype != "float32"
        or not config.enable_tf32
        or config.expected_dataset_revision != 3
        or config.expected_opls_version != "2.0.0"
    ):
        raise RuntimeError("Y14 no longer matches the historical E311 protocol")
    output_directory = output_directory.resolve()
    paths = e_common.TrialPaths.create(output_directory)
    if paths.summary.exists():
        raise FileExistsError(paths.summary)
    spec = replace(
        control_spec,
        epochs=resolved_epochs,
        graph_batch_size=resolved_batch_size,
    )
    config = replace(
        config,
        study_directory=output_directory.parent,
        epochs=resolved_epochs,
        effective_batch_size=resolved_batch_size,
        micro_batch_size=resolved_batch_size,
        comet=_formal_comet_config(
            comet_project,
            comet_workspace,
            "Y14",
        ),
    )
    experiment_name = _resolved_experiment_name(
        "Y14_E311_OddGraph_400K",
        control_spec,
        resolved_epochs,
        resolved_batch_size,
    )
    logger: Any = NullCometTrialLogger()
    try:
        logger = _create_strict_y_series_comet_logger(
            comet_path=paths.comet,
            model_id="Y14",
            experiment_name=experiment_name,
            project=config.comet.project_name,
            workspace=config.comet.workspace,
        )
        e_common.run_trial(
            config,
            spec,
            paths,
            split,
            split_manifest,
            logger,
            device=str(_resolve_device(device)),
            epochs=resolved_epochs,
            model_builder=_build_y14_model,
            selected_model_audit_hook=_y14_selected_audit,
            source_sha256=_source_sha256(),
            study_metadata={
                "series": "Y13_Y15_pair_controls_v1",
                "experiment_id": "Y14",
                "comet_project": config.comet.project_name,
                "shared_split_source": str(split_directory),
                "shared_split_manifest_hash": split_manifest["manifest_hash"],
                "only_trainable_module": "historical_E311",
                "running_rms_population": "both_endpoint_orientations",
                "epochs": resolved_epochs,
                "batch_size": resolved_batch_size,
                "train_labeled_pair_exposures": (
                    resolved_epochs * split.counts()["train"]
                ),
                "train_ordered_kernel_evaluations": (
                    2 * resolved_epochs * split.counts()["train"]
                ),
            },
        )
    except KeyboardInterrupt:
        logger.finish("interrupted")
        raise
    except BaseException as error:
        e_common._record_error(paths, spec, error)
        try:
            logger.log_error(error, stage="Y14")
        finally:
            logger.finish("error")
        raise
    if not paths.summary.is_file():
        raise RuntimeError("Y14 did not produce a summary")
    return paths.summary


@dataclass(frozen=True, slots=True)
class Y15NodeArraysV2:
    centers_world: np.ndarray
    frames_body_to_world: np.ndarray
    node_force_world: np.ndarray
    pair_index: np.ndarray
    group_id: np.ndarray
    records: Mapping[str, Any]

    @property
    def sample_count(self) -> int:
        return int(self.centers_world.shape[0])


@dataclass(frozen=True, slots=True)
class SplitIndicesV1:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
        }


def _deterministic_group_split(
    group_id: np.ndarray,
    seed: int,
) -> SplitIndicesV1:
    groups = np.asarray(group_id, dtype=np.int64)
    if groups.ndim != 1:
        raise ValueError("group_id must be one-dimensional")
    unique = np.unique(groups)
    if unique.size < 10:
        raise ValueError("at least ten independent groups are required")
    order = np.random.default_rng(seed).permutation(unique)
    train_count = math.floor(0.8 * len(order))
    validation_count = math.floor(0.1 * len(order))
    train_groups = order[:train_count]
    validation_groups = order[train_count : train_count + validation_count]
    test_groups = order[train_count + validation_count :]
    result = SplitIndicesV1(
        np.flatnonzero(np.isin(groups, train_groups)).astype(np.int64),
        np.flatnonzero(np.isin(groups, validation_groups)).astype(np.int64),
        np.flatnonzero(np.isin(groups, test_groups)).astype(np.int64),
    )
    selected = (
        set(groups[result.train].tolist()),
        set(groups[result.validation].tolist()),
        set(groups[result.test].tolist()),
    )
    if (
        selected[0] & selected[1]
        or selected[0] & selected[2]
        or selected[1] & selected[2]
    ):
        raise RuntimeError("group split leaked configurations")
    combined = np.concatenate((result.train, result.validation, result.test))
    if not np.array_equal(np.sort(combined), np.arange(len(groups))):
        raise RuntimeError("split is not a complete partition")
    return result


def load_y15_node_arrays_v2(
    csv_path: Path,
    *,
    expected_sample_count: int = 100_000,
) -> Y15NodeArraysV2:
    """Load five molecule geometries and molecular force labels."""

    csv_path = csv_path.resolve()
    csv_metadata_path = csv_path.with_suffix(".json")
    csv_validation_path = csv_path.with_suffix(".validation.json")
    for path in (
        csv_path,
        csv_metadata_path,
        csv_validation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    csv_sha = _sha256(csv_path)
    csv_metadata = json.loads(csv_metadata_path.read_text(encoding="utf_8"))
    csv_validation = json.loads(csv_validation_path.read_text(encoding="utf_8"))
    csv_generator = (
        REPOSITORY_ROOT / "experiments" / "gnn" / "data" / "five_benzene.py"
    )
    if (
        csv_metadata.get("schema_name") != "tfenn_five_benzene_gnn_dataset"
        or csv_metadata.get("schema_version") != 1
        or csv_metadata.get("dataset_revision") != 1
        or csv_metadata.get("sample_count") != expected_sample_count
        or csv_metadata.get("molecule_count") != 5
        or csv_metadata.get("row_count") != expected_sample_count * 5
        or csv_metadata.get("rows_per_sample") != 5
        or csv_metadata.get("csv_sha256") != csv_sha
        or csv_metadata.get("generator_sha256") != _sha256(csv_generator)
        or csv_metadata.get("opls", {}).get("runtime_version")
        != Y15_OPLS_VERSION
        or csv_metadata.get("opls", {}).get("source_commit")
        != Y15_OPLS_COMMIT
        or csv_metadata.get("coordinate_convention", {}).get("frame")
        != "single_root_molecule_0_body_frame"
        or csv_metadata.get("coordinate_convention", {}).get(
            "rotation_semantics"
        )
        != "active_body_to_root_frame"
        or csv_metadata.get("units", {}).get("force")
        != "kcal_per_mol_per_angstrom"
        or csv_metadata.get("statistics", {}).get("unique_configuration_count")
        != expected_sample_count
    ):
        raise ValueError("five-benzene CSV metadata hash mismatch")
    if (
        csv_validation.get("passed") is not True
        or csv_validation.get("csv_sha256") != csv_sha
    ):
        raise ValueError("five-benzene CSV validation is missing or stale")
    arrays = load_benzene_cluster_csv(csv_path)
    if arrays.molecule_count != 5 or len(arrays) != expected_sample_count:
        raise ValueError("Y15 requires exactly 100k five-benzene configurations")
    node_force = np.asarray(arrays.forces, dtype=np.float64)
    if node_force.shape != (expected_sample_count, 5, 3):
        raise ValueError("Y15 molecular force array has an unexpected shape")
    if not np.isfinite(node_force).all():
        raise ValueError("Y15 molecular force labels contain nonfinite values")
    group_id = np.arange(expected_sample_count, dtype=np.int64)
    pair_index = complete_pair_index_v1(5).cpu().numpy()
    maximum_net_force_component = float(
        np.max(np.abs(node_force.sum(axis=1)))
    )
    return Y15NodeArraysV2(
        centers_world=np.ascontiguousarray(arrays.centers, dtype=np.float32),
        frames_body_to_world=np.ascontiguousarray(
            arrays.rotations,
            dtype=np.float32,
        ),
        node_force_world=np.ascontiguousarray(node_force, dtype=np.float32),
        pair_index=np.ascontiguousarray(pair_index, dtype=np.int64),
        group_id=np.ascontiguousarray(group_id, dtype=np.int64),
        records={
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha,
            "csv_metadata_sha256": _sha256(csv_metadata_path),
            "csv_validation_sha256": _sha256(csv_validation_path),
            "sample_count": expected_sample_count,
            "molecule_count": 5,
            "supervision_target": "five_molecule_force_world",
            "maximum_target_net_force_component": maximum_net_force_component,
            "group_source": "one_independent_group_per_configuration",
        },
    )


def _make_loader(
    dataset: TensorDataset,
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    subset = Subset(dataset, indices.tolist())
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _move_batch(
    batch: Sequence[Tensor],
    device: torch.device,
) -> tuple[Tensor, ...]:
    return tuple(
        item.to(device=device, non_blocking=device.type == "cuda")
        for item in batch
    )


def _evaluate_y15_node_force_loss_v2(
    model: E311OddGraphCoreV1,
    loader: DataLoader[Any],
    pair_index: Tensor,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    squared_sum = 0.0
    component_count = 0
    with torch.no_grad():
        for batch in loader:
            centers, frames, node_target = _move_batch(batch, device)
            output = model.core_output(centers, frames, pair_index)
            difference = output.normalized_node_force_world - node_target
            squared_sum += float(difference.square().sum().cpu())
            component_count += difference.numel()
    model.train(was_training)
    return squared_sum / component_count


def _save_y15_node_force_checkpoint_v2(
    path: Path,
    model: E311OddGraphCoreV1,
    *,
    epoch: int,
    validation_loss: float,
    target_scale: float,
    source_sha256: str,
) -> None:
    e_common._atomic_torch_save(
        path,
        {
            "schema_name": "tfenn_y15_node_force_selected_checkpoint",
            "schema_version": 2,
            "epoch": epoch,
            "validation_normalized_node_force_mse": validation_loss,
            "node_force_target_scale_component_rms": target_scale,
            "source_sha256": source_sha256,
            "parameter_state_dict": e_common._parameter_state(model),
            "normalization_state_dict": e_common._normalization_state(model),
            "calibration_state_dict": e_common._calibration_state(model),
        },
    )


def _restore_y15_node_force_checkpoint_v2(
    path: Path,
    model: E311OddGraphCoreV1,
    *,
    source_sha256: str,
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_name")
        != "tfenn_y15_node_force_selected_checkpoint"
        or payload.get("schema_version") != 2
    ):
        raise RuntimeError("Y15 checkpoint does not use node force supervision")
    if payload.get("source_sha256") != source_sha256:
        raise RuntimeError("Y15 checkpoint source hash changed")
    e_common._restore_model_state(
        model,
        payload["parameter_state_dict"],
        payload["normalization_state_dict"],
        payload.get("calibration_state_dict"),
    )
    return payload


def _selected_y15_node_force_metrics_v2(
    model: E311OddGraphCoreV1,
    loader: DataLoader[Any],
    pair_index: Tensor,
    device: torch.device,
    target_scale: float,
) -> dict[str, Any]:
    totals = {
        "node_abs": 0.0,
        "node_square": 0.0,
    }
    node_components = 0
    max_net_force = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            centers, frames, node_target_normalized = _move_batch(batch, device)
            output = model.core_output(centers, frames, pair_index)
            node_prediction = output.normalized_node_force_world * target_scale
            node_target = node_target_normalized * target_scale
            node_difference = node_prediction - node_target
            totals["node_abs"] += float(node_difference.abs().sum().cpu())
            totals["node_square"] += float(node_difference.square().sum().cpu())
            node_components += node_difference.numel()
            net = node_prediction.sum(dim=-2)
            max_net_force = max(
                max_net_force,
                float(net.abs().max().cpu()),
            )
    model.train(was_training)
    return {
        "molecular_force": {
            "mae": totals["node_abs"] / node_components,
            "sae": totals["node_abs"],
            "rmse": math.sqrt(totals["node_square"] / node_components),
            "normalized_mse": (
                totals["node_square"]
                / node_components
                / (target_scale * target_scale)
            ),
            "component_count": node_components,
        },
        "maximum_net_force_component": max_net_force,
        "unit": "kcal_per_mol_per_angstrom",
    }


def _tensor_residual_v1(actual: Tensor, expected: Tensor) -> dict[str, float]:
    difference = actual - expected
    difference_l2 = torch.linalg.vector_norm(difference)
    expected_l2 = torch.linalg.vector_norm(expected)
    return {
        "maximum_absolute": float(difference.abs().max().cpu()),
        "relative_l2": float(
            (difference_l2 / expected_l2.clamp_min(1.0e-12)).cpu()
        ),
    }


def _selected_y15_symmetry_audit_v1(
    model: E311OddGraphCoreV1,
    loader: DataLoader[Any],
    pair_index: Tensor,
    device: torch.device,
    *,
    tolerance: float = 1.0e-4,
) -> dict[str, Any]:
    """Audit the selected graph model once, without changing its statistics."""

    try:
        batch = next(iter(loader))
    except StopIteration as error:
        raise RuntimeError("Y15 test loader is empty") from error
    centers, frames, _node_target = _move_batch(batch, device)
    audit_count = min(4, int(centers.shape[0]))
    centers = centers[:audit_count]
    frames = frames[:audit_count]
    was_training = model.training
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    model.eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            reference = model.core_output(centers, frames, pair_index)

            angle = torch.tensor(0.37, device=device, dtype=centers.dtype)
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            zero = torch.zeros((), device=device, dtype=centers.dtype)
            identity = torch.eye(3, device=device, dtype=centers.dtype)
            axis = torch.tensor(
                (0.31, -0.47, 0.82),
                device=device,
                dtype=centers.dtype,
            )
            axis = axis / torch.linalg.vector_norm(axis)
            axis_x, axis_y, axis_z = axis.unbind()
            cross_matrix = torch.stack(
                (
                    torch.stack((zero, -axis_z, axis_y)),
                    torch.stack((axis_z, zero, -axis_x)),
                    torch.stack((-axis_y, axis_x, zero)),
                )
            )
            rotation = (
                cosine * identity
                + (1.0 - cosine) * torch.outer(axis, axis)
                + sine * cross_matrix
            )
            translation = torch.tensor(
                (0.41, -0.73, 1.17),
                device=device,
                dtype=centers.dtype,
            )
            transformed_centers = torch.einsum(
                "ij,bnj->bni",
                rotation,
                centers,
            ) + translation
            transformed_frames = torch.einsum(
                "ij,bnjk->bnik",
                rotation,
                frames,
            )
            transformed = model.core_output(
                transformed_centers,
                transformed_frames,
                pair_index,
            )
            expected_transformed_node = torch.einsum(
                "ij,bnj->bni",
                rotation,
                reference.normalized_node_force_world,
            )

            generators = e_common._proper_d6_generators().to(
                device=device,
                dtype=centers.dtype,
            )
            gauges = torch.stack(
                (
                    identity,
                    generators[0],
                    generators[1],
                    generators[0] @ generators[0],
                    generators[0] @ generators[1],
                )
            )
            gauged_frames = torch.einsum("bnij,njk->bnik", frames, gauges)
            gauged = model.core_output(centers, gauged_frames, pair_index)

            permutation = torch.tensor(
                (2, 4, 0, 3, 1),
                device=device,
                dtype=torch.int64,
            )
            permuted = model.core_output(
                centers.index_select(-2, permutation),
                frames.index_select(-3, permutation),
                pair_index,
            )
            expected_permuted_node = (
                reference.normalized_node_force_world.index_select(
                    -2,
                    permutation,
                )
            )

            zero_net_force = torch.zeros_like(
                reference.normalized_node_force_world.sum(dim=-2)
            )

        residuals = {
            "global_se3_node_force": _tensor_residual_v1(
                transformed.normalized_node_force_world,
                expected_transformed_node,
            ),
            "independent_d6_gauge_node_force": _tensor_residual_v1(
                gauged.normalized_node_force_world,
                reference.normalized_node_force_world,
            ),
            "node_permutation": _tensor_residual_v1(
                permuted.normalized_node_force_world,
                expected_permuted_node,
            ),
            "zero_total_force": _tensor_residual_v1(
                reference.normalized_node_force_world.sum(dim=-2),
                zero_net_force,
            ),
        }
        return {
            "passed": all(
                record["maximum_absolute"] <= tolerance
                for record in residuals.values()
            ),
            "absolute_tolerance": tolerance,
            "sample_count": audit_count,
            "tf32_disabled_during_audit": device.type == "cuda",
            "residuals": residuals,
        }
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        model.train(was_training)


def _run_y15_odd_graph_5b100k_node_force_v2(
    csv_path: Path,
    output_directory: Path,
    device_value: str,
    comet_logger: Any,
    epochs: int,
    batch_size: int,
) -> Path:
    """Train the shared E311 OddGraph on 100k five-benzene configurations."""

    control_spec = get_y_pair_control_spec_v1("Y15")
    spec = replace(
        control_spec,
        epochs=int(epochs),
        graph_batch_size=int(batch_size),
    )
    device = _resolve_device(device_value)
    output_directory = output_directory.resolve()
    summary_path = output_directory / "summary.json"
    history_path = output_directory / "history.csv"
    status_path = output_directory / "status.json"
    best_path = output_directory / "best.pt"
    final_path = output_directory / "final.pt"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    source_sha = _source_sha256()
    arrays = load_y15_node_arrays_v2(
        csv_path,
        expected_sample_count=spec.sample_count,
    )
    split = _deterministic_group_split(arrays.group_id, HISTORICAL_SPLIT_SEED)
    if split.counts() != {"train": 80_000, "validation": 10_000, "test": 10_000}:
        raise RuntimeError("Y15 formal split counts changed")
    target_scale = float(
        np.sqrt(np.mean(np.square(arrays.node_force_world[split.train])))
    )
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise RuntimeError("Y15 target RMS is invalid")
    dataset = TensorDataset(
        torch.from_numpy(arrays.centers_world),
        torch.from_numpy(arrays.frames_body_to_world),
        torch.from_numpy(arrays.node_force_world / target_scale),
    )
    pin_memory = device.type == "cuda"
    train_loader = _make_loader(
        dataset,
        split.train,
        spec.graph_batch_size,
        shuffle=True,
        seed=HISTORICAL_SHUFFLE_SEED,
        pin_memory=pin_memory,
    )
    warm_loader = _make_loader(
        dataset,
        split.train,
        spec.graph_batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
    )
    validation_loader = _make_loader(
        dataset,
        split.validation,
        spec.graph_batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
    )
    test_loader = _make_loader(
        dataset,
        split.test,
        spec.graph_batch_size,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
    )

    _set_seed(HISTORICAL_MODEL_SEED)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = Y15_TF32
        torch.backends.cudnn.allow_tf32 = Y15_TF32
    model = build_e311_odd_graph_core_v1(
        dtype=torch.float32,
        device=device,
        seed=HISTORICAL_MODEL_SEED,
    )
    if _parameter_count(model) != HISTORICAL_E311_PARAMETER_COUNT:
        raise RuntimeError("Y15 parameter count changed")
    comet_logger.log_config(
        study_config={
            "study_name": output_directory.parent.name,
            "series": "Y13_Y15_node_force_control_v2",
            "experiment_id": "Y15",
            "data": dict(arrays.records),
            "source_sha256": source_sha,
        },
        trial_config={
            "experiment": spec.as_dict(),
            "output_directory": str(output_directory),
        },
        parameters={
            "compiled_parameter_count": _parameter_count(model),
            "optimizer": "AdamW",
            "learning_rate": Y15_LEARNING_RATE,
            "weight_decay": Y15_WEIGHT_DECAY,
            "scheduler": "StepLR",
            "scheduler_step_size": spec.scheduler_step_size,
            "scheduler_gamma": Y15_SCHEDULER_GAMMA,
            "epochs": spec.epochs,
            "graph_batch_size": spec.graph_batch_size,
            "split_seed": HISTORICAL_SPLIT_SEED,
            "model_seed": HISTORICAL_MODEL_SEED,
            "shuffle_seed": HISTORICAL_SHUFFLE_SEED,
            "dtype": "float32",
            "enable_tf32_during_training": Y15_TF32,
            "node_force_target_scale_component_rms": target_scale,
            "training_supervision": "five_molecule_force_world",
            "validation_target": "five_molecule_force_world",
            "test_target": "five_molecule_force_world",
            "pair_force_labels_loaded": False,
        },
    )
    pair_index = torch.from_numpy(arrays.pair_index).to(device)
    model.reset_normalization_stats()
    model.train()
    with torch.no_grad():
        for batch in warm_loader:
            centers, frames, _node_target = _move_batch(batch, device)
            model.core_output(centers, frames, pair_index)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Y15_LEARNING_RATE,
        weight_decay=Y15_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=spec.scheduler_step_size,
        gamma=Y15_SCHEDULER_GAMMA,
    )
    best_epoch = 0
    best_validation = _evaluate_y15_node_force_loss_v2(
        model,
        validation_loader,
        pair_index,
        device,
    )
    _save_y15_node_force_checkpoint_v2(
        best_path,
        model,
        epoch=best_epoch,
        validation_loss=best_validation,
        target_scale=target_scale,
        source_sha256=source_sha,
    )
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "learning_rate": Y15_LEARNING_RATE,
            "train_normalized_node_force_mse": "",
            "validation_normalized_node_force_mse": best_validation,
            "epoch_duration_seconds": 0.0,
        }
    ]
    _write_history_atomic(history_path, history)
    _write_json_atomic(
        status_path,
        {
            "status": "running",
            "experiment_id": "Y15",
            "epoch": 0,
            "epochs": spec.epochs,
            "best_epoch": 0,
            "best_validation_normalized_node_force_mse": best_validation,
        },
    )

    global_step = 0
    for epoch in range(1, spec.epochs + 1):
        started = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        squared_sum = 0.0
        component_count = 0
        for batch in train_loader:
            centers, frames, node_target = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model.core_output(centers, frames, pair_index)
            prediction = output.normalized_node_force_world
            loss = torch.nn.functional.mse_loss(prediction, node_target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Y15 training loss became nonfinite")
            loss.backward()
            global_step += 1
            if global_step <= 4 or global_step % 100 == 0:
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                if any(
                    gradient is None or not bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                ):
                    raise RuntimeError(
                        "Y15 encountered a missing or nonfinite gradient"
                    )
            optimizer.step()
            difference = prediction.detach() - node_target
            squared_sum += float(difference.square().sum().cpu())
            component_count += difference.numel()
        scheduler.step()
        train_loss = squared_sum / component_count
        validation_loss = _evaluate_y15_node_force_loss_v2(
            model,
            validation_loader,
            pair_index,
            device,
        )
        if validation_loss < best_validation:
            best_epoch = epoch
            best_validation = validation_loss
            _save_y15_node_force_checkpoint_v2(
                best_path,
                model,
                epoch=epoch,
                validation_loss=validation_loss,
                target_scale=target_scale,
                source_sha256=source_sha,
            )
        epoch_duration = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_normalized_node_force_mse": train_loss,
            "validation_normalized_node_force_mse": validation_loss,
            "epoch_duration_seconds": epoch_duration,
        }
        history.append(row)
        _write_history_atomic(history_path, history)
        _write_json_atomic(
            status_path,
            {
                "status": "running",
                "experiment_id": "Y15",
                "epoch": epoch,
                "epochs": spec.epochs,
                "best_epoch": best_epoch,
                "best_validation_normalized_node_force_mse": best_validation,
            },
        )
        comet_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            extra_metrics={"epoch_duration_seconds": epoch_duration},
        )
        _update_strict_comet_epoch_record(output_directory / "comet.json", epoch)
        print(json.dumps({"experiment_id": "Y15", **row}), flush=True)

    selected = _restore_y15_node_force_checkpoint_v2(
        best_path,
        model,
        source_sha256=source_sha,
    )
    selected_metrics = _selected_y15_node_force_metrics_v2(
        model,
        test_loader,
        pair_index,
        device,
        target_scale,
    )
    selected_symmetry_audit = _selected_y15_symmetry_audit_v1(
        model,
        test_loader,
        pair_index,
        device,
    )
    e_common._atomic_torch_save(final_path, dict(selected))
    train_graphs = split.counts()["train"]
    optimizer_updates = spec.epochs * math.ceil(
        train_graphs / spec.graph_batch_size
    )
    if global_step != optimizer_updates:
        raise RuntimeError("Y15 optimizer update count changed")
    summary = {
        "schema_name": "tfenn_y15_e311_odd_graph_node_force_result",
        "schema_version": 2,
        "status": "complete",
        "experiment": spec.as_dict(),
        "model": {
            "parameter_count": _parameter_count(model),
            "architecture": dict(model.architecture_metadata),
            "pair_kernel": HISTORICAL_E311_MODEL_ID,
        },
        "protocol": {
            "epochs": spec.epochs,
            "graph_batch_size": spec.graph_batch_size,
            "unordered_edges_per_graph": spec.unordered_edge_count,
            "ordered_kernel_evaluations_per_graph": 2
            * spec.unordered_edge_count,
            "learning_rate": Y15_LEARNING_RATE,
            "weight_decay": Y15_WEIGHT_DECAY,
            "scheduler": "StepLR",
            "scheduler_step_size": spec.scheduler_step_size,
            "scheduler_gamma": Y15_SCHEDULER_GAMMA,
            "node_force_target_scale_component_rms": target_scale,
            "training_supervision": "five_molecule_force_world",
            "validation_target": "five_molecule_force_world",
            "test_target": "five_molecule_force_world",
            "pair_force_labels_loaded": False,
            "dtype": "float32",
            "enable_tf32_during_training": Y15_TF32,
            "torch_threads": 4,
            "split_seed": HISTORICAL_SPLIT_SEED,
            "model_seed": HISTORICAL_MODEL_SEED,
            "shuffle_seed": HISTORICAL_SHUFFLE_SEED,
            "optimizer_updates": optimizer_updates,
            "train_graph_exposures": spec.epochs * train_graphs,
            "train_ordered_kernel_evaluations": (
                2 * spec.epochs * train_graphs * spec.unordered_edge_count
            ),
            "running_rms_population": "both_endpoint_orientations",
        },
        "split": split.counts(),
        "selected_checkpoint": {
            "rule": "minimum validation normalized molecular force MSE",
            "best_epoch": int(selected["epoch"]),
            "best_validation_normalized_node_force_mse": float(
                selected["validation_normalized_node_force_mse"]
            ),
            "test_metrics_evaluated_once": True,
            "symmetry_audit_reuses_four_unlabeled_test_geometries": True,
        },
        "selected_test": selected_metrics,
        "selected_symmetry_audit": selected_symmetry_audit,
        "data": dict(arrays.records),
        "source_sha256": source_sha,
        "git_commit": _git_commit(),
        "comet": comet_logger.identity,
        "artifacts": {
            "history": str(history_path),
            "history_sha256": _sha256(history_path),
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": _sha256(best_path),
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": _sha256(final_path),
        },
        "completed_at_utc": _utc_now(),
    }
    comet_logger.log_final(
        metrics={
            "test": {
                "mae": selected_metrics["molecular_force"]["mae"],
                "sae": selected_metrics["molecular_force"]["sae"],
            }
        },
        relative_force_norm_stats={},
        summary=summary,
    )
    comet_logger.finish("complete")
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(
        status_path,
        {
            "status": "complete",
            "experiment_id": "Y15",
            "epoch": spec.epochs,
            "best_epoch": best_epoch,
            "best_validation_normalized_node_force_mse": best_validation,
            "completed_at_utc": _utc_now(),
        },
    )
    return summary_path


def run_y15_odd_graph_5b100k_node_force_v2(
    csv_path: Path,
    output_directory: Path,
    device_value: str,
    comet_project: str = DEFAULT_COMET_PROJECT,
    comet_workspace: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> Path:
    """Run formal Y15 with the strict Y-series Comet metric contract."""

    control_spec, resolved_epochs, resolved_batch_size = _resolve_training_protocol(
        "Y15",
        epochs,
        batch_size,
    )
    output_directory = output_directory.resolve()
    summary_path = output_directory / "summary.json"
    status_path = output_directory / "status.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    experiment_name = _resolved_experiment_name(
        "Y15_E311_OddGraph_5B100K_NodeForce",
        control_spec,
        resolved_epochs,
        resolved_batch_size,
    )
    logger: Any = NullCometTrialLogger()
    try:
        logger = _create_strict_y_series_comet_logger(
            comet_path=output_directory / "comet.json",
            model_id="Y15",
            experiment_name=experiment_name,
            project=_formal_comet_config(
                comet_project,
                comet_workspace,
                "Y15",
            ).project_name,
            workspace=comet_workspace,
        )
        return _run_y15_odd_graph_5b100k_node_force_v2(
            csv_path,
            output_directory,
            device_value,
            logger,
            resolved_epochs,
            resolved_batch_size,
        )
    except KeyboardInterrupt:
        current = (
            json.loads(status_path.read_text(encoding="utf_8"))
            if status_path.is_file()
            else {}
        )
        _write_json_atomic(
            status_path,
            {
                **current,
                "status": "interrupted",
                "experiment_id": "Y15",
                "updated_at_utc": _utc_now(),
            },
        )
        logger.finish("interrupted")
        raise
    except BaseException as error:
        current = (
            json.loads(status_path.read_text(encoding="utf_8"))
            if status_path.is_file()
            else {}
        )
        _write_json_atomic(
            status_path,
            {
                **current,
                "status": "error",
                "experiment_id": "Y15",
                "exception_type": type(error).__name__,
                "updated_at_utc": _utc_now(),
            },
        )
        try:
            logger.log_error(error, stage="Y15")
        finally:
            logger.finish("error")
        raise


def build_argument_parser_v1() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    y13 = commands.add_parser("y13", help="exact historical E311 reproduction")
    y13.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "Y13_exact_e311_400k",
    )
    y13.add_argument("--output-directory", type=Path)
    y13.add_argument("--device", default="cuda")

    y14 = commands.add_parser("y14", help="two-node E311 OddGraph control")
    y14.add_argument("--e-study-root", type=Path, required=True)
    y14.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "Y14_e311_odd_graph_400k",
    )
    y14.add_argument("--device", default="cuda")

    y15 = commands.add_parser("y15", help="five molecule force E311 OddGraph")
    y15.add_argument("--csv", type=Path, required=True)
    y15.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "Y15_e311_odd_graph_5b100k_node_force",
    )
    y15.add_argument("--device", default="cuda")
    for command in (y13, y14, y15):
        command.add_argument("--epochs", type=int)
        command.add_argument("--batch-size", type=int)
        command.add_argument("--comet-project", default=DEFAULT_COMET_PROJECT)
        command.add_argument("--comet-workspace")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser_v1().parse_args(argv)
    if arguments.command == "y13":
        result = run_y13_exact_reproduction_v1(
            arguments.study_root,
            arguments.device,
            arguments.comet_project,
            arguments.comet_workspace,
            output_directory=arguments.output_directory,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
        )
    elif arguments.command == "y14":
        result = run_y14_odd_graph_400k_v1(
            arguments.e_study_root,
            arguments.output_directory,
            arguments.device,
            arguments.comet_project,
            arguments.comet_workspace,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
        )
    else:
        result = run_y15_odd_graph_5b100k_node_force_v2(
            arguments.csv,
            arguments.output_directory,
            arguments.device,
            arguments.comet_project,
            arguments.comet_workspace,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
        )
    print(json.dumps({"status": "complete", "result": str(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COMET_PROJECT",
    "SplitIndicesV1",
    "Y15NodeArraysV2",
    "build_argument_parser_v1",
    "load_y15_node_arrays_v2",
    "main",
    "run_y13_exact_reproduction_v1",
    "run_y14_odd_graph_400k_v1",
    "run_y15_odd_graph_5b100k_node_force_v2",
]
