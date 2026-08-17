"""Run the three D series benzene pair studies with one shared training core."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from experiments.benzene_pair import sweep30 as common
from experiments.benzene_pair.comet_logging import NullCometTrialLogger
from experiments.benzene_pair.d_series.catalog import (
    get_experiment_specs,
    get_model_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "experiments" / "benzene_pair" / "runs" / "d_series_400k_v1"
)
CALIBRATED_MODEL_IDS = frozenset({"D63", "D67", "D68", "D69"})
D_STUDY_METADATA = {
    "concurrent_run": True,
    "shared_gpu_process_count": 3,
}
DEFAULT_CONFIG_PATHS = {
    experiment_id: Path(__file__).resolve().parent / f"experiment_{experiment_id}.json"
    for experiment_id in (1, 2, 3)
}


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    experiment_id: int
    directory_name: str
    comet_project: str
    purpose: str


EXPERIMENTS = {
    1: ExperimentDefinition(
        1,
        "experiment_1_dense_residual",
        "tfenn_d_series_experiment_1_dense_bypass",
        "dense bypass and typed residual necessity",
    ),
    2: ExperimentDefinition(
        2,
        "experiment_2_architecture_paths",
        "tfenn_d_series_experiment_2_architecture_paths",
        "typed ordering path family and polynomial degree",
    ),
    3: ExperimentDefinition(
        3,
        "experiment_3_invariant_gate",
        "tfenn_d_series_experiment_3_invariant_gate",
        "invariant gate activation descriptor and capacity",
    ),
}


def _experiment(value: int) -> ExperimentDefinition:
    try:
        return EXPERIMENTS[int(value)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("experiment must be one two or three") from error


def _shared_split_directory(study_root: str | Path) -> Path:
    return Path(study_root).resolve() / "shared_split"


def make_config(
    experiment_id: int,
    *,
    study_root: str | Path = DEFAULT_STUDY_ROOT,
) -> common.SweepConfig:
    """Build the fixed four hundred thousand sample D series protocol."""
    experiment = _experiment(experiment_id)
    root = Path(study_root).resolve()
    path = DEFAULT_CONFIG_PATHS[experiment_id]
    value = json.loads(path.read_text(encoding="utf_8"))
    if value.get("schema_name") != "tfenn_benzene_pair_d_series":
        raise ValueError("unexpected D series config schema")
    if value.get("schema_version") != 1:
        raise ValueError("unexpected D series config version")
    if int(value.get("experiment_id", 0)) != experiment_id:
        raise ValueError("D series config experiment does not match")
    if value.get("study_directory_name") != experiment.directory_name:
        raise ValueError("D series study directory name does not match")
    expected_model_ids = tuple(
        item.model_id for item in get_experiment_specs(experiment_id)
    )
    if tuple(value.get("model_ids", ())) != expected_model_ids:
        raise ValueError("D series config model ids do not match the catalog")
    for name, expected in D_STUDY_METADATA.items():
        if value.get(name) != expected:
            raise ValueError(f"D series config {name} does not match")
    config = common.SweepConfig(
        shard_paths=tuple(
            (REPOSITORY_ROOT / str(item)).resolve() for item in value["shard_paths"]
        ),
        study_directory=root / experiment.directory_name,
        epochs=int(value["epochs"]),
        effective_batch_size=int(value["effective_batch_size"]),
        micro_batch_size=int(value["micro_batch_size"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        scheduler_step_size=int(value["scheduler_step_size"]),
        scheduler_gamma=float(value["scheduler_gamma"]),
        validation_every=int(value["validation_every"]),
        split_seed=int(value["split_seed"]),
        model_seed=int(value["model_seed"]),
        shuffle_seed=int(value["shuffle_seed"]),
        split_fractions=tuple(float(item) for item in value["split_fractions"]),
        device=str(value["device"]),
        dtype=str(value["dtype"]),
        threads=int(value["threads"]),
        symmetry_tolerance=float(value["symmetry_tolerance"]),
        symmetry_probe_count=int(value["symmetry_probe_count"]),
        expected_sample_count=int(value["expected_sample_count"]),
        expected_dataset_revision=int(value["expected_dataset_revision"]),
        expected_opls_version=str(value["expected_opls_version"]),
        enable_tf32=bool(value["enable_tf32"]),
        relative_force_norm_sample_count=int(value["relative_force_norm_sample_count"]),
        relative_force_norm_seed=int(value["relative_force_norm_seed"]),
        comet=common.CometConfig.from_mapping(value["comet"]),
        schema_name=str(value["schema_name"]),
        schema_version=int(value["schema_version"]),
    )
    config.validate()
    if config.comet.project_name != experiment.comet_project:
        raise ValueError("D series Comet project does not match")
    return config


def _source_sha256(dependency_sha256: str | None = None) -> str:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "catalog.py",
        Path(__file__).resolve().parent / "model_factory.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "sweep30.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "comet_logging.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "metrics.py",
        REPOSITORY_ROOT / "experiments" / "benzene_pair" / "train.py",
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
    if dependency_sha256 is not None:
        digest.update(b"dependency_checkpoint_sha256")
        digest.update(dependency_sha256.encode("ascii"))
    return digest.hexdigest()


def _build_model(spec: Any, device: str) -> nn.Module:
    from experiments.benzene_pair.d_series.catalog import build_d_series_model

    model = build_d_series_model(
        spec,
        common._proper_d6_generators(),
        generator_names=("sixfold", "twofold"),
    )
    if not isinstance(model, nn.Module):
        raise TypeError("D series model builder must return a torch module")
    return model.to(device=torch.device(device), dtype=torch.float32)


def _descriptor_statistics(
    model: nn.Module,
    data: common.TrainingData,
    indices: Tensor,
    *,
    batch_size: int,
    device: str,
) -> dict[str, dict[str, Tensor | int]]:
    collector = getattr(model, "collect_normalized_descriptors", None)
    if not callable(collector):
        raise TypeError("descriptor calibration requires a descriptor collector")
    accumulators: dict[str, dict[str, Tensor | int]] = {}
    stage_names = tuple(stage.name for stage in model.config.stages)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, int(indices.numel()), batch_size):
                selection = indices[start : start + batch_size]
                centers, frames, _target = common._batch_inputs(data, selection, device)
                descriptors = collector(centers, frames)
                if tuple(descriptors) != stage_names:
                    raise RuntimeError("descriptor collector stage order changed")
                for stage_name, descriptor in descriptors.items():
                    if descriptor.ndim != 2:
                        raise ValueError(
                            "normalized descriptor must be a batch by feature tensor"
                        )
                    value = descriptor.detach().to(dtype=torch.float32)
                    dimension = int(value.shape[1])
                    if dimension > 4096:
                        raise RuntimeError(
                            f"descriptor dimension {dimension} exceeds calibration guard"
                        )
                    batch_sum = value.sum(dim=0).to(dtype=torch.float64, device="cpu")
                    batch_cross = (value.T @ value).to(
                        dtype=torch.float64, device="cpu"
                    )
                    current = accumulators.get(stage_name)
                    if current is None:
                        accumulators[stage_name] = {
                            "count": int(value.shape[0]),
                            "sum": batch_sum,
                            "cross": batch_cross,
                        }
                    else:
                        current["count"] = int(current["count"]) + int(value.shape[0])
                        current["sum"] = current["sum"] + batch_sum
                        current["cross"] = current["cross"] + batch_cross
    finally:
        model.train(was_training)
    if tuple(accumulators) != stage_names:
        raise RuntimeError("not every stage supplied descriptor statistics")
    return accumulators


def _radial_count(model: nn.Module) -> int:
    radial = model.config.radial
    return 2 + len(radial.rbf_centers) + len(radial.inverse_powers)


def _projection_components(
    model: nn.Module,
    statistics: Mapping[str, Mapping[str, Tensor | int]],
    *,
    policy: str,
    seed: int,
) -> tuple[dict[str, tuple[Tensor, Tensor | None]], dict[str, Any]]:
    radial_count = _radial_count(model)
    projections: dict[str, tuple[Tensor, Tensor | None]] = {}
    report: dict[str, Any] = {"policy": policy, "stages": {}}
    for stage_name, values in statistics.items():
        count = int(values["count"])
        total = values["sum"]
        cross = values["cross"]
        assert isinstance(total, Tensor) and isinstance(cross, Tensor)
        mean = total / count
        covariance = (cross / count) - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        dimension = int(mean.numel())
        if dimension < radial_count:
            raise RuntimeError("descriptor is smaller than its radial prefix")
        tail_dimension = dimension - radial_count
        radial = torch.eye(dimension, dtype=torch.float64)[:radial_count]
        tail_covariance = covariance[radial_count:, radial_count:]
        if tail_dimension:
            eigenvalues, eigenvectors = torch.linalg.eigh(tail_covariance)
            order = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[order].clamp_min(0.0)
            eigenvectors = eigenvectors[:, order]
        else:
            eigenvalues = torch.empty(0, dtype=torch.float64)
            eigenvectors = torch.empty((0, 0), dtype=torch.float64)
        if policy == "rank_revealing":
            maximum = float(eigenvalues.max()) if eigenvalues.numel() else 0.0
            tolerance = (
                maximum
                * max(count, max(tail_dimension, 1))
                * torch.finfo(torch.float32).eps
            )
            retained_tail = int((eigenvalues > tolerance).sum())
            tail_components = eigenvectors[:, :retained_tail].T
            projection_mean = mean.clone()
            projection_mean[:radial_count] = 0.0
        elif policy in {"pca_99", "random_orthogonal"}:
            total_variance = float(eigenvalues.sum())
            if total_variance <= 0.0:
                retained_tail = 0
            else:
                cumulative = torch.cumsum(eigenvalues, dim=0) / total_variance
                retained_tail = int(
                    torch.searchsorted(
                        cumulative,
                        torch.tensor(0.99, dtype=cumulative.dtype),
                    ).item()
                    + 1
                )
            if policy == "pca_99":
                tail_components = eigenvectors[:, :retained_tail].T
                projection_mean = mean.clone()
                projection_mean[:radial_count] = 0.0
            else:
                generator = torch.Generator().manual_seed(seed + len(projections))
                random = torch.randn(
                    tail_dimension,
                    retained_tail,
                    generator=generator,
                    dtype=torch.float64,
                )
                tail_components = (
                    torch.linalg.qr(random, mode="reduced").Q.T
                    if retained_tail
                    else torch.empty((0, tail_dimension), dtype=torch.float64)
                )
                projection_mean = None
        else:
            raise ValueError(f"unknown descriptor projection policy {policy}")
        padded_tail = torch.nn.functional.pad(
            tail_components,
            (radial_count, 0),
        )
        components = torch.cat((radial, padded_tail), dim=0)
        projections[stage_name] = (components, projection_mean)
        report["stages"][stage_name] = {
            "sample_count": count,
            "input_dimension": dimension,
            "radial_dimension": radial_count,
            "retained_dimension": int(components.shape[0]),
            "retained_tail_dimension": retained_tail,
            "retained_variance_fraction": 1.0
            if not eigenvalues.numel() or float(eigenvalues.sum()) <= 0.0
            else float(eigenvalues[:retained_tail].sum() / eigenvalues.sum()),
        }
    return projections, report


def _calibrate_model(**values: Any) -> Mapping[str, Any] | None:
    model: nn.Module = values["model"]
    spec = values["spec"]
    data: common.TrainingData = values["data"]
    split: common.SplitIndices = values["split"]
    config: common.SweepConfig = values["config"]
    paths: common.TrialPaths = values["paths"]
    device: str = values["device"]
    model_id = str(spec.model_id).upper()
    if model_id == "D63":
        dependency_path = paths.directory.parent / "D51" / "best.pt"
        if not dependency_path.is_file():
            raise FileNotFoundError(
                f"D63 requires the completed D51 checkpoint {dependency_path}"
            )
        dependency = torch.load(
            dependency_path,
            map_location="cpu",
            weights_only=True,
        )
        dense = _build_model(get_model_spec("D51"), device)
        common._restore_model_state(
            dense,
            dependency["parameter_state_dict"],
            dependency["normalization_state_dict"],
            dependency.get("calibration_state_dict"),
        )
        initializer = getattr(model, "initialize_coefficient_heads_from", None)
        if not callable(initializer):
            raise TypeError("D63 model does not support SVD initialization")
        initialized_head_count = int(initializer(dense))
        del dense
        common._warm_normalization(
            model,
            data,
            split.train,
            batch_size=config.micro_batch_size,
            device=device,
        )
        return {
            "kind": "truncated_svd_from_D51",
            "initialized_head_count": initialized_head_count,
            "dependency_checkpoint": str(dependency_path),
            "dependency_checkpoint_sha256": common.sha256_file(dependency_path),
        }
    policies = {
        "D67": "rank_revealing",
        "D68": "pca_99",
        "D69": "random_orthogonal",
    }
    policy = policies.get(model_id)
    if policy is None:
        return None
    setter = getattr(model, "set_descriptor_projection", None)
    if not callable(setter):
        raise TypeError(f"{model_id} does not support descriptor projection")
    if model_id == "D69":
        dependency_path = paths.directory.parent / "D68" / "summary.json"
        if not dependency_path.is_file():
            raise FileNotFoundError(
                f"D69 requires the completed D68 summary {dependency_path}"
            )
        dependency = common._load_json(dependency_path)
        stage_report = (
            dependency.get("training", {}).get("calibration", {}).get("stages", {})
        )
        transforms = getattr(model, "descriptor_transforms", None)
        if not isinstance(transforms, nn.ModuleDict):
            raise TypeError("D69 requires named descriptor transforms")
        radial_count = _radial_count(model)
        report: dict[str, Any] = {
            "policy": "random_orthogonal",
            "dependency_summary": str(dependency_path),
            "dependency_summary_sha256": common.sha256_file(dependency_path),
            "stages": {},
        }
        for stage_index, (stage_name, transform) in enumerate(transforms.items()):
            if stage_name not in stage_report:
                raise ValueError(f"D68 summary has no calibration for {stage_name}")
            dimension = int(transform.dimension)
            retained = int(stage_report[stage_name]["retained_dimension"])
            retained_tail = retained - radial_count
            if retained_tail < 0 or retained_tail > dimension - radial_count:
                raise ValueError("D68 retained descriptor rank is invalid")
            generator = torch.Generator().manual_seed(
                config.model_seed + 69 + stage_index
            )
            random = torch.randn(
                dimension - radial_count,
                retained_tail,
                generator=generator,
                dtype=torch.float64,
            )
            tail = (
                torch.linalg.qr(random, mode="reduced").Q.T
                if retained_tail
                else torch.empty((0, dimension - radial_count), dtype=torch.float64)
            )
            radial = torch.eye(dimension, dtype=torch.float64)[:radial_count]
            components = torch.cat(
                (radial, torch.nn.functional.pad(tail, (radial_count, 0))),
                dim=0,
            )
            setter(stage_name, components, mean=None)
            report["stages"][stage_name] = {
                "input_dimension": dimension,
                "radial_dimension": radial_count,
                "retained_dimension": retained,
                "retained_tail_dimension": retained_tail,
            }
        return report
    statistics = _descriptor_statistics(
        model,
        data,
        split.train,
        batch_size=config.micro_batch_size,
        device=device,
    )
    projections, report = _projection_components(
        model,
        statistics,
        policy=policy,
        seed=config.model_seed + int(model_id[1:]),
    )
    for stage_name, (components, mean) in projections.items():
        setter(stage_name, components, mean=mean)
    return report


def _dependency_sha256(spec: Any, paths: common.TrialPaths) -> str | None:
    dependencies = {"D63": ("D51", "best.pt"), "D69": ("D68", "summary.json")}
    dependency_spec = dependencies.get(str(spec.model_id).upper())
    if dependency_spec is None:
        return None
    dependency = paths.directory.parent / dependency_spec[0] / dependency_spec[1]
    return common.sha256_file(dependency) if dependency.is_file() else None


def _select_specs(experiment_id: int, values: Sequence[str]) -> tuple[Any, ...]:
    available = tuple(get_experiment_specs(experiment_id))
    if len(available) != 25:
        raise RuntimeError(
            "each D sub experiment must contain exactly twenty five models"
        )
    if not values:
        return available
    allowed = {item.model_id: item for item in available}
    selected = []
    for value in values:
        key = str(value).upper()
        if key not in allowed:
            raise ValueError(f"model {value} is outside experiment {experiment_id}")
        selected.append(allowed[key])
    if len({item.model_id for item in selected}) != len(selected):
        raise ValueError("model selection contains duplicates")
    return tuple(selected)


def _result_row(config: common.SweepConfig, spec: Any) -> dict[str, Any]:
    return common._result_row(config, spec)


def _refresh_results(config: common.SweepConfig, specs: Sequence[Any]) -> Path:
    config.study_directory.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(config, spec) for spec in specs]
    path = config.study_directory / "results.csv"
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with partial.open("w", encoding="utf_8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=common.RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)
    completed = [row for row in rows if row["status"] == "complete"]
    ranking = sorted(
        completed,
        key=lambda row: float(row["best_validation_normalized_mse"]),
    )
    common._atomic_json(
        config.study_directory / "comparison.json",
        {
            "schema_name": "tfenn_d_series_comparison",
            "schema_version": 1,
            "completed_model_count": len(completed),
            "error_model_count": sum(row["status"] == "error" for row in rows),
            "ranking_by_validation": ranking,
            "updated_at_utc": common._utc_now(),
        },
    )
    return path


def run_study(arguments: argparse.Namespace) -> int:
    experiment = _experiment(arguments.experiment)
    config = make_config(experiment.experiment_id, study_root=arguments.study_root)
    if not os.environ.get("COMET_API_KEY", "").strip():
        raise RuntimeError("COMET_API_KEY must be set for a formal D series run")
    device = common._resolve_device(arguments.device or config.device)
    all_specs = tuple(get_experiment_specs(experiment.experiment_id))
    selected = _select_specs(experiment.experiment_id, arguments.model)
    split_config = replace(
        config,
        study_directory=_shared_split_directory(arguments.study_root),
    )
    split_manifest = common._prepare_split(split_config)
    manifest = {
        "schema_name": "tfenn_d_series_study",
        "schema_version": 1,
        "experiment_id": experiment.experiment_id,
        "experiment_purpose": experiment.purpose,
        "model_count": len(all_specs),
        "models": [item.as_dict() for item in all_specs],
        "config": config.as_dict(device=device),
        "shared_split_directory": str(_shared_split_directory(arguments.study_root)),
        "split_manifest_hash": split_manifest["manifest_hash"],
        "source_sha256": _source_sha256(),
        **D_STUDY_METADATA,
    }
    manifest["study_hash"] = common._canonical_sha256(manifest)
    manifest_path = config.study_directory / "manifest.json"
    if manifest_path.is_file() and common._load_json(manifest_path) != manifest:
        raise RuntimeError("existing D series manifest does not match this run")
    common._atomic_json(manifest_path, manifest)
    _refresh_results(config, all_specs)
    for spec in selected:
        paths = common.TrialPaths.create(
            config.study_directory / "models" / spec.model_id
        )
        if paths.summary.is_file():
            completed = common._load_json(paths.summary)
            dependency_sha = _dependency_sha256(spec, paths)
            expected_hash = common._trial_hash(
                config,
                spec,
                split_manifest,
                device=device,
                epochs=config.epochs,
                source_sha256=_source_sha256(dependency_sha),
                study_metadata=D_STUDY_METADATA,
            )
            if completed.get("status") != "complete":
                raise RuntimeError("existing D trial summary is not complete")
            if completed.get("trial_hash") != expected_hash:
                raise RuntimeError("existing D trial summary hash does not match")
            if not paths.best.is_file() or not paths.final.is_file():
                raise RuntimeError("completed D trial is missing a checkpoint")
            continue
        paths.directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.d_series.runner",
            "trial",
            "--experiment",
            str(experiment.experiment_id),
            "--study_root",
            str(Path(arguments.study_root).resolve()),
            "--model",
            spec.model_id,
            "--device",
            device,
        ]
        with (
            paths.stdout.open("a", encoding="utf_8") as stdout,
            paths.stderr.open("a", encoding="utf_8") as stderr,
        ):
            process = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        _refresh_results(config, all_specs)
        if process.returncode == 130:
            return 130
    _refresh_results(config, all_specs)
    return 0


def run_trial_command(arguments: argparse.Namespace) -> int:
    experiment = _experiment(arguments.experiment)
    config = make_config(experiment.experiment_id, study_root=arguments.study_root)
    spec = get_model_spec(arguments.model)
    if spec not in tuple(get_experiment_specs(experiment.experiment_id)):
        raise ValueError("model does not belong to the selected experiment")
    device = common._resolve_device(arguments.device or config.device)
    epochs = config.epochs if arguments.epochs is None else int(arguments.epochs)
    if epochs < 1 or epochs > config.epochs:
        raise ValueError("epoch override is outside the D series protocol")
    paths = common.TrialPaths.create(
        config.study_directory / "models" / spec.model_id
        if arguments.output_directory is None
        else arguments.output_directory
    )
    logger: Any = NullCometTrialLogger()
    try:
        if arguments.disable_comet and arguments.sample_limit is None:
            raise ValueError("Comet can only be disabled for a sampled smoke trial")
        logger = common._create_trial_comet_logger(
            config,
            spec,
            paths,
            disabled=arguments.disable_comet,
        )
        if arguments.sample_limit is None:
            split, split_manifest = common._load_split(
                _shared_split_directory(arguments.study_root)
            )
        else:
            data = common.load_data(config, sample_limit=arguments.sample_limit)
            split, report = common.create_group_aware_split(
                data.centers,
                data.frames,
                seed=config.split_seed,
                fractions=config.split_fractions,
            )
            split_manifest = common._write_split(
                paths.directory,
                split,
                report,
                data.provenance,
            )
        dependency_sha = _dependency_sha256(spec, paths)
        summary = common.run_trial(
            config,
            spec,
            paths,
            split,
            split_manifest,
            logger,
            device=device,
            epochs=epochs,
            sample_limit=arguments.sample_limit,
            model_builder=_build_model,
            calibration_hook=_calibrate_model
            if spec.model_id in CALIBRATED_MODEL_IDS
            else None,
            source_sha256=_source_sha256(dependency_sha),
            study_metadata=D_STUDY_METADATA,
        )
        print(json.dumps({"status": "complete", "summary": str(paths.summary)}))
        return 0 if summary["status"] == "complete" else 1
    except KeyboardInterrupt:
        logger.finish("interrupted")
        return 130
    except BaseException as error:
        common._record_error(paths, spec, error)
        try:
            logger.log_error(error, stage="trial")
            logger.finish("error")
        except BaseException:
            traceback.print_exc(file=sys.stderr)
        traceback.print_exception(error, file=sys.stderr)
        return 1


def run_smoke(arguments: argparse.Namespace) -> int:
    experiment = _experiment(arguments.experiment)
    config = make_config(experiment.experiment_id, study_root=arguments.study_root)
    device = common._resolve_device(arguments.device or config.device)
    defaults = {
        1: ("D01", "D11", "D22"),
        2: ("D26", "D46", "D50"),
        3: ("D51", "D55", "D75"),
    }
    selected = _select_specs(
        experiment.experiment_id,
        arguments.model or defaults[experiment.experiment_id],
    )
    smoke_root = (
        Path(arguments.output_directory).resolve()
        if arguments.output_directory is not None
        else config.study_directory / "smoke"
    )
    for spec in selected:
        output = smoke_root / spec.model_id
        command = [
            sys.executable,
            "-m",
            "experiments.benzene_pair.d_series.runner",
            "trial",
            "--experiment",
            str(experiment.experiment_id),
            "--study_root",
            str(Path(arguments.study_root).resolve()),
            "--model",
            spec.model_id,
            "--device",
            device,
            "--epochs",
            str(arguments.epochs),
            "--sample_limit",
            str(arguments.sample_limit),
            "--output_directory",
            str(output),
            "--disable_comet",
        ]
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


def run_prepare(arguments: argparse.Namespace) -> int:
    config = make_config(1, study_root=arguments.study_root)
    shared = replace(
        config,
        study_directory=_shared_split_directory(arguments.study_root),
    )
    manifest = common._prepare_split(shared)
    print(
        json.dumps(
            {
                "status": "complete",
                "shared_split_directory": str(shared.study_directory),
                "manifest_hash": manifest["manifest_hash"],
                "partition_counts": manifest["partition_counts"],
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    prepare.set_defaults(handler=run_prepare)
    run = commands.add_parser("run")
    run.add_argument(
        "--experiment", type=int, choices=tuple(EXPERIMENTS), required=True
    )
    run.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    run.add_argument("--device", default=None)
    run.add_argument("--model", action="append", default=[])
    run.set_defaults(handler=run_study)
    trial = commands.add_parser("trial")
    trial.add_argument(
        "--experiment", type=int, choices=tuple(EXPERIMENTS), required=True
    )
    trial.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    trial.add_argument("--model", required=True)
    trial.add_argument("--device", default=None)
    trial.add_argument("--epochs", type=int, default=None)
    trial.add_argument("--sample_limit", type=int, default=None)
    trial.add_argument("--output_directory", type=Path, default=None)
    trial.add_argument("--disable_comet", action="store_true")
    trial.set_defaults(handler=run_trial_command)
    smoke = commands.add_parser("smoke")
    smoke.add_argument(
        "--experiment", type=int, choices=tuple(EXPERIMENTS), required=True
    )
    smoke.add_argument("--study_root", type=Path, default=DEFAULT_STUDY_ROOT)
    smoke.add_argument("--device", default=None)
    smoke.add_argument("--model", action="append", default=[])
    smoke.add_argument("--epochs", type=int, default=1)
    smoke.add_argument("--sample_limit", type=int, default=16000)
    smoke.add_argument("--output_directory", type=Path, default=None)
    smoke.set_defaults(handler=run_smoke)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())
