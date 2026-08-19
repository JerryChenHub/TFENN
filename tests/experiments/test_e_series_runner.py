"""Validate the five E series runners and fixed protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.benzene_pair.e_series import runner


def test_five_subexperiments_use_independent_comet_projects_and_one_protocol(
    tmp_path: Path,
) -> None:
    configs = tuple(
        runner.make_config(index, study_root=tmp_path) for index in range(5)
    )
    assert len({config.study_directory for config in configs}) == 5
    assert len({config.comet.project_name for config in configs}) == 5
    assert {config.epochs for config in configs} == {500}
    assert {config.effective_batch_size for config in configs} == {10_000}
    assert {config.micro_batch_size for config in configs} == {10_000}
    assert {config.expected_sample_count for config in configs} == {400_000}
    assert {config.split_seed for config in configs} == {20260821}
    assert {config.model_seed for config in configs} == {20260822}
    assert {config.shuffle_seed for config in configs} == {20260823}
    assert len({config.shard_paths for config in configs}) == 1
    assert all(
        config.comet.enabled and config.comet.required_online for config in configs
    )


def test_config_files_match_groups_and_fixed_gpu_concurrency_metadata() -> None:
    for experiment, path in runner.DEFAULT_CONFIG_PATHS.items():
        value = json.loads(path.read_text(encoding="utf_8"))
        specs = runner._select_specs(experiment, ())
        assert value["experiment_id"] == experiment
        assert tuple(value["model_ids"]) == tuple(spec.model_id for spec in specs)
        assert (
            value["study_directory_name"]
            == runner.EXPERIMENTS[experiment].directory_name
        )
        assert (
            value["comet"]["project_name"]
            == runner.EXPERIMENTS[experiment].comet_project
        )
        assert value["concurrent_run"] is True
        assert value["shared_gpu_process_count"] == 5
        assert value["epochs"] == 500
        assert value["effective_batch_size"] == 10_000
        assert value["micro_batch_size"] == 10_000


def test_group_selection_counts_boundaries_and_duplicates() -> None:
    groups = tuple(runner._select_specs(index, ()) for index in range(5))
    assert tuple(map(len, groups)) == (8, 25, 25, 25, 25)
    assert tuple(spec.model_id for spec in groups[0]) == tuple(
        f"E{index:03d}" for index in range(1, 9)
    )
    for experiment in range(1, 5):
        assert tuple(spec.model_id for spec in groups[experiment]) == tuple(
            f"E{experiment}{index:02d}" for index in range(1, 26)
        )
    assert len({spec.model_id for group in groups for spec in group}) == 108
    with pytest.raises(ValueError, match="outside experiment"):
        runner._select_specs(2, ("E101",))
    with pytest.raises(ValueError, match="duplicates"):
        runner._select_specs(4, ("E425", "e425"))


def test_parser_supports_prepare_formal_trial_and_sampled_smoke() -> None:
    parser = runner.build_parser()
    assert parser.parse_args(("prepare",)).handler is runner.run_prepare
    preflight = parser.parse_args(("preflight", "--force"))
    assert preflight.force is True
    formal = parser.parse_args(("run", "--experiment", "2"))
    assert formal.handler is runner.run_study
    assert formal.experiment == 2
    trial = parser.parse_args(
        (
            "trial",
            "--experiment",
            "4",
            "--model",
            "E425",
            "--sample_limit",
            "32",
            "--disable_comet",
        )
    )
    assert trial.handler is runner.run_trial_command
    assert trial.disable_comet is True
    smoke = parser.parse_args(
        (
            "smoke",
            "--experiment",
            "4",
            "--model",
            "E424",
            "--epochs",
            "1",
            "--sample_limit",
            "32",
        )
    )
    assert smoke.handler is runner.run_smoke
    assert smoke.model == ["E424"]
    assert smoke.epochs == 1
    assert smoke.sample_limit == 32


def test_smoke_spawns_a_sampled_trial_with_comet_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def record(command: list[str], **_values: object) -> SimpleNamespace:
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", record)
    arguments = runner.build_parser().parse_args(
        (
            "smoke",
            "--experiment",
            "4",
            "--model",
            "E424",
            "--epochs",
            "1",
            "--sample_limit",
            "64",
            "--study_root",
            str(tmp_path / "study"),
            "--output_directory",
            str(tmp_path / "smoke"),
            "--device",
            "cpu",
        )
    )
    assert runner.run_smoke(arguments) == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == (
        runner.sys.executable,
        "-m",
        "experiments.benzene_pair.e_series.runner",
    )
    assert command[command.index("--experiment") + 1] == "4"
    assert command[command.index("--model") + 1] == "E424"
    assert command[command.index("--epochs") + 1] == "1"
    assert command[command.index("--sample_limit") + 1] == "64"
    assert command[command.index("--device") + 1] == "cpu"
    assert "--disable_comet" in command


class _DisconnectedCovariant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.unused = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, centers: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        del frames
        return centers[:, 1] * self.used


def test_covariant_preflight_rejects_a_disconnected_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.common,
        "symmetry_metrics",
        lambda *_values, **_options: {"passed": True},
    )
    with pytest.raises(RuntimeError, match="gradient connectivity"):
        runner._covariant_unit_check(_DisconnectedCovariant())


def test_source_hash_is_order_stable_and_sensitive_to_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = runner._source_sha256()
    original_rglob = Path.rglob

    def reversed_rglob(path: Path, pattern: str):
        return iter(reversed(tuple(original_rglob(path, pattern))))

    with monkeypatch.context() as local:
        local.setattr(Path, "rglob", reversed_rglob)
        assert runner._source_sha256() == baseline

    dependency_roots = (
        runner.REPOSITORY_ROOT / "src" / "TFENN" / "tensor_math",
        runner.REPOSITORY_ROOT / "src" / "TFENN" / "data",
    )
    targets = tuple(next(root.rglob("*.py")) for root in dependency_roots) + (
        runner.REPOSITORY_ROOT
        / "src"
        / "TFENN"
        / "models"
        / "model_level_group_conv_mlp.py",
        runner.REPOSITORY_ROOT / "experiments" / "benzene_pair" / "sweep30.py",
        runner.REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "d_series"
        / "model_factory.py",
    )
    original_read_bytes = Path.read_bytes
    for target in targets:
        resolved_target = target.resolve()

        def altered_read_bytes(path: Path) -> bytes:
            value = original_read_bytes(path)
            if path.resolve() == resolved_target:
                return value + b"source hash sensitivity probe"
            return value

        with monkeypatch.context() as local:
            local.setattr(Path, "read_bytes", altered_read_bytes)
            assert runner._source_sha256() != baseline, target
