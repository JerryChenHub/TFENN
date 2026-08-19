"""Validate the F series runner, split gates, and selected model audit."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.benzene_pair import sweep30 as common
from experiments.benzene_pair.comet_logging import NullCometTrialLogger
from experiments.benzene_pair.f_series import runner
from experiments.benzene_pair.f_series.gate_audit import export_selected_gate_audit
from experiments.benzene_pair.f_series.model_factory import build_f_series_model


def test_five_execution_shards_have_one_shared_protocol(
    tmp_path: Path,
) -> None:
    configs = tuple(
        runner.make_config(index, study_root=tmp_path) for index in range(5)
    )
    assert len({config.study_directory for config in configs}) == 1
    assert len({config.comet.project_name for config in configs}) == 1
    assert {config.epochs for config in configs} == {500}
    assert {config.effective_batch_size for config in configs} == {10_000}
    assert {config.micro_batch_size for config in configs} == {10_000}
    assert {config.expected_sample_count for config in configs} == {400_000}
    assert len({config.shard_paths for config in configs}) == 1
    assert all(config.comet.required_online for config in configs)
    assert [
        len(runner.get_execution_shard_specs(index)) for index in range(5)
    ] == [1, 25, 25, 25, 25]
    assert [
        runner.EXECUTION_SHARDS[index].tmux_session_name for index in range(5)
    ] == [
        "tfenn_f_control",
        "tfenn_f1_a",
        "tfenn_f1_b",
        "tfenn_f2_a",
        "tfenn_f2_b",
    ]


def test_config_files_match_execution_shards_and_concurrency() -> None:
    for shard_id, path in runner.DEFAULT_CONFIG_PATHS.items():
        value = json.loads(path.read_text(encoding="utf_8"))
        specs = runner._select_specs(shard_id, ())
        assert tuple(value["model_ids"]) == tuple(spec.model_id for spec in specs)
        assert value["concurrent_run"] is True
        assert value["tmux_session_count"] == 5
        assert len(specs) == runner.EXPECTED_SHARD_COUNTS[shard_id]


def test_parser_exposes_prepare_run_trial_smoke_aggregate_and_tmux(
    tmp_path: Path,
) -> None:
    parser = runner.build_parser()
    prepare = parser.parse_args(
        (
            "prepare",
            "--reference_split_directory",
            str(tmp_path / "reference"),
            "--force_preflight",
        )
    )
    assert prepare.handler is runner.run_prepare
    assert prepare.reference_split_directory == tmp_path / "reference"
    assert prepare.force_preflight is True
    assert parser.parse_args(("preflight", "--force")).force is True
    assert parser.parse_args(("run", "--shard", "f2b")).handler is runner.run_study
    trial = parser.parse_args(
        (
            "trial",
            "--model",
            "F201",
            "--sample_limit",
            "32",
            "--disable_comet",
        )
    )
    assert trial.handler is runner.run_trial_command
    smoke = parser.parse_args(
        ("smoke", "--shard", "f2a", "--model", "F218", "--epochs", "1")
    )
    assert smoke.handler is runner.run_smoke
    assert smoke.model == ["F218"]
    assert parser.parse_args(("aggregate",)).handler is runner.run_aggregate
    assert parser.parse_args(("launch-tmux", "--dry-run")).handler is (
        runner.run_launch_tmux
    )


def test_tmux_launcher_has_exactly_five_disjoint_batches(tmp_path: Path) -> None:
    commands = runner.tmux_launch_commands(study_root=tmp_path, devices=("cuda",))
    assert len(commands) == 5
    assert len({session for session, _command in commands}) == 5
    for shard_id, (session, command) in enumerate(commands):
        shard = runner.EXECUTION_SHARDS[shard_id]
        assert session == shard.tmux_session_name
        pane = command[-1]
        assert f"--shard {shard.key}" in pane
        assert "--device cuda" in pane
        assert "COMET_API_KEY is unavailable inside tmux" in pane


def test_tmux_launcher_auto_maps_five_visible_devices_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 5)
    assert runner._tmux_devices(()) == tuple(
        f"cuda:{index}" for index in range(5)
    )
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 4)
    with pytest.raises(ValueError, match="five visible CUDA devices"):
        runner._tmux_devices(())


def test_source_hash_covers_package_initializers_and_all_f_runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = runner._source_sha256()
    targets = (
        runner.REPOSITORY_ROOT / "experiments" / "benzene_pair" / "__init__.py",
        runner.REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "e_series"
        / "__init__.py",
        runner.REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "f_series"
        / "__init__.py",
        runner.REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "f_series"
        / "gate_audit.py",
    )
    original_read_bytes = Path.read_bytes
    for target in targets:
        resolved_target = target.resolve()

        def altered_read_bytes(path: Path) -> bytes:
            value = original_read_bytes(path)
            if path.resolve() == resolved_target:
                return value + b"F source closure probe"
            return value

        with monkeypatch.context() as local:
            local.setattr(Path, "read_bytes", altered_read_bytes)
            assert runner._source_sha256() != baseline, target


def test_smoke_spawns_sampled_trials_without_comet(
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
            "--shard",
            "f1a",
            "--model",
            "F101",
            "--model",
            "F125",
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
    assert len(calls) == 2
    assert {call[call.index("--model") + 1] for call in calls} == {"F101", "F125"}
    assert all("--disable_comet" in call for call in calls)
    assert all(call[call.index("--sample_limit") + 1] == "64" for call in calls)


def test_reference_split_copy_records_and_rechecks_exact_e311_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = (tmp_path / "e311" / "shared_split").resolve()
    reference.mkdir(parents=True)
    source_indices = reference / "split_indices.npz"
    source_indices.write_bytes(b"exact E311 split bytes")
    split = common.SplitIndices(
        train=torch.arange(0, 320_000),
        validation=torch.arange(320_000, 360_000),
        test=torch.arange(360_000, 400_000),
    )
    reference_manifest = {
        "schema_name": "tfenn_benzene_pair_group_aware_split",
        "schema_version": 1,
        "sample_count": 400_000,
        "partition_counts": split.counts(),
        "indices_path": str(source_indices),
        "indices_sha256": runner.E311_SPLIT_INDICES_SHA256,
        "data_provenance": {"shards": ["exact E311"]},
        "manifest_hash": runner.E311_SPLIT_MANIFEST_HASH,
    }

    def load_split(directory: Path):
        if Path(directory).resolve() == reference:
            return split, dict(reference_manifest)
        manifest = json.loads(
            (Path(directory) / "split_manifest.json").read_text(encoding="utf_8")
        )
        return split, manifest

    monkeypatch.setattr(runner.common, "_load_split", load_split)
    monkeypatch.setattr(
        runner.common,
        "sha256_file",
        lambda _path: runner.E311_SPLIT_INDICES_SHA256,
    )
    copied, manifest = runner._prepare_reference_split(tmp_path / "f", reference)
    assert copied.counts() == split.counts()
    assert manifest["reference_manifest_hash"] == runner.E311_SPLIT_MANIFEST_HASH
    assert manifest["reference_indices_sha256"] == runner.E311_SPLIT_INDICES_SHA256
    assert len(runner.E311_SPLIT_MANIFEST_HASH) == 64
    assert len(runner.E311_SPLIT_INDICES_SHA256) == 64
    target = tmp_path / "f" / "shared_split" / "split_indices.npz"
    assert target.read_bytes() == source_indices.read_bytes()
    runner._require_e311_split(manifest)


def test_pair_preflight_has_same_initialization_schema_and_declared_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_spec = runner.get_model_spec("F101")
    second_spec = runner.get_model_spec("F201")
    first = runner._preflight_model(first_spec)
    second = runner._preflight_model(second_spec)
    records = [first, second]
    monkeypatch.setattr(runner, "F_SERIES_SPECS", (first_spec, second_spec))
    runner._apply_pair_gates(records)
    assert all(record["status"] == "passed" for record in records)
    for record in records:
        assert record["planned_parameter_count_matches"] is True
        assert record["pair_parameter_count_equal"] is True
        assert record["pair_initialization_equal"] is True
        assert record["pair_descriptor_schema_equal"] is True
        assert record["pair_candidate_manifest_equal"] is True
        assert record["pair_coefficient_manifest_equal"] is True
        assert record["covariant_unit_check"][
            "all_trainable_parameter_tensors_have_gradients"
        ]


def test_f100_preflight_reproduces_e311_exactly() -> None:
    record = runner._preflight_model(runner.get_model_spec("F100"))
    assert record["status"] == "passed"
    assert record["actual_parameter_count"] == 14_926
    assert record["planned_parameter_count_matches"] is True
    parity = record["e311_reference_parity"]
    assert parity == {
        "builder_reference_model_id": "E311",
        "compiled_config_equal": True,
        "candidate_manifest_equal": True,
        "initialization_equal": True,
        "forward_bitwise_equal": True,
        "parameter_count": 14_926,
    }


def test_gate_audit_marks_only_raw_only_descriptor_columns_ineligible(
    tmp_path: Path,
) -> None:
    generator = torch.Generator().manual_seed(7)
    centers = torch.zeros(8, 2, 3)
    centers[:, 1] = torch.randn(8, 3, generator=generator) + 5.0
    frames = torch.eye(3).expand(8, 2, 3, 3).clone()
    data = common.TrainingData(
        centers=centers,
        frames=frames,
        root_force=torch.randn(8, 3, generator=generator),
        provenance={"source": "synthetic"},
    )
    split = common.SplitIndices(
        train=torch.arange(0, 2),
        validation=torch.arange(2, 8),
        test=torch.arange(6, 8),
    )
    config = runner.make_config(1, study_root=tmp_path)
    reports = {}
    for model_id in ("F100", "F101", "F201"):
        torch.manual_seed(20260822)
        model = build_f_series_model(
            model_id,
            common._proper_d6_generators(),
            generator_names=("sixfold", "twofold"),
        ).to(dtype=torch.float32)
        paths = common.TrialPaths.create(tmp_path / model_id)
        report = export_selected_gate_audit(
            model=model,
            spec=runner.get_model_spec(model_id),
            data=data,
            split=split,
            config=config,
            paths=paths,
            device="cpu",
            comet_logger=NullCometTrialLogger(),
        )
        reports[model_id] = report
        payload = json.loads(
            (paths.directory / "gate_audit.json").read_text(encoding="utf_8")
        )
        assert payload["validation_probe_sample_count"] == 6
        assert payload["validation_probe_seed"] == 20260822
        assert payload["validation_probe_batch_size"] == 2_000
        assert payload["validation_gamma_statistics"]
        assert all(
            row["pre_projection_branch_rms"] >= 0.0
            for row in payload["validation_gamma_statistics"]
        )
        assert all(
            "primitive_index" in row and "basis_index" not in row
            for row in payload["coefficient_parameter_statistics"]
        )
        parameter_path = Path(report["invariant_gate_parameter_path"])
        assert parameter_path.is_file()
        snapshot = torch.load(parameter_path)
        assert snapshot["model_id"] == model_id
        assert snapshot["stage_trunks"]
        assert snapshot["coefficient_heads_by_role"]
        assert snapshot["typed_channel_projections"]
        assert all(
            row["ranking_eligible"] == row["active"]
            for row in payload["trunk_input_column_statistics"]
        )
    assert reports["F100"]["masked_descriptor_column_count"] == 0
    assert reports["F101"]["masked_descriptor_column_count"] == 0
    assert reports["F201"]["masked_descriptor_column_count"] > 0
