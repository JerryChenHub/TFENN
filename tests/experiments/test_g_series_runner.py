"""Validate the G CPU runner, paired seeds, split gate, and Gate artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.benzene_pair import sweep30 as common
from experiments.benzene_pair.comet_logging import NullCometTrialLogger
from experiments.benzene_pair.g_series import runner
from experiments.benzene_pair.g_series.gate_audit import (
    VALIDATION_PROBE_BATCH_SIZE,
    export_selected_gate_audit,
)


def test_formal_config_is_cpu_batch_one_thousand_and_within_budget(
    tmp_path: Path,
) -> None:
    config = runner.make_config(study_root=tmp_path)
    assert isinstance(config, runner.GSeriesConfig)
    assert config.study_directory == tmp_path.resolve()
    assert config.device == "cpu"
    assert config.effective_batch_size == 1_000
    assert config.micro_batch_size == 1_000
    assert config.epochs == 500
    assert config.learning_rate == 0.003
    assert config.weight_decay == 0.0001
    assert config.scheduler_step_size == 125
    assert config.scheduler_gamma == 0.5
    assert config.dtype == "float32"
    assert config.enable_tf32 is False
    assert config.comet.enabled and config.comet.required_online
    assert config.comet.project_name == runner.EXPECTED_COMET_PROJECT
    assert len(runner.G_SERIES_SPECS) == 70 < runner.MODEL_BUDGET


def test_trial_config_uses_catalog_paired_seeds_without_changing_split(
    tmp_path: Path,
) -> None:
    base = runner.make_config(study_root=tmp_path)
    first = runner._config_for_spec(base, runner.get_model_spec("G101"))
    fifth = runner._config_for_spec(base, runner.get_model_spec("G501"))
    assert (first.model_seed, first.shuffle_seed) == (20260822, 20260823)
    assert (fifth.model_seed, fifth.shuffle_seed) == (20260862, 20260863)
    assert first.split_seed == fifth.split_seed == 20260821
    assert first.study_directory == fifth.study_directory
    with pytest.raises(ValueError, match="CPU-only"):
        replace(base, device="cuda").validate()
    with pytest.raises(ValueError, match="one thousand"):
        replace(base, effective_batch_size=10_000).validate()


def test_group_and_model_selection_preserve_catalog_order() -> None:
    factorial = runner._select_specs(("factorial",), ())
    assert len(factorial) == 40
    assert {spec.variant_id for spec in factorial} == set(range(1, 9))
    combined = runner._select_specs(("legacy", "carrier"), ())
    assert len(combined) == 35
    assert len({spec.model_id for spec in combined}) == len(combined)
    assert runner._select_specs(("legacy",), ("g109",))[0].model_id == "G109"
    with pytest.raises(ValueError, match="outside selected G groups"):
        runner._select_specs(("legacy",), ("G102",))
    with pytest.raises(ValueError, match="duplicates"):
        runner._select_specs((), ("G101", "g101"))


def test_study_manifest_is_group_restart_safe_and_selections_are_separate(
    tmp_path: Path,
) -> None:
    config = runner.make_config(study_root=tmp_path)
    split_manifest = {
        "manifest_hash": "shared_manifest",
        "indices_sha256": runner.E311_SPLIT_INDICES_SHA256,
    }
    manifest = runner._study_manifest(
        config,
        split_manifest,
        device="cpu",
        source_sha256="source_hash",
    )
    assert "selected_model_ids" not in manifest
    factorial = runner._select_specs(("factorial",), ("G101",))
    legacy = runner._select_specs(("legacy",), ("G109",))
    first = runner._write_invocation_record(
        config,
        manifest,
        factorial,
        groups=("factorial",),
        requested_models=("G101",),
    )
    second = runner._write_invocation_record(
        config,
        manifest,
        legacy,
        groups=("legacy",),
        requested_models=("G109",),
    )
    assert first != second
    assert json.loads(first.read_text(encoding="utf_8"))["study_hash"] == manifest[
        "study_hash"
    ]
    assert json.loads(second.read_text(encoding="utf_8"))["study_hash"] == manifest[
        "study_hash"
    ]


def test_parser_exposes_prepare_run_trial_smoke_and_aggregate() -> None:
    parser = runner.build_parser()
    assert parser.parse_args(("prepare",)).handler is runner.run_prepare
    formal = parser.parse_args(
        ("run", "--group", "factorial", "--model", "G101")
    )
    assert formal.handler is runner.run_study
    assert formal.group == ["factorial"]
    assert formal.model == ["G101"]
    trial = parser.parse_args(
        ("trial", "--model", "G101", "--sample_limit", "32", "--disable_comet")
    )
    assert trial.handler is runner.run_trial_command
    assert trial.disable_comet is True
    assert parser.parse_args(("smoke", "--group", "carrier")).handler is (
        runner.run_smoke
    )
    assert parser.parse_args(("aggregate",)).handler is runner.run_aggregate


def test_cpu_gate_rejects_every_non_cpu_device(tmp_path: Path) -> None:
    config = runner.make_config(study_root=tmp_path)
    assert runner._require_cpu_device(None, config) == "cpu"
    assert runner._require_cpu_device("cpu", config) == "cpu"
    with pytest.raises(ValueError, match="CPU-only"):
        runner._require_cpu_device("mps", config)


def test_nonformal_trial_requires_an_isolated_output_directory(
    tmp_path: Path,
) -> None:
    arguments = runner.build_parser().parse_args(
        (
            "trial",
            "--study_root",
            str(tmp_path),
            "--model",
            "G101",
            "--sample_limit",
            "32",
            "--disable_comet",
        )
    )
    with pytest.raises(ValueError, match="explicit output directory"):
        runner.run_trial_command(arguments)
    assert not (tmp_path / "models" / "G101").exists()


def test_smoke_spawns_sampled_cpu_trials_without_comet(
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
            "--group",
            "legacy",
            "--model",
            "G109",
            "--epochs",
            "1",
            "--sample_limit",
            "64",
            "--study_root",
            str(tmp_path / "study"),
            "--output_directory",
            str(tmp_path / "smoke"),
        )
    )
    assert runner.run_smoke(arguments) == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == (
        runner.sys.executable,
        "-m",
        "experiments.benzene_pair.g_series.runner",
    )
    assert command[command.index("--model") + 1] == "G109"
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--sample_limit") + 1] == "64"
    assert "--disable_comet" in command


def test_reference_split_copy_records_exact_e311_hashes(
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
    copied, manifest = runner._prepare_reference_split(tmp_path / "g", reference)
    assert copied.counts() == split.counts()
    assert manifest["reference_manifest_hash"] == runner.E311_SPLIT_MANIFEST_HASH
    assert manifest["reference_indices_sha256"] == runner.E311_SPLIT_INDICES_SHA256
    assert (tmp_path / "g" / "shared_split" / "split_indices.npz").read_bytes() == (
        source_indices.read_bytes()
    )
    runner._require_e311_split(manifest)


def test_aggregate_emits_same_seed_controls_and_factorial_interactions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.make_config(study_root=tmp_path)

    def complete_row(_config, spec):
        row = {field: "" for field in runner.RESULT_FIELDS}
        scale = 1.0 + 0.01 * spec.variant_id + 0.001 * spec.seed_index
        row.update(
            {
                "model_id": spec.model_id,
                "variant_id": spec.variant_id,
                "seed_index": spec.seed_index,
                "status": "complete",
                "validation_mae": scale,
                "test_mae": scale + 0.05,
            }
        )
        return row

    monkeypatch.setattr(runner, "_result_row", complete_row)
    runner._refresh_results(config)
    comparison = json.loads(
        (tmp_path / "comparison.json").read_text(encoding="utf_8")
    )
    assert len(comparison["paired_vs_same_seed_gx01_control"]) == 65
    assert len(comparison["factorial_log_mae_contrasts"]) == 40
    assert len(comparison["factorial_orthogonal_log_mae_contrasts"]) == 35
    assert len(comparison["mechanism_log_mae_contrasts"]) == 35
    assert len(comparison["factorial_orthogonal_contrast_summary"]) == 7
    assert len(comparison["mechanism_contrast_summary"]) == 7
    assert comparison["mechanism_contrast_summary"][0][
        "practical_equivalence_mae_percent"
    ] == 2.0
    assert comparison["g5_control_gate_parameter_importance"] == {
        "completed_control_seed_count": 0,
        "control_model_ids": [],
        "descriptor_role_summary": [],
        "interpretation": (
            "descriptive parameter allocation only; V@W omits sample-dependent "
            "SiLU derivatives and is not a causal invariant ablation"
        ),
        "ranking_metric": "mean active-head V@W RMS within each stage",
    }
    assert {
        row["contrast"] for row in comparison["factorial_log_mae_contrasts"]
    } >= {
        "generic_by_both_stf_interaction",
        "a1_by_out_stf_interaction_generic_off",
        "generic_by_a1_by_out_three_way_interaction",
    }
    assert {
        row["contrast"]
        for row in comparison["factorial_orthogonal_log_mae_contrasts"]
    } == {
        "generic_pair_main_effect",
        "a1_stf_main_effect",
        "out_stf_main_effect",
        "generic_pair_by_a1_stf",
        "generic_pair_by_out_stf",
        "a1_stf_by_out_stf",
        "generic_pair_by_a1_stf_by_out_stf",
    }


def test_control_gate_importance_aggregates_fresh_gx01_seeds(
    tmp_path: Path,
) -> None:
    rows = []
    for seed_index, high_value in ((1, 0.8), (2, 0.6)):
        path = tmp_path / f"G{seed_index}01_gate_audit.json"
        path.write_text(
            json.dumps(
                {
                    "variant_id": 1,
                    "trunk_input_column_statistics": [
                        {
                            "stage": "out",
                            "role": "out.scalar.high",
                            "kind": "pair",
                            "source_names": ["x", "r"],
                            "ranking_eligible": True,
                            "weight": {"rms": high_value},
                        },
                        {
                            "stage": "out",
                            "role": "out.scalar.low",
                            "kind": "unary",
                            "source_names": ["x"],
                            "ranking_eligible": True,
                            "weight": {"rms": 0.2},
                        },
                    ],
                    "head_composed_weight_statistics": [
                        {
                            "stage": "out",
                            "descriptor_role": "out.scalar.high",
                            "ranking_eligible": True,
                            "composed_weight": {"rms": high_value},
                        },
                        {
                            "stage": "out",
                            "descriptor_role": "out.scalar.low",
                            "ranking_eligible": True,
                            "composed_weight": {"rms": 0.1},
                        },
                    ],
                }
            ),
            encoding="utf_8",
        )
        rows.append(
            {
                "model_id": f"G{seed_index}01",
                "variant_id": 1,
                "gate_audit_path": str(path),
            }
        )
    summary = runner._control_gate_importance(rows)
    assert summary["completed_control_seed_count"] == 2
    assert summary["control_model_ids"] == ["G101", "G201"]
    descriptors = summary["descriptor_role_summary"]
    assert [item["descriptor_role"] for item in descriptors] == [
        "out.scalar.high",
        "out.scalar.low",
    ]
    assert descriptors[0]["mean_within_stage_rank"] == 1
    assert descriptors[0]["top_five_seed_frequency"] == 1.0


class _TinyGateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_trunks = nn.ModuleDict(
            {"out": nn.Sequential(nn.Linear(2, 3), nn.SiLU())}
        )
        self.path_heads = nn.ModuleDict({"head": nn.Linear(3, 1)})
        self.channel_projections = nn.ModuleDict(
            {"projection": nn.Linear(1, 1, bias=False)}
        )

    @property
    def descriptor_role_manifest(self):
        return (
            {
                "stage": "out",
                "role": "out.scalar.radial",
                "kind": "radial",
                "source_names": ("x",),
                "start": 0,
                "stop": 2,
                "active": True,
                "active_column_count": 2,
                "basis_role": "radial",
            },
        )

    @property
    def coefficient_head_role_manifest(self):
        return (
            {
                "role": "out.covariant.example",
                "stage": "out",
                "target": "A",
                "source_names": ("x",),
                "source_types": ("A",),
                "path_family": "unary",
                "basis_role": "A<-A",
                "primitive_channels": 1,
                "target_channels": 1,
                "coefficient_channels": 1,
                "module_name": "path_heads.head",
            },
        )

    @property
    def candidate_manifest(self):
        return ()

    def coefficient_head_modules_by_role(self):
        return {"out.covariant.example": self.path_heads["head"]}

    def first_trunk_linear_modules_by_stage(self):
        return {"out": self.stage_trunks["out"][0]}

    def _coefficient(self, centers: torch.Tensor) -> torch.Tensor:
        descriptor = centers[:, 1, :2]
        return self.path_heads["head"](self.stage_trunks["out"](descriptor))

    def collect_coefficient_activations(
        self,
        centers: torch.Tensor,
        frames: torch.Tensor,
    ):
        del frames
        return {"out.covariant.example": self._coefficient(centers)}

    def debug_forward(self, centers: torch.Tensor, frames: torch.Tensor):
        del frames
        return SimpleNamespace(
            direct_paths={"out.covariant.example": self._coefficient(centers)}
        )


def test_gate_audit_exports_weight_matrices_and_role_statistics(
    tmp_path: Path,
) -> None:
    generator = torch.Generator().manual_seed(7)
    centers = torch.randn(8, 2, 3, generator=generator)
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
    spec = runner.get_model_spec("G101")
    paths = common.TrialPaths.create(tmp_path / spec.model_id)
    report = export_selected_gate_audit(
        model=_TinyGateModel(),
        spec=spec,
        data=data,
        split=split,
        config=SimpleNamespace(),
        paths=paths,
        device="cpu",
        comet_logger=NullCometTrialLogger(),
    )
    assert VALIDATION_PROBE_BATCH_SIZE == 1_000
    assert Path(report["invariant_gate_parameter_path"]).is_file()
    payload = json.loads(
        (paths.directory / "gate_audit.json").read_text(encoding="utf_8")
    )
    assert payload["variant_id"] == 1
    assert payload["validation_probe_sample_count"] == 6
    assert "g_series_manifest" in payload
    assert "causal_mask_manifest" in payload
    assert "projection_input_role_manifest" in payload
    assert "legacy_carrier_gate_parameter_statistics" in payload
    assert payload["trunk_input_column_statistics"]
    assert payload["coefficient_parameter_statistics"]
    assert payload["head_composed_weight_statistics"]
    assert payload["validation_gamma_statistics"]
    assert payload["validation_carrier_gate_statistics"] == []


def test_source_hash_covers_g_and_model_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = runner._source_sha256()
    target = (
        runner.REPOSITORY_ROOT
        / "experiments"
        / "benzene_pair"
        / "g_series"
        / "model_factory.py"
    ).resolve()
    original_read_bytes = Path.read_bytes

    def altered_read_bytes(path: Path) -> bytes:
        value = original_read_bytes(path)
        return value + b"G source closure probe" if path.resolve() == target else value

    monkeypatch.setattr(Path, "read_bytes", altered_read_bytes)
    assert runner._source_sha256() != baseline
