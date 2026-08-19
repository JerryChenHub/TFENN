"""Run the explicit full compiled E4 acceptance audit."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.benzene_pair.e_series.catalog import E4_SPECS
from experiments.benzene_pair.e_series.catalog import get_model_spec
from experiments.benzene_pair.e_series.model_factory import build_e_series_model
from experiments.benzene_pair.e_series.runner import _covariant_unit_check
from experiments.benzene_pair.train import _proper_d6_generators
from TFENN.models.e_series import E4_FROZEN_BUDGETS, compact_blueprint_manifest


def _selected_source_sets(model, stage_name: str) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(endpoint.source for endpoint in path.candidate.endpoints)
        for paths in model._stage_paths[stage_name].values()
        for path in paths
    )


def _audit_one(model_id: str) -> dict[str, object]:
    spec = get_model_spec(model_id)
    assert spec in E4_SPECS
    assert "implementation" not in spec.options
    model = build_e_series_model(
        spec,
        _proper_d6_generators(),
        generator_names=("sixfold", "twofold"),
    )
    actual = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_width, expected_readout, expected_count = E4_FROZEN_BUDGETS[spec.model_id]
    assert actual == expected_count
    assert 7_800 <= actual <= 8_200
    budget = model.budget_compilation_manifest
    assert budget["selected_parameter_count"] == actual
    assert budget["selected_base_width"] == expected_width
    assert budget["selected_readout_width"] == expected_readout
    assert budget["requested_mechanism"]["full_invariant_context"] is True
    assert budget["requested_mechanism"]["implemented_mechanism"] == (
        spec.architecture_name
    )
    assert budget["blueprint"] == compact_blueprint_manifest(spec.model_id)
    assert budget["compiled_stage_config"] == [
        stage.as_dict() for stage in model.config.stages
    ]
    assert tuple(budget["selected_covariant_roles"]) == (model.selected_covariant_roles)
    assert budget["factorization_ranks"] == tuple(
        {
            "stage": stage.name,
            "coefficient_rank": stage.coefficient_rank,
            "channel_projection_rank": stage.channel_projection_rank,
        }
        for stage in model.config.stages
        if stage.coefficient_rank is not None
        or stage.channel_projection_rank is not None
    )
    coverage = budget["coverage_audit"]
    assert coverage["status"] == "passed"
    assert set(coverage["requested_inputs"]) == {"x", "r"}
    assert set(coverage["represented_inputs"]) == {"x", "r"}
    assert coverage["final_a_reachable"] is True
    assert coverage["a_b_bridge_count"] > 0
    assert min(coverage["required_type_live_lanes"].values()) >= 1
    per_stage_source_coverage: dict[str, tuple[str, ...]] = {}
    for stage in model.config.stages:
        selected_sources = set().union(*_selected_source_sets(model, stage.name))
        skip_sources = set().union(
            *(
                set(model._skip_sources[(stage.name, target)])
                for target in model._stage_paths[stage.name]
            )
        )
        represented_sources = selected_sources | skip_sources
        assert set(stage.source_names).issubset(represented_sources), (
            stage.name,
            stage.source_names,
            represented_sources,
        )
        per_stage_source_coverage[stage.name] = tuple(sorted(represented_sources))
    mechanism_audit: dict[str, object] | None = None
    hidden_stages = model.config.stages[:-1]
    if spec.model_id == "E401":
        assert all(stage.skip_policy == "id" for stage in hidden_stages)
        assert all(stage.trunk_residual for stage in hidden_stages)
        assert all(stage.coefficient_head == "factorized" for stage in hidden_stages)
        assert all(stage.coefficient_rank == 1 for stage in hidden_stages)
    if spec.model_id == "E407":
        assert all(stage.skip_policy == "id" for stage in hidden_stages)
        assert all(stage.trunk_residual for stage in hidden_stages)
        assert {stage.parameter_share_group for stage in hidden_stages} == {"tied_cell"}
        assert len({id(model.stage_trunks[stage.name]) for stage in hidden_stages}) == 1
    if spec.model_id == "E404":
        typed_b_stages = tuple(
            stage
            for stage in model.config.stages
            if stage.output_stream == "B" and stage.type_channel_overrides
        )
        assert typed_b_stages
        typed_widths: dict[str, dict[int, int]] = {}
        for stage in typed_b_stages:
            widths = {
                key.component: model._channels[(stage.name, key)]
                for key in model._b_keys
            }
            assert min(widths.values()) >= 1
            typed_widths[stage.name] = widths
        assert any(len(set(widths.values())) > 1 for widths in typed_widths.values())
        hidden_schedule = tuple(stage.channels for stage in model.config.stages[:-1])
        assert hidden_schedule == tuple(budget["blueprint"]["channels"])
        assert hidden_schedule[0] > min(hidden_schedule)
        assert hidden_schedule[-1] > min(hidden_schedule)
        mechanism_audit = {"typed_b_channels": typed_widths}
    if spec.model_id in {"E409", "E414"}:
        expected_schedule = tuple(
            tuple(sources) for sources in budget["blueprint"]["source_schedule"]
        )
        actual_schedule = tuple(stage.source_names for stage in hidden_stages)
        assert actual_schedule == expected_schedule
        selected_by_stage: dict[str, tuple[tuple[str, ...], ...]] = {}
        for stage, allowed in zip(hidden_stages, expected_schedule):
            selected = _selected_source_sets(model, stage.name)
            assert selected
            assert all(sources.issubset(set(allowed)) for sources in selected)
            represented = set().union(*selected)
            assert set(allowed).issubset(represented), (
                stage.name,
                allowed,
                represented,
                selected,
            )
            selected_by_stage[stage.name] = tuple(
                tuple(sorted(sources)) for sources in selected
            )
        mechanism_audit = {
            "source_schedule": actual_schedule,
            "selected_endpoint_sources": selected_by_stage,
        }
    if spec.model_id == "E411":
        expected_sources = {
            "a1": ("x",),
            "b1": ("r",),
            "a2": ("x", "a1"),
            "b2": ("r", "b1"),
            "a3": ("x", "a2"),
            "b3": ("r", "b2"),
            "fusion1": ("x", "r", "a3", "b3"),
            "out": ("x", "r", "a3", "b3", "fusion1"),
        }
        assert {
            stage.name: stage.source_names for stage in model.config.stages
        } == expected_sources
        for stage_name in ("a1", "b1", "a2", "b2", "a3", "b3"):
            selected = _selected_source_sets(model, stage_name)
            assert selected
            assert all(
                sources.issubset(set(expected_sources[stage_name]))
                for sources in selected
            )
        fusion_sources = set().union(*_selected_source_sets(model, "fusion1"))
        assert {"a3", "b3"}.issubset(fusion_sources)
        output_sources = set().union(*_selected_source_sets(model, "out"))
        assert "fusion1" in output_sources
        mechanism_audit = {"source_schedule": expected_sources}
    if spec.model_id == "E421":
        assert all(
            stage.channel_projection == "toeplitz" for stage in model.config.stages
        )
        assert all(
            stage.coefficient_head == "factorized" for stage in model.config.stages
        )
        assert all(stage.coefficient_rank == 1 for stage in model.config.stages)
    if spec.model_id == "E425":
        assert model.architecture_metadata["shared_module_reference_count"] > 0
        assert any(
            type(module).__name__ == "_CayleyChannelProjection"
            for module in model.channel_projections.values()
        )
    runtime = None
    if spec.model_id in {"E423", "E424", "E425"}:
        model = model.to(dtype=torch.float32)
        runtime = _covariant_unit_check(model)
    result: dict[str, object] = {
        "model_id": spec.model_id,
        "actual_parameter_count": actual,
        "base_width": expected_width,
        "readout_width": expected_readout,
        "blueprint": budget["blueprint"],
        "per_stage_source_coverage": per_stage_source_coverage,
        "mechanism_audit": mechanism_audit,
        "tier_c_runtime": runtime,
    }
    del model
    gc.collect()
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_e_series_e4_compiled.py --worker MODEL_ID")
    print(
        "E4_WORKER="
        + json.dumps(_audit_one(sys.argv[2]), sort_keys=True, allow_nan=False),
        flush=True,
    )
    raise SystemExit(0)


@pytest.mark.skipif(
    os.environ.get("TFENN_RUN_E4_COMPILED_AUDIT") != "1",
    reason="explicit full E4 compilation audit",
)
def test_all_e4_frozen_budgets_manifests_and_tier_c_runtime() -> None:
    assert tuple(E4_FROZEN_BUDGETS) == tuple(spec.model_id for spec in E4_SPECS)
    actual_counts: dict[str, int] = {}
    runtime_checks: dict[str, object] = {}
    blueprint_signatures = set()
    worker = Path(__file__).resolve()
    repository_root = worker.parents[2]
    worker_environment = dict(os.environ)
    worker_environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(repository_root),
            str(repository_root / "src"),
            worker_environment.get("PYTHONPATH", ""),
        )
        if value
    )
    for spec in E4_SPECS:
        completed = subprocess.run(
            (sys.executable, str(worker), "--worker", spec.model_id),
            cwd=repository_root,
            env=worker_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, (
            spec.model_id,
            completed.stdout,
            completed.stderr,
        )
        line = next(
            value
            for value in completed.stdout.splitlines()
            if value.startswith("E4_WORKER=")
        )
        print(line, flush=True)
        record = json.loads(line.removeprefix("E4_WORKER="))
        blueprint_signatures.add(
            json.dumps(record["blueprint"], sort_keys=True, allow_nan=False)
        )
        actual_counts[spec.model_id] = int(record["actual_parameter_count"])
        if record["tier_c_runtime"] is not None:
            runtime_checks[spec.model_id] = record["tier_c_runtime"]
    assert len(blueprint_signatures) == 25
    assert tuple(runtime_checks) == ("E423", "E424", "E425")
    print(
        "E4_COMPILED_AUDIT="
        + json.dumps(
            {
                "actual_counts": actual_counts,
                "tier_c_runtime": runtime_checks,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
