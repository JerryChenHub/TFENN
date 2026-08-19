"""Verify exact declared edge strict dual stream compilation."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

import TFENN.models.strict_dual_stream_flow as strict_module
from TFENN.models import (
    StrictDualStreamFlowConfig,
    StrictFlowCompilationError,
    StrictFlowStageConfig,
    build_invariant_gate_pipeline_v2,
    build_strict_dual_stream_flow,
    compile_strict_dual_stream_config,
)
from TFENN.models.invariant_gate_pipeline_v2 import A


DTYPE = torch.float64


def proper_d6_generators() -> Tensor:
    cosine = math.cos(math.pi / 3.0)
    sine = math.sin(math.pi / 3.0)
    sixfold = torch.tensor(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=DTYPE,
    )
    twofold = torch.diag(torch.tensor((1.0, -1.0, -1.0), dtype=DTYPE))
    return torch.stack((sixfold, twofold))


def proper_d6_group() -> Tensor:
    sixfold, twofold = proper_d6_generators()
    powers = tuple(torch.linalg.matrix_power(sixfold, power) for power in range(6))
    return torch.stack((*powers, *(power @ twofold for power in powers)))


def skew(vector: Tensor) -> Tensor:
    x_value, y_value, z_value = vector.unbind(dim=-1)
    zero = torch.zeros_like(x_value)
    return torch.stack(
        (
            zero,
            -z_value,
            y_value,
            z_value,
            zero,
            -x_value,
            -y_value,
            x_value,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def rotation(vector: Tensor) -> Tensor:
    return torch.matrix_exp(skew(vector))


def pairs(count: int = 2) -> tuple[Tensor, Tensor]:
    index = torch.arange(count, dtype=DTYPE)
    centers = torch.zeros((count, 2, 3), dtype=DTYPE)
    centers[:, 1] = torch.stack(
        (4.4 + 0.1 * index, -0.6 + 0.04 * index, 3.8 - 0.03 * index),
        dim=-1,
    )
    root = rotation(
        torch.stack((0.13 + 0.01 * index, -0.08 * index, 0.07 * index), dim=-1)
    )
    sender = rotation(
        torch.stack((-0.11 * index, 0.17 + 0.02 * index, 0.09 * index), dim=-1)
    )
    return centers, torch.stack((root, sender), dim=1)


def strict_config(
    topology: str,
    *,
    descriptor_mask: str = "full",
) -> StrictDualStreamFlowConfig:
    stage = StrictFlowStageConfig
    raw = (
        stage("a1", "A", ("x",), 1, ("x", "r"), 0),
        stage("b1", "B", ("r",), 2, ("x", "r"), 0),
    )
    if topology == "T1":
        hidden = (
            stage(
                "a2",
                "A",
                ("a1", "b1"),
                1,
                ("x", "r", "a1", "b1"),
                1,
            ),
            stage(
                "b2",
                "B",
                ("b1", "a1"),
                1,
                ("x", "r", "b1", "a1"),
                1,
            ),
            stage(
                "a3",
                "A",
                ("a2", "b2"),
                1,
                ("x", "r", "a2", "b2"),
                2,
            ),
        )
    elif topology == "T2":
        hidden = (
            stage(
                "a2",
                "A",
                ("a1", "b1"),
                1,
                ("x", "r", "a1", "b1"),
                1,
            ),
            stage("a3", "A", ("a2",), 1, ("x", "r", "a2"), 2),
        )
    elif topology == "T3":
        hidden = (
            stage("a2", "A", ("a1",), 1, ("x", "r", "a1"), 1),
            stage("b2", "B", ("b1",), 1, ("x", "r", "b1"), 1),
            stage(
                "a3",
                "A",
                ("a2", "b2"),
                1,
                ("x", "r", "a2", "b2"),
                2,
            ),
        )
    else:
        raise ValueError(topology)
    stages = (
        *raw,
        *hidden,
        stage("out", "A", ("a3",), 1, ("x", "r", "a3"), 3),
    )
    return StrictDualStreamFlowConfig(
        stages,
        architecture_id=f"strict_{topology.lower()}_{descriptor_mask}",
        descriptor_mask=descriptor_mask,
    )


def test_strict_config_round_trip_and_exact_lowering() -> None:
    config = strict_config("T1")
    assert StrictDualStreamFlowConfig.from_dict(config.as_dict()) == config
    lowered = compile_strict_dual_stream_config(config)
    assert tuple(stage.source_names for stage in lowered.stages) == tuple(
        stage.source_names for stage in config.stages
    )
    assert tuple(stage.invariant_source_names for stage in lowered.stages) == tuple(
        stage.invariant_source_names for stage in config.stages
    )
    assert all(stage.skip_policy == "legacy" for stage in lowered.stages)
    assert all(
        stage.covariant_required_source_names == stage.source_names
        for stage in lowered.stages
    )
    assert all(
        stage.covariant_include_raw_mixed_pairs is False
        and stage.invariant_include_raw_mixed_pairs is True
        and stage.degree3_policy == "none"
        for stage in lowered.stages
    )


def test_strict_config_rejects_raw_rereads_and_same_level_dependencies() -> None:
    stage = StrictFlowStageConfig
    with pytest.raises(ValueError, match="raw x"):
        StrictDualStreamFlowConfig(
            (
                stage("a1", "A", ("x",), 1, ("x", "r"), 0),
                stage("b1", "B", ("r",), 2, ("x", "r"), 0),
                stage("a2", "A", ("x", "a1"), 1, ("x", "r", "a1"), 1),
                stage("out", "A", ("a2",), 1, ("x", "r", "a2"), 2),
            )
        )
    with pytest.raises(ValueError, match="same level"):
        StrictDualStreamFlowConfig(
            (
                stage("a1", "A", ("x",), 1, ("x", "r"), 0),
                stage("b1", "B", ("r",), 2, ("x", "r", "a1"), 0),
                stage("out", "A", ("a1",), 1, ("x", "r", "a1"), 1),
            )
        )


@pytest.mark.parametrize(
    ("topology", "expected_count"),
    (("T1", 12_635), ("T2", 6_972), ("T3", 11_137)),
)
def test_planned_baseline_parameter_counts_and_strict_edge_manifest(
    topology: str,
    expected_count: int,
) -> None:
    model = build_strict_dual_stream_flow(
        proper_d6_generators(),
        strict_config(topology),
        generator_names=("sixfold", "twofold"),
    )
    assert model.trainable_parameter_count == expected_count
    edge_audit = model.strict_flow_manifest["edge_audit"]
    assert edge_audit
    assert all(not item["missing_edges"] for item in edge_audit)
    assert all(not item["undeclared_sources"] for item in edge_audit)
    assert all(not item["live_live_path_roles"] for item in edge_audit)
    assert all(
        all(edge["covered"] for edge in item["same_type_edges"])
        and all(edge["covered"] for edge in item["cross_type_edges"])
        for item in edge_audit
    )


def test_full_and_raw_only_share_schema_parameters_and_initialization() -> None:
    generators = proper_d6_generators()
    torch.manual_seed(20260818)
    full = build_strict_dual_stream_flow(
        generators,
        strict_config("T2", descriptor_mask="full"),
    )
    torch.manual_seed(20260818)
    raw_only = build_strict_dual_stream_flow(
        generators,
        strict_config("T2", descriptor_mask="raw_only"),
    )
    assert full.trainable_parameter_count == raw_only.trainable_parameter_count
    assert full.candidate_manifest == raw_only.candidate_manifest
    assert full.coefficient_head_role_manifest == (
        raw_only.coefficient_head_role_manifest
    )
    full_parameters = dict(full.named_parameters())
    raw_parameters = dict(raw_only.named_parameters())
    assert full_parameters.keys() == raw_parameters.keys()
    for name in full_parameters:
        torch.testing.assert_close(full_parameters[name], raw_parameters[name])
    full_roles = {
        (item["stage"], item["role"]): item
        for item in full.descriptor_role_manifest
    }
    raw_roles = {
        (item["stage"], item["role"]): item
        for item in raw_only.descriptor_role_manifest
    }
    assert full_roles.keys() == raw_roles.keys()
    assert all(
        (item["start"], item["stop"])
        == (raw_roles[key]["start"], raw_roles[key]["stop"])
        for key, item in full_roles.items()
    )
    hidden_roles = tuple(
        key
        for key, item in raw_roles.items()
        if any(source not in {"x", "r"} for source in item["source_names"])
    )
    assert hidden_roles
    assert all(full_roles[key]["active"] for key in hidden_roles)
    assert all(not raw_roles[key]["active"] for key in hidden_roles)
    scalar_pairs = tuple(
        item
        for item in full.descriptor_role_manifest
        if item["kind"] == "pair"
    )
    assert any(
        any(source in {"x", "r"} for source in item["source_names"])
        and any(source not in {"x", "r"} for source in item["source_names"])
        for item in scalar_pairs
    )
    assert not any(
        all(source not in {"x", "r"} for source in item["source_names"])
        for item in scalar_pairs
    )


def test_postcompile_audit_rejects_a_missing_cross_type_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generators = proper_d6_generators()
    config = strict_config("T2")
    lowered = compile_strict_dual_stream_config(config)
    broken = build_invariant_gate_pipeline_v2(generators, lowered)
    broken._stage_paths["a2"][A] = tuple(
        path
        for path in broken._stage_paths["a2"][A]
        if all(endpoint.source != "b1" for endpoint in path.candidate.endpoints)
    )
    monkeypatch.setattr(
        strict_module,
        "build_invariant_gate_pipeline_v2",
        lambda *_args, **_kwargs: broken,
    )
    with pytest.raises(StrictFlowCompilationError, match="edge audit failed"):
        build_strict_dual_stream_flow(generators, config)


def test_gate_audit_hooks_preserve_mode_and_report_reshaped_gamma() -> None:
    model = build_strict_dual_stream_flow(
        proper_d6_generators(),
        strict_config("T2", descriptor_mask="raw_only"),
    )
    model.train()
    centers, frames = pairs(2)
    activations = model.collect_coefficient_activations(centers, frames)
    assert model.training is True
    assert activations.keys() == model.coefficient_head_modules_by_role().keys()
    manifest = {
        item["role"]: item for item in model.coefficient_head_role_manifest
    }
    assert all(
        value.shape[-2:]
        == (
            manifest[role]["target_channels"],
            manifest[role]["primitive_channels"],
        )
        for role, value in activations.items()
    )
    assert model.first_trunk_linear_modules_by_stage().keys() == {
        stage.name for stage in model.config.stages
    }


def test_strict_model_has_finite_gradients_and_exact_covariance() -> None:
    model = build_strict_dual_stream_flow(
        proper_d6_generators(),
        strict_config("T2"),
    )
    centers, frames = pairs(2)
    output = model(centers, frames)
    assert output.shape == (2, 3)
    assert bool(torch.isfinite(output).all())
    output.square().mean().backward()
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    model.eval()
    one_center, one_frame = pairs(1)
    with torch.no_grad():
        reference = model(one_center, one_frame)
        moved = tuple(
            torch.stack(
                (
                    one_frame[0, 0] @ root_gauge,
                    one_frame[0, 1] @ sender_gauge,
                )
            )
            for root_gauge in proper_d6_group()
            for sender_gauge in proper_d6_group()
        )
        gauge_frames = torch.stack(moved)
        gauge_centers = one_center.expand(len(moved), 2, 3).clone()
        torch.testing.assert_close(
            model(gauge_centers, gauge_frames),
            reference.expand(len(moved), 3),
            rtol=4.0e-8,
            atol=4.0e-8,
        )
        world_rotation = rotation(torch.tensor((0.21, -0.14, 0.09), dtype=DTYPE))
        torch.testing.assert_close(
            model(
                one_center @ world_rotation.T,
                world_rotation @ one_frame,
            ),
            reference @ world_rotation.T,
            rtol=4.0e-8,
            atol=4.0e-8,
        )
