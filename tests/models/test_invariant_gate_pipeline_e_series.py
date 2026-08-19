"""Exercise E series synchronous execution and path selection."""

from __future__ import annotations

import math
from types import MethodType

import pytest
import torch
from torch import Tensor

from TFENN.models import (
    InvariantGatePipelineV2,
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    build_invariant_gate_pipeline_v2,
)


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
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


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


def dual_stream_config(*, quota: int = 3) -> InvariantGatePipelineV2Config:
    common = {
        "trunk_width": 2,
        "include_symmetric_unary": False,
        "include_raw_mixed_pairs": True,
        "include_stf_shortcuts": False,
        "covariant_include_symmetric_unary": False,
        "covariant_include_raw_mixed_pairs": True,
        "covariant_include_stf_shortcuts": False,
        "invariant_include_symmetric_unary": False,
        "invariant_include_raw_mixed_pairs": True,
        "invariant_include_stf_shortcuts": False,
    }
    return InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config(
                "a0",
                "A",
                ("x",),
                1,
                invariant_source_names=("x", "r"),
                execution_level=0,
                **common,
            ),
            InvariantGateStageV2Config(
                "b0",
                "B",
                ("r",),
                1,
                invariant_source_names=("x", "r"),
                execution_level=0,
                **common,
            ),
            InvariantGateStageV2Config(
                "a1",
                "A",
                ("x", "r", "a0", "b0"),
                1,
                invariant_source_names=("x", "r", "a0", "b0"),
                execution_level=1,
                covariant_live_mixed_only=True,
                covariant_path_quota=quota,
                **common,
            ),
            InvariantGateStageV2Config(
                "b1",
                "B",
                ("x", "r", "a0", "b0"),
                1,
                invariant_source_names=("x", "r", "a0", "b0"),
                execution_level=1,
                covariant_live_mixed_only=True,
                covariant_path_quota=quota,
                **common,
            ),
            InvariantGateStageV2Config(
                "out",
                "A",
                ("x", "r", "a1", "b1"),
                1,
                invariant_source_names=("x", "r", "a0", "b0", "a1", "b1"),
                execution_level=2,
                covariant_live_mixed_only=True,
                covariant_path_quota=quota,
                **common,
            ),
        ),
        output_stage="out",
        architecture_id="e2_unit_dual_stream",
        anchor_ranks=(2,),
        max_constraint_entries=2_000_000,
        max_gate_coefficients=100_000,
        max_invariant_channels=10_000,
    )


@pytest.fixture(scope="module")
def dual_stream_model() -> InvariantGatePipelineV2:
    model = build_invariant_gate_pipeline_v2(
        proper_d6_generators(),
        dual_stream_config(),
        generator_names=("sixfold", "twofold"),
    )
    model.eval()
    return model


def test_new_stage_options_round_trip_without_changing_defaults() -> None:
    config = dual_stream_config(quota=7)
    restored = InvariantGatePipelineV2Config.from_dict(config.as_dict())
    assert restored == config
    stage = restored.stages[2]
    assert stage.execution_level == 1
    assert stage.covariant_live_mixed_only is True
    assert stage.covariant_path_quota == 7
    legacy = InvariantGateStageV2Config("out", "A", ("x",), 1)
    legacy_dict = legacy.as_dict()
    assert "execution_level" not in legacy_dict
    assert "covariant_live_mixed_only" not in legacy_dict
    assert "covariant_path_quota" not in legacy_dict
    required = InvariantGateStageV2Config(
        "required",
        "A",
        ("x", "r"),
        1,
        covariant_required_source_names=("x",),
    )
    assert InvariantGateStageV2Config.from_dict(required.as_dict()) == required
    with pytest.raises(ValueError, match="required sources"):
        InvariantGateStageV2Config(
            "invalid",
            "A",
            ("x",),
            1,
            covariant_required_source_names=("r",),
        )


def test_same_level_sibling_dependencies_are_rejected() -> None:
    common = {"execution_level": 0, "trunk_width": 1}
    with pytest.raises(ValueError, match="frozen prelevel snapshot"):
        InvariantGatePipelineV2Config(
            stages=(
                InvariantGateStageV2Config("a0", "A", ("x",), 1, **common),
                InvariantGateStageV2Config("out", "A", ("x", "a0"), 1, **common),
            ),
            output_stage="out",
            anchor_ranks=(2,),
        )


def test_compute_then_commit_exposes_one_frozen_snapshot_per_level(
    dual_stream_model: InvariantGatePipelineV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, tuple[str, ...]] = {}
    original = dual_stream_model._stage_invariants

    def recording(
        _self: InvariantGatePipelineV2,
        stage: InvariantGateStageV2Config,
        radial: Tensor,
        state: dict[str, object],
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        seen[stage.name] = tuple(state)
        return original(stage, radial, state)

    monkeypatch.setattr(
        dual_stream_model,
        "_stage_invariants",
        MethodType(recording, dual_stream_model),
    )
    dual_stream_model(*pairs(1))
    assert seen["a0"] == seen["b0"] == ("x", "r")
    assert seen["a1"] == seen["b1"] == ("x", "r", "a0", "b0")
    assert seen["out"] == ("x", "r", "a0", "b0", "a1", "b1")
    summary = dual_stream_model.offline_compilation_summary
    assert summary["execution_group_count"] == 3
    assert summary["synchronous_stage_count"] == 4


def test_post_stem_candidates_keep_live_paths_and_remove_raw_only_paths(
    dual_stream_model: InvariantGatePipelineV2,
) -> None:
    audits = tuple(
        item
        for item in dual_stream_model.candidate_manifest
        if item["bank"] == "covariant" and item["stage"] in {"a1", "b1", "out"}
    )
    roles = tuple(item["role"] for item in audits)
    assert any(".unary.a0." in role or ".unary.b0." in role for role in roles)
    assert any(".pair.a0." in role and ".b0." in role for role in roles)
    assert not any(".unary.x." in role or ".unary.r." in role for role in roles)
    assert not any(".pair.x." in role and ".r." in role for role in roles)


def test_covariant_path_quota_is_applied_per_target(
    dual_stream_model: InvariantGatePipelineV2,
) -> None:
    quota = 3
    for stage_name in ("a1", "b1", "out"):
        target_paths = dual_stream_model._stage_paths[stage_name]
        assert target_paths
        assert all(0 < len(paths) <= quota for paths in target_paths.values())
    assert dual_stream_model.offline_compilation_summary[
        "selected_covariant_path_count"
    ] == len(dual_stream_model.selected_covariant_roles)


def test_dual_stream_has_finite_gradients_and_exact_d6_covariance(
    dual_stream_model: InvariantGatePipelineV2,
) -> None:
    centers, frames = pairs(2)
    dual_stream_model.zero_grad(set_to_none=True)
    output = dual_stream_model(centers, frames)
    assert output.shape == (2, 3)
    assert bool(torch.isfinite(output).all())
    output.square().mean().backward()
    for name, parameter in dual_stream_model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name

    dual_stream_model.eval()
    one_center, one_frame = pairs(1)
    with torch.no_grad():
        reference = dual_stream_model(one_center, one_frame)
        moved = []
        for root_gauge in proper_d6_group():
            for sender_gauge in proper_d6_group():
                moved.append(
                    torch.stack(
                        (
                            one_frame[0, 0] @ root_gauge,
                            one_frame[0, 1] @ sender_gauge,
                        )
                    )
                )
        gauge_frames = torch.stack(moved)
        gauge_centers = one_center.expand(len(moved), 2, 3).clone()
        torch.testing.assert_close(
            dual_stream_model(gauge_centers, gauge_frames),
            reference.expand(len(moved), 3),
            rtol=4.0e-8,
            atol=4.0e-8,
        )
        world_rotation = rotation(torch.tensor((0.21, -0.14, 0.09), dtype=DTYPE))
        torch.testing.assert_close(
            dual_stream_model(
                one_center @ world_rotation.T,
                world_rotation @ one_frame,
            ),
            reference @ world_rotation.T,
            rtol=4.0e-8,
            atol=4.0e-8,
        )


def one_stage_config(
    *,
    path_aggregation: str = "linear",
    channel_projection: str = "dense",
    coefficient_head: str = "dense",
    coefficient_rank: int | None = None,
) -> InvariantGatePipelineV2Config:
    return InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config(
                "hidden",
                "A",
                ("x", "r"),
                3,
                trunk_width=4,
                path_aggregation=path_aggregation,
                channel_projection=channel_projection,
                coefficient_head=coefficient_head,
                coefficient_rank=coefficient_rank,
            ),
            InvariantGateStageV2Config(
                "out",
                "A",
                ("x", "r", "hidden"),
                1,
                trunk_width=4,
                path_aggregation=path_aggregation,
                channel_projection=channel_projection,
                coefficient_head=coefficient_head,
                coefficient_rank=coefficient_rank,
            ),
        ),
        output_stage="out",
        architecture_id=f"e_mechanism_{path_aggregation}_{channel_projection}_{coefficient_head}",
        anchor_ranks=(2,),
        max_constraint_entries=2_000_000,
        max_gate_coefficients=100_000,
    )


def test_attention_and_soft_moe_change_actual_branch_aggregation() -> None:
    models = {
        policy: build_invariant_gate_pipeline_v2(
            proper_d6_generators(), one_stage_config(path_aggregation=policy)
        ).eval()
        for policy in ("linear", "attention", "soft_moe")
    }
    reference_parameters = dict(models["linear"].named_parameters())
    with torch.no_grad():
        for model in (models["attention"], models["soft_moe"]):
            parameters = dict(model.named_parameters())
            assert parameters.keys() == reference_parameters.keys()
            for name, parameter in parameters.items():
                parameter.copy_(reference_parameters[name])
    centers, frames = pairs(2)
    with torch.no_grad():
        outputs = {name: model(centers, frames) for name, model in models.items()}
    assert not torch.allclose(outputs["linear"], outputs["attention"])
    assert not torch.allclose(outputs["attention"], outputs["soft_moe"])
    assert all(bool(torch.isfinite(value).all()) for value in outputs.values())
    for model in (models["attention"], models["soft_moe"]):
        path_role = model.coefficient_head_role_manifest[0]["role"]
        with pytest.raises(ValueError, match="linear path aggregation"):
            model.set_covariant_path_activity(path_role, False)
        carrier_role = next(
            item["role"]
            for item in model.projection_input_role_manifest
            if item["kind"] == "legacy_carrier"
        )
        with pytest.raises(ValueError, match="linear path aggregation"):
            model.set_legacy_carrier_activity(carrier_role, False)


def test_parameter_share_group_reuses_module_objects_and_parameters() -> None:
    def build(shared: bool) -> InvariantGatePipelineV2:
        stages = tuple(
            InvariantGateStageV2Config(
                name,
                "A",
                ("x", "r"),
                1,
                trunk_width=2,
                parameter_share_group="tied" if shared else None,
                covariant_include_symmetric_unary=False,
                covariant_include_raw_mixed_pairs=False,
                covariant_include_stf_shortcuts=False,
                invariant_include_symmetric_unary=False,
                invariant_include_raw_mixed_pairs=False,
                invariant_include_stf_shortcuts=False,
            )
            for name in ("cell0", "out")
        )
        return build_invariant_gate_pipeline_v2(
            proper_d6_generators(),
            InvariantGatePipelineV2Config(
                stages=stages,
                output_stage="out",
                anchor_ranks=(2,),
                max_constraint_entries=2_000_000,
                max_gate_coefficients=100_000,
            ),
        )

    tied = build(True)
    untied = build(False)
    assert tied.stage_trunks["cell0"] is tied.stage_trunks["out"]
    assert tied.architecture_metadata["shared_module_reference_count"] > 0
    assert tied.trainable_parameter_count < untied.trainable_parameter_count
    assert tied(*pairs(1)).shape == (1, 3)


def test_cayley_mixer_has_an_orthogonal_learned_multiplicity_factor() -> None:
    model = build_invariant_gate_pipeline_v2(
        proper_d6_generators(), one_stage_config(channel_projection="cayley")
    )
    cayley = next(
        module
        for module in model.channel_projections.values()
        if type(module).__name__ == "_CayleyChannelProjection"
    )
    with torch.no_grad():
        cayley.skew_values.copy_(
            torch.linspace(
                -0.3,
                0.3,
                cayley.skew_values.numel(),
                dtype=cayley.skew_values.dtype,
            )
        )
        skew = cayley.base.new_zeros((cayley.output_channels, cayley.output_channels))
        row, column = cayley.upper_indices.unbind(dim=0)
        skew[row, column] = cayley.skew_values
        skew[column, row] = -cayley.skew_values
        identity = torch.eye(cayley.output_channels, dtype=cayley.base.dtype)
        factor = torch.linalg.solve(identity + skew, identity - skew)
    torch.testing.assert_close(
        factor.mT @ factor,
        identity,
        rtol=2.0e-10,
        atol=2.0e-10,
    )


def test_context_lora_head_has_nonzero_context_conditioned_update() -> None:
    model = build_invariant_gate_pipeline_v2(
        proper_d6_generators(),
        one_stage_config(coefficient_head="context_lora", coefficient_rank=2),
    )
    head = next(
        module
        for module in model.path_heads.values()
        if type(module).__name__ == "_ContextLoRACoefficientHead"
    )
    value = torch.tensor(
        ((0.2, -0.4, 0.7, 0.1), (-0.5, 0.3, 0.2, 0.8)),
        dtype=DTYPE,
    )
    update = head(value) - head.base(value)
    assert bool(torch.isfinite(update).all())
    assert float(update.detach().abs().sum()) > 0.0
    assert not torch.allclose(update[0], update[1])


def test_reversible_coupling_changes_forward_and_has_finite_gradients() -> None:
    config = InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config(
                "hidden",
                "A",
                ("x", "r"),
                2,
                trunk_width=3,
                skip_policy="local_proj",
                reversible_coupling=True,
            ),
            InvariantGateStageV2Config(
                "out",
                "A",
                ("x", "r", "hidden"),
                1,
                trunk_width=3,
                skip_policy="local_proj",
            ),
        ),
        output_stage="out",
        architecture_id="e_reversible_coupling_unit",
        anchor_ranks=(2,),
        max_constraint_entries=2_000_000,
        max_gate_coefficients=100_000,
    )
    model = build_invariant_gate_pipeline_v2(proper_d6_generators(), config)
    assert model.reversible_couplings
    centers, frames = pairs(2)
    with torch.no_grad():
        for parameter in model.reversible_couplings.parameters():
            parameter.zero_()
        baseline = model(centers, frames)
        for parameter in model.reversible_couplings.parameters():
            parameter.fill_(0.1)
        changed = model(centers, frames)
    assert not torch.allclose(baseline, changed)
    model.zero_grad(set_to_none=True)
    model(centers, frames).square().mean().backward()
    coupling_parameter_ids = {
        id(parameter) for parameter in model.reversible_couplings.parameters()
    }
    coupling_gradients = tuple(
        parameter.grad
        for parameter in model.parameters()
        if id(parameter) in coupling_parameter_ids
    )
    assert coupling_gradients
    assert all(gradient is not None for gradient in coupling_gradients)
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in coupling_gradients
        if gradient is not None
    )
