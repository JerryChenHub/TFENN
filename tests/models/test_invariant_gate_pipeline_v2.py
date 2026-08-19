from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import Tensor

import TFENN.models.invariant_gate_pipeline_v2 as pipeline_module
from TFENN.models.invariant_gate_pipeline_v2 import (
    InvariantGatePipelineV2,
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    PipelineV2CompilationError,
    build_invariant_gate_pipeline_v2,
)
from TFENN.tensor_math import TypeKey, stf_representation


DTYPE = torch.float64


def skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def rotation(vector: Tensor) -> Tensor:
    return torch.matrix_exp(skew(vector))


def d3_generators() -> Tensor:
    angle = 2.0 * math.pi / 3.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        (
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
        ),
        dtype=DTYPE,
    )


def d3_group() -> Tensor:
    threefold, twofold = d3_generators()
    powers = tuple(torch.linalg.matrix_power(threefold, power) for power in range(3))
    return torch.stack(powers + tuple(value @ twofold for value in powers))


def c3_generators() -> Tensor:
    return rotation(torch.tensor((0.0, 0.0, 2.0 * math.pi / 3.0), dtype=DTYPE))[None]


def tetrahedral_generators() -> Tensor:
    return torch.stack(
        (
            rotation(torch.tensor((math.pi, 0.0, 0.0), dtype=DTYPE)),
            torch.tensor(
                ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                dtype=DTYPE,
            ),
        )
    )


def octahedral_generators() -> Tensor:
    return torch.stack(
        (
            rotation(torch.tensor((math.pi / 2.0, 0.0, 0.0), dtype=DTYPE)),
            rotation(torch.tensor((0.0, 0.0, math.pi / 2.0), dtype=DTYPE)),
        )
    )


def sample_pairs(count: int) -> tuple[Tensor, Tensor]:
    index = torch.arange(count, dtype=DTYPE)
    centers = torch.zeros((count, 2, 3), dtype=DTYPE)
    centers[:, 0] = torch.stack(
        (0.07 * index, -0.04 * index, 0.03 * index),
        dim=-1,
    )
    centers[:, 1] = centers[:, 0] + torch.stack(
        (4.6 + 0.08 * index, -0.9 + 0.04 * index, 4.2 - 0.05 * index),
        dim=-1,
    )
    root_vectors = torch.stack(
        (0.13 + 0.01 * index, -0.18 + 0.02 * index, 0.09 - 0.01 * index),
        dim=-1,
    )
    sender_vectors = torch.stack(
        (-0.22 + 0.02 * index, 0.12 - 0.01 * index, 0.17 + 0.01 * index),
        dim=-1,
    )
    frames = torch.stack((rotation(root_vectors), rotation(sender_vectors)), dim=1)
    return centers, frames


def compact_config(
    *,
    max_gate_coefficients: int = 100_000,
    mixed: bool = True,
) -> InvariantGatePipelineV2Config:
    common = {
        "trunk_width": 8,
        "include_raw_mixed_pairs": mixed,
        "include_stf_shortcuts": mixed,
    }
    return InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config(
                "hidden",
                "A",
                ("x", "r"),
                2,
                **common,
            ),
            InvariantGateStageV2Config(
                "bridge",
                "B",
                ("x", "r", "hidden"),
                2,
                **common,
            ),
            InvariantGateStageV2Config(
                "readout",
                "A",
                ("x", "r", "hidden", "bridge"),
                1,
                **common,
            ),
        ),
        output_stage="readout",
        anchor_ranks=(2, 3),
        max_constraint_entries=2_000_000,
        max_gate_coefficients=max_gate_coefficients,
        max_invariant_channels=10_000,
    )


def one_stage_config(
    anchor_ranks: tuple[int, ...],
    *,
    trunk_width: int = 8,
) -> InvariantGatePipelineV2Config:
    return InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config(
                "readout",
                "A",
                ("x", "r"),
                1,
                trunk_width=trunk_width,
            ),
        ),
        output_stage="readout",
        anchor_ranks=anchor_ranks,
        max_constraint_entries=2_000_000,
        max_gate_coefficients=100_000,
        max_invariant_channels=10_000,
    )


@pytest.fixture(scope="module")
def network() -> InvariantGatePipelineV2:
    torch.manual_seed(20260813)
    model = build_invariant_gate_pipeline_v2(
        d3_generators(),
        compact_config(),
        generator_names=("threefold", "twofold"),
    )
    model.eval()
    return model


def _raw_snapshot(debug: object) -> dict[str, dict[TypeKey, Tensor]]:
    state = debug.state
    return {
        source: {key: value.detach().clone() for key, value in state[source].items()}
        for source in ("x", "r")
    }


def test_typed_state_retains_raw_sources_and_every_stage(
    network: InvariantGatePipelineV2,
) -> None:
    centers, frames = sample_pairs(3)
    debug = network.debug_forward(centers, frames)

    assert tuple(debug.state) == ("x", "r", "hidden", "bridge", "readout")
    assert tuple(debug.state["x"]) == (TypeKey("A"),)
    assert tuple(debug.state["r"]) == tuple(item.key for item in network.manifest)
    assert debug.state["x"][TypeKey("A")].shape == (3, 1, 3)
    assert all(value.shape[0] == 3 for value in debug.state["r"].values())


def test_zero_stage_updates_do_not_mutate_raw_entries(
    network: InvariantGatePipelineV2,
) -> None:
    model = build_invariant_gate_pipeline_v2(d3_generators(), compact_config())
    centers, frames = sample_pairs(2)
    before = _raw_snapshot(model.debug_forward(centers, frames))
    with torch.no_grad():
        for projection in model.channel_projections.values():
            projection.weight.zero_()
    after_debug = model.debug_forward(centers, frames)
    after = _raw_snapshot(after_debug)

    for source in before:
        for key in before[source]:
            torch.testing.assert_close(
                after[source][key], before[source][key], rtol=0, atol=0
            )
    for source in ("hidden", "bridge", "readout"):
        for value in after_debug.state[source].values():
            torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)


def test_stage_branches_are_concatenated_without_early_summation(
    network: InvariantGatePipelineV2,
) -> None:
    centers, frames = sample_pairs(2)
    debug = network.debug_forward(centers, frames)
    stage_by_name = {stage.name: stage for stage in network.config.stages}

    for stage_name, target_blocks in debug.branches.items():
        stage = stage_by_name[stage_name]
        skip_names = (
            stage.source_names
            if stage.skip_source_names is None
            else stage.skip_source_names
        )
        for target, branches in target_blocks.items():
            skip = tuple(
                debug.state[name][target]
                for name in skip_names
                if target in debug.state[name]
            )
            assert len(branches) >= 2
            expected = torch.cat((*skip, *branches), dim=-2)
            actual = debug.concats[stage_name][target]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.shape[-2] == sum(
                value.shape[-2] for value in (*skip, *branches)
            )


def test_final_readout_manifest_names_raw_and_hidden_sources(
    network: InvariantGatePipelineV2,
) -> None:
    assert network.readout_source_manifest == ("x", "r", "hidden", "bridge")
    manifest = network.get_extra_state()
    assert manifest["readout_sources"] == network.readout_source_manifest


def test_mixed_features_respond_to_pose_and_displacement_direction(
    network: InvariantGatePipelineV2,
) -> None:
    centers, frames = sample_pairs(1)
    reference = network.debug_forward(centers, frames)

    changed_pose = frames.clone()
    changed_pose[:, 1] = rotation(torch.tensor(((0.71, -0.33, 0.26),), dtype=DTYPE))
    pose_debug = network.debug_forward(centers, changed_pose)
    assert any(
        not torch.allclose(reference.invariants[name], pose_debug.invariants[name])
        for name in reference.invariants
    )

    changed_direction = centers.clone()
    distance = torch.linalg.vector_norm(centers[:, 1] - centers[:, 0], dim=-1)
    changed_direction[:, 1] = centers[:, 0] + torch.stack(
        (torch.zeros_like(distance), distance, torch.zeros_like(distance)), dim=-1
    )
    direction_debug = network.debug_forward(changed_direction, frames)
    torch.testing.assert_close(
        torch.linalg.vector_norm(
            changed_direction[:, 1] - changed_direction[:, 0], dim=-1
        ),
        distance,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    mixed_roles = tuple(
        role
        for role in reference.direct_paths
        if ".pair.x." in role and role.split(".")[1] == "a"
    )
    assert mixed_roles
    assert any(
        not torch.allclose(
            reference.direct_paths[role], direction_debug.direct_paths[role]
        )
        for role in mixed_roles
    )


def test_each_scalar_and_direct_feature_obeys_its_transformation_law(
    network: InvariantGatePipelineV2,
) -> None:
    centers, frames = sample_pairs(1)
    reference = network.debug_forward(centers, frames)
    root_gauge = d3_group()[1]
    gauged_frames = frames.clone()
    gauged_frames[:, 0] = frames[:, 0] @ root_gauge
    gauged = network.debug_forward(centers, gauged_frames)

    for stage_name in reference.invariants:
        torch.testing.assert_close(
            gauged.invariants[stage_name],
            reference.invariants[stage_name],
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    targets = {
        path.candidate.role: path.candidate.target
        for target_paths in network._stage_paths.values()
        for paths in target_paths.values()
        for path in paths
    }
    rank_by_key = {item.key: item.stf_rank for item in network.manifest}
    for role, reference_value in reference.direct_paths.items():
        target = targets[role]
        assert target is not None
        representation = (
            root_gauge
            if target.stream == "A"
            else stf_representation(root_gauge, rank_by_key[target])
        )
        torch.testing.assert_close(
            gauged.direct_paths[role],
            reference_value @ representation,
            rtol=2.0e-9,
            atol=2.0e-9,
        )


def test_whole_pipeline_obeys_all_d3_gauges_and_world_rotation(
    network: InvariantGatePipelineV2,
) -> None:
    centers, frames = sample_pairs(1)
    with torch.no_grad():
        reference_local = network.forward_local(centers, frames)
        reference_world = network(centers, frames)
        moved_frames = []
        expected_local = []
        for root_gauge in d3_group():
            for sender_gauge in d3_group():
                moved_frames.append(
                    torch.stack(
                        (
                            frames[0, 0] @ root_gauge,
                            frames[0, 1] @ sender_gauge,
                        )
                    )
                )
                expected_local.append(reference_local[0] @ root_gauge)
        gauge_frames = torch.stack(moved_frames)
        gauge_centers = centers.expand(len(moved_frames), 2, 3).clone()
        torch.testing.assert_close(
            network.forward_local(gauge_centers, gauge_frames),
            torch.stack(expected_local),
            rtol=2.0e-9,
            atol=2.0e-9,
        )
        torch.testing.assert_close(
            network(gauge_centers, gauge_frames),
            reference_world.expand(len(moved_frames), 3),
            rtol=2.0e-9,
            atol=2.0e-9,
        )

        world_rotation = rotation(torch.tensor((0.23, -0.17, 0.11), dtype=DTYPE))
        translation = torch.tensor((0.4, -0.3, 0.2), dtype=DTYPE)
        torch.testing.assert_close(
            network(
                centers @ world_rotation.T + translation,
                world_rotation @ frames,
            ),
            reference_world @ world_rotation.T,
            rtol=2.0e-9,
            atol=2.0e-9,
        )


@pytest.mark.parametrize(
    ("group_name", "generators", "anchor_ranks"),
    (
        ("C3", c3_generators(), (1,)),
        ("D3", d3_generators(), (2, 3)),
        ("T", tetrahedral_generators(), (3,)),
        ("O", octahedral_generators(), (4,)),
    ),
)
def test_group_general_pipeline_on_representative_groups(
    group_name: str,
    generators: Tensor,
    anchor_ranks: tuple[int, ...],
) -> None:
    model = build_invariant_gate_pipeline_v2(
        generators,
        one_stage_config(anchor_ranks),
        generator_names=tuple(
            f"{group_name}_generator_{index}" for index in range(len(generators))
        ),
    )
    centers, frames = sample_pairs(1)
    reference = model.debug_forward(centers, frames)
    for gauge in generators:
        root_frames = frames.clone()
        root_frames[:, 0] = frames[:, 0] @ gauge
        root_debug = model.debug_forward(centers, root_frames)
        for stage_name in reference.invariants:
            torch.testing.assert_close(
                root_debug.invariants[stage_name],
                reference.invariants[stage_name],
                rtol=3.0e-8,
                atol=3.0e-8,
            )
        torch.testing.assert_close(
            root_debug.output_local,
            reference.output_local @ gauge,
            rtol=3.0e-8,
            atol=3.0e-8,
        )
        torch.testing.assert_close(
            root_debug.output_world,
            reference.output_world,
            rtol=3.0e-8,
            atol=3.0e-8,
        )

        sender_frames = frames.clone()
        sender_frames[:, 1] = frames[:, 1] @ gauge
        torch.testing.assert_close(
            model(centers, sender_frames),
            reference.output_world,
            rtol=3.0e-8,
            atol=3.0e-8,
        )

    world_rotation = rotation(torch.tensor((0.21, -0.16, 0.12), dtype=DTYPE))
    torch.testing.assert_close(
        model(centers @ world_rotation.T, world_rotation @ frames),
        reference.output_world @ world_rotation.T,
        rtol=3.0e-8,
        atol=3.0e-8,
    )


def test_pose_encoder_right_orbits_and_current_fixture_separation(
    network: InvariantGatePipelineV2,
) -> None:
    first = rotation(torch.tensor((0.13, -0.24, 0.31), dtype=DTYPE))
    second = rotation(torch.tensor((-0.41, 0.22, 0.17), dtype=DTYPE))
    encoded_first = network.pose_encoder(first)
    for group_element in d3_group():
        torch.testing.assert_close(
            network.pose_encoder(first @ group_element),
            encoded_first,
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    orbit_distance = min(
        float(torch.linalg.vector_norm(second - first @ group_element))
        for group_element in d3_group()
    )
    encoding_distance = float(
        torch.linalg.vector_norm(network.pose_encoder(second) - encoded_first)
    )
    assert orbit_distance > 0.1
    assert encoding_distance > 0.1


def test_each_stage_trunk_runs_once_per_forward(
    network: InvariantGatePipelineV2,
) -> None:
    calls = {name: 0 for name in network.trunks}
    handles = []
    for name, trunk in network.trunks.items():
        handles.append(
            trunk.register_forward_hook(
                lambda _module, _inputs, _output, stage=name: calls.__setitem__(
                    stage, calls[stage] + 1
                )
            )
        )
    try:
        network(*sample_pairs(2))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {name: 1 for name in network.trunks}


def test_parameters_are_auditable_and_compiled_assets_are_buffers(
    network: InvariantGatePipelineV2,
) -> None:
    parameter_count = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    assert network.trainable_parameter_count == parameter_count
    assert (
        network.offline_compilation_summary["trainable_parameter_count"]
        == parameter_count
    )
    assert not any(
        name.startswith(("pose_encoder.", "covariants."))
        for name, _parameter in network.named_parameters()
    )
    buffer_names = tuple(name for name, _buffer in network.named_buffers())
    assert any(name.startswith("pose_encoder._anchor_rank_") for name in buffer_names)
    assert any(
        name.startswith("covariants.") and "basis" in name for name in buffer_names
    )
    assert all(not buffer.requires_grad for _name, buffer in network.named_buffers())

    reduced = build_invariant_gate_pipeline_v2(
        d3_generators(), compact_config(mixed=False)
    )
    assert reduced.trainable_parameter_count < network.trainable_parameter_count


def test_running_rms_preserves_scalar_magnitude_and_freezes_in_eval(
    network: InvariantGatePipelineV2,
) -> None:
    original_state = network.normalization_state_dict()
    was_training = network.training
    try:
        network.reset_normalization_stats()
        network.train()
        values = torch.tensor(((1.0,), (2.0,), (4.0,)), dtype=DTYPE)
        normalized = network._normalize_schema(values, "hidden", 0)
        expected_rms = math.sqrt(7.0 + network.config.radial.rms_epsilon)
        torch.testing.assert_close(normalized, values / expected_rms)
        assert normalized[1, 0] == pytest.approx(2.0 * normalized[0, 0])
        assert normalized[2, 0] == pytest.approx(4.0 * normalized[0, 0])

        network._normalize_schema(torch.tensor(((8.0,),), dtype=DTYPE), "hidden", 0)
        trained_state = network.normalization_state_dict()
        assert trained_state["hidden.0.mean_square"].item() == pytest.approx(21.25)
        assert trained_state["hidden.0.sample_count"].item() == 4

        network.eval()
        evaluated = network._normalize_schema(
            torch.tensor(((16.0,),), dtype=DTYPE), "hidden", 0
        )
        expected_eval_rms = math.sqrt(21.25 + network.config.radial.rms_epsilon)
        torch.testing.assert_close(
            evaluated, evaluated.new_tensor(((16.0 / expected_eval_rms,),))
        )
        frozen_state = network.normalization_state_dict()
        assert frozen_state.keys() == trained_state.keys()
        for key in frozen_state:
            torch.testing.assert_close(frozen_state[key], trained_state[key])
    finally:
        network.load_normalization_state_dict(original_state)
        network.train(was_training)


def test_normalization_state_is_nontrainable_compact_and_restorable(
    network: InvariantGatePipelineV2,
) -> None:
    original_state = network.normalization_state_dict()
    was_training = network.training
    parameter_count = network.trainable_parameter_count
    try:
        network.reset_normalization_stats()
        network.train()
        network(*sample_pairs(3))
        trained_state = network.normalization_state_dict()
        assert trained_state
        assert all(value.device.type == "cpu" for value in trained_state.values())
        assert any(
            int(value.item()) > 0
            for key, value in trained_state.items()
            if key.endswith("sample_count")
        )
        assert not any(
            name.startswith("normalization.")
            for name, _parameter in network.named_parameters()
        )
        assert network.trainable_parameter_count == parameter_count

        network.reset_normalization_stats()
        assert all(
            int(value.item()) == 0
            for key, value in network.normalization_state_dict().items()
            if key.endswith("sample_count")
        )
        network.load_normalization_state_dict(trained_state)
        restored = network.normalization_state_dict()
        for key in trained_state:
            torch.testing.assert_close(restored[key], trained_state[key])

        missing = dict(trained_state)
        missing.pop(next(iter(missing)))
        with pytest.raises(ValueError, match="keys do not match"):
            network.load_normalization_state_dict(missing)
    finally:
        network.load_normalization_state_dict(original_state)
        network.train(was_training)


def test_forward_never_calls_offline_compilers_or_svd(
    network: InvariantGatePipelineV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forward called an offline operation")

    monkeypatch.setattr(pipeline_module, "compile_anchors", forbidden)
    monkeypatch.setattr(pipeline_module, "compile_covariant_basis", forbidden)
    monkeypatch.setattr(pipeline_module, "build_type_catalog", forbidden)
    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    output = network(*sample_pairs(2))
    assert output.shape == (2, 3)
    assert network.offline_compilation_summary["forward_compilation"] is False
    assert network.offline_compilation_summary["runtime_group_expansion"] is False


def test_candidate_manifest_records_statuses_and_budget_failures() -> None:
    model = build_invariant_gate_pipeline_v2(d3_generators(), compact_config())
    allowed = {"compiled", "empty_hom", "over_budget", "failed"}
    assert model.candidate_manifest
    assert {item["status"] for item in model.candidate_manifest} <= allowed
    assert all(
        item["stage"] and item["bank"] and item["role"] and item["signature"]
        for item in model.candidate_manifest
    )
    assert any(item["status"] == "compiled" for item in model.candidate_manifest)
    assert any(item["status"] == "empty_hom" for item in model.candidate_manifest)

    with pytest.raises(PipelineV2CompilationError) as error:
        build_invariant_gate_pipeline_v2(
            d3_generators(), compact_config(max_gate_coefficients=1)
        )
    over_budget = tuple(
        item
        for item in error.value.candidate_manifest
        if item["status"] == "over_budget"
    )
    assert over_budget
    assert all(item["estimated_parameter_count"] is not None for item in over_budget)
    assert all(item["reason"] for item in over_budget)


def test_failed_compilation_is_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pipeline_module.compile_covariant_basis
    calls = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected compiler failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "compile_covariant_basis", fail_once)
    with pytest.raises(PipelineV2CompilationError) as error:
        build_invariant_gate_pipeline_v2(d3_generators(), compact_config())
    failed = tuple(
        item for item in error.value.candidate_manifest if item["status"] == "failed"
    )
    assert failed
    assert "injected compiler failure" in failed[0]["reason"]


def test_output_shapes_and_all_trainable_gradients_are_finite(
    network: InvariantGatePipelineV2,
) -> None:
    network.zero_grad(set_to_none=True)
    centers, frames = sample_pairs(4)
    local = network.forward_local(centers, frames)
    world = network(centers, frames)
    assert local.shape == world.shape == (4, 3)
    assert bool(torch.isfinite(local).all() and torch.isfinite(world).all())

    probe = torch.tensor(
        (
            (0.7, -0.2, 0.5),
            (-0.3, 0.8, 0.4),
            (0.6, 0.1, -0.9),
            (-0.5, -0.4, 0.7),
        ),
        dtype=DTYPE,
    )
    ((world * probe).sum() + 0.1 * world.square().sum()).backward()
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.abs().sum()) > 0.0, name


@pytest.fixture(scope="module")
def causal_mask_network() -> InvariantGatePipelineV2:
    torch.manual_seed(20260823)
    model = build_invariant_gate_pipeline_v2(
        d3_generators(),
        one_stage_config((2,)),
        generator_names=("threefold", "twofold"),
    )
    model.eval()
    return model


def _parameter_schema(
    model: InvariantGatePipelineV2,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (name, tuple(parameter.shape))
        for name, parameter in model.named_parameters()
    )


def test_default_causal_activity_preserves_the_legacy_runtime(
    causal_mask_network: InvariantGatePipelineV2,
) -> None:
    model = causal_mask_network
    assert not model.legacy_carrier_gates
    assert model.causal_mask_manifest["legacy_carrier_gate_mode"] is None
    assert all(
        item["active"] for item in model.coefficient_head_role_manifest
    )
    assert all(
        item["active"] for item in model.projection_input_role_manifest
    )

    debug = model.debug_forward(*sample_pairs(2))
    carrier = next(
        item
        for item in model.projection_input_role_manifest
        if item["kind"] == "legacy_carrier"
    )
    source_name = carrier["source_names"][0]
    target = TypeKey("A")
    torch.testing.assert_close(
        debug.concats[carrier["stage"]][target][
            ..., carrier["start"] : carrier["stop"], :
        ],
        debug.state[source_name][target],
        rtol=0,
        atol=0,
    )


def test_pair_and_stf_activity_masks_keep_shapes_and_zero_only_the_roles(
    causal_mask_network: InvariantGatePipelineV2,
) -> None:
    model = causal_mask_network
    roles = {
        item["path_family"]: item["role"]
        for item in model.coefficient_head_role_manifest
        if item["path_family"] in {"pair", "stf"}
    }
    assert roles.keys() == {"pair", "stf"}
    centers, frames = sample_pairs(2)
    parameter_schema = _parameter_schema(model)
    state_schema = tuple(
        (name, tuple(value.shape)) for name, value in model.state_dict().items()
        if isinstance(value, Tensor)
    )
    reference = model(centers, frames)
    try:
        for role in roles.values():
            model.set_covariant_path_activity(role, False)
        debug = model.debug_forward(centers, frames)
        for role in roles.values():
            torch.testing.assert_close(
                debug.direct_paths[role],
                torch.zeros_like(debug.direct_paths[role]),
                rtol=0,
                atol=0,
            )
        active_by_role = {
            item["role"]: item["active"]
            for item in model.coefficient_head_role_manifest
        }
        assert all(not active_by_role[role] for role in roles.values())
        assert any(
            item["kind"] == "pair" and item["active"]
            for item in model.descriptor_role_manifest
        )
        assert any(
            item["kind"] == "stf" and item["active"]
            for item in model.descriptor_role_manifest
        )
        assert _parameter_schema(model) == parameter_schema
        assert tuple(
            (name, tuple(value.shape))
            for name, value in model.state_dict().items()
            if isinstance(value, Tensor)
        ) == state_schema
    finally:
        for role in roles.values():
            model.set_covariant_path_activity(role, True)
    torch.testing.assert_close(model(centers, frames), reference, rtol=0, atol=0)


def test_legacy_carrier_activity_mask_keeps_its_projection_slot(
    causal_mask_network: InvariantGatePipelineV2,
) -> None:
    model = causal_mask_network
    carrier = next(
        item
        for item in model.projection_input_role_manifest
        if item["kind"] == "legacy_carrier"
    )
    centers, frames = sample_pairs(2)
    reference = model(centers, frames)
    parameter_schema = _parameter_schema(model)
    try:
        model.set_legacy_carrier_activity(carrier["role"], False)
        updated = next(
            item
            for item in model.projection_input_role_manifest
            if item["role"] == carrier["role"]
        )
        assert updated["active"] is False
        debug = model.debug_forward(centers, frames)
        target = TypeKey("A")
        value = debug.concats[carrier["stage"]][target][
            ..., carrier["start"] : carrier["stop"], :
        ]
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
        assert _parameter_schema(model) == parameter_schema
    finally:
        model.set_legacy_carrier_activity(carrier["role"], True)
    torch.testing.assert_close(model(centers, frames), reference, rtol=0, atol=0)


def test_projection_input_manifest_is_contiguous_and_matches_module_shapes(
    causal_mask_network: InvariantGatePipelineV2,
) -> None:
    model = causal_mask_network
    modules = model.channel_projection_modules_by_role()
    rows = model.projection_input_role_manifest
    assert modules
    assert {item["projection_role"] for item in rows} == set(modules)
    for projection_role, module in modules.items():
        selected = tuple(
            item for item in rows if item["projection_role"] == projection_role
        )
        assert selected
        cursor = 0
        for item in selected:
            assert item["start"] == cursor
            assert item["stop"] > item["start"]
            cursor = item["stop"]
        assert hasattr(module, "weight")
        assert cursor == module.weight.shape[1]
    mask_roles = {
        item["role"]
        for family in ("covariant_paths", "legacy_carriers")
        for item in model.causal_mask_manifest[family]
    }
    assert mask_roles == {item["role"] for item in rows}


def test_preallocated_legacy_carrier_gate_modes_share_one_schema(
    causal_mask_network: InvariantGatePipelineV2,
) -> None:
    base = causal_mask_network
    models = {}
    for mode in ("direct", "residual_zero", "default"):
        model = copy.deepcopy(base)
        torch.manual_seed(20260824)
        model.configure_legacy_carrier_gates(mode)
        model.eval()
        models[mode] = model

    schemas = {_parameter_schema(model) for model in models.values()}
    assert len(schemas) == 1
    assert models["direct"].trainable_parameter_count > base.trainable_parameter_count
    gate_roles = {
        tuple(model.legacy_carrier_gate_modules_by_role()) for model in models.values()
    }
    assert len(gate_roles) == 1
    assert gate_roles.pop()
    for role, direct_head in (
        models["direct"].legacy_carrier_gate_modules_by_role().items()
    ):
        default_head = models["default"].legacy_carrier_gate_modules_by_role()[role]
        torch.testing.assert_close(
            direct_head.weight, default_head.weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            direct_head.bias, default_head.bias, rtol=0, atol=0
        )
    assert all(
        int(torch.count_nonzero(parameter)) == 0
        for parameter in models["residual_zero"].legacy_carrier_gates.parameters()
    )

    centers, frames = sample_pairs(2)
    reference = base(centers, frames)
    torch.testing.assert_close(
        models["direct"](centers, frames), reference, rtol=0, atol=0
    )
    torch.testing.assert_close(
        models["residual_zero"](centers, frames), reference, rtol=0, atol=0
    )
    default = models["default"]
    debug = default.debug_forward(centers, frames)
    carrier = next(
        item
        for item in default.projection_input_role_manifest
        if item["kind"] == "legacy_carrier"
    )
    target = TypeKey("A")
    source_name = carrier["source_names"][0]
    trunk = default.stage_trunks[carrier["stage"]](
        debug.invariants[carrier["stage"]]
    )
    head = default.legacy_carrier_gate_modules_by_role()[carrier["role"]]
    expected = debug.state[source_name][target] * (
        1.0 + torch.tanh(head(trunk))
    ).unsqueeze(-1)
    actual = debug.concats[carrier["stage"]][target][
        ..., carrier["start"] : carrier["stop"], :
    ]
    torch.testing.assert_close(actual, expected)
    assert default.causal_mask_manifest["legacy_carrier_gate_mode"] == "default"


@pytest.mark.parametrize(
    ("group_name", "generators", "anchor_ranks"),
    (
        ("C3", c3_generators(), (1,)),
        ("D3", d3_generators(), (2, 3)),
        ("T", tetrahedral_generators(), (3,)),
        ("O", octahedral_generators(), (4,)),
    ),
)
def test_compiled_mixed_teacher_can_be_overfit_on_32_samples(
    group_name: str,
    generators: Tensor,
    anchor_ranks: tuple[int, ...],
) -> None:
    torch.manual_seed(20260814)
    model = build_invariant_gate_pipeline_v2(
        generators, one_stage_config(anchor_ranks, trunk_width=8)
    )
    centers, frames = sample_pairs(32)
    with torch.no_grad():
        debug = model.debug_forward(centers, frames)
        scalar_path = next(
            path
            for path in model._stage_scalars["readout"]
            if ".scalar.pair.x." in path.candidate.role
            and path.candidate.shortcut_rank is None
        )
        covariant_path = next(
            path
            for path in model._stage_paths["readout"][TypeKey("A")]
            if ".a.pair.x." in path.candidate.role
            and path.candidate.shortcut_rank is None
        )
        scalar_index = model._stage_scalars["readout"].index(scalar_path)
        scalar_offset = (
            2
            + len(model.config.radial.rbf_centers)
            + len(model.config.radial.inverse_powers)
            + sum(
                path.primitive_channels
                for path in model._stage_scalars["readout"][:scalar_index]
            )
        )
        mixed_scalar = debug.invariants["readout"][..., scalar_offset]
        mixed_covariant = model._primitive(covariant_path, debug.state)[..., 0, :]
        target = mixed_scalar.unsqueeze(-1) * mixed_covariant
        target_scale = target.square().mean().clamp_min(1.0e-12)
    model.eval()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=500,
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_local(centers, frames)
        loss = (prediction - target).square().mean() / target_scale
        loss.backward()
        return loss

    optimizer.step(closure)
    normalized_loss = (
        model.forward_local(centers, frames) - target
    ).square().mean() / target_scale
    iteration_count = max(
        int(state.get("n_iter", 0)) for state in optimizer.state.values()
    )
    print(group_name, iteration_count, float(normalized_loss.detach()))
    assert 0 < iteration_count <= 500
    assert float(normalized_loss.detach()) < 1.0e-5, group_name
