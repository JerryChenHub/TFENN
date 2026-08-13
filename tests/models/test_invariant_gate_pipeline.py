"""Validate the configurable benzene pair pipeline."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

import TFENN.models.invariant_gate_pipeline as pipeline_module
from TFENN.models import (
    InvariantGatePipeline,
    MLPConfig,
    PairPipelineConfig,
    StageConfig,
    build_invariant_gate_pipeline,
    default_pair_pipeline_config,
)


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


def benzene_generators() -> Tensor:
    angle = math.pi / 3.0
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


def d6_group() -> Tensor:
    sixfold, twofold = benzene_generators()
    powers = tuple(torch.linalg.matrix_power(sixfold, power) for power in range(6))
    return torch.stack(powers + tuple(value @ twofold for value in powers))


def sample_pairs(count: int) -> tuple[Tensor, Tensor]:
    index = torch.arange(count, dtype=DTYPE)
    centers = torch.zeros((count, 2, 3), dtype=DTYPE)
    centers[:, 0] = torch.stack(
        (0.2 * index, -0.1 * index, 0.05 * index),
        dim=-1,
    )
    centers[:, 1] = centers[:, 0] + torch.stack(
        (4.8 + 0.2 * index, -1.1 + 0.1 * index, 5.7 - 0.15 * index),
        dim=-1,
    )
    root_vectors = torch.stack(
        (0.11 + 0.01 * index, -0.17 + 0.02 * index, 0.08 - 0.01 * index),
        dim=-1,
    )
    sender_vectors = torch.stack(
        (-0.19 + 0.01 * index, 0.07 - 0.01 * index, 0.13 + 0.02 * index),
        dim=-1,
    )
    frames = torch.stack((rotation(root_vectors), rotation(sender_vectors)), dim=1)
    return centers, frames


@pytest.fixture(scope="module")
def network() -> InvariantGatePipeline:
    torch.manual_seed(20260810)
    return build_invariant_gate_pipeline(
        benzene_generators(),
        default_pair_pipeline_config(),
        generator_names=("sixfold", "twofold"),
    )


def test_config_round_trip_and_graph_validation() -> None:
    config = default_pair_pipeline_config()
    assert PairPipelineConfig.from_dict(config.as_dict()) == config
    with pytest.raises(ValueError, match="unavailable"):
        PairPipelineConfig(
            stages=(StageConfig("a1", "A", ("later",), 1),),
            output_stage="a1",
        )


def test_default_graph_shapes_and_offline_contract(
    network: InvariantGatePipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers, frames = sample_pairs(3)
    monkeypatch.setattr(
        pipeline_module,
        "compile_covariant_basis",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("forward attempted compilation")
        ),
    )
    local = network.forward_local(centers, frames)
    world = network(centers, frames)
    assert local.shape == (3, 3)
    assert world.shape == (3, 3)
    assert network.b_ranks == (2, 6)
    assert network.typed_channel_schedule == {
        "x": 1,
        "r": (1, 1),
        "a1": 4,
        "a2": 4,
        "b2": (4, 4),
        "a3": 1,
    }
    assert len(network.named_gates()) == 27
    assert sum(item["degree"] == 2 for item in network.gate_manifest) == 17
    assert all(item["hidden_widths"] == [32] for item in network.gate_manifest)
    assert network.offline_compilation_summary["forward_compilation"] is False
    assert network.offline_compilation_summary["disk_artifact_cache"] is False
    assert network.offline_compilation_summary["orbit_storage"] is False
    assert bool(torch.isfinite(local).all() and torch.isfinite(world).all())


def test_reordered_a_b_graph_and_variable_mlp_widths() -> None:
    config = PairPipelineConfig(
        stages=(
            StageConfig(
                "b1",
                "B",
                ("r",),
                2,
                lift_orders=(1,),
                mlp=MLPConfig((12, 7)),
            ),
            StageConfig(
                "a1",
                "A",
                ("x", "b1"),
                3,
                lift_orders=(1,),
                mlp=MLPConfig((9,)),
            ),
            StageConfig(
                "b2",
                "B",
                ("a1",),
                2,
                lift_orders=(1,),
                mlp=MLPConfig(()),
            ),
            StageConfig(
                "a2",
                "A",
                ("b2",),
                1,
                lift_orders=(1,),
                mlp=MLPConfig((8, 4)),
            ),
        ),
        output_stage="a2",
        anchor_ranks=(1, 2),
    )
    model = build_invariant_gate_pipeline(
        benzene_generators(),
        config,
        generator_names=("sixfold", "twofold"),
    )
    centers, frames = sample_pairs(2)
    assert model(centers, frames).shape == (2, 3)
    assert tuple(model.typed_channel_schedule) == (
        "x",
        "r",
        "b1",
        "a1",
        "b2",
        "a2",
    )
    widths = {item["stage"]: item["hidden_widths"] for item in model.gate_manifest}
    assert widths == {"b1": [12, 7], "a1": [9], "b2": [], "a2": [8, 4]}


def test_all_d6_gauges_and_global_rotation(network: InvariantGatePipeline) -> None:
    centers, frames = sample_pairs(1)
    with torch.no_grad():
        reference_local = network.forward_local(centers, frames)
        reference_world = network(centers, frames)
        moved_frames = []
        expected_local = []
        for root_gauge in d6_group():
            for sender_gauge in d6_group():
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
        gauge_centers = centers.expand(144, 2, 3).clone()
        torch.testing.assert_close(
            network.forward_local(gauge_centers, gauge_frames),
            torch.stack(expected_local),
            atol=5.0e-8,
            rtol=5.0e-8,
        )
        torch.testing.assert_close(
            network(gauge_centers, gauge_frames),
            reference_world.expand(144, 3),
            atol=5.0e-8,
            rtol=5.0e-8,
        )
        world_rotation = rotation(torch.tensor((0.23, -0.17, 0.11), dtype=DTYPE))
        translation = torch.tensor((0.4, -0.3, 0.2), dtype=DTYPE)
        torch.testing.assert_close(
            network(centers @ world_rotation.T + translation, world_rotation @ frames),
            reference_world @ world_rotation.T,
            atol=5.0e-8,
            rtol=5.0e-8,
        )


def test_every_parameter_receives_finite_gradient(
    network: InvariantGatePipeline,
) -> None:
    network.zero_grad(set_to_none=True)
    centers, frames = sample_pairs(4)
    output = network(centers, frames)
    probe = torch.tensor(
        (
            (0.7, -0.2, 0.5),
            (-0.3, 0.8, 0.4),
            (0.6, 0.1, -0.9),
            (-0.5, -0.4, 0.7),
        ),
        dtype=DTYPE,
    )
    ((output * probe).sum() + 0.1 * output.square().sum()).backward()
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.abs().sum()) > 0.0, name
