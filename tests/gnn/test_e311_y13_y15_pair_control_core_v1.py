from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from experiments.benzene_pair.e_series.catalog import get_model_spec
from experiments.benzene_pair.e_series.model_factory import pipeline_config_from_spec
from experiments.benzene_pair.train import symmetry_metrics
from experiments.gnn.e311_y13_y15_pair_control_core_v1 import (
    HISTORICAL_E311_PARAMETER_COUNT,
    Y_PAIR_CONTROL_SPECS_V1,
    E311OddGraphCoreV1,
    build_e311_odd_graph_core_v1,
    build_y14_two_node_control_v1,
    complete_pair_index_v1,
    signed_scatter_pair_force_v1,
)
from experiments.gnn.e311_y13_y15_pair_control_runner_v1 import (
    _selected_y15_symmetry_audit_v1,
)


@pytest.fixture(scope="module")
def graph_core() -> E311OddGraphCoreV1:
    model = build_e311_odd_graph_core_v1(dtype=torch.float64, device="cpu")
    model.eval()
    return model


def _geometry(node_count: int) -> tuple[Tensor, Tensor]:
    centers = torch.tensor(
        (
            (
                (0.0, 0.0, 0.0),
                (5.1, 0.2, -0.1),
                (0.4, 5.3, 0.3),
                (-0.2, 0.5, 5.7),
                (4.0, 4.2, 4.8),
            ),
        ),
        dtype=torch.float64,
    )[:, :node_count]
    identity = torch.eye(3, dtype=torch.float64)
    frames = identity.repeat(1, node_count, 1, 1)
    return centers, frames


def _rotation(angle: float) -> Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )


def test_registry_locks_the_three_causal_controls() -> None:
    assert tuple(spec.experiment_id for spec in Y_PAIR_CONTROL_SPECS_V1) == (
        "Y13",
        "Y14",
        "Y15",
    )
    y13, y14, y15 = Y_PAIR_CONTROL_SPECS_V1
    assert y13.expected_parameter_count == y14.expected_parameter_count
    assert y14.expected_parameter_count == y15.expected_parameter_count == 14_926
    assert y13.sample_count == y14.sample_count == 400_000
    assert y15.sample_count == 100_000
    assert y13.epochs * 320_000 == y14.epochs * 320_000
    assert y13.epochs * 320_000 == y15.epochs * 80_000 * 10
    assert 500 * (320_000 // 10_000) == 200 * (80_000 // 1_000) == 16_000


def test_exact_e311_math_is_locked_and_generic_covariant_raw_mix_is_off() -> None:
    spec = get_model_spec("E311")
    config = pipeline_config_from_spec(spec)
    assert spec.options["path_policy"] == "NO_RAW_MIXED"
    assert spec.planned_parameter_count == HISTORICAL_E311_PARAMETER_COUNT
    assert tuple(stage.output_stream for stage in config.stages) == (
        "A",
        "B",
        "B",
        "A",
    )
    assert tuple(stage.channels for stage in config.stages) == (1, 2, 1, 1)
    assert tuple(stage.trunk_width for stage in config.stages) == (8, 8, 8, 8)
    assert all(
        not stage.covariant_include_raw_mixed_pairs for stage in config.stages
    )
    assert all(
        stage.invariant_include_raw_mixed_pairs for stage in config.stages
    )


def test_wrapper_has_exactly_one_e311_and_no_multibody_message_block(
    graph_core: E311OddGraphCoreV1,
) -> None:
    count = sum(
        parameter.numel()
        for parameter in graph_core.parameters()
        if parameter.requires_grad
    )
    assert count == HISTORICAL_E311_PARAMETER_COUNT
    assert graph_core.trainable_parameter_count == HISTORICAL_E311_PARAMETER_COUNT
    names = tuple(name for name, _parameter in graph_core.named_parameters())
    assert names and all(name.startswith("pair_kernel.") for name in names)
    module_names = tuple(type(module).__name__ for module in graph_core.modules())
    assert not any("MultibodyMessageBlock" in name for name in module_names)
    metadata = dict(graph_core.architecture_metadata)
    assert metadata["uses_receiver_local_multibody_message_block"] is False
    assert metadata["has_hidden_node_state"] is False


def test_two_node_output_is_world_odd_pair_and_signed_scatter(
    graph_core: E311OddGraphCoreV1,
) -> None:
    centers, frames = _geometry(2)
    pair_index = complete_pair_index_v1(2)
    output = graph_core.core_output(centers, frames, pair_index)
    direct_forward = graph_core.pair_kernel(centers, frames)
    reverse_centers = centers.flip(-2)
    reverse_frames = frames.flip(-3)
    direct_reverse = graph_core.pair_kernel(reverse_centers, reverse_frames)
    expected = 0.5 * (direct_forward - direct_reverse)
    torch.testing.assert_close(
        output.raw_forward_world[..., 0, :],
        direct_forward,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        output.raw_reverse_world[..., 0, :],
        direct_reverse,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        output.normalized_pair_force_world[..., 0, :],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        output.normalized_node_force_world,
        torch.stack((expected, -expected), dim=-2),
        rtol=0.0,
        atol=0.0,
    )
    swapped = graph_core.core_output(
        reverse_centers,
        reverse_frames,
        pair_index,
    )
    torch.testing.assert_close(
        swapped.normalized_pair_force_world,
        -output.normalized_pair_force_world,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_two_node_historical_runner_adapter_returns_one_pair_vector() -> None:
    model = build_y14_two_node_control_v1(dtype=torch.float64, device="cpu")
    assert model._pair_index.device == next(model.parameters()).device
    model.eval()
    centers, frames = _geometry(2)
    prediction = model(centers, frames)
    expected = model.core_output(centers, frames).normalized_pair_force_world[..., 0, :]
    assert prediction.shape == (1, 3)
    torch.testing.assert_close(prediction, expected, rtol=0.0, atol=0.0)
    assert model.trainable_parameter_count == HISTORICAL_E311_PARAMETER_COUNT
    symmetry = symmetry_metrics(model, centers, frames, tolerance=1.0e-9)
    assert symmetry["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_two_node_historical_runner_adapter_places_pair_index_on_cuda() -> None:
    model = build_y14_two_node_control_v1(dtype=torch.float32, device="cuda")
    assert model._pair_index.device.type == "cuda"
    assert model._pair_index.device == next(model.parameters()).device
    centers, frames = _geometry(2)
    prediction = model(centers.float().cuda(), frames.float().cuda())
    assert prediction.shape == (1, 3)
    assert bool(torch.isfinite(prediction).all())


def test_five_node_complete_graph_is_permutation_equivariant_and_conservative(
    graph_core: E311OddGraphCoreV1,
) -> None:
    centers, frames = _geometry(5)
    pair_index = complete_pair_index_v1(5)
    assert pair_index.shape == (2, 10)
    output = graph_core.core_output(centers, frames, pair_index)
    assert output.normalized_pair_force_world.shape == (1, 10, 3)
    assert output.normalized_node_force_world.shape == (1, 5, 3)
    torch.testing.assert_close(
        output.normalized_node_force_world.sum(dim=-2),
        torch.zeros((1, 3), dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-12,
    )

    permutation = torch.tensor((2, 4, 0, 3, 1), dtype=torch.int64)
    permuted = graph_core.core_output(
        centers.index_select(-2, permutation),
        frames.index_select(-3, permutation),
        complete_pair_index_v1(5),
    )
    torch.testing.assert_close(
        permuted.normalized_node_force_world,
        output.normalized_node_force_world.index_select(-2, permutation),
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_global_rotation_translation_covariance(
    graph_core: E311OddGraphCoreV1,
) -> None:
    centers, frames = _geometry(5)
    output = graph_core.core_output(centers, frames)
    rotation = _rotation(0.37)
    translation = torch.tensor((1.2, -0.7, 0.4), dtype=torch.float64)
    transformed_centers = torch.einsum(
        "ij,...nj->...ni",
        rotation,
        centers,
    ) + translation
    transformed_frames = torch.einsum(
        "ij,...njk->...nik",
        rotation,
        frames,
    )
    transformed = graph_core.core_output(
        transformed_centers,
        transformed_frames,
    )
    expected = torch.einsum(
        "ij,...nj->...ni",
        rotation,
        output.normalized_node_force_world,
    )
    torch.testing.assert_close(
        transformed.normalized_node_force_world,
        expected,
        rtol=3.0e-11,
        atol=3.0e-11,
    )


def test_selected_y15_symmetry_audit_covers_the_graph_contract(
    graph_core: E311OddGraphCoreV1,
) -> None:
    centers, frames = _geometry(5)
    pair_index = complete_pair_index_v1(5)
    loader = DataLoader(
        TensorDataset(
            centers,
            frames,
            torch.zeros((1, 10, 3), dtype=torch.float64),
            torch.zeros((1, 5, 3), dtype=torch.float64),
        ),
        batch_size=1,
    )
    audit = _selected_y15_symmetry_audit_v1(
        graph_core,
        loader,
        pair_index,
        torch.device("cpu"),
        tolerance=1.0e-8,
    )
    assert audit["passed"]
    assert graph_core.training is False
    assert set(audit["residuals"]) == {
        "global_se3_node_force",
        "independent_d6_gauge_node_force",
        "node_permutation",
        "oddpair_definition",
        "zero_total_force",
    }


def test_signed_scatter_and_shared_kernel_backward_are_finite(
    graph_core: E311OddGraphCoreV1,
) -> None:
    pair_index = complete_pair_index_v1(5)
    synthetic = torch.arange(30, dtype=torch.float64).reshape(1, 10, 3)
    scattered = signed_scatter_pair_force_v1(synthetic, pair_index, 5)
    torch.testing.assert_close(
        scattered.sum(dim=-2),
        torch.zeros((1, 3), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )

    centers, frames = _geometry(5)
    graph_core.zero_grad(set_to_none=True)
    output = graph_core.core_output(centers, frames, pair_index)
    output.normalized_pair_force_world.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in graph_core.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    )
