"""Validate the network level D6 Reynolds averaged MLP baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from TFENN.models.network_group_conv import (
    build_benzene_pair_network_group_conv_mlp,
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


def explicit_reynolds_average(
    model: nn.Module, centers: Tensor, frames: Tensor
) -> Tensor:
    root = frames[:, 0]
    displacement = (
        torch.einsum(
            "bij,bj->bi",
            root.mT,
            centers[:, 1] - centers[:, 0],
        )
        / model.config.distance_scale
    )
    relative_frame = root.mT @ frames[:, 1]
    result = torch.zeros_like(displacement)
    for receiver in model._group:
        transformed_displacement = displacement @ receiver
        for sender in model._group:
            transformed_frame = receiver.mT @ relative_frame @ sender
            features = torch.cat(
                (transformed_frame.flatten(start_dim=1), transformed_displacement),
                dim=-1,
            )
            result = result + model.mlp(features) @ receiver.mT
    return result / (model._group.shape[0] ** 2)


def test_default_parameter_budget_shape_and_fixed_group() -> None:
    model = build_benzene_pair_network_group_conv_mlp().to(dtype=DTYPE)
    centers, frames = sample_pairs(3)

    assert model(centers, frames).shape == (3, 3)
    assert model.forward_local(centers, frames).shape == (3, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) == 20160
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        == 20160
    )
    assert model._group.shape == (12, 3, 3)
    assert "_group" not in model.state_dict()

    linear_layers = tuple(
        module for module in model.mlp.modules() if isinstance(module, nn.Linear)
    )
    assert tuple(
        (layer.in_features, layer.out_features) for layer in linear_layers
    ) == ((12, 96), (96, 96), (96, 96), (96, 3))
    assert all(layer.bias is not None for layer in linear_layers[:-1])
    assert linear_layers[-1].bias is None


def test_vectorized_forward_matches_explicit_one_hundred_forty_four_term_sum() -> None:
    model = build_benzene_pair_network_group_conv_mlp().to(dtype=DTYPE)
    centers, frames = sample_pairs(2)
    with torch.no_grad():
        expected = explicit_reynolds_average(model, centers, frames)
        actual = model.forward_local(centers, frames)
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_all_d6_gauges_world_rotation_and_translation() -> None:
    model = build_benzene_pair_network_group_conv_mlp().to(dtype=DTYPE)
    centers, frames = sample_pairs(1)
    with torch.no_grad():
        reference_local = model.forward_local(centers, frames)
        reference_world = model(centers, frames)
        moved_frames = []
        expected_local = []
        for root_gauge in model._group:
            for sender_gauge in model._group:
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
            model.forward_local(gauge_centers, gauge_frames),
            torch.stack(expected_local),
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        torch.testing.assert_close(
            model(gauge_centers, gauge_frames),
            reference_world.expand(144, 3),
            atol=3.0e-11,
            rtol=3.0e-11,
        )

        world_rotation = rotation(torch.tensor((0.23, -0.17, 0.11), dtype=DTYPE))
        translation = torch.tensor((0.4, -0.3, 0.2), dtype=DTYPE)
        moved_centers = centers @ world_rotation.T + translation
        moved_frames = world_rotation @ frames
        torch.testing.assert_close(
            model(moved_centers, moved_frames),
            reference_world @ world_rotation.T,
            atol=3.0e-11,
            rtol=3.0e-11,
        )


def test_gradients_and_learned_checkpoint_round_trip() -> None:
    first = build_benzene_pair_network_group_conv_mlp().to(dtype=DTYPE)
    centers, frames = sample_pairs(2)
    centers.requires_grad_(True)
    frames.requires_grad_(True)
    loss = first(centers, frames).square().sum()
    loss.backward()

    assert centers.grad is not None and bool(torch.isfinite(centers.grad).all())
    assert frames.grad is not None and bool(torch.isfinite(frames.grad).all())
    for name, parameter in first.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name

    torch.manual_seed(19)
    second = build_benzene_pair_network_group_conv_mlp().to(dtype=DTYPE)
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(first(centers, frames), second(centers, frames))
