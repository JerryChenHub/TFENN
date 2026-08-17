"""Tests for the model level D6 Reynolds averaged MLP."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from TFENN.models import (
    ModelLevelGroupConvMLP,
    ModelLevelGroupConvMLPConfig,
    build_model_level_group_conv_mlp,
)


DTYPE = torch.float64


def proper_d6_generators() -> Tensor:
    cosine = math.cos(math.pi / 3.0)
    sine = math.sin(math.pi / 3.0)
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


def rotation(vector: tuple[float, float, float]) -> Tensor:
    x, y, z = torch.tensor(vector, dtype=DTYPE)
    skew = torch.stack(
        (
            torch.tensor(0.0, dtype=DTYPE),
            -z,
            y,
            z,
            torch.tensor(0.0, dtype=DTYPE),
            -x,
            -y,
            x,
            torch.tensor(0.0, dtype=DTYPE),
        )
    ).reshape(3, 3)
    return torch.matrix_exp(skew)


def sample_pairs(batch_size: int = 2) -> tuple[Tensor, Tensor]:
    centers = torch.tensor(
        (
            ((0.2, -0.1, 0.3), (3.4, 1.2, -0.7)),
            ((-0.4, 0.5, 0.1), (2.8, -1.1, 1.4)),
        ),
        dtype=DTYPE,
    )[:batch_size]
    frames = torch.stack(
        (
            torch.stack((rotation((0.13, -0.21, 0.08)), rotation((-0.17, 0.04, 0.29)))),
            torch.stack((rotation((-0.08, 0.16, 0.11)), rotation((0.24, -0.12, 0.05)))),
        )
    )[:batch_size]
    return centers, frames


@pytest.fixture(scope="module")
def model() -> ModelLevelGroupConvMLP:
    return build_model_level_group_conv_mlp(proper_d6_generators())


def test_default_model_has_exact_parameter_budget_and_fixed_group(
    model: ModelLevelGroupConvMLP,
) -> None:
    assert model.trainable_parameter_count == 20160
    assert sum(parameter.numel() for parameter in model.parameters()) == 20160
    assert model.architecture_metadata == {
        "model_family": "model_level_group_conv_mlp",
        "symmetry_projection": "network_level_reynolds_average_d6xd6",
        "group_order": 12,
        "group_pair_count": 144,
        "input_width": 12,
        "output_width": 3,
        "trainable_parameter_count": 20160,
        "config": ModelLevelGroupConvMLPConfig().as_dict(),
    }
    assert "group_actions" in dict(model.named_buffers())
    assert "group_actions" not in model.state_dict()
    torch.testing.assert_close(
        model.group_actions @ model.group_actions.mT,
        torch.eye(3, dtype=DTYPE).expand(12, 3, 3),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_configuration_round_trip_and_validation() -> None:
    config = ModelLevelGroupConvMLPConfig(
        hidden_widths=(16, 24), activation="gelu", distance_scale=4.5, seed=17
    )
    assert ModelLevelGroupConvMLPConfig.from_dict(config.as_dict()) == config
    with pytest.raises(ValueError, match="hidden_widths"):
        ModelLevelGroupConvMLPConfig(hidden_widths=())
    with pytest.raises(ValueError, match="activation"):
        ModelLevelGroupConvMLPConfig(activation="relu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="order six"):
        build_model_level_group_conv_mlp(torch.eye(3, dtype=DTYPE).repeat(2, 1, 1))


def test_forward_shapes_world_reconstruction_and_gradients(
    model: ModelLevelGroupConvMLP,
) -> None:
    centers, frames = sample_pairs()
    centers.requires_grad_()
    local = model.forward_local(centers, frames)
    world = model(centers, frames)
    assert local.shape == (2, 3)
    assert world.shape == (2, 3)
    torch.testing.assert_close(
        world,
        torch.einsum("...ij,...j->...i", frames[:, 0], local),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    world.square().sum().backward()
    assert centers.grad is not None
    assert bool(torch.isfinite(centers.grad).all())
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_complete_d6_pair_reynolds_projection(
    model: ModelLevelGroupConvMLP,
) -> None:
    centers, frames = sample_pairs(1)
    with torch.no_grad():
        reference_local = model.forward_local(centers, frames)
        reference_world = model(centers, frames)
        moved_frames = []
        expected_local = []
        for root_gauge in model.group_actions:
            for sender_gauge in model.group_actions:
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
            rtol=3.0e-12,
            atol=3.0e-12,
        )
        torch.testing.assert_close(
            model(gauge_centers, gauge_frames),
            reference_world.expand(144, 3),
            rtol=3.0e-12,
            atol=3.0e-12,
        )


def test_world_rotation_and_parameter_only_checkpoint(
    model: ModelLevelGroupConvMLP,
) -> None:
    centers, frames = sample_pairs()
    world_rotation = rotation((0.23, -0.17, 0.11))
    translation = torch.tensor((0.4, -0.3, 0.2), dtype=DTYPE)
    with torch.no_grad():
        reference = model(centers, frames)
        moved = model(
            centers @ world_rotation.T + translation,
            world_rotation @ frames,
        )
        torch.testing.assert_close(
            moved,
            reference @ world_rotation.T,
            rtol=3.0e-12,
            atol=3.0e-12,
        )

        restored = build_model_level_group_conv_mlp(
            proper_d6_generators(), ModelLevelGroupConvMLPConfig(seed=91)
        )
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            restored(centers, frames),
            reference,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
