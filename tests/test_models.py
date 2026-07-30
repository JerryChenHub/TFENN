from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from TFENN.data import BenzenePairDataset
from TFENN.models import (
    D6GroupAverageNetV1,
    D6SymmetrizedMLPBaselineV1,
    D6TensorBasisNetV1,
    D6TensorBasisNetV2,
    MLPBaselineV1,
)
from TFENN.symmetry import d6_rotations


MODEL_ATOL = 2e-10
MODEL_RTOL = 2e-10


def _compact_tensor_basis_model(
    model_type: type[D6TensorBasisNetV1] | type[D6TensorBasisNetV2],
) -> nn.Module:
    return model_type(
        x_hidden_channels=3,
        r_hidden_channels=2,
        num_x_layers=1,
        num_r_layers=1,
        r_to_x_channels=3,
        out_channels=2,
        vector_activation="tanh",
        matrix_gate="block_norm",
        head_activation="tanh",
        num_head_layers=2,
        head_hidden_channels=4,
        init_policy="xavier_uniform",
    ).double()


def _representative_input(
    benzene_csv_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = BenzenePairDataset(benzene_csv_path, dtype=torch.float64)
    (displacement, relative_rotation), _target = dataset[0]
    return displacement.unsqueeze(0), relative_rotation.unsqueeze(0)


def test_tensor_basis_v1_covers_all_d6_pairs_in_one_batch(
    benzene_csv_path: Path,
) -> None:
    torch.manual_seed(29)
    model = _compact_tensor_basis_model(D6TensorBasisNetV1).eval()
    displacement, relative_rotation = _representative_input(benzene_csv_path)
    rotations = d6_rotations(dtype=torch.float64)
    left = rotations.repeat_interleave(12, dim=0)
    right = rotations.repeat(12, 1, 1)

    transformed_x = torch.einsum(
        "bi,bij->bj",
        displacement.expand(144, 3),
        left,
    )
    transformed_r = (
        left.transpose(1, 2)
        @ relative_rotation.expand(144, 3, 3)
        @ right
    )
    with torch.no_grad():
        reference = model(displacement, relative_rotation)
        actual = model(transformed_x, transformed_r)
    expected = torch.einsum(
        "bci,bij->bcj",
        reference.expand(144, 2, 3),
        left,
    )

    assert actual.shape == (144, 2, 3)
    torch.testing.assert_close(
        actual,
        expected,
        atol=MODEL_ATOL,
        rtol=MODEL_RTOL,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _compact_tensor_basis_model(D6TensorBasisNetV2),
        lambda: D6GroupAverageNetV1(
            x_hidden_channels=2,
            r_hidden_channels=2,
            num_x_layers=1,
            num_r_layers=1,
            r_to_x_channels=2,
            out_channels=2,
            vector_activation="tanh",
            head_activation="tanh",
            num_head_layers=1,
            head_hidden_channels=2,
            init_policy="xavier_uniform",
        ).double(),
        lambda: D6SymmetrizedMLPBaselineV1(
            out_channels=2,
            hidden_dim=8,
            num_hidden_layers=1,
            activation="tanh",
            init_policy="xavier_uniform",
        ).double(),
    ],
    ids=(
        "tensor_basis_v2",
        "group_average_v1",
        "symmetrized_mlp_v1",
    ),
)
def test_other_symmetric_models_preserve_representative_d6_action(
    factory: Callable[[], nn.Module],
    benzene_csv_path: Path,
) -> None:
    torch.manual_seed(31)
    model = factory().eval()
    displacement, relative_rotation = _representative_input(benzene_csv_path)
    rotations = d6_rotations(dtype=torch.float64)
    left = rotations[4]
    right = rotations[9]

    transformed_x = displacement @ left
    transformed_r = left.T @ relative_rotation @ right
    with torch.no_grad():
        reference = model(displacement, relative_rotation)
        actual = model(transformed_x, transformed_r)
    expected = reference @ left

    assert actual.shape == (1, 2, 3)
    torch.testing.assert_close(
        actual,
        expected,
        atol=MODEL_ATOL,
        rtol=MODEL_RTOL,
    )


def test_plain_mlp_baseline_has_the_common_output_shape(
    benzene_csv_path: Path,
) -> None:
    model = MLPBaselineV1(
        out_channels=2,
        hidden_dim=8,
        num_hidden_layers=1,
        activation="tanh",
        init_policy="xavier_uniform",
    ).double()
    displacement, relative_rotation = _representative_input(benzene_csv_path)

    output = model(
        displacement.expand(4, 3),
        relative_rotation.expand(4, 3, 3),
    )

    assert output.shape == (4, 2, 3)
    assert torch.isfinite(output).all()
