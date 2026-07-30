from __future__ import annotations

import torch

from TFENN.symmetry import d6_rotations


def test_d6_contains_twelve_distinct_proper_rotations() -> None:
    rotations = d6_rotations(dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)

    assert rotations.shape == (12, 3, 3)
    torch.testing.assert_close(
        rotations.transpose(1, 2) @ rotations,
        identity.expand(12, 3, 3),
        atol=1e-14,
        rtol=1e-14,
    )
    torch.testing.assert_close(
        torch.linalg.det(rotations),
        torch.ones(12, dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )

    distances = (
        rotations[:, None].sub(rotations[None, :]).square().sum(dim=(-1, -2))
    )
    off_diagonal = distances + torch.eye(12, dtype=torch.float64)
    assert off_diagonal.min().item() > 1e-12


def test_d6_is_closed_under_matrix_multiplication() -> None:
    rotations = d6_rotations(dtype=torch.float64)

    for first in rotations:
        for second in rotations:
            product = first @ second
            distance = (rotations - product).abs().amax(dim=(-1, -2))
            assert distance.min().item() < 1e-13
