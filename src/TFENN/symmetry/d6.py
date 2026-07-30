"""D6 rotations and representation actions used throughout TFENN."""

from __future__ import annotations

import math

import torch


def _build_d6_rotations() -> torch.Tensor:
    root_three_over_two = math.sqrt(3.0) / 2.0
    planar_entries = (
        (1.0, 0.0),
        (0.5, root_three_over_two),
        (-0.5, root_three_over_two),
        (-1.0, 0.0),
        (-0.5, -root_three_over_two),
        (0.5, -root_three_over_two),
    )
    rotations = []
    for cosine, sine in planar_entries:
        rotations.append(
            torch.tensor(
                (
                    (cosine, -sine, 0.0),
                    (sine, cosine, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=torch.float64,
            )
        )

    rotation_x_pi = torch.diag(torch.tensor((1.0, -1.0, -1.0), dtype=torch.float64))
    planar_rotations = tuple(rotations)
    rotations.extend(rotation_x_pi @ rotation for rotation in planar_rotations)
    return torch.stack(rotations)


_D6_ROTATIONS = _build_d6_rotations()
_D6_PROJECTORS = torch.stack(
    (
        torch.diag(torch.tensor((1.0, 1.0, 0.0), dtype=torch.float64)),
        torch.diag(torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)),
    )
)


def d6_rotations(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the 12 proper rotations of the benzene D6 group."""
    if dtype is None:
        dtype = torch.float64
    return _D6_ROTATIONS.to(dtype=dtype, device=device).clone()


def d6_projectors(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the planar and axial projectors with shape ``(2, 3, 3)``."""
    if dtype is None:
        dtype = torch.float64
    return _D6_PROJECTORS.to(dtype=dtype, device=device).clone()


def transform_vector(vector: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply the stored row vector action ``x_new = x @ rotation``."""
    return vector @ rotation


def transform_bi_tensor(
    tensor: torch.Tensor,
    left_rotation: torch.Tensor,
    right_rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply ``R_new = left_rotation.T @ R @ right_rotation``."""
    return left_rotation.transpose(-1, -2) @ tensor @ right_rotation


__all__ = [
    "d6_projectors",
    "d6_rotations",
    "transform_bi_tensor",
    "transform_vector",
]
