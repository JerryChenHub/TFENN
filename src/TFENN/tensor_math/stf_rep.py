"""Matrix representations on Cartesian STF spaces."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .stf_space import (
    stf_basis,
    symmetric_dimension,
    symmetric_multi_indices,
)


__all__ = ["stf_representation", "symmetric_representation"]


def _validate_input(matrix: Tensor, rank: int) -> None:
    """Validate a batched matrix and tensor rank."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a nonnegative integer, got {rank!r}")
    if not isinstance(matrix, Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have final shape (3, 3)")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise TypeError("matrix must use torch.float32 or torch.float64")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must contain only finite values")


def _rotation_tolerance(
    value: float | None,
    dtype: torch.dtype,
    name: str,
) -> float:
    """Resolve one independent rotation validation tolerance."""
    if value is None:
        return 1e-5 if dtype == torch.float32 else 1e-10
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _validate_rotation(
    matrix: Tensor,
    *,
    rotation_atol: float | None,
    rotation_rtol: float | None,
) -> None:
    """Require a proper orthogonal action on the STF space."""
    atol = _rotation_tolerance(rotation_atol, matrix.dtype, "rotation_atol")
    rtol = _rotation_tolerance(rotation_rtol, matrix.dtype, "rotation_rtol")
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    gram = matrix.mT @ matrix
    if not bool(torch.isclose(gram, identity, atol=atol, rtol=rtol).all()):
        raise ValueError("matrix must be orthogonal")
    determinant = torch.linalg.det(matrix)
    if not bool(
        torch.isclose(
            determinant,
            torch.ones_like(determinant),
            atol=atol,
            rtol=rtol,
        ).all()
    ):
        raise ValueError("matrix must contain only proper rotations")


def symmetric_representation(matrix: Tensor, rank: int) -> Tensor:
    """Return the normalized symmetric power of batched matrices."""
    _validate_input(matrix, rank)
    representation = torch.ones(
        (*matrix.shape[:-2], 1, 1),
        dtype=matrix.dtype,
        device=matrix.device,
    )

    for degree in range(1, rank + 1):
        previous = symmetric_multi_indices(degree - 1)
        current = symmetric_multi_indices(degree)
        positions = {alpha: index for index, alpha in enumerate(previous)}
        columns = []
        for beta in current:
            input_axis = next(axis for axis, count in enumerate(beta) if count)
            gamma = list(beta)
            gamma[input_axis] -= 1
            gamma_index = positions[tuple(gamma)]
            entries = []
            for alpha in current:
                terms = []
                for output_axis, count in enumerate(alpha):
                    if count == 0:
                        continue
                    delta = list(alpha)
                    delta[output_axis] -= 1
                    delta_index = positions[tuple(delta)]
                    scale = math.sqrt(count / beta[input_axis])
                    terms.append(
                        scale
                        * matrix[..., output_axis, input_axis]
                        * representation[..., delta_index, gamma_index]
                    )
                entries.append(torch.stack(terms, dim=-1).sum(dim=-1))
            columns.append(torch.stack(entries, dim=-1))
        representation = torch.stack(columns, dim=-1)
    expected = symmetric_dimension(rank)
    return representation.reshape((*matrix.shape[:-2], expected, expected))


def stf_representation(
    matrix: Tensor,
    rank: int,
    *,
    rotation_atol: float | None = None,
    rotation_rtol: float | None = None,
) -> Tensor:
    """Return the rank STF representation of batched rotations."""
    _validate_input(matrix, rank)
    _validate_rotation(
        matrix,
        rotation_atol=rotation_atol,
        rotation_rtol=rotation_rtol,
    )
    basis = stf_basis(rank, dtype=matrix.dtype, device=matrix.device)
    symmetric = symmetric_representation(matrix, rank)
    return torch.einsum("ia,...ij,jb->...ab", basis, symmetric, basis)
