"""Active matrix representations on canonical Cartesian STF spaces.

Responsibility and major public API
    This module evaluates normalized symmetric powers and irreducible STF
    rotation representations.  ``stf_representation`` is the primary runtime
    API.  ``symmetric_representation`` exposes the complete normalized
    symmetric power.  ``validate_rotation`` is the explicit data validation
    path for finite proper rotations.

Equations, convention, and shapes
    Python ``rank`` denotes ``l`` in the nonnegative integers, with
    ``d_l = 2 * l + 1`` and zero through ``MAX_STF_RANK`` supported.  Active
    column vectors are used.  ``R`` maps body frame coordinates to common frame
    coordinates and ``D_l(R1 @ R2) = D_l(R1) @ D_l(R2)``.  An input matrix has
    trailing shape ``(3, 3)``.  ``symmetric_representation`` returns trailing
    shape ``(comb(l + 2, 2), comb(l + 2, 2))`` and ``stf_representation``
    returns ``(2 * l + 1, 2 * l + 1)``.  Every leading batch axis is preserved,
    including empty batches.

Validation, tensor behavior, and determinism
    Runtime representation defaults to ``validate=False``.  This path performs
    only type, dtype, trailing shape, and rank checks, so it does not read
    Tensor data and remains suitable for autograd and ``torch.compile``.
    Call ``validate_rotation`` explicitly, or pass ``validate=True``, when
    finite values and membership in SO(3) must be checked.  Validation tolerance
    keywords are used only when validation is enabled.

    Inputs support ``torch.float32`` and ``torch.float64``.  Outputs preserve
    dtype, device, batches, and gradients.  Canonical bases originate as CPU
    ``torch.float64`` constants and are cast without changing their convention.
    Bitwise identity across PyTorch versions, devices, and linear algebra
    backends is not promised.  ``TypeError`` reports unsupported tensor types,
    dtypes, and validation flags.  ``ValueError`` reports invalid ranks, shapes,
    tolerances, nonfinite data, nonorthogonal matrices, and improper rotations.

Mapped references
    Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li,
    Kai Kohlhoff, Patrick Riley, Tensor Field Networks, Rotation and Translation
    Equivariant Neural Networks for 3D Point Clouds,
    https://arxiv.org/abs/1802.08219
    Leon Lang, Maurice Weiler, A Wigner Eckart Theorem for Group Equivariant
    Convolution Kernels, https://openreview.net/forum?id=ajOrOhQOsYx
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor

from .stf_space import (
    MAX_STF_RANK,
    stf_basis,
    symmetric_dimension,
    symmetric_multi_indices,
)


__all__ = [
    "stf_representation",
    "symmetric_representation",
    "validate_rotation",
]


def _validate_input(matrix: Tensor, rank: int) -> None:
    """Validate structural properties without reading Tensor data."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a nonnegative integer, got {rank!r}")
    if rank > MAX_STF_RANK:
        raise ValueError(
            f"rank {rank} exceeds MAX_STF_RANK {MAX_STF_RANK} before allocation"
        )
    if not isinstance(matrix, Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "matrix must have trailing shape (3, 3), "
            f"got {tuple(matrix.shape)}"
        )
    if matrix.dtype not in (torch.float32, torch.float64):
        raise TypeError("matrix must use torch.float32 or torch.float64")


def _rotation_tolerance(
    value: float | None,
    dtype: torch.dtype,
    name: str,
) -> float:
    """Resolve one independent rotation validation tolerance."""
    if value is None:
        return 1e-5 if dtype == torch.float32 else 1e-10
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number and cannot be bool")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    result = float(value)
    return result


def validate_rotation(
    matrix: Tensor,
    *,
    rotation_atol: float | None = None,
    rotation_rtol: float | None = None,
) -> None:
    """Require finite proper orthogonal matrices with trailing shape three."""
    _validate_input(matrix, 0)
    atol = _rotation_tolerance(rotation_atol, matrix.dtype, "rotation_atol")
    rtol = _rotation_tolerance(rotation_rtol, matrix.dtype, "rotation_rtol")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must contain only finite values")
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    gram = matrix.mT @ matrix
    if not bool(torch.isclose(gram, identity, atol=atol, rtol=rtol).all()):
        residual = float((gram - identity).abs().amax())
        raise ValueError(
            "matrix must be orthogonal: "
            f"residual {residual:.6e}, threshold {atol + rtol:.6e}, "
            "actual trailing shape (3, 3), expected trailing shape (3, 3)"
        )
    determinant = torch.linalg.det(matrix)
    if not bool(
        torch.isclose(
            determinant,
            torch.ones_like(determinant),
            atol=atol,
            rtol=rtol,
        ).all()
    ):
        residual = float((determinant - 1.0).abs().amax())
        raise ValueError(
            "matrix must contain only proper rotations: "
            f"residual {residual:.6e}, threshold {atol + rtol:.6e}, "
            "actual trailing shape (3, 3), expected trailing shape (3, 3)"
        )


def symmetric_representation(matrix: Tensor, rank: int) -> Tensor:
    """Return the normalized symmetric power of batched matrices."""
    _validate_input(matrix, rank)
    representation = matrix[..., 0, 0, None, None] * 0.0 + 1.0

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
    validate: bool = False,
    rotation_atol: float | None = None,
    rotation_rtol: float | None = None,
) -> Tensor:
    """Return the rank STF representation of batched rotations."""
    _validate_input(matrix, rank)
    if not isinstance(validate, bool):
        raise TypeError("validate must be bool")
    if validate:
        validate_rotation(
            matrix,
            rotation_atol=rotation_atol,
            rotation_rtol=rotation_rtol,
        )
    basis = stf_basis(rank, dtype=matrix.dtype, device=matrix.device)
    symmetric = symmetric_representation(matrix, rank)
    return torch.einsum("ia,...ij,jb->...ab", basis, symmetric, basis)
