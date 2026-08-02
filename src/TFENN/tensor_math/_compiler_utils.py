"""Shared deterministic numerical tools for offline compilers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor


@dataclass(frozen=True)
class NullspaceResult:
    """Store a canonical nullspace basis and its numerical diagnostics."""

    basis: Tensor
    dimension: int
    singular_values: Tensor
    threshold: float
    singular_value_gap: float
    residual: float


@dataclass(frozen=True)
class RangeResult:
    """Store a canonical column space basis and its rank diagnostics."""

    basis: Tensor
    dimension: int
    singular_values: Tensor
    threshold: float


def positive_finite_tolerance(
    value: float | None,
    name: str,
    default: float,
) -> float:
    """Return one strictly positive finite numerical tolerance."""
    candidate = default if value is None else value
    if isinstance(candidate, bool) or not isinstance(candidate, Real):
        raise TypeError(f"{name} must be a real number and cannot be bool")
    resolved = float(candidate)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def singular_threshold(
    singular_values: Tensor,
    atol: float,
    rtol: float,
) -> float:
    """Return atol plus rtol times the largest singular value."""
    sigma_max = float(singular_values[0]) if singular_values.numel() else 0.0
    return atol + rtol * sigma_max


def _canonical_range(projector: Tensor, dimension: int) -> Tensor:
    """Return a deterministic orthonormal basis for a projector range."""
    size = projector.shape[0]
    if dimension == 0:
        return projector.new_empty((size, 0))
    threshold = 64.0 * torch.finfo(projector.dtype).eps * max(1, size)
    columns: list[Tensor] = []
    for index in range(size):
        vector = projector[:, index].clone()
        for _ in range(2):
            for column in columns:
                vector = vector - torch.dot(column, vector) * column
        norm = torch.linalg.vector_norm(vector)
        if float(norm) <= threshold:
            continue
        vector = vector / norm
        pivot = torch.nonzero(vector.abs() > threshold, as_tuple=False)
        if pivot.numel() and float(vector[pivot[0, 0]]) < 0.0:
            vector = -vector
        columns.append(vector)
        if len(columns) == dimension:
            break
    if len(columns) != dimension:
        raise RuntimeError("failed to construct a stable canonical subspace basis")
    return torch.stack(columns, dim=1)


def canonical_nullspace(
    matrix: Tensor,
    atol: float,
    rtol: float,
) -> NullspaceResult:
    """Compile a canonical nullspace with reduced CPU float64 SVD."""
    if matrix.ndim != 2:
        raise ValueError("matrix must have two dimensions")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must contain only finite values")
    work = matrix.detach().to(device="cpu", dtype=torch.float64)
    rows, columns = work.shape
    if rows == 0:
        basis = torch.eye(columns, dtype=torch.float64)
        return NullspaceResult(
            basis=basis,
            dimension=columns,
            singular_values=work.new_empty((0,)),
            threshold=atol,
            singular_value_gap=float("inf"),
            residual=0.0,
        )

    _, singular_values, right_vectors_h = torch.linalg.svd(work, full_matrices=False)
    threshold = singular_threshold(singular_values, atol, rtol)
    numerical_rank = int((singular_values > threshold).sum().item())
    dimension = columns - numerical_rank
    right_range = right_vectors_h[:numerical_rank].T
    projector = torch.eye(columns, dtype=torch.float64)
    if numerical_rank:
        projector = projector - right_range @ right_range.T
    projector = 0.5 * (projector + projector.T)
    basis = _canonical_range(projector, dimension)

    if 0 < numerical_rank < singular_values.numel():
        gap = float(
            singular_values[numerical_rank - 1] - singular_values[numerical_rank]
        )
    elif numerical_rank:
        gap = float(singular_values[numerical_rank - 1])
    else:
        gap = float("inf")
    if dimension:
        residual = float(torch.linalg.vector_norm(work @ basis, dim=0).max())
    else:
        residual = 0.0
    return NullspaceResult(
        basis=basis,
        dimension=dimension,
        singular_values=singular_values,
        threshold=threshold,
        singular_value_gap=gap,
        residual=residual,
    )


def canonical_column_space(
    matrix: Tensor,
    atol: float,
    rtol: float,
) -> RangeResult:
    """Compile a canonical basis for a column span with reduced SVD."""
    if matrix.ndim != 2:
        raise ValueError("matrix must have two dimensions")
    work = matrix.detach().to(device="cpu", dtype=torch.float64)
    rows = work.shape[0]
    if work.shape[1] == 0:
        return RangeResult(
            basis=work.new_empty((rows, 0)),
            dimension=0,
            singular_values=work.new_empty((0,)),
            threshold=atol,
        )
    left, singular_values, _ = torch.linalg.svd(work, full_matrices=False)
    threshold = singular_threshold(singular_values, atol, rtol)
    dimension = int((singular_values > threshold).sum().item())
    if dimension == 0:
        basis = work.new_empty((rows, 0))
    else:
        projector = left[:, :dimension] @ left[:, :dimension].T
        projector = 0.5 * (projector + projector.T)
        basis = _canonical_range(projector, dimension)
    return RangeResult(
        basis=basis,
        dimension=dimension,
        singular_values=singular_values,
        threshold=threshold,
    )


def canonical_projector_basis(projector: Tensor, dimension: int) -> Tensor:
    """Return the canonical basis of a known projector range."""
    work = projector.detach().to(device="cpu", dtype=torch.float64)
    return _canonical_range(0.5 * (work + work.T), dimension)
