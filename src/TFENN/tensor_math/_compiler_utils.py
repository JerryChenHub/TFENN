"""Internal numerical support for deterministic offline compilation.

Responsibility
    This module reduces matrix constraints to canonical nullspace and range
    records.  It has no public API.  Compiler modules consume the records and
    attach problem specific rank and shape context to user facing diagnostics.

Equations and shapes
    For an input matrix with shape ``(rows, columns)``, singular values are
    classified with ``threshold = atol + rtol * sigma_max``.  A nullspace basis
    has shape ``(columns, nullity)`` and a range basis has shape
    ``(rows, numerical_rank)``.  Columns are orthonormal and are ordered by
    scanning the ambient coordinate projector from first coordinate to last.

Rank, batching, and tensor behavior
    STF rank does not apply in this internal module.  Batched matrices are not
    accepted.  Inputs must use ``torch.float32`` or ``torch.float64``.  Every
    compiler input is detached, copied to CPU, and converted to
    ``torch.float64``.  Returned tensors therefore do not preserve device,
    dtype, or gradients.  Applying the resulting fixed tensors is the
    responsibility of differentiable runtime modules.

Determinism and exceptions
    Canonical projector scanning removes arbitrary singular vector phases and
    rotations for a fixed numerical subspace.  Rank decisions can still vary
    near the threshold across PyTorch versions and linear algebra backends.
    The records expose the spectral gap, threshold margin, residual, and
    singular values so callers can reject unstable artifacts.  A stable rank
    decision requires its threshold margin to be at least one half of the
    numerical threshold.  ``TypeError``
    reports unsupported values and dtypes.  ``ValueError`` reports invalid
    shapes, tolerances, dimensions, or nonfinite data.  ``RuntimeError``
    reports failure to construct the requested canonical basis.

Mapped reference
    Marc Finzi, Max Welling, Andrew Gordon Wilson, A Practical Method for
    Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix
    Groups, https://proceedings.mlr.press/v139/finzi21a.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor


__all__: tuple[str, ...] = ()


_MINIMUM_THRESHOLD_MARGIN_FRACTION = 0.5


@dataclass(frozen=True)
class NullspaceResult:
    """Store a canonical nullspace basis and its numerical diagnostics."""

    basis: Tensor
    dimension: int
    singular_values: Tensor
    threshold: float
    singular_value_gap: float
    threshold_margin: float
    residual: float
    numerical_rank: int


@dataclass(frozen=True)
class RangeResult:
    """Store a canonical column space basis and its rank diagnostics."""

    basis: Tensor
    dimension: int
    singular_values: Tensor
    threshold: float
    singular_value_gap: float
    threshold_margin: float
    residual: float
    numerical_rank: int


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


def minimum_stable_margin(threshold: float) -> float:
    """Return the required distance from a numerical rank boundary."""
    return _MINIMUM_THRESHOLD_MARGIN_FRACTION * threshold


def _spectral_diagnostics(
    singular_values: Tensor,
    numerical_rank: int,
    threshold: float,
) -> tuple[float, float]:
    """Return separation across the decision and distance from its threshold."""
    count = singular_values.numel()
    above = (
        float(singular_values[numerical_rank - 1])
        if numerical_rank > 0
        else float("inf")
    )
    below = (
        float(singular_values[numerical_rank])
        if numerical_rank < count
        else 0.0
    )
    gap = above - below
    upper_margin = above - threshold
    lower_margin = threshold - below
    margin = min(upper_margin, lower_margin)
    return gap, margin


def _offline_matrix(matrix: Tensor, name: str) -> Tensor:
    """Validate one compiler matrix and return detached CPU double storage."""
    if not isinstance(matrix, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} must have shape (rows, columns), got {tuple(matrix.shape)}"
        )
    if matrix.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use torch.float32 or torch.float64")
    work = matrix.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(work).all()):
        raise ValueError(f"{name} must contain only finite values")
    return work


def _canonical_range(projector: Tensor, dimension: int) -> Tensor:
    """Return a deterministic orthonormal basis for a projector range."""
    if not isinstance(projector, Tensor):
        raise TypeError("projector must be a torch.Tensor")
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ValueError(
            "projector must have square shape, "
            f"got {tuple(projector.shape)}"
        )
    size = projector.shape[0]
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 0 <= dimension <= size
    ):
        raise ValueError(
            f"dimension must be an integer from zero through {size}, got {dimension!r}"
        )
    if not bool(torch.isfinite(projector).all()):
        raise ValueError("projector must contain only finite values")
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
        raise RuntimeError(
            "failed to construct a stable canonical subspace basis: "
            f"actual shape ({size}, {len(columns)}), "
            f"expected shape ({size}, {dimension}), threshold {threshold:.6e}"
        )
    return torch.stack(columns, dim=1)


def canonical_nullspace(
    matrix: Tensor,
    atol: float,
    rtol: float,
) -> NullspaceResult:
    """Compile a canonical nullspace with reduced CPU float64 SVD."""
    resolved_atol = positive_finite_tolerance(atol, "atol", 1e-10)
    resolved_rtol = positive_finite_tolerance(rtol, "rtol", 1e-12)
    work = _offline_matrix(matrix, "matrix")
    rows, columns = work.shape
    if rows == 0:
        basis = torch.eye(columns, dtype=torch.float64)
        return NullspaceResult(
            basis=basis,
            dimension=columns,
            singular_values=work.new_empty((0,)),
            threshold=resolved_atol,
            singular_value_gap=float("inf"),
            threshold_margin=resolved_atol,
            residual=0.0,
            numerical_rank=0,
        )

    _, singular_values, right_vectors_h = torch.linalg.svd(work, full_matrices=False)
    threshold = singular_threshold(singular_values, resolved_atol, resolved_rtol)
    numerical_rank = int((singular_values > threshold).sum().item())
    dimension = columns - numerical_rank
    right_range = right_vectors_h[:numerical_rank].T
    projector = torch.eye(columns, dtype=torch.float64)
    if numerical_rank:
        projector = projector - right_range @ right_range.T
    projector = 0.5 * (projector + projector.T)
    basis = _canonical_range(projector, dimension)

    gap, margin = _spectral_diagnostics(
        singular_values,
        numerical_rank,
        threshold,
    )
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
        threshold_margin=margin,
        residual=residual,
        numerical_rank=numerical_rank,
    )


def canonical_column_space(
    matrix: Tensor,
    atol: float,
    rtol: float,
) -> RangeResult:
    """Compile a canonical basis for a column span with reduced SVD."""
    resolved_atol = positive_finite_tolerance(atol, "atol", 1e-10)
    resolved_rtol = positive_finite_tolerance(rtol, "rtol", 1e-12)
    work = _offline_matrix(matrix, "matrix")
    rows = work.shape[0]
    if work.shape[1] == 0:
        return RangeResult(
            basis=work.new_empty((rows, 0)),
            dimension=0,
            singular_values=work.new_empty((0,)),
            threshold=resolved_atol,
            singular_value_gap=float("inf"),
            threshold_margin=resolved_atol,
            residual=0.0,
            numerical_rank=0,
        )
    left, singular_values, _ = torch.linalg.svd(work, full_matrices=False)
    threshold = singular_threshold(singular_values, resolved_atol, resolved_rtol)
    dimension = int((singular_values > threshold).sum().item())
    if dimension == 0:
        basis = work.new_empty((rows, 0))
    else:
        projector = left[:, :dimension] @ left[:, :dimension].T
        projector = 0.5 * (projector + projector.T)
        basis = _canonical_range(projector, dimension)
    gap, margin = _spectral_diagnostics(singular_values, dimension, threshold)
    if dimension:
        projected = basis @ (basis.T @ work)
        residual = float(torch.linalg.matrix_norm(work - projected))
    else:
        residual = float(torch.linalg.matrix_norm(work))
    return RangeResult(
        basis=basis,
        dimension=dimension,
        singular_values=singular_values,
        threshold=threshold,
        singular_value_gap=gap,
        threshold_margin=margin,
        residual=residual,
        numerical_rank=dimension,
    )


def canonical_projector_basis(projector: Tensor, dimension: int) -> Tensor:
    """Return the canonical basis of a known projector range."""
    work = _offline_matrix(projector, "projector")
    if work.shape[0] != work.shape[1]:
        raise ValueError(
            f"projector must have square shape, got {tuple(work.shape)}"
        )
    return _canonical_range(0.5 * (work + work.T), dimension)
