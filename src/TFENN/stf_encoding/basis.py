"""Cartesian STF bases and representations built only with PyTorch."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor


__all__ = [
    "stf_basis",
    "stf_power_components",
    "stf_representation",
    "symmetric_multi_indices",
    "symmetric_power_components",
    "symmetric_power_representation",
    "trace_matrix",
]


def _validate_rank(rank: int) -> None:
    """Require a nonnegative integer tensor rank."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a nonnegative integer, got {rank!r}")


def _validate_tensor(x: Tensor, trailing_shape: tuple[int, ...], name: str) -> None:
    """Require a floating or complex tensor with the requested trailing shape."""
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(
            f"{name} must have trailing shape {trailing_shape}, got {tuple(x.shape)}"
        )
    if not (x.is_floating_point() or x.is_complex()):
        raise TypeError(f"{name} must have a floating or complex dtype")


@lru_cache(maxsize=None)
def symmetric_multi_indices(rank: int) -> tuple[tuple[int, int, int], ...]:
    """Return rank three multi indices in descending lexicographic order."""
    _validate_rank(rank)
    return tuple(
        (a, b, rank - a - b)
        for a in range(rank, -1, -1)
        for b in range(rank - a, -1, -1)
    )


def _multiplicity(rank: int, alpha: tuple[int, int, int]) -> int:
    """Return the number of index words represented by one multi index."""
    result = math.factorial(rank)
    for count in alpha:
        result //= math.factorial(count)
    return result


@lru_cache(maxsize=None)
def _trace_matrix_cpu(rank: int) -> Tensor:
    """Build the normalized symmetric trace matrix on CPU in double precision."""
    _validate_rank(rank)
    high = symmetric_multi_indices(rank)
    if rank < 2:
        return torch.empty((0, len(high)), dtype=torch.float64)
    low = symmetric_multi_indices(rank - 2)
    low_position = {alpha: index for index, alpha in enumerate(low)}
    matrix = torch.zeros((len(low), len(high)), dtype=torch.float64)
    denominator = rank * (rank - 1)
    for column, alpha in enumerate(high):
        for axis, count in enumerate(alpha):
            if count < 2:
                continue
            beta = list(alpha)
            beta[axis] -= 2
            row = low_position[tuple(beta)]
            matrix[row, column] = math.sqrt(count * (count - 1) / denominator)
    return matrix


def trace_matrix(
    rank: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the trace map between normalized symmetric component spaces."""
    _validate_rank(rank)
    target_dtype = dtype or torch.get_default_dtype()
    probe = torch.empty((), dtype=target_dtype)
    if not (probe.is_floating_point() or probe.is_complex()):
        raise TypeError("dtype must be floating or complex")
    return _trace_matrix_cpu(rank).to(dtype=target_dtype, device=device).clone()


@lru_cache(maxsize=None)
def _stf_basis_cpu(rank: int) -> Tensor:
    """Construct a deterministic trace nullspace basis in double precision."""
    _validate_rank(rank)
    size = len(symmetric_multi_indices(rank))
    target_size = 2 * rank + 1
    if rank < 2:
        return torch.eye(size, dtype=torch.float64)

    trace = _trace_matrix_cpu(rank)
    gram = trace @ trace.T
    projection = torch.eye(size, dtype=torch.float64) - trace.T @ torch.linalg.solve(
        gram, trace
    )
    projection = 0.5 * (projection + projection.T)

    columns: list[Tensor] = []
    tolerance = 512.0 * torch.finfo(torch.float64).eps * size
    for index in range(size):
        vector = projection[:, index].clone()
        for _ in range(2):
            for column in columns:
                vector = vector - torch.dot(column, vector) * column
        norm = torch.linalg.vector_norm(vector)
        if norm.item() <= tolerance:
            continue
        vector = vector / norm
        pivot_tolerance = tolerance * vector.abs().max().item()
        pivot = torch.nonzero(vector.abs() > pivot_tolerance, as_tuple=False)[0, 0]
        if vector[pivot].item() < 0.0:
            vector = -vector
        columns.append(vector)
        if len(columns) == target_size:
            break

    if len(columns) != target_size:
        raise RuntimeError(f"failed to construct the rank {rank} STF basis")
    basis = torch.stack(columns, dim=1)
    identity = torch.eye(target_size, dtype=torch.float64)
    if not torch.allclose(basis.T @ basis, identity, atol=2e-12, rtol=2e-12):
        raise RuntimeError(f"rank {rank} STF basis lost orthonormality")
    if not torch.allclose(
        trace @ basis, torch.zeros_like(trace @ basis), atol=2e-12, rtol=0.0
    ):
        raise RuntimeError(f"rank {rank} STF basis is not trace free")
    return basis


def stf_basis(
    rank: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a deterministic Frobenius orthonormal Cartesian STF basis."""
    _validate_rank(rank)
    target_dtype = dtype or torch.get_default_dtype()
    probe = torch.empty((), dtype=target_dtype)
    if not (probe.is_floating_point() or probe.is_complex()):
        raise TypeError("dtype must be floating or complex")
    return _stf_basis_cpu(rank).to(dtype=target_dtype, device=device).clone()


def symmetric_power_components(vector: Tensor, rank: int) -> Tensor:
    """Return normalized symmetric components of the pure power of a vector."""
    _validate_rank(rank)
    _validate_tensor(vector, (3,), "vector")
    components = []
    for alpha in symmetric_multi_indices(rank):
        value = torch.ones_like(vector[..., 0])
        for axis, power in enumerate(alpha):
            if power:
                value = value * vector[..., axis].pow(power)
        scale = math.sqrt(_multiplicity(rank, alpha))
        components.append(value * scale)
    return torch.stack(components, dim=-1)


def symmetric_power_representation(matrix: Tensor, rank: int) -> Tensor:
    """Return the normalized symmetric rank representation of batched matrices."""
    _validate_rank(rank)
    _validate_tensor(matrix, (3, 3), "matrix")
    batch_shape = matrix.shape[:-2]
    representation = torch.ones(
        (*batch_shape, 1, 1), dtype=matrix.dtype, device=matrix.device
    )

    for degree in range(1, rank + 1):
        previous = symmetric_multi_indices(degree - 1)
        current = symmetric_multi_indices(degree)
        previous_position = {alpha: index for index, alpha in enumerate(previous)}
        columns = []
        for beta in current:
            axis_in = next(axis for axis, count in enumerate(beta) if count)
            gamma = list(beta)
            gamma[axis_in] -= 1
            gamma_position = previous_position[tuple(gamma)]
            entries = []
            for alpha in current:
                terms = []
                for axis_out, count in enumerate(alpha):
                    if not count:
                        continue
                    delta = list(alpha)
                    delta[axis_out] -= 1
                    delta_position = previous_position[tuple(delta)]
                    scale = math.sqrt(count / beta[axis_in])
                    terms.append(
                        scale
                        * matrix[..., axis_out, axis_in]
                        * representation[..., delta_position, gamma_position]
                    )
                entries.append(torch.stack(terms, dim=-1).sum(dim=-1))
            columns.append(torch.stack(entries, dim=-1))
        representation = torch.stack(columns, dim=-1)
    return representation


def _coerce_basis(
    basis: Tensor | None,
    rank: int,
    reference: Tensor,
) -> Tensor:
    """Return a validated STF basis on the reference dtype and device."""
    expected = (len(symmetric_multi_indices(rank)), 2 * rank + 1)
    if basis is None:
        return stf_basis(rank, dtype=reference.dtype, device=reference.device)
    if not isinstance(basis, Tensor) or tuple(basis.shape) != expected:
        raise ValueError(f"basis must have shape {expected}")
    return basis.to(dtype=reference.dtype, device=reference.device)


def stf_representation(
    matrix: Tensor,
    rank: int,
    *,
    basis: Tensor | None = None,
) -> Tensor:
    """Return the rank STF representation of batched orthogonal matrices."""
    _validate_rank(rank)
    _validate_tensor(matrix, (3, 3), "matrix")
    coordinates = _coerce_basis(basis, rank, matrix)
    symmetric = symmetric_power_representation(matrix, rank)
    return coordinates.T @ (symmetric @ coordinates)


def stf_power_components(
    vector: Tensor,
    rank: int,
    *,
    basis: Tensor | None = None,
) -> Tensor:
    """Return STF components of the orthogonal projection of a vector power."""
    _validate_rank(rank)
    _validate_tensor(vector, (3,), "vector")
    coordinates = _coerce_basis(basis, rank, vector)
    symmetric = symmetric_power_components(vector, rank)
    return symmetric @ coordinates
