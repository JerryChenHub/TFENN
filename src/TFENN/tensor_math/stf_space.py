"""Cartesian symmetric trace free tensor spaces."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import product

import torch
from torch import Tensor


__all__ = [
    "STF_BASIS_VERSION",
    "dense_to_stf",
    "dense_to_symmetric",
    "project_symmetric",
    "stf_basis",
    "stf_dimension",
    "stf_power_components",
    "stf_symmetric_product",
    "stf_tensor_coupling",
    "stf_to_dense",
    "stf_to_symmetric",
    "symmetric_dimension",
    "symmetric_multi_indices",
    "symmetric_power_components",
    "symmetric_to_dense",
    "symmetric_to_stf",
    "trace_matrix",
]


STF_BASIS_VERSION = "cartesian-stf-v1"


def _validate_rank(rank: int) -> None:
    """Require a nonnegative integer rank."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a nonnegative integer, got {rank!r}")


def _resolve_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Resolve and validate a real floating dtype."""
    result = dtype or torch.get_default_dtype()
    if result not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    return result


def _validate_float_tensor(value: Tensor, name: str) -> None:
    """Require a real single or double precision tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use torch.float32 or torch.float64")


def _validate_components(
    components: Tensor,
    size: int,
    name: str,
) -> None:
    """Require coordinates in the final tensor dimension."""
    _validate_float_tensor(components, name)
    if components.ndim == 0 or components.shape[-1] != size:
        raise ValueError(f"{name} must have final dimension {size}")


@lru_cache(maxsize=None)
def _symmetric_multi_indices_master(
    rank: int,
) -> tuple[tuple[int, int, int], ...]:
    """Build immutable multi indices in canonical order."""
    return tuple(
        (a, b, rank - a - b)
        for a in range(rank, -1, -1)
        for b in range(rank - a, -1, -1)
    )


def symmetric_multi_indices(rank: int) -> tuple[tuple[int, int, int], ...]:
    """Return rank three multi indices in fixed lexicographic order."""
    _validate_rank(rank)
    return _symmetric_multi_indices_master(rank)


def symmetric_dimension(rank: int) -> int:
    """Return the dimension of the rank symmetric tensor space."""
    _validate_rank(rank)
    return math.comb(rank + 2, 2)


def stf_dimension(rank: int) -> int:
    """Return the dimension of the rank STF tensor space."""
    _validate_rank(rank)
    return 2 * rank + 1


@lru_cache(maxsize=None)
def _multiplicity(rank: int, alpha: tuple[int, int, int]) -> int:
    """Count index words represented by a multi index."""
    result = math.factorial(rank)
    for count in alpha:
        result //= math.factorial(count)
    return result


@lru_cache(maxsize=None)
def _trace_matrix_master(rank: int) -> Tensor:
    """Build the canonical trace map on CPU in double precision."""
    high = symmetric_multi_indices(rank)
    if rank < 2:
        return torch.empty((0, len(high)), dtype=torch.float64)

    low = symmetric_multi_indices(rank - 2)
    positions = {alpha: index for index, alpha in enumerate(low)}
    result = torch.zeros((len(low), len(high)), dtype=torch.float64)
    denominator = rank * (rank - 1)
    for column, alpha in enumerate(high):
        for axis, count in enumerate(alpha):
            if count < 2:
                continue
            beta = list(alpha)
            beta[axis] -= 2
            row = positions[tuple(beta)]
            result[row, column] = math.sqrt(count * (count - 1) / denominator)
    return result


def trace_matrix(
    rank: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the canonical trace map in requested tensor properties."""
    _validate_rank(rank)
    target_dtype = _resolve_dtype(dtype)
    return _trace_matrix_master(rank).to(dtype=target_dtype, device=device).clone()


@lru_cache(maxsize=None)
def _stf_basis_master(rank: int) -> Tensor:
    """Build the canonical STF basis on CPU in double precision."""
    size = symmetric_dimension(rank)
    target_size = stf_dimension(rank)
    if rank < 2:
        return torch.eye(size, dtype=torch.float64)

    trace = _trace_matrix_master(rank)
    gram = trace @ trace.mT
    projection = torch.eye(size, dtype=torch.float64)
    projection = projection - trace.mT @ torch.linalg.solve(gram, trace)
    projection = 0.5 * (projection + projection.mT)

    columns: list[Tensor] = []
    tolerance = 128.0 * torch.finfo(torch.float64).eps * size
    for index in range(size):
        vector = projection[:, index].clone()
        for _ in range(2):
            for column in columns:
                vector = vector - torch.dot(column, vector) * column
        norm = torch.linalg.vector_norm(vector)
        if norm.item() <= tolerance:
            continue
        vector = vector / norm
        pivot = int(torch.argmax(vector.abs()).item())
        if vector[pivot].item() < 0.0:
            vector = -vector
        columns.append(vector)
        if len(columns) == target_size:
            break

    if len(columns) != target_size:
        raise RuntimeError(f"could not construct the rank {rank} STF basis")
    return torch.stack(columns, dim=1)


def stf_basis(
    rank: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the canonical STF basis in requested tensor properties."""
    _validate_rank(rank)
    target_dtype = _resolve_dtype(dtype)
    return _stf_basis_master(rank).to(dtype=target_dtype, device=device).clone()


def symmetric_to_stf(components: Tensor, rank: int) -> Tensor:
    """Project normalized symmetric coordinates into STF coordinates."""
    _validate_rank(rank)
    _validate_components(components, symmetric_dimension(rank), "components")
    basis = stf_basis(rank, dtype=components.dtype, device=components.device)
    return components @ basis


def stf_to_symmetric(components: Tensor, rank: int) -> Tensor:
    """Expand STF coordinates into normalized symmetric coordinates."""
    _validate_rank(rank)
    _validate_components(components, stf_dimension(rank), "components")
    basis = stf_basis(rank, dtype=components.dtype, device=components.device)
    return components @ basis.mT


def project_symmetric(components: Tensor, rank: int) -> Tensor:
    """Project normalized symmetric coordinates onto the STF subspace."""
    return stf_to_symmetric(symmetric_to_stf(components, rank), rank)


@lru_cache(maxsize=None)
def _dense_symmetric_basis_master(rank: int) -> Tensor:
    """Build the dense symmetric basis on CPU in double precision."""
    indices = symmetric_multi_indices(rank)
    positions = {alpha: index for index, alpha in enumerate(indices)}
    if rank == 0:
        return torch.ones((1, 1), dtype=torch.float64)

    words = tuple(product(range(3), repeat=rank))
    result = torch.zeros((3**rank, len(indices)), dtype=torch.float64)
    for row, word in enumerate(words):
        alpha = tuple(word.count(axis) for axis in range(3))
        column = positions[alpha]
        result[row, column] = 1.0 / math.sqrt(_multiplicity(rank, alpha))
    return result


def _dense_symmetric_basis(
    rank: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return the dense symmetric basis in requested tensor properties."""
    return _dense_symmetric_basis_master(rank).to(dtype=dtype, device=device)


def symmetric_to_dense(components: Tensor, rank: int) -> Tensor:
    """Expand normalized symmetric coordinates into a dense tensor."""
    _validate_rank(rank)
    _validate_components(components, symmetric_dimension(rank), "components")
    if rank == 0:
        return components[..., 0]
    basis = _dense_symmetric_basis(
        rank, dtype=components.dtype, device=components.device
    )
    flat = components @ basis.mT
    return flat.reshape((*components.shape[:-1], *((3,) * rank)))


def dense_to_symmetric(tensor: Tensor, rank: int) -> Tensor:
    """Project a dense tensor into normalized symmetric coordinates."""
    _validate_rank(rank)
    _validate_float_tensor(tensor, "tensor")
    if rank == 0:
        return tensor.unsqueeze(-1)
    if tensor.ndim < rank or tensor.shape[-rank:] != (3,) * rank:
        raise ValueError(f"tensor must end in {rank} dimensions of size 3")
    basis = _dense_symmetric_basis(rank, dtype=tensor.dtype, device=tensor.device)
    flat = tensor.reshape((*tensor.shape[:-rank], 3**rank))
    return flat @ basis


def stf_to_dense(components: Tensor, rank: int) -> Tensor:
    """Expand STF coordinates into a dense tensor."""
    return symmetric_to_dense(stf_to_symmetric(components, rank), rank)


def dense_to_stf(tensor: Tensor, rank: int) -> Tensor:
    """Project a dense tensor into STF coordinates."""
    return symmetric_to_stf(dense_to_symmetric(tensor, rank), rank)


def symmetric_power_components(vector: Tensor, rank: int) -> Tensor:
    """Return normalized symmetric coordinates of a vector power."""
    _validate_rank(rank)
    _validate_float_tensor(vector, "vector")
    if vector.ndim == 0 or vector.shape[-1] != 3:
        raise ValueError("vector must have final dimension 3")

    values = []
    for alpha in symmetric_multi_indices(rank):
        value = torch.ones_like(vector[..., 0])
        for axis, power in enumerate(alpha):
            if power:
                value = value * vector[..., axis].pow(power)
        values.append(value * math.sqrt(_multiplicity(rank, alpha)))
    return torch.stack(values, dim=-1)


def stf_power_components(vector: Tensor, rank: int) -> Tensor:
    """Return STF coordinates of a projected vector power."""
    return symmetric_to_stf(symmetric_power_components(vector, rank), rank)


def stf_symmetric_product(
    left: Tensor,
    right: Tensor,
    rank_left: int,
    rank_right: int,
) -> Tensor:
    """Return the STF projection of a symmetric tensor product."""
    return stf_tensor_coupling(
        left,
        right,
        rank_left,
        rank_right,
        rank_left + rank_right,
    )


@lru_cache(maxsize=1)
def _levi_civita_master() -> Tensor:
    """Build the Cartesian Levi Civita tensor on CPU."""
    epsilon = torch.zeros((3, 3, 3), dtype=torch.float64)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0
    return epsilon


def _levi_civita(*, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return the Cartesian Levi Civita tensor after a master cast."""
    return _levi_civita_master().to(dtype=dtype, device=device)


def _unbatched_tensor_coupling(
    left: Tensor,
    right: Tensor,
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> Tensor:
    """Couple two unbatched STF tensors through delta and epsilon."""
    left_dense = stf_to_dense(left, rank_left)
    right_dense = stf_to_dense(right, rank_right)
    difference = rank_left + rank_right - output_rank
    contraction_count = difference // 2
    left_axes = tuple(range(rank_left - contraction_count, rank_left))
    right_axes = tuple(range(contraction_count))
    coupled = torch.tensordot(
        left_dense,
        right_dense,
        dims=(left_axes, right_axes) if contraction_count else 0,
    )
    if difference % 2:
        left_remaining = rank_left - contraction_count
        right_remaining = rank_right - contraction_count
        left_axis = left_remaining - 1
        right_axis = left_remaining
        other_axes = tuple(range(left_axis)) + tuple(
            range(right_axis + 1, left_remaining + right_remaining)
        )
        coupled = coupled.permute(other_axes + (left_axis, right_axis))
        coupled = torch.tensordot(
            coupled,
            _levi_civita(dtype=left.dtype, device=left.device),
            dims=((-2, -1), (1, 2)),
        )
    return dense_to_stf(coupled, output_rank)


def _validate_coupling_inputs(
    left: Tensor,
    right: Tensor,
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> torch.dtype:
    """Validate coupling inputs and return their promoted dtype."""
    _validate_rank(rank_left)
    _validate_rank(rank_right)
    _validate_rank(output_rank)
    _validate_components(left, stf_dimension(rank_left), "left")
    _validate_components(right, stf_dimension(rank_right), "right")
    if not abs(rank_left - rank_right) <= output_rank <= rank_left + rank_right:
        raise ValueError("output_rank violates the angular momentum bounds")
    if left.device != right.device:
        raise ValueError("left and right must use the same device")
    return torch.promote_types(left.dtype, right.dtype)


def _stf_tensor_coupling_reference(
    left: Tensor,
    right: Tensor,
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> Tensor:
    """Evaluate the dense reference coupling for verification."""
    dtype = _validate_coupling_inputs(left, right, rank_left, rank_right, output_rank)

    batch_shape = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    left_flat = (
        left.to(dtype=dtype)
        .expand(batch_shape + (stf_dimension(rank_left),))
        .reshape(-1, stf_dimension(rank_left))
    )
    right_flat = (
        right.to(dtype=dtype)
        .expand(batch_shape + (stf_dimension(rank_right),))
        .reshape(-1, stf_dimension(rank_right))
    )
    if left_flat.shape[0] == 0:
        return left_flat.new_empty(batch_shape + (stf_dimension(output_rank),))
    coupled = [
        _unbatched_tensor_coupling(
            left_value,
            right_value,
            rank_left,
            rank_right,
            output_rank,
        )
        for left_value, right_value in zip(left_flat, right_flat)
    ]
    return torch.stack(coupled).reshape(batch_shape + (stf_dimension(output_rank),))


@lru_cache(maxsize=None)
def _coupling_coefficients_master(
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> Tensor:
    """Build canonical coupling coefficients on CPU in double precision."""
    left_size = stf_dimension(rank_left)
    right_size = stf_dimension(rank_right)
    left = torch.eye(left_size, dtype=torch.float64)[:, None, :]
    right = torch.eye(right_size, dtype=torch.float64)[None, :, :]
    with torch.no_grad():
        values = _stf_tensor_coupling_reference(
            left, right, rank_left, rank_right, output_rank
        )
    return values.permute(2, 0, 1).contiguous()


@lru_cache(maxsize=None)
def _coupling_coefficients(
    rank_left: int,
    rank_right: int,
    output_rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return cached coupling coefficients after a master cast."""
    return _coupling_coefficients_master(rank_left, rank_right, output_rank).to(
        dtype=dtype, device=device
    )


def stf_tensor_coupling(
    left: Tensor,
    right: Tensor,
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> Tensor:
    """Return one SO(3) coupling channel through batched contraction."""
    dtype = _validate_coupling_inputs(left, right, rank_left, rank_right, output_rank)
    coefficients = _coupling_coefficients(
        rank_left,
        rank_right,
        output_rank,
        dtype,
        left.device,
    )
    return torch.einsum(
        "...i,...j,oij->...o",
        left.to(dtype=dtype),
        right.to(dtype=dtype),
        coefficients,
    )
