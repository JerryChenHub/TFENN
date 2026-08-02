"""SO(3) reference contractions of STF pose blocks with positions.

These functions provide reference scalar and vector covariants. They do not
construct a complete polynomial basis for a finite group G. The vector result
is the partial derivative with respect to position while pose_block is fixed.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor

from .stf_space import (
    stf_basis,
    stf_power_components,
    symmetric_multi_indices,
    symmetric_power_components,
)


__all__ = ["scalar_contraction", "vector_covariant"]


def _validate_inputs(
    position: Tensor,
    pose_block: Tensor,
    rank: int,
    strict_same_edge: bool,
) -> tuple[Tensor, Tensor, torch.Size]:
    """Validate one position and pose contraction request."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if not isinstance(strict_same_edge, bool):
        raise TypeError("strict_same_edge must be a bool")
    if not isinstance(position, Tensor):
        raise TypeError("position must be a torch.Tensor")
    if position.ndim == 0 or position.shape[-1] != 3:
        raise ValueError("position must have trailing shape (3,)")
    if not isinstance(pose_block, Tensor):
        raise TypeError("pose_block must be a torch.Tensor")
    if pose_block.ndim < 2:
        raise ValueError("pose_block must have STF and anchor dimensions")

    expected = 2 * rank + 1
    if pose_block.shape[-2] != expected:
        raise ValueError(
            f"rank {rank} pose_block must have shape (..., {expected}, anchors)"
        )
    if pose_block.shape[-1] == 0:
        raise ValueError("pose_block must contain at least one anchor")
    if position.dtype not in (torch.float32, torch.float64):
        raise TypeError("position must use float32 or float64")
    if pose_block.dtype not in (torch.float32, torch.float64):
        raise TypeError("pose_block must use float32 or float64")
    if position.dtype != pose_block.dtype:
        raise TypeError("position and pose_block must use the same dtype")
    if position.device != pose_block.device:
        raise ValueError("position and pose_block must use the same device")
    if not torch.isfinite(position).all().item():
        raise ValueError("position must contain only finite values")
    if not torch.isfinite(pose_block).all().item():
        raise ValueError("pose_block must contain only finite values")

    position_batch = position.shape[:-1]
    pose_batch = pose_block.shape[:-2]
    if strict_same_edge:
        if position_batch != pose_batch:
            raise ValueError(
                "strict_same_edge requires identical position and pose batch shapes"
            )
        batch_shape = position_batch
    else:
        try:
            batch_shape = torch.broadcast_shapes(position_batch, pose_batch)
        except RuntimeError as error:
            raise ValueError(
                "position and pose batch shapes are not broadcast compatible"
            ) from error
    return position, pose_block, batch_shape


@lru_cache(maxsize=None)
def _symmetric_derivatives_master(rank: int) -> Tensor:
    """Return the canonical CPU float64 derivative master."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("derivative rank must be a positive integer")
    high = symmetric_multi_indices(rank)
    low = symmetric_multi_indices(rank - 1)
    low_positions = {multi_index: index for index, multi_index in enumerate(low)}
    derivatives = torch.zeros((3, len(low), len(high)), dtype=torch.float64)
    for high_index, alpha in enumerate(high):
        for axis, count in enumerate(alpha):
            if count == 0:
                continue
            beta = list(alpha)
            beta[axis] -= 1
            low_index = low_positions[tuple(beta)]
            derivatives[axis, low_index, high_index] = math.sqrt(rank * count)
    return derivatives


@lru_cache(maxsize=None)
def _symmetric_derivatives(
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Cast and cache a canonical derivative master."""
    return _symmetric_derivatives_master(rank).to(dtype=dtype, device=device)


def scalar_contraction(
    position: Tensor,
    pose_block: Tensor,
    rank: int,
    *,
    strict_same_edge: bool = True,
) -> Tensor:
    """Return the SO(3) reference scalar for each anchor.

    The final output dimension indexes anchors. By default, position and
    pose_block must have identical batch shapes so every contraction belongs
    to the same edge. Set strict_same_edge to False to request broadcasting.
    """
    position, pose_block, _ = _validate_inputs(
        position, pose_block, rank, strict_same_edge
    )
    if rank == 0:
        position_one = position[..., 0].unsqueeze(-1) * 0.0 + 1.0
        return position_one * pose_block[..., 0, :]
    power = stf_power_components(position, rank)
    return torch.einsum("...d,...da->...a", power, pose_block)


def vector_covariant(
    position: Tensor,
    pose_block: Tensor,
    rank: int,
    *,
    strict_same_edge: bool = True,
) -> Tensor:
    """Return the SO(3) reference vector for each anchor.

    This is the partial derivative of scalar_contraction with respect to
    position while pose_block is fixed. The final dimensions index anchors
    and Cartesian components. By default, both inputs must describe the same
    edges. Set strict_same_edge to False to request broadcasting.
    """
    position, pose_block, batch_shape = _validate_inputs(
        position, pose_block, rank, strict_same_edge
    )
    anchor_count = pose_block.shape[-1]
    if rank == 0:
        position_zero = position[..., 0].unsqueeze(-1) * 0.0
        pose_zero = pose_block[..., 0, :] * 0.0
        zero = position_zero + pose_zero
        return zero.unsqueeze(-1).expand(batch_shape + (anchor_count, 3))

    basis = stf_basis(rank, dtype=pose_block.dtype, device=pose_block.device)
    symmetric_pose = basis @ pose_block
    lower_power = symmetric_power_components(position, rank - 1)
    derivatives = _symmetric_derivatives(rank, pose_block.dtype, pose_block.device)
    return torch.einsum(
        "...b,jbn,...na->...aj", lower_power, derivatives, symmetric_pose
    )
