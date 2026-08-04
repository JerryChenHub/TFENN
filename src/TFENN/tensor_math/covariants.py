"""Evaluate scalar contractions and their position covariants.

This file owns two major public runtime APIs, ``scalar_contraction`` and
``vector_covariant``.  They contract one Cartesian position with one STF pose
block.  They are reference SO(3) covariants and do not claim to form a
complete polynomial basis for a finite molecular group.

For Python ``rank`` equal to the mathematical degree ``l``, let
``d_l = 2 * l + 1`` and let ``p_l(x)`` be the normalized STF coordinates of
the symmetric vector power.  The public formulas are

``s_a(x, B) = sum_mu p_l(x)[mu] B[mu, a]``

and

``q_a,j(x, B) = partial s_a(x, B) / partial x_j`` with ``B`` fixed.

Thus ``scalar_contraction`` maps ``(..., 3)`` and
``(..., d_l, m_l)`` to ``(..., m_l)``.  ``vector_covariant`` maps the same
inputs to ``(..., m_l, 3)``.  The vector is neither a total derivative when
the pose block depends on position nor an automatically negated force.

Actions use active column vectors from body frame to common frame, with
``D_l(R1 R2) = D_l(R1) D_l(R2)``.  Under simultaneous action on position and
pose block, the scalar is invariant and each stored row vector transforms on
the right by the transpose of the active Cartesian action.

The degree ``l`` is an integer from zero through ``MAX_STF_RANK``, currently
ten.  Rank zero has ``d_l = 1`` and an exactly zero vector covariant whose
autograd graph remains connected to both inputs.  The convention identity
``TENSOR_CONVENTION_VERSION`` covers multi index order, the normalized
symmetric and STF bases, active action, epsilon sign, coupling scale and
phase, canonical subspaces, rank order, and pose flattening.

With ``strict_same_edge=True``, both inputs must have identical batch axes.
With ``strict_same_edge=False``, their batch axes use ordinary PyTorch
broadcasting.  Empty batch axes are supported.  The multiplicity ``m_l``
must be positive and is never broadcast as a batch axis.

Both inputs must use the same device and the same float32 or float64 dtype.
Outputs preserve that dtype and device.  Every tensor contraction preserves
gradients with respect to position and pose block.  The default hot path
checks only static metadata and does not synchronize on tensor data.
``validate_finite=True`` explicitly checks all data values before evaluation
and may synchronize devices, so it is intended for validation outside
ordinary autograd and compiled network cores.

Canonical derivative constants are built deterministically on CPU in
float64 and cast to the runtime dtype and device.  Exact floating results may
vary across PyTorch, device, and arithmetic library versions.  TypeError
reports unsupported objects, dtypes, and flags.  ValueError reports rank,
trailing shape, multiplicity, device, batch compatibility, and requested
finite data failures, including actual and expected shape information.

Mapped references:

Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li,
Kai Kohlhoff, Patrick Riley. Tensor field networks: Rotation and translation
equivariant neural networks for 3D point clouds.
https://arxiv.org/abs/1802.08219

Andrea Grisafi, David M. Wilkins, Gabor Csanyi, Michele Ceriotti.
Symmetry Adapted Machine Learning for Tensorial Properties of Atomistic
Systems. https://arxiv.org/abs/1709.06757
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor

from .stf_space import (
    MAX_STF_RANK,
    stf_basis,
    stf_power_components,
    symmetric_multi_indices,
    symmetric_power_components,
)


__all__ = ["scalar_contraction", "vector_covariant"]


_assume_constant_result = torch.compiler.assume_constant_result


def _validate_inputs(
    position: Tensor,
    pose_block: Tensor,
    rank: int,
    strict_same_edge: bool,
    validate_finite: bool,
) -> tuple[Tensor, Tensor, torch.Size]:
    """Validate one position and pose contraction request."""
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank > MAX_STF_RANK
    ):
        raise ValueError(
            f"rank actual {rank!r} must be an integer from zero through "
            f"{MAX_STF_RANK}"
        )
    if not isinstance(strict_same_edge, bool):
        raise TypeError("strict_same_edge must be a bool")
    if not isinstance(validate_finite, bool):
        raise TypeError("validate_finite must be a bool")
    if not isinstance(position, Tensor):
        raise TypeError("position must be a torch.Tensor")
    if position.ndim == 0 or position.shape[-1] != 3:
        raise ValueError(
            f"position actual shape {tuple(position.shape)} must have expected "
            "trailing shape (3,)"
        )
    if not isinstance(pose_block, Tensor):
        raise TypeError("pose_block must be a torch.Tensor")
    if pose_block.ndim < 2:
        raise ValueError(
            f"rank {rank} pose_block actual shape {tuple(pose_block.shape)} must "
            f"have expected trailing shape ({2 * rank + 1}, m_l)"
        )

    expected = 2 * rank + 1
    if pose_block.shape[-2] != expected:
        raise ValueError(
            f"rank {rank} pose_block actual shape {tuple(pose_block.shape)} must "
            f"have expected trailing shape ({expected}, m_l)"
        )
    if pose_block.shape[-1] == 0:
        raise ValueError(
            f"rank {rank} pose_block actual shape {tuple(pose_block.shape)} must "
            "have expected positive m_l"
        )
    if position.dtype not in (torch.float32, torch.float64):
        raise TypeError("position must use float32 or float64")
    if pose_block.dtype not in (torch.float32, torch.float64):
        raise TypeError("pose_block must use float32 or float64")
    if position.dtype != pose_block.dtype:
        raise TypeError("position and pose_block must use the same dtype")
    if position.device != pose_block.device:
        raise ValueError("position and pose_block must use the same device")
    if validate_finite:
        if not bool(torch.isfinite(position).all()):
            raise ValueError("position must contain only finite values")
        if not bool(torch.isfinite(pose_block).all()):
            raise ValueError("pose_block must contain only finite values")

    position_batch = position.shape[:-1]
    pose_batch = pose_block.shape[:-2]
    if strict_same_edge:
        if position_batch != pose_batch:
            raise ValueError(
                "strict_same_edge requires identical position and pose batch "
                f"shapes: actual {position_batch} and {pose_batch}"
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
def _symmetric_derivatives_cached(
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Cast and cache a canonical derivative master."""
    return _symmetric_derivatives_master(rank).to(dtype=dtype, device=device)


@_assume_constant_result
def _symmetric_derivatives(
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return one cached derivative map as a graph constant."""
    return _symmetric_derivatives_cached(rank, dtype, device)


def scalar_contraction(
    position: Tensor,
    pose_block: Tensor,
    rank: int,
    *,
    strict_same_edge: bool = True,
    validate_finite: bool = False,
) -> Tensor:
    """Return the SO(3) reference scalar for each anchor.

    The final output dimension indexes anchors. By default, position and
    pose_block must have identical batch shapes so every contraction belongs
    to the same edge. Set strict_same_edge to False to request broadcasting.
    Set validate_finite to True for an explicit data validation pass.
    """
    position, pose_block, _ = _validate_inputs(
        position,
        pose_block,
        rank,
        strict_same_edge,
        validate_finite,
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
    validate_finite: bool = False,
) -> Tensor:
    """Return the SO(3) reference vector for each anchor.

    This is the partial derivative of scalar_contraction with respect to
    position while pose_block is fixed. The final dimensions index anchors
    and Cartesian components. By default, both inputs must describe the same
    edges. Set strict_same_edge to False to request broadcasting.
    Set validate_finite to True for an explicit data validation pass.
    """
    position, pose_block, _ = _validate_inputs(
        position,
        pose_block,
        rank,
        strict_same_edge,
        validate_finite,
    )
    if rank == 0:
        position_zero = position[..., 0].unsqueeze(-1) * 0.0
        pose_zero = pose_block[..., 0, :] * 0.0
        zero = position_zero + pose_zero
        return zero.unsqueeze(-1).expand(zero.shape + (3,))

    basis = stf_basis(rank, dtype=pose_block.dtype, device=pose_block.device)
    symmetric_pose = basis @ pose_block
    lower_power = symmetric_power_components(position, rank - 1)
    derivatives = _symmetric_derivatives(rank, pose_block.dtype, pose_block.device)
    return torch.einsum(
        "...b,jbn,...na->...aj", lower_power, derivatives, symmetric_pose
    )
