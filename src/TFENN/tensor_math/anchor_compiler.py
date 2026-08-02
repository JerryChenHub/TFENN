"""Compile primitive invariant STF anchors from generator constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from ._compiler_utils import (
    canonical_column_space,
    canonical_nullspace,
    canonical_projector_basis,
    positive_finite_tolerance,
)
from .stf_rep import stf_representation
from .stf_space import STF_BASIS_VERSION, stf_tensor_coupling


__all__ = [
    "AnchorBlock",
    "AnchorCompilation",
    "AnchorDimensions",
    "compile_anchors",
]


@dataclass(frozen=True)
class AnchorDimensions:
    """Store fixed, generated, and primitive subspace dimensions."""

    fixed: int
    generated: int
    primitive: int


@dataclass(frozen=True)
class AnchorBlock:
    """Store the invariant subspaces compiled at one STF rank."""

    rank: int
    fixed_basis: Tensor
    generated_basis: Tensor
    primitive_basis: Tensor
    constraint_residual: float
    generated_projection_residual: float
    dimensions: AnchorDimensions
    singular_values: Tensor
    nullspace_threshold: float
    singular_value_gap: float


@dataclass(frozen=True)
class AnchorCompilation:
    """Store primitive anchor compilation results through the maximum rank."""

    blocks: dict[int, AnchorBlock]
    requested_ranks: tuple[int, ...]
    generator_count: int
    generators: Tensor
    basis_version: str
    nullspace_atol: float
    nullspace_rtol: float
    rotation_atol: float
    rotation_rtol: float

    @property
    def tolerance(self) -> float:
        """Return the legacy absolute nullspace tolerance."""
        return self.nullspace_atol

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return compiled ranks in ascending order."""
        return tuple(self.blocks)

    @property
    def primitive_anchors(self) -> dict[int, Tensor]:
        """Return the retained primitive anchor matrices."""
        return {
            rank: block.primitive_basis
            for rank, block in self.blocks.items()
            if block.primitive_basis.shape[1] > 0
        }

    @property
    def encoding_dimension(self) -> int:
        """Return the dimension of the retained primitive B space."""
        return sum(
            (2 * rank + 1) * block.dimensions.primitive
            for rank, block in self.blocks.items()
        )

    @property
    def residual(self) -> float:
        """Return the largest generator constraint residual."""
        return max(
            (block.constraint_residual for block in self.blocks.values()), default=0.0
        )

    @property
    def dimensions(self) -> dict[int, AnchorDimensions]:
        """Return the three subspace dimensions at each rank."""
        return {rank: block.dimensions for rank, block in self.blocks.items()}


def _as_generators(generators: Tensor | Sequence[Tensor]) -> Tensor:
    """Coerce generators to one real floating tensor."""
    if isinstance(generators, Tensor):
        matrices = generators.unsqueeze(0) if generators.ndim == 2 else generators
    else:
        items = list(generators)
        if not items:
            return torch.empty((0, 3, 3), dtype=torch.float64)
        matrices = torch.stack([torch.as_tensor(item) for item in items])
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("generators must have shape (count, 3, 3)")
    if matrices.dtype not in (torch.float32, torch.float64):
        raise TypeError("generators must use float32 or float64")
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError("generators must contain only finite values")
    return matrices


def _validate_rotations(matrices: Tensor, atol: float, rtol: float) -> None:
    """Require proper orthogonal generator matrices."""
    if matrices.shape[0] == 0:
        return
    identity = torch.eye(3, dtype=matrices.dtype, device=matrices.device)
    gram_error = (
        (matrices.transpose(-1, -2) @ matrices - identity).abs().amax(dim=(-2, -1))
    )
    determinant_error = (torch.linalg.det(matrices) - 1.0).abs()
    threshold = atol + rtol
    if bool((gram_error > threshold).any()) or bool(
        (determinant_error > threshold).any()
    ):
        raise ValueError("generators must belong to SO(3)")


def _selected_ranks(ranks: int | Iterable[int]) -> tuple[int, ...]:
    """Validate and sort the requested positive STF ranks."""
    values = (ranks,) if isinstance(ranks, int) else tuple(ranks)
    if not values:
        raise ValueError("ranks must not be empty")
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
        for rank in values
    ):
        raise ValueError("ranks must contain positive integers")
    return tuple(sorted(set(values)))


def _orthonormal_span(matrix: Tensor, atol: float, rtol: float) -> Tensor:
    """Return a deterministic orthonormal basis for a column span."""
    return canonical_column_space(matrix, atol, rtol).basis


def _generated_coupling_closure(
    primitive: dict[int, Tensor],
    fixed_spaces: dict[int, Tensor],
    target_rank: int,
    atol: float,
    rtol: float,
) -> tuple[Tensor, float]:
    """Close lower primitive anchors under every STF tensor coupling."""
    known = {
        rank: (
            primitive[rank].clone()
            if rank in primitive and rank < target_rank
            else fixed.new_empty((2 * rank + 1, 0))
        )
        for rank, fixed in fixed_spaces.items()
    }
    frontier = {rank: space.clone() for rank, space in known.items()}
    capacity = sum(fixed.shape[1] for fixed in fixed_spaces.values())
    largest_leakage = 0.0

    for _ in range(capacity + 1):
        candidates: dict[int, list[Tensor]] = {rank: [] for rank in known}
        for rank_left, left in frontier.items():
            if left.shape[1] == 0:
                continue
            for rank_right, right in known.items():
                if right.shape[1] == 0:
                    continue
                highest = min(rank_left + rank_right, target_rank)
                for output_rank in range(
                    max(1, abs(rank_left - rank_right)), highest + 1
                ):
                    if (
                        known[output_rank].shape[1]
                        == fixed_spaces[output_rank].shape[1]
                    ):
                        continue
                    coupled = stf_tensor_coupling(
                        left.T[:, None],
                        right.T[None],
                        rank_left,
                        rank_right,
                        output_rank,
                    )
                    candidates[output_rank].append(
                        coupled.reshape(-1, 2 * output_rank + 1).T
                    )

        updated: dict[int, Tensor] = {}
        next_frontier: dict[int, Tensor] = {}
        grew = False
        for rank, fixed in fixed_spaces.items():
            if candidates[rank]:
                raw_candidates = torch.cat(candidates[rank], dim=1)
                projected = fixed @ (fixed.T @ raw_candidates)
                largest_leakage = max(
                    largest_leakage,
                    float(torch.linalg.matrix_norm(raw_candidates - projected)),
                )
                updated[rank] = _orthonormal_span(
                    torch.cat((known[rank], projected), dim=1), atol, rtol
                )
            else:
                updated[rank] = known[rank]
            increase = updated[rank].shape[1] - known[rank].shape[1]
            if increase < 0 or updated[rank].shape[1] > fixed.shape[1]:
                raise RuntimeError("generated invariant space changed inconsistently")
            if increase:
                grew = True
                projector = updated[rank] @ updated[rank].T
                projector = projector - known[rank] @ known[rank].T
                projector = 0.5 * (projector + projector.T)
                next_frontier[rank] = canonical_projector_basis(projector, increase)
            else:
                next_frontier[rank] = fixed.new_empty((2 * rank + 1, 0))
        known = updated
        frontier = next_frontier
        if known[target_rank].shape[1] == fixed_spaces[target_rank].shape[1]:
            return known[target_rank], largest_leakage
        if not grew:
            return known[target_rank], largest_leakage

    raise RuntimeError("generated tensor coupling closure did not converge")


def compile_anchors(
    generators: Tensor | Sequence[Tensor],
    ranks: int | Iterable[int],
    *,
    atol: float | None = None,
    nullspace_atol: float | None = None,
    nullspace_rtol: float | None = None,
    rotation_atol: float | None = None,
    rotation_rtol: float | None = None,
) -> AnchorCompilation:
    """Compile every primitive invariant through the largest requested rank."""
    source = _as_generators(generators)
    source_dtype = source.dtype
    if atol is not None and nullspace_atol is not None:
        raise ValueError("atol and nullspace_atol cannot both be provided")
    requested_nullspace_atol = nullspace_atol if nullspace_atol is not None else atol
    default_atol = 1e-6 if source_dtype == torch.float32 else 1e-10
    resolved_nullspace_atol = positive_finite_tolerance(
        requested_nullspace_atol,
        "nullspace_atol",
        default_atol,
    )
    resolved_nullspace_rtol = positive_finite_tolerance(
        nullspace_rtol,
        "nullspace_rtol",
        1e-12,
    )
    resolved_rotation_atol = positive_finite_tolerance(
        rotation_atol,
        "rotation_atol",
        default_atol,
    )
    resolved_rotation_rtol = positive_finite_tolerance(
        rotation_rtol,
        "rotation_rtol",
        default_atol,
    )
    matrices = source.detach().to(device="cpu", dtype=torch.float64)
    selected = _selected_ranks(ranks)
    scan_ranks = range(1, max(selected) + 1)
    _validate_rotations(
        matrices,
        resolved_rotation_atol,
        resolved_rotation_rtol,
    )
    blocks: dict[int, AnchorBlock] = {}
    primitive: dict[int, Tensor] = {}
    fixed_spaces: dict[int, Tensor] = {}

    with torch.no_grad():
        for rank in scan_ranks:
            dimension = 2 * rank + 1
            identity = torch.eye(
                dimension, dtype=matrices.dtype, device=matrices.device
            )
            represented = stf_representation(
                matrices,
                rank,
                rotation_atol=resolved_rotation_atol,
                rotation_rtol=resolved_rotation_rtol,
            )
            constraints = (represented - identity).reshape(-1, dimension)
            nullspace = canonical_nullspace(
                constraints,
                resolved_nullspace_atol,
                resolved_nullspace_rtol,
            )
            fixed = nullspace.basis
            fixed_spaces[rank] = fixed
            generated, leakage = _generated_coupling_closure(
                primitive,
                fixed_spaces,
                rank,
                resolved_nullspace_atol,
                resolved_nullspace_rtol,
            )
            fixed_projector = fixed @ fixed.T
            primitive_dimension = fixed.shape[1] - generated.shape[1]
            if primitive_dimension < 0:
                raise RuntimeError("generated invariant space exceeds the fixed space")
            primitive_projector = fixed_projector - generated @ generated.T
            primitive_projector = 0.5 * (primitive_projector + primitive_projector.T)
            retained = canonical_projector_basis(
                primitive_projector,
                primitive_dimension,
            )
            primitive[rank] = retained
            blocks[rank] = AnchorBlock(
                rank=rank,
                fixed_basis=fixed,
                generated_basis=generated,
                primitive_basis=retained,
                constraint_residual=nullspace.residual,
                generated_projection_residual=leakage,
                dimensions=AnchorDimensions(
                    fixed=fixed.shape[1],
                    generated=generated.shape[1],
                    primitive=retained.shape[1],
                ),
                singular_values=nullspace.singular_values,
                nullspace_threshold=nullspace.threshold,
                singular_value_gap=nullspace.singular_value_gap,
            )
    return AnchorCompilation(
        blocks=blocks,
        requested_ranks=selected,
        generator_count=matrices.shape[0],
        generators=matrices.clone(),
        basis_version=STF_BASIS_VERSION,
        nullspace_atol=resolved_nullspace_atol,
        nullspace_rtol=resolved_nullspace_rtol,
        rotation_atol=resolved_rotation_atol,
        rotation_rtol=resolved_rotation_rtol,
    )
