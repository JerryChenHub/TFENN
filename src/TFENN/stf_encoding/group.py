"""Finite rotation groups and their complete invariant STF spaces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .basis import stf_basis, stf_representation


__all__ = [
    "FiniteSO3Classification",
    "StabilizerCertificate",
    "classify_finite_so3_group",
    "enumerate_finite_group",
    "invariant_stf_anchors",
    "stabilizer_certificate",
    "validate_so3_generators",
]


@dataclass(frozen=True)
class FiniteSO3Classification:
    """Store one family from the finite subgroup classification of SO(3)."""

    family: str
    order: int
    n: int | None = None

    @property
    def label(self) -> str:
        """Return the conventional concrete group label."""
        return f"{self.family[0]}{self.n}" if self.n is not None else self.family

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return the complete fixed space rank recipe for this family."""
        if self.family == "C1":
            return (1,)
        if self.family == "Cn":
            return (1, int(self.n))
        if self.family == "D2":
            return (2,)
        if self.family == "Dn":
            return (2, int(self.n))
        if self.family == "T":
            return (3,)
        if self.family == "O":
            return (4,)
        if self.family == "I":
            return (6,)
        raise ValueError(f"unsupported family {self.family!r}")


@dataclass(frozen=True)
class StabilizerCertificate:
    """Certify a joint stabilizer using complete fixed STF spaces."""

    classification: FiniteSO3Classification
    ranks: tuple[int, ...]
    certifying_ranks: tuple[int, ...]
    fixed_dimensions: tuple[int, ...]
    generator_constraint_residual: float
    complete_fixed_spaces: bool
    theorem: str
    proof: str

    @property
    def exact(self) -> bool:
        """Report whether the certificate proves global equality."""
        return self.complete_fixed_spaces

    @property
    def group_name(self) -> str:
        """Return the classified finite rotation group label."""
        return self.classification.label

    @property
    def group_order(self) -> int:
        """Return the classified finite group order."""
        return self.classification.order

    @property
    def encoding_dimension(self) -> int:
        """Return the dimension certified by the complete fixed spaces."""
        return sum(
            (2 * rank + 1) * dimension
            for rank, dimension in zip(self.ranks, self.fixed_dimensions)
        )


def _tol(dtype: torch.dtype, atol: float | None) -> float:
    """Choose a tolerance suitable for numerical group operations."""
    if atol is not None:
        if atol <= 0.0:
            raise ValueError("atol must be positive")
        return float(atol)
    return 1e-6 if dtype == torch.float32 else 1e-10


def _as_matrices(generators: Tensor | Sequence[Tensor]) -> Tensor:
    """Coerce generators to one real floating tensor."""
    if isinstance(generators, Tensor):
        matrices = generators.unsqueeze(0) if generators.ndim == 2 else generators
    else:
        items = list(generators)
        if not items:
            return torch.empty((0, 3, 3), dtype=torch.get_default_dtype())
        matrices = torch.stack([torch.as_tensor(item) for item in items])
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("generators must have shape (count, 3, 3)")
    if not matrices.is_floating_point():
        raise TypeError("generators must have a real floating dtype")
    return matrices


def validate_so3_generators(
    generators: Tensor | Sequence[Tensor], *, atol: float | None = None
) -> Tensor:
    """Validate and return a stacked set of SO(3) generators."""
    matrices = _as_matrices(generators)
    if matrices.numel() == 0:
        return matrices
    tolerance = _tol(matrices.dtype, atol)
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError("generators must contain only finite values")
    identity = torch.eye(3, dtype=matrices.dtype, device=matrices.device)
    gram = matrices.transpose(-1, -2) @ matrices
    orthogonal_error = torch.linalg.matrix_norm(gram - identity, dim=(-2, -1))
    determinant_error = (torch.linalg.det(matrices) - 1.0).abs()
    if bool((orthogonal_error > 8.0 * tolerance).any()) or bool(
        (determinant_error > 8.0 * tolerance).any()
    ):
        raise ValueError(
            "generators must be proper orthogonal matrices; "
            f"maximum errors are {orthogonal_error.max().item():.3e} and "
            f"{determinant_error.max().item():.3e}"
        )
    return matrices


def _project_so3(matrix: Tensor) -> Tensor:
    """Remove accumulated roundoff while preserving the proper component."""
    left, _, right = torch.linalg.svd(matrix)
    correction = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    correction[-1, -1] = torch.sign(torch.linalg.det(left @ right))
    return left @ correction @ right


def enumerate_finite_group(
    generators: Tensor | Sequence[Tensor],
    *,
    atol: float | None = None,
    max_order: int = 512,
) -> Tensor:
    """Enumerate the finite SO(3) group generated by the input matrices."""
    matrices = validate_so3_generators(generators, atol=atol).detach()
    if max_order < 1:
        raise ValueError("max_order must be positive")
    tolerance = _tol(matrices.dtype, atol)
    identity = torch.eye(3, dtype=matrices.dtype, device=matrices.device)
    steps = [matrix for matrix in matrices]
    steps.extend(matrix.T for matrix in matrices)
    members = [identity]
    cursor = 0
    with torch.no_grad():
        while cursor < len(members):
            for step in steps:
                candidate = _project_so3(members[cursor] @ step)
                known = any(
                    torch.linalg.matrix_norm(candidate - member).item()
                    <= 8.0 * tolerance
                    for member in members
                )
                if known:
                    continue
                if len(members) >= max_order:
                    raise ValueError(
                        "group closure exceeded max_order; the group may be infinite "
                        "or the tolerance may be too small"
                    )
                members.append(candidate)
            cursor += 1
    return torch.stack(members)


def _nullspace(matrix: Tensor, tolerance: float) -> Tensor:
    """Return a full numerical right nullspace."""
    columns = matrix.shape[1]
    if matrix.shape[0] == 0:
        return torch.eye(columns, dtype=matrix.dtype, device=matrix.device)
    _, singular, right = torch.linalg.svd(matrix, full_matrices=True)
    scale = max(1.0, singular[0].item())
    threshold = tolerance * max(matrix.shape) * scale
    rank = int((singular > threshold).sum().item())
    return right[rank:].T


def _element_order(matrix: Tensor, group_order: int, tolerance: float) -> int:
    """Find the order of one numerical rotation."""
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    power = identity
    for order in range(1, group_order + 1):
        power = _project_so3(power @ matrix)
        if torch.linalg.matrix_norm(power - identity).item() <= 8.0 * tolerance:
            return order
    raise ValueError("an element order does not divide the enumerated group order")


def classify_finite_so3_group(
    group: Tensor | Sequence[Tensor], *, atol: float | None = None
) -> FiniteSO3Classification:
    """Classify a finite SO(3) group as cyclic, dihedral, or polyhedral."""
    matrices = validate_so3_generators(group, atol=atol).detach()
    order = len(matrices)
    if order == 0:
        raise ValueError("group must contain the identity")
    tolerance = _tol(matrices.dtype, atol)
    identity = torch.eye(3, dtype=matrices.dtype, device=matrices.device)
    if not bool(
        (torch.linalg.matrix_norm(matrices - identity, dim=(-2, -1)) <= 8.0 * tolerance).any()
    ):
        raise ValueError("group must contain the identity")
    orders = Counter(_element_order(g, order, tolerance) for g in matrices)
    polyhedral = {
        (12, ((1, 1), (2, 3), (3, 8))): "T",
        (24, ((1, 1), (2, 9), (3, 8), (4, 6))): "O",
        (60, ((1, 1), (2, 15), (3, 20), (5, 24))): "I",
    }
    family = polyhedral.get((order, tuple(sorted(orders.items()))))
    if family is not None:
        return FiniteSO3Classification(family, order)
    if order == 1:
        return FiniteSO3Classification("C1", 1)
    constraints = torch.cat([g - identity for g in matrices], dim=0)
    if _nullspace(constraints, tolerance).shape[1] == 1:
        if max(orders) != order:
            raise ValueError("a finite axial subgroup must be cyclic")
        return FiniteSO3Classification("Cn", order, order)
    if order == 4 and orders == Counter({2: 3, 1: 1}):
        return FiniteSO3Classification("D2", 4, 2)
    if order % 2 == 0:
        n = order // 2
        expected_twos = n + int(n % 2 == 0)
        if n >= 3 and max(orders) == n and orders[2] == expected_twos:
            return FiniteSO3Classification("Dn", order, n)
    raise ValueError("the matrices do not match a finite SO(3) subgroup family")


def invariant_stf_anchors(
    generators: Tensor | Sequence[Tensor],
    ranks: int | Iterable[int],
    *,
    atol: float | None = None,
) -> dict[int, Tensor]:
    """Find every invariant STF anchor from generator constraints only."""
    matrices = validate_so3_generators(generators, atol=atol)
    selected = (ranks,) if isinstance(ranks, int) else tuple(dict.fromkeys(ranks))
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in selected):
        raise ValueError("ranks must contain nonnegative integers")
    tolerance = _tol(matrices.dtype, atol)
    anchors: dict[int, Tensor] = {}
    for rank in selected:
        dimension = 2 * rank + 1
        if len(matrices) == 0:
            anchors[rank] = torch.eye(
                dimension, dtype=matrices.dtype, device=matrices.device
            )
            continue
        basis = stf_basis(rank, dtype=matrices.dtype, device=matrices.device)
        constraints = torch.cat(
            [stf_representation(g, rank, basis=basis) - torch.eye(
                dimension, dtype=matrices.dtype, device=matrices.device
            ) for g in matrices],
            dim=0,
        )
        anchors[rank] = _nullspace(constraints, tolerance)
    return anchors


def _recipe(
    classification: FiniteSO3Classification,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Return classification ranks, dimensions, and the global proof."""
    family, n = classification.family, classification.n
    if family == "C1":
        return (1,), (3,), "Fixing the complete vector space pointwise leaves only identity."
    if family == "Cn":
        assert n is not None
        return (1, n), (1, 3), "Rank one fixes the oriented axis and rank n fixes its n fold phase."
    if family == "D2":
        return (2,), (2,), "The complete diagonal traceless space has pointwise stabilizer D2."
    if family == "Dn":
        assert n is not None
        return (2, n), (1, 1 + int(n % 2 == 0)), (
            "Rank two fixes the unoriented axis and rank n fixes exactly the n horizontal axes."
        )
    if family == "T":
        return (3,), (1,), (
            "A nonzero rank three harmonic has a proper closed stabilizer. Its "
            "T subgroup is neither cyclic nor dihedral, so it cannot lie in an "
            "infinite proper closed SO(3) subgroup. O and I have no rank three "
            "fixed vector. Thus the stabilizer is T."
        )
    if family == "O":
        return (4,), (1,), (
            "A nonzero rank four harmonic has a proper closed stabilizer. Its "
            "O subgroup is neither cyclic nor dihedral, so it cannot lie in an "
            "infinite proper closed SO(3) subgroup. O has no proper finite "
            "rotation overgroup."
        )
    if family == "I":
        return (6,), (1,), (
            "A nonzero rank six harmonic has a proper closed stabilizer. Its "
            "I subgroup is neither cyclic nor dihedral, so it cannot lie in an "
            "infinite proper closed SO(3) subgroup. I has no proper finite "
            "rotation overgroup."
        )
    raise ValueError(f"unsupported family {family!r}")


def stabilizer_certificate(
    generators: Tensor | Sequence[Tensor],
    anchors: Mapping[int, Tensor] | None = None,
    *,
    atol: float | None = None,
    max_order: int = 512,
) -> StabilizerCertificate:
    """Certify the exact global joint stabilizer of complete STF fixed spaces."""
    matrices = validate_so3_generators(generators, atol=atol)
    group = enumerate_finite_group(matrices, atol=atol, max_order=max_order)
    classification = classify_finite_so3_group(group, atol=atol)
    certifying_ranks, certifying_dimensions, proof = _recipe(classification)
    raw_supplied = (
        invariant_stf_anchors(matrices, certifying_ranks, atol=atol)
        if anchors is None
        else dict(anchors)
    )
    supplied: dict[int, Tensor] = {}
    for rank, value in raw_supplied.items():
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("anchor ranks must be nonnegative integers")
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError("anchor blocks must be real floating tensors")
        anchor = value.to(dtype=matrices.dtype, device=matrices.device)
        if anchor.ndim != 2 or anchor.shape[0] != 2 * rank + 1:
            raise ValueError(
                f"rank {rank} fixed space must have shape ({2 * rank + 1}, count)"
            )
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError("anchor blocks must contain only finite values")
        supplied[rank] = anchor
    supplied = dict(sorted(supplied.items()))
    ranks = tuple(supplied)
    complete = invariant_stf_anchors(matrices, ranks, atol=atol)
    tolerance = _tol(matrices.dtype, atol)
    residual = 0.0
    for rank, expected_dimension in zip(certifying_ranks, certifying_dimensions):
        if rank not in supplied:
            raise ValueError(f"missing required rank {rank} fixed space")
        if complete[rank].shape[1] != expected_dimension:
            raise ValueError(
                f"rank {rank} fixed dimension {complete[rank].shape[1]} does not "
                f"match the classified dimension {expected_dimension}"
            )
    dimensions = tuple(supplied[rank].shape[1] for rank in ranks)
    for rank, dimension in zip(ranks, dimensions):
        anchor = supplied[rank]
        expected = complete[rank]
        if expected.shape[1] != dimension:
            raise ValueError(
                f"rank {rank} numerical fixed dimension {expected.shape[1]} "
                f"does not match the classified dimension {dimension}"
            )
        projector_error = torch.linalg.matrix_norm(
            anchor @ anchor.T - expected @ expected.T
        ).item()
        if projector_error > 64.0 * tolerance:
            raise ValueError(f"rank {rank} anchors are not the complete fixed space")
        if len(matrices):
            basis = stf_basis(rank, dtype=matrices.dtype, device=matrices.device)
            for generator in matrices:
                error = torch.linalg.matrix_norm(
                    stf_representation(generator, rank, basis=basis) @ anchor - anchor
                ).item()
                residual = max(residual, error)
    if residual > 64.0 * tolerance:
        raise ValueError("generator constraint residual exceeds tolerance")
    return StabilizerCertificate(
        classification=classification,
        ranks=ranks,
        certifying_ranks=certifying_ranks,
        fixed_dimensions=dimensions,
        generator_constraint_residual=residual,
        complete_fixed_spaces=True,
        theorem=(
            "Closed subgroups of SO(3) and their fixed harmonic spaces classify "
            "the joint stabilizer; finite subgroups are C1, Cn, D2, Dn, T, O, or I."
        ),
        proof=proof,
    )
