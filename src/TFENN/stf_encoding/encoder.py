"""STF component encoders for finite rotation quotients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .basis import stf_representation
from .d6 import d6_analytic_anchors, d6_generators, d6_group_elements
from .verification import (
    InverseFiberResult,
    encoding_jacobian,
    jacobian_pseudoinverse,
    numerical_inverse_fiber,
    verify_gradients,
)


@dataclass(frozen=True)
class AnalyticStabilizerCertificate:
    """Record a theorem backed global stabilizer equality."""

    exact: bool
    group_name: str
    group_order: int
    ranks: tuple[int, ...]
    certifying_ranks: tuple[int, ...]
    encoding_dimension: int
    generator_constraint_residual: float
    proof: str


class STFEncoder:
    """Encode rotations by the orbit of fixed STF anchor columns."""

    def __init__(
        self,
        generators: torch.Tensor,
        group_elements: torch.Tensor,
        anchors: Mapping[int, torch.Tensor],
        certificate: object,
    ) -> None:
        """Store validated generators, finite orbit elements, and anchors."""
        if generators.ndim != 3 or generators.shape[-2:] != (3, 3):
            raise ValueError("generators must have shape (count, 3, 3)")
        if group_elements.ndim != 3 or group_elements.shape[-2:] != (3, 3):
            raise ValueError("group_elements must have shape (order, 3, 3)")
        if not anchors:
            raise ValueError("at least one STF anchor block is required")
        ordered: dict[int, torch.Tensor] = {}
        for rank, anchor in sorted(anchors.items()):
            if anchor.ndim == 1:
                anchor = anchor[:, None]
            expected_rows = 2 * rank + 1
            if anchor.ndim != 2 or anchor.shape[0] != expected_rows:
                raise ValueError(
                    f"rank {rank} anchors must have shape ({expected_rows}, count)"
                )
            ordered[rank] = anchor.clone()
        self.generators = generators.clone()
        self.group_elements = group_elements.clone()
        self.anchors = ordered
        self.certificate = certificate

    @classmethod
    def from_generators(
        cls,
        generators: torch.Tensor | Sequence[torch.Tensor],
        *,
        ranks: Sequence[int] | None = None,
        tolerance: float | None = None,
        max_group_order: int = 512,
    ) -> STFEncoder:
        """Build a globally certified encoder from finite SO(3) generators."""
        from .group import (
            classify_finite_so3_group,
            enumerate_finite_group,
            invariant_stf_anchors,
            stabilizer_certificate,
            validate_so3_generators,
        )

        checked = validate_so3_generators(generators, atol=tolerance)
        elements = enumerate_finite_group(
            checked, atol=tolerance, max_order=max_group_order
        )
        classification = classify_finite_so3_group(elements, atol=tolerance)
        selected = tuple(ranks) if ranks is not None else classification.ranks
        anchors = invariant_stf_anchors(checked, selected, atol=tolerance)
        certificate = stabilizer_certificate(
            checked,
            anchors,
            atol=tolerance,
            max_order=max_group_order,
        )
        return cls(checked, elements, anchors, certificate)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return encoded STF ranks in block order."""
        return tuple(self.anchors)

    @property
    def encoding_dimension(self) -> int:
        """Return the flattened rotation encoding dimension."""
        return sum(anchor.numel() for anchor in self.anchors.values())

    @property
    def encoding_dim(self) -> int:
        """Return a short alias for the encoding dimension."""
        return self.encoding_dimension

    def encode_blocks(self, rotation: torch.Tensor) -> dict[int, torch.Tensor]:
        """Return transformed STF anchors grouped by rank."""
        if rotation.shape[-2:] != (3, 3):
            raise ValueError("rotation must have trailing shape (3, 3)")
        blocks = {}
        for rank, anchor in self.anchors.items():
            fixed = anchor.to(dtype=rotation.dtype, device=rotation.device)
            blocks[rank] = stf_representation(rotation, rank) @ fixed
        return blocks

    def encode_rotation(self, rotation: torch.Tensor) -> torch.Tensor:
        """Return flattened STF components for one rotation or a batch."""
        batch_shape = rotation.shape[:-2]
        blocks = tuple(
            block.reshape(batch_shape + (block.shape[-2] * block.shape[-1],))
            for block in self.encode_blocks(rotation).values()
        )
        return torch.cat(blocks, dim=-1)

    def encode(self, position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """Concatenate a covariant position with the rotation encoding."""
        if position.shape[-1:] != (3,):
            raise ValueError("position must have trailing shape (3,)")
        code = self.encode_rotation(rotation)
        if position.shape[:-1] != code.shape[:-1]:
            raise ValueError("position and rotation batch shapes must match")
        return torch.cat((position, code), dim=-1)

    def generator_representations(self) -> dict[int, torch.Tensor]:
        """Return every input generator in each STF component basis."""
        return {
            rank: stf_representation(self.generators, rank) for rank in self.ranks
        }

    def inverse_fiber(
        self,
        code: torch.Tensor,
        *,
        reference_rotation: torch.Tensor | None = None,
        num_starts: int = 32,
        adam_steps: int = 180,
        learning_rate: float = 0.08,
        seed: int = 0,
    ) -> InverseFiberResult:
        """Numerically recover one representative and its complete right fiber."""
        if not bool(getattr(self.certificate, "exact", False)):
            raise ValueError("inverse fiber requires an exact global certificate")
        return numerical_inverse_fiber(
            self,
            code,
            reference_rotation=reference_rotation,
            num_starts=num_starts,
            adam_steps=adam_steps,
            learning_rate=learning_rate,
            seed=seed,
        )

    def jacobian(self, rotation: torch.Tensor) -> torch.Tensor:
        """Return the three dimensional left tangent Jacobian."""
        return encoding_jacobian(self, rotation)

    def jacobian_pseudoinverse(self, rotation: torch.Tensor) -> torch.Tensor:
        """Return the local tangent Moore Penrose inverse."""
        return jacobian_pseudoinverse(self, rotation)

    def verify_gradients(self, rotation: torch.Tensor) -> bool:
        """Verify PyTorch gradients on the SO(3) tangent space."""
        return verify_gradients(self, rotation)


def d6_benzene_encoder(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> STFEncoder:
    """Build the analytic eighteen component proper D6 benzene encoder."""
    generators = d6_generators(dtype=dtype, device=device)
    anchors = d6_analytic_anchors(dtype=dtype, device=device)
    residual = max(
        float(
            torch.linalg.vector_norm(
                stf_representation(generator, rank) @ anchor - anchor
            )
        )
        for rank, anchor in anchors.items()
        for generator in generators
    )
    proof = (
        "K2 fixes the unoriented hexagon normal. The nonzero cos(6 theta) "
        "component of K6 reduces its SO(3) stabilizer exactly to proper D6."
    )
    certificate = AnalyticStabilizerCertificate(
        exact=True,
        group_name="D6_tilde",
        group_order=12,
        ranks=(2, 6),
        certifying_ranks=(2, 6),
        encoding_dimension=18,
        generator_constraint_residual=residual,
        proof=proof,
    )
    return STFEncoder(
        generators,
        d6_group_elements(dtype=dtype, device=device),
        anchors,
        certificate,
    )


__all__ = [
    "AnalyticStabilizerCertificate",
    "STFEncoder",
    "d6_benzene_encoder",
]
