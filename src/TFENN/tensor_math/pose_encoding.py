"""Differentiable STF pose encoding with verified group provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ._compiler_utils import positive_finite_tolerance
from .anchor_compiler import AnchorCompilation, compile_anchors
from .intertwiner_compiler import direct_sum_representation
from .stf_rep import stf_representation
from .stf_space import STF_BASIS_VERSION


__all__ = ["PoseEncoder"]


def _required_positive_tolerance(value: Any, name: str) -> float:
    """Validate one required compilation tolerance."""
    if value is None:
        raise TypeError(f"{name} must be present")
    return positive_finite_tolerance(value, name, 1.0)


def _validate_compilation(compilation: AnchorCompilation) -> dict[int, Tensor]:
    """Return finite primitive anchors from current canonical compilation."""
    if not isinstance(compilation, AnchorCompilation):
        raise TypeError("PoseEncoder requires an AnchorCompilation")
    if compilation.basis_version != STF_BASIS_VERSION:
        raise ValueError("anchor basis version is obsolete and must be recompiled")
    anchors = compilation.primitive_anchors
    if not anchors:
        raise ValueError("compilation must contain at least one primitive anchor")
    ordered: dict[int, Tensor] = {}
    for rank, anchor in sorted(anchors.items()):
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("anchor ranks must be nonnegative integers")
        if not isinstance(anchor, Tensor):
            raise TypeError("every anchor block must be a torch.Tensor")
        if anchor.ndim != 2 or anchor.shape[0] != 2 * rank + 1:
            raise ValueError(
                f"rank {rank} anchors must have shape ({2 * rank + 1}, count)"
            )
        if anchor.shape[1] == 0:
            raise ValueError("every anchor block must contain at least one anchor")
        if anchor.dtype not in (torch.float32, torch.float64):
            raise TypeError("anchor matrices must use float32 or float64")
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError("anchor matrices must contain only finite values")
        ordered[rank] = anchor.detach().clone()
    return ordered


def _validate_generator_anchors(
    generators: Tensor,
    anchors: Mapping[int, Tensor],
    *,
    rotation_atol: float,
    rotation_rtol: float,
) -> None:
    """Validate generators and their invariant anchor blocks."""
    if not isinstance(generators, Tensor):
        raise TypeError("generators must be a torch.Tensor")
    if generators.ndim != 3 or generators.shape[-2:] != (3, 3):
        raise ValueError("generators must have shape (count, 3, 3)")
    if generators.dtype not in (torch.float32, torch.float64):
        raise TypeError("generators must use float32 or float64")
    if not bool(torch.isfinite(generators).all()):
        raise ValueError("generators must contain only finite values")

    epsilon = torch.finfo(generators.dtype).eps
    validation_atol = max(1e-10, min(rotation_atol, 1e-5), 512.0 * epsilon)
    validation_rtol = max(1e-10, min(rotation_rtol, 1e-5), 512.0 * epsilon)
    for rank, anchor in anchors.items():
        if not isinstance(anchor, Tensor):
            raise TypeError("every anchor block must be a torch.Tensor")
        if anchor.ndim != 2 or anchor.shape[0] != 2 * rank + 1:
            raise ValueError("anchor shape does not match its rank")
        if anchor.shape[1] == 0:
            raise ValueError("every anchor block must contain at least one anchor")
        if anchor.dtype != generators.dtype:
            raise TypeError("anchors and generators must use the same dtype")
        if anchor.device != generators.device:
            raise ValueError("anchors and generators must use the same device")
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError("anchors must contain only finite values")

        identity = torch.eye(anchor.shape[1], dtype=anchor.dtype, device=anchor.device)
        gram_residual = torch.linalg.matrix_norm(anchor.mT @ anchor - identity)
        gram_scale = float(torch.linalg.matrix_norm(identity))
        gram_threshold = validation_atol + validation_rtol * gram_scale
        if float(gram_residual) > gram_threshold:
            raise ValueError("anchor columns must be orthonormal")

        represented = stf_representation(
            generators,
            rank,
            rotation_atol=max(1e-10, min(rotation_atol, 1e-5), 128.0 * epsilon),
            rotation_rtol=max(1e-10, min(rotation_rtol, 1e-5), 128.0 * epsilon),
        )
        residual = torch.linalg.vector_norm(represented @ anchor - anchor, dim=-2)
        anchor_scale = float(torch.linalg.matrix_norm(anchor))
        threshold = validation_atol + validation_rtol * anchor_scale
        if residual.numel() and float(residual.amax()) > threshold:
            raise ValueError(
                f"rank {rank} anchors violate generator invariance and must be recompiled"
            )


def _validate_compilation_subspaces(
    compilation: AnchorCompilation,
    nullspace_atol: float,
    nullspace_rtol: float,
    rotation_atol: float,
    rotation_rtol: float,
) -> None:
    """Recompile and validate every canonical anchor subspace."""
    ranks = tuple(compilation.blocks)
    if not ranks or ranks != tuple(range(1, max(ranks) + 1)):
        raise ValueError("compiled ranks must be consecutive from rank one")
    canonical = compile_anchors(
        compilation.generators,
        ranks=compilation.requested_ranks,
        nullspace_atol=nullspace_atol,
        nullspace_rtol=nullspace_rtol,
        rotation_atol=rotation_atol,
        rotation_rtol=rotation_rtol,
    )
    if tuple(canonical.blocks) != ranks:
        raise ValueError("compiled rank layout is inconsistent")
    for rank in ranks:
        block = compilation.blocks[rank]
        expected = canonical.blocks[rank]
        expected_rows = 2 * rank + 1
        values = (block.fixed_basis, block.generated_basis, block.primitive_basis)
        expected_values = (
            expected.fixed_basis,
            expected.generated_basis,
            expected.primitive_basis,
        )
        for value in values:
            if not isinstance(value, Tensor) or value.ndim != 2:
                raise ValueError("compiled subspaces must be matrices")
            if value.shape[0] != expected_rows:
                raise ValueError("compiled subspace shape does not match rank")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("compiled subspaces must contain only finite values")
        actual_dimensions = (
            block.dimensions.fixed,
            block.dimensions.generated,
            block.dimensions.primitive,
        )
        expected_dimensions = (
            expected.dimensions.fixed,
            expected.dimensions.generated,
            expected.dimensions.primitive,
        )
        if actual_dimensions != expected_dimensions:
            raise ValueError("compiled subspace dimensions are inconsistent")
        for value, expected_value in zip(values, expected_values):
            if value.shape != expected_value.shape or not bool(
                torch.allclose(
                    value.to(device="cpu", dtype=torch.float64),
                    expected_value,
                    atol=2e-11,
                    rtol=2e-11,
                )
            ):
                raise ValueError("compiled anchor subspace is not canonical")


def _require_matching_state_tensor(
    incoming: Any,
    reference: Tensor,
    name: str,
) -> None:
    """Require state constants to match the current canonical module."""
    if not isinstance(incoming, Tensor) or incoming.shape != reference.shape:
        raise ValueError(f"PoseEncoder {name} shape does not match")
    if incoming.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"PoseEncoder {name} must use float32 or float64")
    if not bool(torch.isfinite(incoming).all()):
        raise ValueError(f"PoseEncoder {name} must contain only finite values")
    epsilon = max(torch.finfo(incoming.dtype).eps, torch.finfo(reference.dtype).eps)
    tolerance = 128.0 * epsilon
    if not bool(
        torch.allclose(
            incoming.detach().to(device="cpu", dtype=torch.float64),
            reference.detach().to(device="cpu", dtype=torch.float64),
            atol=tolerance,
            rtol=tolerance,
        )
    ):
        raise ValueError(f"PoseEncoder {name} does not match canonical compilation")


class PoseEncoder(nn.Module):
    """Encode rotations using verified invariant STF anchor buffers."""

    def __init__(self, compilation: AnchorCompilation) -> None:
        """Register compiled anchors and verify their generator invariance."""
        super().__init__()
        anchors = _validate_compilation(compilation)
        generators = compilation.generators.detach().clone()
        if (
            isinstance(compilation.generator_count, bool)
            or not isinstance(compilation.generator_count, int)
            or compilation.generator_count != generators.shape[0]
        ):
            raise ValueError("compiled generator count does not match generators")

        self._ranks = tuple(anchors)
        self._anchor_names = tuple(
            (rank, f"anchor_rank_{rank}") for rank in self._ranks
        )
        self._nullspace_atol = _required_positive_tolerance(
            compilation.nullspace_atol, "nullspace_atol"
        )
        self._nullspace_rtol = _required_positive_tolerance(
            compilation.nullspace_rtol, "nullspace_rtol"
        )
        self._rotation_atol = _required_positive_tolerance(
            compilation.rotation_atol, "rotation_atol"
        )
        self._rotation_rtol = _required_positive_tolerance(
            compilation.rotation_rtol, "rotation_rtol"
        )
        _validate_generator_anchors(
            generators,
            anchors,
            rotation_atol=self._rotation_atol,
            rotation_rtol=self._rotation_rtol,
        )
        _validate_compilation_subspaces(
            compilation,
            self._nullspace_atol,
            self._nullspace_rtol,
            self._rotation_atol,
            self._rotation_rtol,
        )
        self.register_buffer("generators", generators)
        for rank, name in self._anchor_names:
            self.register_buffer(name, anchors[rank])
        self._validate_registered_anchors()

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return STF ranks in flattened block order."""
        return self._ranks

    @property
    def anchors(self) -> dict[int, Tensor]:
        """Return registered anchor buffers indexed by rank."""
        return {rank: getattr(self, name) for rank, name in self._anchor_names}

    @property
    def multiplicities(self) -> dict[int, int]:
        """Return the number of primitive anchors at each rank."""
        return {rank: anchor.shape[1] for rank, anchor in self.anchors.items()}

    @property
    def encoding_dimension(self) -> int:
        """Return the dimension of the flattened B space."""
        return sum(anchor.numel() for anchor in self.anchors.values())

    def _runtime_rotation_tolerances(self) -> tuple[float, float]:
        """Return rotation tolerances compatible with the buffer dtype."""
        epsilon = torch.finfo(self.generators.dtype).eps
        return (
            max(self._rotation_atol, 128.0 * epsilon),
            max(self._rotation_rtol, 128.0 * epsilon),
        )

    def gauge_residuals(self) -> dict[int, Tensor]:
        """Return maximum generator invariance residual for each anchor rank."""
        rotation_atol, rotation_rtol = self._runtime_rotation_tolerances()
        residuals: dict[int, Tensor] = {}
        for rank, anchor in self.anchors.items():
            if self.generators.shape[0] == 0:
                residuals[rank] = anchor.new_zeros(())
                continue
            represented = stf_representation(
                self.generators,
                rank,
                rotation_atol=rotation_atol,
                rotation_rtol=rotation_rtol,
            )
            residuals[rank] = torch.linalg.vector_norm(
                represented @ anchor - anchor, dim=-2
            ).amax()
        return residuals

    def _validate_registered_anchors(self) -> None:
        """Reject malformed, nonfinite, or noninvariant registered anchors."""
        _validate_generator_anchors(
            self.generators,
            self.anchors,
            rotation_atol=self._rotation_atol,
            rotation_rtol=self._rotation_rtol,
        )

    def _validate_runtime_rotation(self, rotation: Tensor) -> None:
        """Require runtime rotations to match registered buffer placement."""
        if not isinstance(rotation, Tensor) or rotation.shape[-2:] != (3, 3):
            raise ValueError("rotation must have trailing shape (3, 3)")
        if rotation.dtype != self.generators.dtype:
            raise TypeError("rotation dtype must match PoseEncoder buffers")
        if rotation.device != self.generators.device:
            raise ValueError("rotation device must match PoseEncoder buffers")

    def encode_blocks(self, rotation: Tensor) -> dict[int, Tensor]:
        """Return verified pose blocks with STF and anchor trailing axes."""
        self._validate_runtime_rotation(rotation)
        rotation_atol, rotation_rtol = self._runtime_rotation_tolerances()
        return {
            rank: stf_representation(
                rotation,
                rank,
                rotation_atol=rotation_atol,
                rotation_rtol=rotation_rtol,
            )
            @ anchor
            for rank, anchor in self.anchors.items()
        }

    def encode(self, rotation: Tensor) -> Tensor:
        """Return anchor major flattened B coordinates for each rotation."""
        blocks = self.encode_blocks(rotation)
        batch_shape = rotation.shape[:-2]
        flattened = [
            block.transpose(-2, -1).reshape(
                batch_shape + (block.shape[-2] * block.shape[-1],)
            )
            for block in blocks.values()
        ]
        return torch.cat(flattened, dim=-1)

    def representation(self, group_action: Tensor) -> Tensor:
        """Return the B representation matching anchor major flattening."""
        self._validate_runtime_rotation(group_action)
        rotation_atol, rotation_rtol = self._runtime_rotation_tolerances()
        blocks = []
        for rank, anchor in self.anchors.items():
            representation = stf_representation(
                group_action,
                rank,
                rotation_atol=rotation_atol,
                rotation_rtol=rotation_rtol,
            )
            blocks.extend(representation for _ in range(anchor.shape[1]))
        return direct_sum_representation(*blocks)

    def forward(self, rotation: Tensor) -> Tensor:
        """Encode rotations using registered primitive anchors."""
        return self.encode(rotation)

    def get_extra_state(self) -> dict[str, Any]:
        """Store coordinate convention metadata in the module state."""
        return {
            "basis_version": STF_BASIS_VERSION,
            "ranks": self.ranks,
        }

    def set_extra_state(self, state: Any) -> None:
        """Reject state created under a different coordinate convention."""
        if not isinstance(state, Mapping):
            raise RuntimeError("PoseEncoder state lacks coordinate metadata")
        if state.get("basis_version") != STF_BASIS_VERSION:
            raise RuntimeError("PoseEncoder state uses an obsolete STF basis")
        if tuple(state.get("ranks", ())) != self.ranks:
            raise RuntimeError("PoseEncoder state rank layout does not match")

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Validate versioned state during direct and nested module loads."""
        extra_key = f"{prefix}_extra_state"
        if extra_key not in state_dict:
            raise RuntimeError(
                "PoseEncoder state is unversioned and must be recompiled"
            )
        self.set_extra_state(state_dict[extra_key])
        generator_key = f"{prefix}generators"
        anchor_keys = {rank: f"{prefix}{name}" for rank, name in self._anchor_names}
        required = (generator_key, *anchor_keys.values())
        if any(key not in state_dict for key in required):
            raise RuntimeError("PoseEncoder state is incomplete and must be recompiled")
        incoming_anchors = {rank: state_dict[key] for rank, key in anchor_keys.items()}
        _validate_generator_anchors(
            state_dict[generator_key],
            incoming_anchors,
            rotation_atol=self._rotation_atol,
            rotation_rtol=self._rotation_rtol,
        )
        _require_matching_state_tensor(
            state_dict[generator_key], self.generators, "generators"
        )
        for rank, anchor in incoming_anchors.items():
            _require_matching_state_tensor(
                anchor,
                self.anchors[rank],
                f"rank {rank} anchors",
            )
        previous = {
            name: getattr(self, name).detach().clone()
            for name in ("generators", *(name for _, name in self._anchor_names))
        }
        try:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            self._validate_registered_anchors()
        except Exception:
            for name, value in previous.items():
                setattr(self, name, value)
            raise
