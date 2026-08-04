r"""Encode body frame rotations with requested primitive STF anchor blocks.

This module owns the differentiable network runtime for primitive pose anchors.
Its only major public API is :class:`PoseEncoder`.  Python uses ``rank`` while
the equations use ``l`` in the integers from one through ``MAX_STF_RANK``, with
``d_l = 2 l + 1``.  Rotations act on active column vectors and map a body frame
to a common frame.  The convention is

``D_l(R1 R2) = D_l(R1) D_l(R2)``,
``B_l(R) = D_l(R) K_l_prim``,
``B_l(R h) = B_l(R)`` for ``h`` in ``H``, and
``B_l(g R) = D_l(g) B_l(R)``.

An input rotation has exact trailing shape ``(..., 3, 3)``.  Each returned
block has shape ``(..., 2 * rank + 1, multiplicity)``.  The flat encoding has
shape ``(..., encoding_dimension)`` and orders coordinates by increasing rank,
then anchor, then STF component.  The matching representation is the direct
sum of ``I_m tensor_product D_l`` blocks and has trailing shape
``(encoding_dimension, encoding_dimension)``.  Arbitrary leading batch axes,
including empty axes, are supported.  Rotation batches do not broadcast with
any anchor batch because anchors are fixed module buffers.

Compiled constants enter as detached CPU float64 artifacts.  Module conversion
may move them to supported float32 or float64 dtype and any PyTorch device.
Runtime inputs must match the buffer dtype and device, and gradients with
respect to rotations are preserved.  Public anchor and generator properties
return clones rather than live buffers.  Data dependent finite and SO(3) checks
are disabled in the ordinary hot path so autograd and ``torch.compile`` remain
usable.  Pass ``validate_runtime=True`` or call :meth:`validate_rotation` for
explicit data validation.

Offline anchor ordering is deterministic only within the limits described by
:mod:`anchor_compiler`.  Type errors report invalid module records and dtypes.
Value errors report shapes, placement, nonfinite data, invalid rotations, and
inconsistent compiled subspaces.  Runtime errors report obsolete or malformed
versioned state.  A zero width primitive output is valid and produces empty
blocks, vectors, and representations with the documented trailing shapes.

Mapped references: Taco S. Cohen, Mario Geiger, Maurice Weiler,
``A General Theory of Equivariant CNNs on Homogeneous Spaces``,
https://proceedings.neurips.cc/paper/2019/hash/b9cfe8b6042cf759dc4c0cccb27a6737-Abstract.html;
Risi Kondor, Shubhendu Trivedi,
``On the Generalization of Equivariance and Convolution in Neural Networks to
the Action of Compact Groups``, https://proceedings.mlr.press/v80/kondor18a.html.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ._compiler_utils import positive_finite_tolerance
from .anchor_compiler import (
    ANCHOR_CONVENTION_VERSION,
    AnchorCompilation,
    compile_anchors,
)
from .intertwiner_compiler import direct_sum_representation
from .stf_rep import stf_representation


__all__ = ["PoseEncoder"]


POSE_STATE_SCHEMA_VERSION = 1
POSE_FLATTEN_ORDER = "rank_anchor_stf"
POSE_CONVENTION_VERSION = (
    f"{ANCHOR_CONVENTION_VERSION}_{POSE_FLATTEN_ORDER}_pose_v1"
)


def _required_positive_tolerance(value: Any, name: str) -> float:
    """Validate one required compilation tolerance."""
    if value is None:
        raise TypeError(f"{name} must be present")
    return positive_finite_tolerance(value, name, 1.0)


def _runtime_representation(matrix: Tensor, rank: int) -> Tensor:
    """Evaluate one STF representation without data dependent validation."""
    return stf_representation(matrix, rank, validate=False)


def _validate_rotation_data(
    rotation: Tensor,
    *,
    rotation_atol: float,
    rotation_rtol: float,
    name: str,
) -> None:
    """Validate finite proper rotations outside the ordinary tensor core."""
    if not bool(torch.isfinite(rotation).all()):
        raise ValueError(f"{name} must contain only finite values")
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    gram = rotation.transpose(-1, -2) @ rotation
    if not bool(
        torch.isclose(
            gram,
            identity,
            atol=rotation_atol,
            rtol=rotation_rtol,
        ).all()
    ):
        raise ValueError(f"{name} must be orthogonal")
    determinant = torch.linalg.det(rotation)
    if not bool(
        torch.isclose(
            determinant,
            torch.ones_like(determinant),
            atol=rotation_atol,
            rtol=rotation_rtol,
        ).all()
    ):
        raise ValueError(f"{name} must contain only proper rotations")


def _validate_compilation_layout(compilation: AnchorCompilation) -> dict[int, Tensor]:
    """Validate record layout and clone requested primitive anchors."""
    if not isinstance(compilation, AnchorCompilation):
        raise TypeError("PoseEncoder requires an AnchorCompilation")
    if compilation.convention_version != ANCHOR_CONVENTION_VERSION:
        raise ValueError("anchor convention is obsolete and must be recompiled")
    generators = compilation.generators
    if not isinstance(generators, Tensor):
        raise TypeError("compiled generators must be a torch.Tensor")
    if generators.device.type != "cpu" or generators.dtype != torch.float64:
        raise ValueError("compiled generators must be a CPU float64 artifact")
    if generators.ndim != 3 or generators.shape[-2:] != (3, 3):
        raise ValueError(
            "compiled generators actual shape "
            f"{tuple(generators.shape)}; expected shape (count, 3, 3)"
        )
    if (
        isinstance(compilation.generator_count, bool)
        or not isinstance(compilation.generator_count, int)
        or compilation.generator_count != generators.shape[0]
    ):
        raise ValueError("compiled generator count does not match generators")

    scanned_ranks = compilation.scanned_ranks
    scanned_blocks = compilation.scanned_blocks
    expected_scanned = (
        tuple(range(1, max(compilation.requested_output_ranks) + 1))
        if compilation.requested_output_ranks
        else ()
    )
    if scanned_ranks != expected_scanned:
        raise ValueError(
            f"scanned ranks actual {scanned_ranks}; expected {expected_scanned}"
        )
    block_ranks = tuple(block.rank for block in scanned_blocks)
    if block_ranks != scanned_ranks:
        raise ValueError(
            f"scanned block ranks actual {block_ranks}; expected {scanned_ranks}"
        )
    requested = compilation.requested_output_ranks
    if requested != tuple(sorted(set(requested))) or any(
        rank not in scanned_ranks for rank in requested
    ):
        raise ValueError("requested output rank layout is inconsistent")

    block_map = {block.rank: block for block in scanned_blocks}
    for block in scanned_blocks:
        rank = block.rank
        expected_rows = 2 * rank + 1
        values = {
            "fixed": block.fixed_basis,
            "generated": block.generated_basis,
            "primitive": block.primitive_basis,
        }
        for label, value in values.items():
            expected_shape = (
                expected_rows,
                getattr(block.dimensions, label),
            )
            if not isinstance(value, Tensor):
                raise TypeError(f"rank {rank} {label} basis must be a torch.Tensor")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"rank {rank} {label} basis actual shape {tuple(value.shape)}; "
                    f"expected shape {expected_shape}"
                )
            if value.device.type != "cpu" or value.dtype != torch.float64:
                raise ValueError(
                    f"rank {rank} {label} basis must be a CPU float64 artifact"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"rank {rank} {label} basis must be finite")
    expected_output = tuple(
        rank
        for rank in requested
        if block_map[rank].primitive_basis.shape[1] > 0
    )
    if compilation.output_ranks != expected_output:
        raise ValueError(
            f"output ranks actual {compilation.output_ranks}; expected {expected_output}"
        )
    anchors = {
        rank: block_map[rank].primitive_basis.detach().clone()
        for rank in compilation.output_ranks
    }
    return anchors


def _validate_generator_anchors(
    generators: Tensor,
    anchors: Mapping[int, Tensor],
    *,
    rotation_atol: float,
    rotation_rtol: float,
) -> None:
    """Validate generators and their invariant primitive anchor blocks."""
    if not isinstance(generators, Tensor):
        raise TypeError("generators must be a torch.Tensor")
    if generators.ndim != 3 or generators.shape[-2:] != (3, 3):
        raise ValueError(
            "generators actual shape "
            f"{tuple(generators.shape)}; expected shape (count, 3, 3)"
        )
    if generators.dtype not in (torch.float32, torch.float64):
        raise TypeError("generators must use float32 or float64")
    epsilon = torch.finfo(generators.dtype).eps
    validation_atol = max(1e-10, min(rotation_atol, 1e-5), 512.0 * epsilon)
    validation_rtol = max(1e-10, min(rotation_rtol, 1e-5), 512.0 * epsilon)
    _validate_rotation_data(
        generators,
        rotation_atol=validation_atol,
        rotation_rtol=validation_rtol,
        name="generators",
    )

    for rank, anchor in anchors.items():
        if not isinstance(anchor, Tensor):
            raise TypeError("every anchor block must be a torch.Tensor")
        expected_shape = (2 * rank + 1, anchor.shape[1] if anchor.ndim == 2 else 0)
        if anchor.ndim != 2 or anchor.shape[0] != 2 * rank + 1:
            raise ValueError(
                f"rank {rank} anchor actual shape {tuple(anchor.shape)}; "
                f"expected shape {expected_shape}"
            )
        if anchor.shape[1] == 0:
            raise ValueError(f"rank {rank} output anchor block must be nonempty")
        if anchor.dtype != generators.dtype:
            raise TypeError("anchors and generators must use the same dtype")
        if anchor.device != generators.device:
            raise ValueError("anchors and generators must use the same device")
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError(f"rank {rank} anchors must contain only finite values")

        identity = torch.eye(anchor.shape[1], dtype=anchor.dtype, device=anchor.device)
        gram_residual = torch.linalg.matrix_norm(anchor.mT @ anchor - identity)
        gram_scale = float(torch.linalg.matrix_norm(identity))
        gram_threshold = validation_atol + validation_rtol * gram_scale
        if float(gram_residual) > gram_threshold:
            raise ValueError(
                f"rank {rank} anchors violate orthonormality; "
                f"residual {float(gram_residual):.6e}; "
                f"threshold {gram_threshold:.6e}; "
                f"actual shape {tuple(anchor.shape)}; expected shape {expected_shape}"
            )

        represented = _runtime_representation(generators, rank)
        residual = torch.linalg.vector_norm(represented @ anchor - anchor, dim=-2)
        anchor_scale = float(torch.linalg.matrix_norm(anchor))
        threshold = validation_atol + validation_rtol * anchor_scale
        maximum = float(residual.amax()) if residual.numel() else 0.0
        if maximum > threshold:
            raise ValueError(
                f"rank {rank} anchors violate generator invariance; "
                f"residual {maximum:.6e}; threshold {threshold:.6e}; "
                f"actual shape {tuple(anchor.shape)}; expected shape {expected_shape}"
            )


def _validate_compilation_subspaces(
    compilation: AnchorCompilation,
    nullspace_atol: float,
    nullspace_rtol: float,
    rotation_atol: float,
    rotation_rtol: float,
) -> None:
    """Recompile and validate every canonical scanned anchor subspace."""
    canonical = compile_anchors(
        compilation.generators,
        output_ranks=compilation.requested_output_ranks,
        nullspace_atol=nullspace_atol,
        nullspace_rtol=nullspace_rtol,
        rotation_atol=rotation_atol,
        rotation_rtol=rotation_rtol,
    )
    if canonical.scanned_ranks != compilation.scanned_ranks:
        raise ValueError(
            f"scanned ranks actual {compilation.scanned_ranks}; "
            f"expected {canonical.scanned_ranks}"
        )
    if canonical.output_ranks != compilation.output_ranks:
        raise ValueError(
            f"output ranks actual {compilation.output_ranks}; "
            f"expected {canonical.output_ranks}"
        )
    for block, expected in zip(
        compilation.scanned_blocks,
        canonical.scanned_blocks,
    ):
        rank = block.rank
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
            raise ValueError(
                f"rank {rank} dimensions actual {actual_dimensions}; "
                f"expected {expected_dimensions}"
            )
        values = (
            ("fixed", block.fixed_basis, expected.fixed_basis),
            ("generated", block.generated_basis, expected.generated_basis),
            ("primitive", block.primitive_basis, expected.primitive_basis),
        )
        for label, value, expected_value in values:
            if value.shape != expected_value.shape or not bool(
                torch.allclose(value, expected_value, atol=2e-11, rtol=2e-11)
            ):
                raise ValueError(
                    f"rank {rank} {label} subspace is not canonical; "
                    f"actual shape {tuple(value.shape)}; "
                    f"expected shape {tuple(expected_value.shape)}"
                )


def _require_matching_state_tensor(
    incoming: Any,
    reference: Tensor,
    name: str,
) -> None:
    """Require state constants to match the current canonical module."""
    actual_shape = tuple(incoming.shape) if isinstance(incoming, Tensor) else ()
    expected_shape = tuple(reference.shape)
    if not isinstance(incoming, Tensor) or actual_shape != expected_shape:
        raise ValueError(
            f"PoseEncoder {name} actual shape {actual_shape}; "
            f"expected shape {expected_shape}"
        )
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
    """Encode rotations with requested primitive STF anchor buffers."""

    def __init__(
        self,
        compilation: AnchorCompilation,
        *,
        validate_runtime: bool = False,
    ) -> None:
        """Register one verified compilation for differentiable runtime use."""
        super().__init__()
        if not isinstance(validate_runtime, bool):
            raise TypeError("validate_runtime must be bool")
        anchors = _validate_compilation_layout(compilation)
        generators = compilation.generators.detach().clone()
        self._ranks = compilation.output_ranks
        self._anchor_names = tuple(
            (rank, f"_anchor_rank_{rank}") for rank in self._ranks
        )
        self._multiplicity_items = tuple(
            (rank, anchors[rank].shape[1]) for rank in self._ranks
        )
        self._encoding_dimension = sum(
            (2 * rank + 1) * multiplicity
            for rank, multiplicity in self._multiplicity_items
        )
        self._validate_runtime = validate_runtime
        self._nullspace_atol = _required_positive_tolerance(
            compilation.nullspace_atol,
            "nullspace_atol",
        )
        self._nullspace_rtol = _required_positive_tolerance(
            compilation.nullspace_rtol,
            "nullspace_rtol",
        )
        self._rotation_atol = _required_positive_tolerance(
            compilation.rotation_atol,
            "rotation_atol",
        )
        self._rotation_rtol = _required_positive_tolerance(
            compilation.rotation_rtol,
            "rotation_rtol",
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
        self.register_buffer("_generators", generators)
        for rank, name in self._anchor_names:
            self.register_buffer(name, anchors[rank])
        self._validate_registered_anchors()

    def _anchor_items(self) -> tuple[tuple[int, Tensor], ...]:
        """Return live private anchor buffers for internal operations."""
        return tuple((rank, getattr(self, name)) for rank, name in self._anchor_names)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return requested nonempty ranks in flattened block order."""
        return self._ranks

    @property
    def anchors(self) -> dict[int, Tensor]:
        """Return detached clones of primitive anchor buffers."""
        return {
            rank: anchor.detach().clone()
            for rank, anchor in self._anchor_items()
        }

    @property
    def generators(self) -> Tensor:
        """Return a detached clone of the compiled group generators."""
        return self._generators.detach().clone()

    @property
    def multiplicities(self) -> dict[int, int]:
        """Return the primitive anchor multiplicity at each output rank."""
        return dict(self._multiplicity_items)

    @property
    def encoding_dimension(self) -> int:
        """Return the dimension of the flattened primitive pose space."""
        return self._encoding_dimension

    def _runtime_rotation_tolerances(self) -> tuple[float, float]:
        """Return rotation tolerances compatible with the buffer dtype."""
        epsilon = torch.finfo(self._generators.dtype).eps
        return (
            max(self._rotation_atol, 128.0 * epsilon),
            max(self._rotation_rtol, 128.0 * epsilon),
        )

    def _validate_runtime_layout(self, rotation: Tensor) -> None:
        """Require runtime matrix shape and registered buffer placement."""
        if not isinstance(rotation, Tensor):
            raise TypeError("rotation must be a torch.Tensor")
        if rotation.shape[-2:] != (3, 3):
            actual_shape = tuple(rotation.shape)
            raise ValueError(
                f"rotation actual shape {actual_shape}; expected trailing shape (3, 3)"
            )
        if rotation.dtype != self._generators.dtype:
            raise TypeError("rotation dtype must match PoseEncoder buffers")
        if rotation.device != self._generators.device:
            raise ValueError("rotation device must match PoseEncoder buffers")

    def validate_rotation(self, rotation: Tensor) -> None:
        """Explicitly validate runtime data as finite proper rotations."""
        self._validate_runtime_layout(rotation)
        rotation_atol, rotation_rtol = self._runtime_rotation_tolerances()
        _validate_rotation_data(
            rotation,
            rotation_atol=rotation_atol,
            rotation_rtol=rotation_rtol,
            name="rotation",
        )

    def gauge_residuals(self) -> dict[int, Tensor]:
        """Return maximum generator invariance residual for each output rank."""
        residuals: dict[int, Tensor] = {}
        for rank, anchor in self._anchor_items():
            if self._generators.shape[0] == 0:
                residuals[rank] = anchor.new_zeros(())
                continue
            represented = _runtime_representation(self._generators, rank)
            residuals[rank] = torch.linalg.vector_norm(
                represented @ anchor - anchor,
                dim=-2,
            ).amax()
        return residuals

    def _validate_registered_anchors(self) -> None:
        """Reject malformed or noninvariant private anchor buffers."""
        _validate_generator_anchors(
            self._generators,
            dict(self._anchor_items()),
            rotation_atol=self._rotation_atol,
            rotation_rtol=self._rotation_rtol,
        )

    def encode_blocks(self, rotation: Tensor) -> dict[int, Tensor]:
        """Return pose blocks with STF and anchor trailing axes."""
        self._validate_runtime_layout(rotation)
        if self._validate_runtime:
            self.validate_rotation(rotation)
        return {
            rank: _runtime_representation(rotation, rank) @ anchor
            for rank, anchor in self._anchor_items()
        }

    def encode(self, rotation: Tensor) -> Tensor:
        """Return increasing rank and anchor major flattened coordinates."""
        blocks = self.encode_blocks(rotation)
        batch_shape = rotation.shape[:-2]
        if not blocks:
            return rotation[..., 0, :0]
        flattened = [
            blocks[rank].transpose(-2, -1).reshape(
                batch_shape + (blocks[rank].shape[-2] * blocks[rank].shape[-1],)
            )
            for rank in self._ranks
        ]
        return torch.cat(flattened, dim=-1)

    def representation(self, group_action: Tensor) -> Tensor:
        """Return the representation matching the documented flat order."""
        self._validate_runtime_layout(group_action)
        if self._validate_runtime:
            self.validate_rotation(group_action)
        if not self._ranks:
            return group_action[..., :0, :0]
        blocks: list[Tensor] = []
        multiplicities = dict(self._multiplicity_items)
        for rank in self._ranks:
            represented = _runtime_representation(group_action, rank)
            blocks.extend(represented for _ in range(multiplicities[rank]))
        return direct_sum_representation(*blocks)

    def forward(self, rotation: Tensor) -> Tensor:
        """Encode rotations using registered primitive anchors."""
        return self.encode(rotation)

    def get_extra_state(self) -> dict[str, Any]:
        """Store complete mathematical and layout convention metadata."""
        return {
            "state_schema_version": POSE_STATE_SCHEMA_VERSION,
            "convention_version": POSE_CONVENTION_VERSION,
            "flatten_order": POSE_FLATTEN_ORDER,
            "output_ranks": self.ranks,
            "multiplicities": self._multiplicity_items,
        }

    def set_extra_state(self, state: Any) -> None:
        """Reject state created under an incompatible convention."""
        if not isinstance(state, Mapping):
            raise RuntimeError("PoseEncoder state lacks convention metadata")
        if state.get("state_schema_version") != POSE_STATE_SCHEMA_VERSION:
            raise RuntimeError("PoseEncoder state schema is obsolete")
        if state.get("convention_version") != POSE_CONVENTION_VERSION:
            raise RuntimeError("PoseEncoder state uses an obsolete convention")
        if state.get("flatten_order") != POSE_FLATTEN_ORDER:
            raise RuntimeError("PoseEncoder state flatten order does not match")
        if tuple(state.get("output_ranks", ())) != self.ranks:
            raise RuntimeError("PoseEncoder state output rank layout does not match")
        incoming_multiplicities = tuple(state.get("multiplicities", ()))
        if incoming_multiplicities != self._multiplicity_items:
            raise RuntimeError("PoseEncoder state multiplicities do not match")

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
        """Validate convention and constants during direct or nested loads."""
        extra_key = f"{prefix}_extra_state"
        if extra_key not in state_dict:
            raise RuntimeError("PoseEncoder state is unversioned and must be recompiled")
        self.set_extra_state(state_dict[extra_key])
        generator_key = f"{prefix}_generators"
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
            state_dict[generator_key],
            self._generators,
            "generators",
        )
        internal_anchors = dict(self._anchor_items())
        for rank, anchor in incoming_anchors.items():
            _require_matching_state_tensor(
                anchor,
                internal_anchors[rank],
                f"rank {rank} anchors",
            )
        names = ("_generators", *(name for _, name in self._anchor_names))
        previous = {name: getattr(self, name).detach().clone() for name in names}
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
