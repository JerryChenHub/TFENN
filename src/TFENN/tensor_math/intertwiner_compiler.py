"""Compile and compose real representation intertwiners.

This file owns the public direct sum and tensor product representation
constructors together with the offline linear and bilinear intertwiner
compilers.  The major public APIs are ``direct_sum_representation``,
``tensor_product_representation``, ``compile_intertwiners``, and
``compile_bilinear_intertwiners``.  ``IntertwinerCompilation`` and
``BilinearIntertwinerCompilation`` are their stable compilation records.

All actions use active column vectors and
``rho(h1 h2) = rho(h1) rho(h2)``.  A linear basis element obeys

``M rho_in(h) = rho_out(h) M``.

The compiler uses column major vectorization of ``M``.  Its canonical
coordinate scan therefore orders the input coordinate before the output
coordinate.  Public linear basis tensors nevertheless have trailing shape
``(out_dim, in_dim)``, and the full stable basis record has shape
``(multiplicity, out_dim, in_dim)``.  The tensor product constructor fixes
``flat(i, j) = i * right_dim + j``, so left coordinates are major and right
coordinates are minor.  It returns a matrix with entries
``P[(i, j), (a, b)] = left[i, a] right[j, b]``.  A bilinear basis element
obeys ``T (rho_left(h) tensor rho_right(h)) = rho_out(h) T`` and has shape
``(out_dim, left_dim, right_dim)``.  Its full stable basis record has shape
``(multiplicity, out_dim, left_dim, right_dim)``.  Its canonical column major
scan orders left, then right, then output coordinates.

Representation constructors accept tensors with square trailing axes and
broadcast all leading axes.  Direct sums return trailing size equal to the
sum of block sizes.  Tensor products return trailing size
``left_dim * right_dim``.  Compiler inputs instead have exact shape
``(generator_count, dim, dim)`` or are one square matrix interpreted as one
generator.  Ordered generator counts must match and are not broadcast.

Python denotes STF degree by ``rank``.  Documentation denotes it by
``l`` in the integers from zero through ``MAX_STF_RANK``, currently ten, with
``d_l = 2 * l + 1``.  This module accepts general real representations and
has no rank argument, but STF representations supplied to it must follow
that project limit.  The convention identity is
``TENSOR_CONVENTION_VERSION`` and includes the active action, STF basis,
coupling, canonical subspace, rank, and pose coordinate orders.

Runtime representation constructors support float32 and float64, require
one device, promote mixed supported dtypes, preserve the resulting device,
and preserve gradients.  Compilers are offline and non differentiable.
They detach every input and return CPU float64 tensors.  Applying a returned
basis with ordinary PyTorch contractions remains differentiable with
respect to runtime features and learned scalar coefficients.

Canonical results are deterministic for a fixed convention and a stable
singular value decision.  Exact floating values may vary across PyTorch,
linear algebra library, and hardware versions.  A singular gap not greater
than its nullspace threshold, a threshold margin smaller than one half of that
threshold, or a constraint residual greater than the threshold raises
RuntimeError and requires a better conditioned artifact.
TypeError reports unsupported values and dtypes.  ValueError reports empty
or nonsquare spaces, generator count mismatches, device mismatches,
nonfinite compiler data, and invalid tolerances.

Mapped references:

Marc Finzi, Max Welling, Andrew Gordon Wilson. A Practical Method for
Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups.
https://proceedings.mlr.press/v139/finzi21a.html

Leon Lang, Maurice Weiler. A Wigner Eckart Theorem for Group Equivariant
Convolution Kernels. https://openreview.net/forum?id=ajOrOhQOsYx
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ._compiler_utils import (
    canonical_nullspace,
    minimum_stable_margin,
    positive_finite_tolerance,
)
from .stf_space import TENSOR_CONVENTION_VERSION


__all__ = [
    "BilinearIntertwinerCompilation",
    "IntertwinerCompilation",
    "a_representation",
    "compile_bilinear_intertwiners",
    "compile_intertwiners",
    "direct_sum_representation",
    "intertwiner_residual",
    "tensor_product_representation",
]


@dataclass(frozen=True)
class IntertwinerCompilation:
    """Store one canonical linear basis and numerical diagnostics."""

    basis: Tensor
    dimension: int
    singular_values: Tensor
    threshold: float
    singular_value_gap: float
    threshold_margin: float
    residual: float
    convention_version: str
    nullspace_atol: float
    nullspace_rtol: float


@dataclass(frozen=True)
class BilinearIntertwinerCompilation:
    """Store one canonical bilinear basis and numerical diagnostics."""

    basis: Tensor
    dimension: int
    left_dimension: int
    right_dimension: int
    output_dimension: int
    singular_values: Tensor
    threshold: float
    singular_value_gap: float
    threshold_margin: float
    residual: float
    convention_version: str
    nullspace_atol: float
    nullspace_rtol: float


def _as_generator_tensor(value: Tensor | Sequence[Tensor], name: str) -> Tensor:
    """Return generator representations with shape count by size by size."""
    if isinstance(value, Tensor):
        result = value.unsqueeze(0) if value.ndim == 2 else value
    else:
        values = tuple(value)
        if not values:
            raise ValueError(f"{name} cannot be an empty sequence")
        result = torch.stack(values)
    if result.ndim != 3 or result.shape[-2] != result.shape[-1]:
        raise ValueError(f"{name} must have shape (count, size, size)")
    if result.shape[-1] == 0:
        raise ValueError(f"{name} representation size must be positive")
    if result.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_runtime_representation(representation: Tensor, name: str) -> None:
    """Validate static metadata for one runtime representation."""
    if not isinstance(representation, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        representation.ndim < 2
        or representation.shape[-2] != representation.shape[-1]
    ):
        raise ValueError(
            f"{name} actual shape {tuple(representation.shape)} must have square "
            "trailing dimensions"
        )
    if representation.shape[-1] == 0:
        raise ValueError(f"{name} representation size must be positive")
    if representation.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")


def direct_sum_representation(*representations: Tensor) -> Tensor:
    """Return the broadcast direct sum with blocks in argument order."""
    if len(representations) == 1 and not isinstance(representations[0], Tensor):
        representations = tuple(representations[0])
    if not representations:
        raise ValueError("at least one representation is required")
    for index, representation in enumerate(representations):
        _validate_runtime_representation(representation, f"representations[{index}]")
    device = representations[0].device
    if any(representation.device != device for representation in representations):
        raise ValueError("all representations must use the same device")
    dtype = representations[0].dtype
    for representation in representations[1:]:
        dtype = torch.promote_types(dtype, representation.dtype)
    batch_shapes = tuple(
        representation.shape[:-2] for representation in representations
    )
    try:
        batch_shape = torch.broadcast_shapes(*batch_shapes)
    except RuntimeError as error:
        raise ValueError(
            f"representation batch shapes {batch_shapes} are not broadcast compatible"
        ) from error
    blocks = [
        representation.to(dtype=dtype).expand(batch_shape + representation.shape[-2:])
        for representation in representations
    ]
    rows = []
    for row_index, row_block in enumerate(blocks):
        row = []
        for column_index, column_block in enumerate(blocks):
            if row_index == column_index:
                row.append(row_block)
            else:
                row.append(
                    row_block.new_zeros(
                        batch_shape + (row_block.shape[-2], column_block.shape[-1])
                    )
                )
        rows.append(torch.cat(row, dim=-1))
    return torch.cat(rows, dim=-2)


def tensor_product_representation(left: Tensor, right: Tensor) -> Tensor:
    """Return the broadcast tensor product in left major coordinate order."""
    _validate_runtime_representation(left, "left")
    _validate_runtime_representation(right, "right")
    if left.device != right.device:
        raise ValueError("left and right must use the same device")
    dtype = torch.promote_types(left.dtype, right.dtype)
    try:
        batch_shape = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    except RuntimeError as error:
        raise ValueError(
            "left and right batch shapes are not broadcast compatible: "
            f"actual {left.shape[:-2]} and {right.shape[:-2]}"
        ) from error
    left_dimension = left.shape[-1]
    right_dimension = right.shape[-1]
    left_work = left.to(dtype=dtype).expand(
        batch_shape + (left_dimension, left_dimension)
    )
    right_work = right.to(dtype=dtype).expand(
        batch_shape + (right_dimension, right_dimension)
    )
    product = torch.einsum("...ia,...jb->...ijab", left_work, right_work)
    product_dimension = left_dimension * right_dimension
    return product.reshape(batch_shape + (product_dimension, product_dimension))


def a_representation(generators: Tensor) -> Tensor:
    """Return the defining three dimensional representation for space A."""
    if not isinstance(generators, Tensor):
        raise TypeError("generators must be a torch.Tensor")
    if generators.shape[-2:] != (3, 3):
        raise ValueError("generators must have trailing shape (3, 3)")
    if generators.dtype not in (torch.float32, torch.float64):
        raise TypeError("generators must use float32 or float64")
    return generators


def compile_intertwiners(
    rho_in: Tensor | Sequence[Tensor],
    rho_out: Tensor | Sequence[Tensor],
    *,
    nullspace_atol: float | None = None,
    nullspace_rtol: float | None = None,
) -> IntertwinerCompilation:
    """Compile every linear intertwiner into one stable result record."""
    inputs = _as_generator_tensor(rho_in, "rho_in")
    outputs = _as_generator_tensor(rho_out, "rho_out")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError(
            "rho_in and rho_out generator counts must match: "
            f"actual {inputs.shape[0]} and {outputs.shape[0]}"
        )
    if inputs.device != outputs.device:
        raise ValueError("rho_in and rho_out must use the same device")
    default_atol = 1e-6 if torch.float32 in (inputs.dtype, outputs.dtype) else 1e-10
    resolved_atol = positive_finite_tolerance(
        nullspace_atol,
        "nullspace_atol",
        default_atol,
    )
    resolved_rtol = positive_finite_tolerance(
        nullspace_rtol,
        "nullspace_rtol",
        1e-12,
    )

    inputs = inputs.detach().to(device="cpu", dtype=torch.float64)
    outputs = outputs.detach().to(device="cpu", dtype=torch.float64)
    input_size = inputs.shape[-1]
    output_size = outputs.shape[-1]
    identity_in = torch.eye(input_size, dtype=torch.float64)
    identity_out = torch.eye(output_size, dtype=torch.float64)
    constraints = []
    for generator_index in range(inputs.shape[0]):
        right_action = torch.kron(
            inputs[generator_index].transpose(0, 1).contiguous(), identity_out
        )
        left_action = torch.kron(identity_in, outputs[generator_index].contiguous())
        constraints.append(right_action - left_action)
    if constraints:
        constraint = torch.cat(constraints, dim=0)
    else:
        constraint = torch.empty((0, input_size * output_size), dtype=torch.float64)
    nullspace = canonical_nullspace(constraint, resolved_atol, resolved_rtol)
    count = nullspace.dimension
    basis = (
        nullspace.basis.transpose(0, 1)
        .reshape(count, input_size, output_size)
        .transpose(-2, -1)
        .contiguous()
    )
    residual = float(intertwiner_residual(basis, inputs, outputs))
    expected_shape = (count, output_size, input_size)
    actual_shape = tuple(basis.shape)
    required_margin = minimum_stable_margin(nullspace.threshold)
    if (
        nullspace.singular_value_gap <= nullspace.threshold
        or nullspace.threshold_margin < required_margin
    ):
        raise RuntimeError(
            "unstable intertwiner nullspace decision: "
            f"singular gap {nullspace.singular_value_gap:.6e}, "
            f"margin {nullspace.threshold_margin:.6e}, "
            f"residual {residual:.6e}, threshold {nullspace.threshold:.6e}, "
            f"required margin {required_margin:.6e}, "
            f"actual shape {actual_shape}, expected shape {expected_shape}"
        )
    if residual > nullspace.threshold:
        raise RuntimeError(
            "intertwiner constraint leakage: "
            f"residual {residual:.6e}, threshold {nullspace.threshold:.6e}, "
            f"actual shape {actual_shape}, expected shape {expected_shape}"
        )
    return IntertwinerCompilation(
        basis=basis,
        dimension=count,
        singular_values=nullspace.singular_values,
        threshold=nullspace.threshold,
        singular_value_gap=nullspace.singular_value_gap,
        threshold_margin=nullspace.threshold_margin,
        residual=residual,
        convention_version=TENSOR_CONVENTION_VERSION,
        nullspace_atol=resolved_atol,
        nullspace_rtol=resolved_rtol,
    )


def compile_bilinear_intertwiners(
    rho_left: Tensor | Sequence[Tensor],
    rho_right: Tensor | Sequence[Tensor],
    rho_out: Tensor | Sequence[Tensor],
    *,
    nullspace_atol: float | None = None,
    nullspace_rtol: float | None = None,
) -> BilinearIntertwinerCompilation:
    """Compile every equivariant bilinear map into one stable result record."""
    left = _as_generator_tensor(rho_left, "rho_left")
    right = _as_generator_tensor(rho_right, "rho_right")
    outputs = _as_generator_tensor(rho_out, "rho_out")
    counts = (left.shape[0], right.shape[0], outputs.shape[0])
    if counts[0] != counts[1] or counts[0] != counts[2]:
        raise ValueError(
            "rho_left, rho_right, and rho_out generator counts must match: "
            f"actual {counts}"
        )
    if left.device != right.device or left.device != outputs.device:
        raise ValueError(
            "rho_left, rho_right, and rho_out must use the same device"
        )
    left_dimension = left.shape[-1]
    right_dimension = right.shape[-1]
    output_dimension = outputs.shape[-1]
    default_atol = (
        1e-6
        if torch.float32 in (left.dtype, right.dtype, outputs.dtype)
        else 1e-10
    )
    resolved_atol = positive_finite_tolerance(
        nullspace_atol,
        "nullspace_atol",
        default_atol,
    )
    resolved_rtol = positive_finite_tolerance(
        nullspace_rtol,
        "nullspace_rtol",
        1e-12,
    )
    left_work = left.detach().to(device="cpu", dtype=torch.float64)
    right_work = right.detach().to(device="cpu", dtype=torch.float64)
    output_work = outputs.detach().to(device="cpu", dtype=torch.float64)
    product = tensor_product_representation(left_work, right_work)
    linear = compile_intertwiners(
        product,
        output_work,
        nullspace_atol=resolved_atol,
        nullspace_rtol=resolved_rtol,
    )
    basis = linear.basis.reshape(
        linear.dimension,
        output_dimension,
        left_dimension,
        right_dimension,
    ).contiguous()
    expected_shape = (
        linear.dimension,
        output_dimension,
        left_dimension,
        right_dimension,
    )
    if tuple(basis.shape) != expected_shape:
        raise RuntimeError(
            "bilinear basis reshape failed: "
            f"actual shape {tuple(basis.shape)}, expected shape {expected_shape}, "
            f"residual {linear.residual:.6e}, threshold {linear.threshold:.6e}"
        )
    return BilinearIntertwinerCompilation(
        basis=basis,
        dimension=linear.dimension,
        left_dimension=left_dimension,
        right_dimension=right_dimension,
        output_dimension=output_dimension,
        singular_values=linear.singular_values,
        threshold=linear.threshold,
        singular_value_gap=linear.singular_value_gap,
        threshold_margin=linear.threshold_margin,
        residual=linear.residual,
        convention_version=linear.convention_version,
        nullspace_atol=linear.nullspace_atol,
        nullspace_rtol=linear.nullspace_rtol,
    )


def intertwiner_residual(
    intertwiners: Tensor,
    rho_in: Tensor | Sequence[Tensor],
    rho_out: Tensor | Sequence[Tensor],
) -> Tensor:
    """Return the largest Frobenius generator constraint residual."""
    inputs = _as_generator_tensor(rho_in, "rho_in")
    outputs = _as_generator_tensor(rho_out, "rho_out")
    if not isinstance(intertwiners, Tensor):
        raise TypeError("intertwiners must be a torch.Tensor")
    if intertwiners.dtype not in (torch.float32, torch.float64):
        raise TypeError("intertwiners must use float32 or float64")
    if not bool(torch.isfinite(intertwiners).all()):
        raise ValueError("intertwiners must contain only finite values")
    basis = intertwiners.unsqueeze(0) if intertwiners.ndim == 2 else intertwiners
    if basis.ndim != 3:
        raise ValueError(
            "intertwiners actual shape "
            f"{tuple(basis.shape)} must have expected shape "
            "(count, out_size, in_size)"
        )
    if basis.shape[-2:] != (outputs.shape[-1], inputs.shape[-1]):
        raise ValueError(
            "intertwiner trailing dimensions do not match representations: "
            f"actual {tuple(basis.shape[-2:])}, expected "
            f"{(outputs.shape[-1], inputs.shape[-1])}"
        )
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError(
            "rho_in and rho_out generator counts must match: "
            f"actual {inputs.shape[0]} and {outputs.shape[0]}"
        )
    if basis.device != inputs.device or outputs.device != inputs.device:
        raise ValueError("all tensors must use the same device")
    dtype = torch.promote_types(
        torch.promote_types(basis.dtype, inputs.dtype), outputs.dtype
    )
    basis = basis.to(dtype=dtype)
    inputs = inputs.to(dtype=dtype)
    outputs = outputs.to(dtype=dtype)
    residuals = basis[:, None] @ inputs[None] - outputs[None] @ basis[:, None]
    if residuals.numel() == 0:
        return basis.new_zeros(())
    return torch.linalg.matrix_norm(residuals, ord="fro", dim=(-2, -1)).amax()
