"""Generator constrained linear intertwiner compilation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ._compiler_utils import canonical_nullspace, positive_finite_tolerance
from .stf_space import STF_BASIS_VERSION


__all__ = [
    "IntertwinerCompilation",
    "a_representation",
    "compile_intertwiners",
    "direct_sum_representation",
    "intertwiner_residual",
]


@dataclass(frozen=True)
class IntertwinerCompilation:
    """Store one canonical intertwiner basis and rank diagnostics."""

    basis: Tensor
    dimension: int
    singular_values: Tensor
    threshold: float
    singular_value_gap: float
    residual: float
    basis_version: str
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


def direct_sum_representation(*representations: Tensor) -> Tensor:
    """Form a batch aware block direct sum of square representations."""
    if len(representations) == 1 and not isinstance(representations[0], Tensor):
        representations = tuple(representations[0])
    if not representations:
        raise ValueError("at least one representation is required")
    for representation in representations:
        if not isinstance(representation, Tensor):
            raise TypeError("every representation must be a torch.Tensor")
        if (
            representation.ndim < 2
            or representation.shape[-2] != representation.shape[-1]
        ):
            raise ValueError(
                "every representation must have square trailing dimensions"
            )
        if representation.dtype not in (torch.float32, torch.float64):
            raise TypeError("representations must use float32 or float64")
    device = representations[0].device
    if any(representation.device != device for representation in representations):
        raise ValueError("all representations must use the same device")
    dtype = representations[0].dtype
    for representation in representations[1:]:
        dtype = torch.promote_types(dtype, representation.dtype)
    batch_shape = torch.broadcast_shapes(
        *(representation.shape[:-2] for representation in representations)
    )
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


def a_representation(generators: Tensor) -> Tensor:
    """Return the defining three dimensional representation for space A."""
    if not isinstance(generators, Tensor) or generators.shape[-2:] != (3, 3):
        raise ValueError("generators must have trailing shape (3, 3)")
    if generators.dtype not in (torch.float32, torch.float64):
        raise TypeError("generators must use float32 or float64")
    return generators


def compile_intertwiners(
    rho_in: Tensor | Sequence[Tensor],
    rho_out: Tensor | Sequence[Tensor],
    *,
    tolerance: float | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    return_compilation: bool = False,
) -> Tensor | IntertwinerCompilation:
    """Return a Frobenius orthonormal basis satisfying M rho_in equals rho_out M."""
    inputs = _as_generator_tensor(rho_in, "rho_in")
    outputs = _as_generator_tensor(rho_out, "rho_out")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("rho_in and rho_out must contain the same generators")
    if inputs.device != outputs.device:
        raise ValueError("rho_in and rho_out must use the same device")
    if tolerance is not None and atol is not None:
        raise ValueError("tolerance and atol cannot both be provided")
    if not isinstance(return_compilation, bool):
        raise TypeError("return_compilation must be bool")
    default_atol = 1e-6 if torch.float32 in (inputs.dtype, outputs.dtype) else 1e-10
    requested_atol = atol if atol is not None else tolerance
    resolved_atol = positive_finite_tolerance(
        requested_atol,
        "tolerance" if tolerance is not None else "atol",
        default_atol,
    )
    resolved_rtol = positive_finite_tolerance(rtol, "rtol", 1e-12)

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
    if not return_compilation:
        return basis
    residual = float(intertwiner_residual(basis, inputs, outputs))
    return IntertwinerCompilation(
        basis=basis,
        dimension=count,
        singular_values=nullspace.singular_values,
        threshold=nullspace.threshold,
        singular_value_gap=nullspace.singular_value_gap,
        residual=residual,
        basis_version=STF_BASIS_VERSION,
        nullspace_atol=resolved_atol,
        nullspace_rtol=resolved_rtol,
    )


def intertwiner_residual(
    intertwiners: Tensor,
    rho_in: Tensor | Sequence[Tensor],
    rho_out: Tensor | Sequence[Tensor],
) -> Tensor:
    """Return the largest Frobenius generator constraint residual."""
    inputs = _as_generator_tensor(rho_in, "rho_in")
    outputs = _as_generator_tensor(rho_out, "rho_out")
    basis = intertwiners.unsqueeze(0) if intertwiners.ndim == 2 else intertwiners
    if basis.ndim != 3:
        raise ValueError("intertwiners must have shape (count, out_size, in_size)")
    if basis.shape[-2:] != (outputs.shape[-1], inputs.shape[-1]):
        raise ValueError("intertwiner dimensions do not match the representations")
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("rho_in and rho_out must contain the same generators")
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
