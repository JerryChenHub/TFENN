"""Numerical verification helpers for STF quotient encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .basis import stf_basis, symmetric_multi_indices

if TYPE_CHECKING:
    from .encoder import STFEncoder


@dataclass(frozen=True)
class InverseFiberResult:
    """Store a numerical representative and its finite right orbit."""

    representative: torch.Tensor
    rotations: torch.Tensor
    code_residual: float
    fiber_residual: float
    inclusion_error: float | None


def skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew matrix for vectors with final dimension three."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def rotation_from_rotvec(vector: torch.Tensor) -> torch.Tensor:
    """Map rotation vectors to SO(3) with the matrix exponential."""
    return torch.matrix_exp(skew(vector))


def rotation_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Map scalar first quaternions to SO(3)."""
    q = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternion.dtype).eps)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _rank_two_tensor(components: torch.Tensor) -> torch.Tensor:
    """Expand five STF coordinates into a symmetric three by three tensor."""
    symmetric = stf_basis(2, dtype=components.dtype, device=components.device) @ components
    tensor = torch.zeros((3, 3), dtype=components.dtype, device=components.device)
    for value, alpha in zip(symmetric, symmetric_multi_indices(2)):
        axes = [axis for axis, count in enumerate(alpha) for _ in range(count)]
        if axes[0] == axes[1]:
            tensor[axes[0], axes[1]] = value
        else:
            entry = value / (2.0**0.5)
            tensor[axes[0], axes[1]] = entry
            tensor[axes[1], axes[0]] = entry
    return tensor


def _d6_inverse_representative(encoder: STFEncoder, code: torch.Tensor) -> torch.Tensor:
    """Recover a D6 representative from its rank two axis and rank six phase."""
    tensor = _rank_two_tensor(code[:5])
    _, eigenvectors = torch.linalg.eigh(tensor)
    normal = eigenvectors[:, 0]
    pivot = normal.abs().argmax()
    normal = normal * torch.where(normal[pivot] < 0, -1.0, 1.0)
    coordinate = torch.eye(3, dtype=code.dtype, device=code.device)[:, normal.abs().argmin()]
    first = coordinate - torch.dot(coordinate, normal) * normal
    first = first / torch.linalg.vector_norm(first)
    second = torch.linalg.cross(normal, first, dim=0)
    frame = torch.stack((first, second, normal), dim=-1)

    def rank_six_at(angle: float) -> torch.Tensor:
        """Evaluate the rank six block after an axial body rotation."""
        vector = torch.tensor((0.0, 0.0, angle), dtype=code.dtype, device=code.device)
        return encoder.encode_blocks(frame @ rotation_from_rotvec(vector))[6][:, 0]

    zero = rank_six_at(0.0)
    opposite = rank_six_at(torch.pi / 6.0)
    quarter = rank_six_at(torch.pi / 12.0)
    center = 0.5 * (zero + opposite)
    cosine_mode = zero - center
    sine_mode = quarter - center
    target = code[5:18] - center
    cosine = torch.dot(target, cosine_mode) / torch.dot(cosine_mode, cosine_mode)
    sine = torch.dot(target, sine_mode) / torch.dot(sine_mode, sine_mode)
    angle = torch.atan2(sine, cosine) / 6.0
    vector = torch.stack((angle * 0.0, angle * 0.0, angle))
    return frame @ rotation_from_rotvec(vector)


def numerical_inverse_fiber(
    encoder: STFEncoder,
    code: torch.Tensor,
    *,
    reference_rotation: torch.Tensor | None = None,
    num_starts: int = 32,
    adam_steps: int = 180,
    learning_rate: float = 0.08,
    seed: int = 0,
) -> InverseFiberResult:
    """Recover one pose numerically and return its complete finite orbit."""
    if code.ndim != 1 or code.shape[0] != encoder.encoding_dim:
        raise ValueError("code must be a single encoder output vector")
    if num_starts < 1 or adam_steps < 1:
        raise ValueError("num_starts and adam_steps must be positive")

    target = code.detach()
    group_elements = encoder.group_elements.to(
        dtype=target.dtype, device=target.device
    )
    if getattr(encoder.certificate, "group_name", "") == "D6_tilde":
        representative = _d6_inverse_representative(encoder, target)
        with torch.no_grad():
            orbit = representative @ group_elements
            encoded_orbit = encoder.encode_rotation(orbit)
            code_residual = float(torch.linalg.vector_norm(
                encoder.encode_rotation(representative) - target
            ))
            fiber_residual = float(
                torch.linalg.vector_norm(encoded_orbit - target, dim=-1).max()
            )
            inclusion_error = None
            if reference_rotation is not None:
                reference = reference_rotation.to(
                    dtype=target.dtype, device=target.device
                )
                inclusion_error = float(
                    torch.linalg.matrix_norm(orbit - reference, dim=(-2, -1)).min()
                )
        return InverseFiberResult(
            representative=representative,
            rotations=orbit,
            code_residual=code_residual,
            fiber_residual=fiber_residual,
            inclusion_error=inclusion_error,
        )

    generator = torch.Generator(device=target.device)
    generator.manual_seed(seed)
    starts = torch.randn(
        (num_starts, 4),
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    starts[0] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0), dtype=target.dtype, device=target.device
    )
    quaternion = torch.nn.Parameter(starts)
    optimizer = torch.optim.Adam((quaternion,), lr=learning_rate)

    for _ in range(adam_steps):
        optimizer.zero_grad()
        rotations = rotation_from_quaternion(quaternion)
        residual = encoder.encode_rotation(rotations) - target
        residual.square().sum(dim=-1).sum().backward()
        optimizer.step()

    with torch.no_grad():
        candidates = rotation_from_quaternion(quaternion)
        errors = (encoder.encode_rotation(candidates) - target).square().sum(dim=-1)
        best = quaternion[errors.argmin()].detach().clone()
    refined = torch.nn.Parameter(best)
    refinement = torch.optim.LBFGS(
        (refined,),
        max_iter=100,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        """Evaluate one representative for the LBFGS refinement."""
        refinement.zero_grad()
        residual = encoder.encode_rotation(rotation_from_quaternion(refined)) - target
        loss = residual.square().sum()
        loss.backward()
        return loss

    refinement.step(closure)

    with torch.no_grad():
        representative = rotation_from_quaternion(refined).clone()
        orbit = representative @ group_elements
        encoded_orbit = encoder.encode_rotation(orbit)
        code_residual = float(torch.linalg.vector_norm(
            encoder.encode_rotation(representative) - target
        ))
        fiber_residual = float(
            torch.linalg.vector_norm(encoded_orbit - target, dim=-1).max()
        )
        inclusion_error = None
        if reference_rotation is not None:
            reference = reference_rotation.to(dtype=target.dtype, device=target.device)
            inclusion_error = float(
                torch.linalg.matrix_norm(orbit - reference, dim=(-2, -1)).min()
            )

    return InverseFiberResult(
        representative=representative,
        rotations=orbit,
        code_residual=code_residual,
        fiber_residual=fiber_residual,
        inclusion_error=inclusion_error,
    )


def encoding_jacobian(encoder: STFEncoder, rotation: torch.Tensor) -> torch.Tensor:
    """Return the left tangent Jacobian of the encoded pose."""
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    base = rotation.detach()
    omega = torch.zeros(3, dtype=base.dtype, device=base.device, requires_grad=True)
    return torch.autograd.functional.jacobian(
        lambda value: encoder.encode_rotation(rotation_from_rotvec(value) @ base),
        omega,
        create_graph=False,
        vectorize=True,
    )


def jacobian_pseudoinverse(
    encoder: STFEncoder, rotation: torch.Tensor
) -> torch.Tensor:
    """Return the Moore Penrose inverse of the tangent Jacobian."""
    return torch.linalg.pinv(encoding_jacobian(encoder, rotation))


def verify_gradients(
    encoder: STFEncoder,
    rotation: torch.Tensor,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> bool:
    """Run double precision gradcheck on the SO(3) tangent parameter."""
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    base = rotation.detach().to(dtype=torch.float64)
    omega = torch.zeros(3, dtype=torch.float64, device=base.device, requires_grad=True)
    return bool(
        torch.autograd.gradcheck(
            lambda value: encoder.encode_rotation(rotation_from_rotvec(value) @ base),
            (omega,),
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
    )


__all__ = [
    "InverseFiberResult",
    "encoding_jacobian",
    "jacobian_pseudoinverse",
    "numerical_inverse_fiber",
    "rotation_from_quaternion",
    "rotation_from_rotvec",
    "skew",
    "verify_gradients",
]
