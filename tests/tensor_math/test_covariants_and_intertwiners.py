"""Validate covariants, gradients, and generator constrained linear maps."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from TFENN.tensor_math import (
    PoseEncoder,
    a_representation,
    compile_anchors,
    compile_intertwiners,
    intertwiner_residual,
    scalar_contraction,
    stf_dimension,
    stf_representation,
    vector_covariant,
)

from ._groups import (
    ATOL,
    DTYPE,
    RTOL,
    benzene_generators,
    c60_generators,
    rotation,
    rotation_from_rotvec,
)


def _pose_encoder(generators: Tensor, ranks: tuple[int, ...]) -> PoseEncoder:
    """Compile one test pose encoder directly from generators."""
    return PoseEncoder(compile_anchors(generators, ranks))


def test_vector_covariant_is_the_analytic_scalar_gradient() -> None:
    """Compare q with an autograd derivative of the defining scalar."""
    generator = torch.Generator().manual_seed(71)
    position = torch.randn(3, dtype=DTYPE, generator=generator)
    pose_block = torch.randn((stf_dimension(6), 2), dtype=DTYPE, generator=generator)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: scalar_contraction(value, pose_block, 6), position
    )
    torch.testing.assert_close(
        vector_covariant(position, pose_block, 6),
        jacobian,
        atol=ATOL,
        rtol=RTOL,
    )


def test_vector_covariant_gradient_in_position_and_pose_block() -> None:
    """Run gradcheck on q with respect to x and Z."""
    generator = torch.Generator().manual_seed(73)
    position = torch.randn(3, dtype=DTYPE, generator=generator, requires_grad=True)
    pose_block = torch.randn(
        (stf_dimension(6), 2),
        dtype=DTYPE,
        generator=generator,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda x, z: vector_covariant(x, z, 6),
        (position, pose_block),
        eps=1e-6,
        atol=3e-5,
        rtol=3e-4,
    )


@pytest.mark.parametrize(
    ("generators", "ranks", "rank"),
    (
        (benzene_generators(), (2, 6), 2),
        (benzene_generators(), (2, 6), 6),
        (c60_generators(), (6,), 6),
    ),
)
def test_q_covariance_and_full_rotation_gradient_chain(
    generators: Tensor, ranks: tuple[int, ...], rank: int
) -> None:
    """Check q(gx,D_l(g)Z)=gq and its R to Z to q gradient."""
    encoder = _pose_encoder(generators, ranks)
    pose = rotation((0.16, -0.22, 0.34))
    position = torch.tensor((0.7, -0.4, 0.3), dtype=DTYPE)
    block = encoder.encode_blocks(pose)[rank]
    action = generators[0]
    transformed_block = stf_representation(action, rank) @ block
    covariant = vector_covariant(position, block, rank)

    torch.testing.assert_close(
        scalar_contraction(action @ position, transformed_block, rank),
        scalar_contraction(position, block, rank),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        vector_covariant(action @ position, transformed_block, rank),
        covariant @ action.T,
        atol=ATOL,
        rtol=RTOL,
    )

    tangent = torch.tensor((0.11, -0.19, 0.23), dtype=DTYPE, requires_grad=True)
    differentiable_position = position.clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda omega, x: vector_covariant(
            x,
            encoder.encode_blocks(rotation_from_rotvec(omega))[rank],
            rank,
        ),
        (tangent, differentiable_position),
        eps=1e-6,
        atol=4e-5,
        rtol=4e-4,
    )


def _assert_intertwiner_space(rho_in: Tensor, rho_out: Tensor) -> Tensor:
    """Compile and validate one full linear intertwiner basis."""
    result = compile_intertwiners(
        rho_in,
        rho_out,
        nullspace_atol=1e-9,
    )
    basis = result.basis
    assert basis.shape[0] > 0
    assert intertwiner_residual(basis, rho_in, rho_out).item() < ATOL
    flattened = basis.reshape(basis.shape[0], -1)
    torch.testing.assert_close(
        flattened @ flattened.T,
        torch.eye(len(basis), dtype=basis.dtype),
        atol=ATOL,
        rtol=RTOL,
    )
    coefficients = torch.linspace(
        0.2, 1.0, len(basis), dtype=basis.dtype, device=basis.device
    )
    combined = torch.einsum("r,rij->ij", coefficients, basis)
    assert intertwiner_residual(combined, rho_in, rho_out).item() < ATOL
    return basis


@pytest.mark.parametrize(
    ("generators", "ranks"),
    ((benzene_generators(), (2, 6)), (c60_generators(), (6,))),
)
def test_a_to_b_and_b_to_b_intertwiner_compilation(
    generators: Tensor, ranks: tuple[int, ...]
) -> None:
    """Check every compiled lift satisfies all generator equations."""
    encoder = _pose_encoder(generators, ranks)
    rho_a = a_representation(generators)
    rho_b = encoder.representation(generators)
    a_to_b = _assert_intertwiner_space(rho_a, rho_b)
    b_to_a = _assert_intertwiner_space(rho_b, rho_a)
    b_to_b = _assert_intertwiner_space(rho_b, rho_b)
    a_to_a = _assert_intertwiner_space(rho_a, rho_a)

    assert a_to_b.shape[-2:] == (encoder.encoding_dimension, 3)
    assert b_to_b.shape[-2:] == (
        encoder.encoding_dimension,
        encoder.encoding_dimension,
    )
    assert len(a_to_b) == len(b_to_a)
    assert a_to_a.shape[-2:] == (3, 3)


def test_intertwiner_compiler_rejects_unsupported_precision() -> None:
    """Check offline SVD inputs fail clearly outside float32 and float64."""
    half = torch.eye(3, dtype=torch.float16)[None]
    with pytest.raises(TypeError, match="float32 or float64"):
        compile_intertwiners(half, half)


def test_intertwiner_compiler_rejects_invalid_representations() -> None:
    """Check degenerate and nonfinite representation inputs fail clearly."""
    empty = torch.empty((1, 0, 0), dtype=DTYPE)
    with pytest.raises(ValueError, match="size must be positive"):
        compile_intertwiners(empty, empty)

    nonfinite = torch.eye(3, dtype=DTYPE)[None]
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="only finite values"):
        compile_intertwiners(nonfinite, nonfinite)

    with pytest.raises(TypeError, match="torch.Tensor"):
        a_representation(None)
    with pytest.raises(ValueError, match="trailing shape"):
        a_representation(torch.eye(2, dtype=DTYPE))
