"""Validate tensor product and bilinear intertwiner public contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from TFENN.tensor_math import PoseEncoder, compile_anchors
from TFENN.tensor_math.intertwiner_compiler import (
    BilinearIntertwinerCompilation,
    compile_bilinear_intertwiners,
    tensor_product_representation,
)

from ._groups import ATOL, DTYPE, RTOL, benzene_generators


def test_tensor_product_uses_left_major_right_minor_order() -> None:
    """Check exact coordinate order, broadcasting, homomorphism, and gradients."""
    generator = torch.Generator().manual_seed(211)
    left = torch.randn(
        (2, 2, 2), dtype=DTYPE, generator=generator, requires_grad=True
    )
    right = torch.randn(
        (1, 3, 3), dtype=DTYPE, generator=generator, requires_grad=True
    )
    product = tensor_product_representation(left, right)
    assert product.shape == (2, 6, 6)
    for left_out in range(2):
        for right_out in range(3):
            for left_in in range(2):
                for right_in in range(3):
                    torch.testing.assert_close(
                        product[:, left_out * 3 + right_out, left_in * 3 + right_in],
                        left[:, left_out, left_in]
                        * right[:, right_out, right_in],
                    )
    product.square().sum().backward()
    assert left.grad is not None and bool(torch.isfinite(left.grad).all())
    assert right.grad is not None and bool(torch.isfinite(right.grad).all())


def test_tensor_product_representation_is_a_homomorphism() -> None:
    """Check tensor products preserve representation multiplication."""
    generators = benzene_generators()
    first, second = generators.unbind()
    product = tensor_product_representation(first, second)
    composed = tensor_product_representation(first @ second, second @ first)
    expected = product @ tensor_product_representation(second, first)
    torch.testing.assert_close(composed, expected, atol=ATOL, rtol=RTOL)

    empty = tensor_product_representation(
        torch.empty((0, 2, 2), dtype=DTYPE),
        torch.eye(3, dtype=DTYPE),
    )
    assert empty.shape == (0, 6, 6)


def _apply_bilinear_basis(
    basis: Tensor,
    coefficients: Tensor,
    left: Tensor,
    right: Tensor,
) -> Tensor:
    """Apply a fixed bilinear basis with learned scalar coefficients."""
    return torch.einsum("m,moij,...i,...j->...o", coefficients, basis, left, right)


def test_d6_bilinear_dimensions_equivariance_and_gradients() -> None:
    """Check known A and B mixing spaces and differentiable application."""
    generators = benzene_generators()
    encoder = PoseEncoder(
        compile_anchors(generators, output_ranks=(2, 6))
    )
    rho_a = generators
    rho_b = encoder.representation(generators)
    specifications = (
        (rho_a, rho_a, rho_a, 3),
        (rho_a, rho_a, rho_b, 16),
        (rho_a, rho_b, rho_a, 16),
    )
    compilations = []
    for left, right, output, expected_dimension in specifications:
        compilation = compile_bilinear_intertwiners(left, right, output)
        assert isinstance(compilation, BilinearIntertwinerCompilation)
        assert compilation.dimension == expected_dimension
        assert compilation.basis.shape == (
            expected_dimension,
            output.shape[-1],
            left.shape[-1],
            right.shape[-1],
        )
        assert compilation.basis.dtype == torch.float64
        assert compilation.basis.device.type == "cpu"
        assert not compilation.basis.requires_grad
        assert compilation.residual < 2e-12
        compilations.append(compilation)

    mixed = compilations[-1]
    generator = torch.Generator().manual_seed(223)
    left = torch.randn((4, 3), dtype=DTYPE, generator=generator)
    right = torch.randn((4, 18), dtype=DTYPE, generator=generator)
    basis_outputs = torch.einsum("moij,bi,bj->bmo", mixed.basis, left, right)
    for rho_left, rho_right, rho_out in zip(rho_a, rho_b, rho_a):
        transformed = torch.einsum(
            "moij,bi,bj->bmo",
            mixed.basis,
            left @ rho_left.T,
            right @ rho_right.T,
        )
        expected = torch.einsum("op,bmp->bmo", rho_out, basis_outputs)
        torch.testing.assert_close(transformed, expected, atol=ATOL, rtol=RTOL)

    coefficients = torch.randn(
        mixed.dimension,
        dtype=DTYPE,
        generator=generator,
        requires_grad=True,
    )
    differentiable_left = left.clone().requires_grad_()
    differentiable_right = right.clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda c, x, y: _apply_bilinear_basis(mixed.basis, c, x, y),
        (coefficients, differentiable_left, differentiable_right),
        eps=1e-6,
        atol=3e-5,
        rtol=3e-4,
    )


def test_bilinear_compiler_promotes_before_tensor_product() -> None:
    """Check single precision inputs enter tensor products as CPU float64."""
    generators = benzene_generators().float()
    single = compile_bilinear_intertwiners(
        generators,
        generators,
        generators,
    )
    promoted = compile_bilinear_intertwiners(
        generators.double(),
        generators.double(),
        generators.double(),
        nullspace_atol=single.nullspace_atol,
        nullspace_rtol=single.nullspace_rtol,
    )
    assert single.dimension == promoted.dimension == 3
    assert single.nullspace_atol == 1e-6
    assert single.basis.dtype == torch.float64
    assert single.basis.device.type == "cpu"
    torch.testing.assert_close(
        single.basis,
        promoted.basis,
        atol=0.0,
        rtol=0.0,
    )
