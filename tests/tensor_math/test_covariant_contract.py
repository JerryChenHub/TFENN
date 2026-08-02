"""Validate the public mathematical contract of the reference covariants."""

from __future__ import annotations

import pytest
import torch

import TFENN.tensor_math.covariants as covariants_module
from TFENN.tensor_math import scalar_contraction, vector_covariant


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_vector_covariant_matches_autograd(dtype: torch.dtype) -> None:
    """Compare the analytic vector with a position Jacobian."""
    generator = torch.Generator().manual_seed(101)
    position = torch.randn(3, dtype=dtype, generator=generator)
    pose_block = torch.randn((5, 3), dtype=dtype, generator=generator)

    jacobian = torch.autograd.functional.jacobian(
        lambda value: scalar_contraction(value, pose_block, 2),
        position,
    )
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        vector_covariant(position, pose_block, 2),
        jacobian,
        atol=tolerance,
        rtol=tolerance,
    )


def test_vector_covariant_matches_centered_finite_difference() -> None:
    """Compare the analytic vector with a centered position difference."""
    generator = torch.Generator().manual_seed(103)
    position = torch.randn(3, dtype=torch.float64, generator=generator)
    pose_block = torch.randn((7, 2), dtype=torch.float64, generator=generator)
    epsilon = 1e-6
    columns = []
    for axis in range(3):
        step = torch.zeros_like(position)
        step[axis] = epsilon
        positive = scalar_contraction(position + step, pose_block, 3)
        negative = scalar_contraction(position - step, pose_block, 3)
        columns.append((positive - negative) / (2.0 * epsilon))
    difference = torch.stack(columns, dim=-1)

    torch.testing.assert_close(
        vector_covariant(position, pose_block, 3),
        difference,
        atol=2e-9,
        rtol=2e-9,
    )


def test_strict_same_edge_rejects_outer_broadcasting() -> None:
    """Reject batch shapes that would pair different edges."""
    position = torch.randn((4, 1, 3), dtype=torch.float64)
    pose_block = torch.randn((1, 4, 5, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="strict_same_edge"):
        scalar_contraction(position, pose_block, 2)
    with pytest.raises(ValueError, match="strict_same_edge"):
        vector_covariant(position, pose_block, 2)


def test_explicit_broadcasting_retains_general_tensor_semantics() -> None:
    """Allow general broadcasting only when it is explicitly requested."""
    generator = torch.Generator().manual_seed(107)
    position = torch.randn((4, 1, 3), dtype=torch.float64, generator=generator)
    pose_block = torch.randn((1, 4, 5, 2), dtype=torch.float64, generator=generator)

    scalar = scalar_contraction(position, pose_block, 2, strict_same_edge=False)
    vector = vector_covariant(position, pose_block, 2, strict_same_edge=False)
    assert scalar.shape == (4, 4, 2)
    assert vector.shape == (4, 4, 2, 3)
    torch.testing.assert_close(
        scalar[2, 3], scalar_contraction(position[2, 0], pose_block[0, 3], 2)
    )
    torch.testing.assert_close(
        vector[2, 3], vector_covariant(position[2, 0], pose_block[0, 3], 2)
    )


@pytest.mark.parametrize("function", (scalar_contraction, vector_covariant))
@pytest.mark.parametrize("target", ("position", "pose_block"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_covariants_reject_nonfinite_inputs(
    function: object,
    target: str,
    value: float,
) -> None:
    """Reject every nonfinite input before contraction."""
    position = torch.ones(3, dtype=torch.float64)
    pose_block = torch.ones((5, 2), dtype=torch.float64)
    if target == "position":
        position[0] = value
    else:
        pose_block[0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        function(position, pose_block, 2)


@pytest.mark.parametrize("function", (scalar_contraction, vector_covariant))
def test_covariants_reject_invalid_shapes(function: object) -> None:
    """Reject malformed coordinate and anchor dimensions."""
    position = torch.ones(3, dtype=torch.float64)
    pose_block = torch.ones((5, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="trailing shape"):
        function(torch.ones(2, dtype=torch.float64), pose_block, 2)
    with pytest.raises(ValueError, match="pose_block must have shape"):
        function(position, torch.ones((4, 2), dtype=torch.float64), 2)
    with pytest.raises(ValueError, match="at least one anchor"):
        function(position, torch.empty((5, 0), dtype=torch.float64), 2)


@pytest.mark.parametrize("function", (scalar_contraction, vector_covariant))
def test_covariants_reject_invalid_dtypes(function: object) -> None:
    """Reject unsupported and inconsistent coordinate dtypes."""
    position = torch.ones(3, dtype=torch.float64)
    pose_block = torch.ones((5, 2), dtype=torch.float64)

    with pytest.raises(TypeError, match="position must use"):
        function(position.to(torch.int64), pose_block, 2)
    with pytest.raises(TypeError, match="pose_block must use"):
        function(position, pose_block.to(torch.int64), 2)
    with pytest.raises(TypeError, match="same dtype"):
        function(position.float(), pose_block, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("function", (scalar_contraction, vector_covariant))
def test_covariants_reject_different_devices(function: object) -> None:
    """Reject coordinate tensors placed on different devices."""
    position = torch.ones(3, dtype=torch.float32, device="cuda")
    pose_block = torch.ones((5, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="same device"):
        function(position, pose_block, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_covariants_match_cpu_and_support_backward() -> None:
    """Check canonical derivative maps support the complete CUDA path."""
    generator = torch.Generator().manual_seed(157)
    position = torch.randn((4, 3), dtype=torch.float32, generator=generator)
    pose_block = torch.randn((4, 5, 2), dtype=torch.float32, generator=generator)
    expected_scalar = scalar_contraction(position, pose_block, 2)
    expected_vector = vector_covariant(position, pose_block, 2)
    cuda_position = position.cuda().requires_grad_()
    cuda_pose = pose_block.cuda().requires_grad_()
    actual_scalar = scalar_contraction(cuda_position, cuda_pose, 2)
    actual_vector = vector_covariant(cuda_position, cuda_pose, 2)
    torch.testing.assert_close(
        actual_scalar.cpu(), expected_scalar, atol=4e-5, rtol=4e-5
    )
    torch.testing.assert_close(
        actual_vector.cpu(), expected_vector, atol=4e-5, rtol=4e-5
    )
    (actual_scalar.square().sum() + actual_vector.square().sum()).backward()
    assert cuda_position.grad is not None
    assert cuda_pose.grad is not None
    assert bool(torch.isfinite(cuda_position.grad).all())
    assert bool(torch.isfinite(cuda_pose.grad).all())


def test_rank_zero_preserves_the_zero_derivative_graph() -> None:
    """Keep constant rank zero results connected to both inputs."""
    generator = torch.Generator().manual_seed(109)
    position = torch.randn(
        (2, 3), dtype=torch.float64, generator=generator, requires_grad=True
    )
    pose_block = torch.randn(
        (2, 1, 3), dtype=torch.float64, generator=generator, requires_grad=True
    )

    scalar = scalar_contraction(position, pose_block, 0)
    vector = vector_covariant(position, pose_block, 0)
    assert scalar.requires_grad
    assert vector.requires_grad
    torch.testing.assert_close(vector, torch.zeros_like(vector))

    position_gradient, pose_gradient = torch.autograd.grad(
        scalar.sum() + vector.sum(), (position, pose_block)
    )
    torch.testing.assert_close(position_gradient, torch.zeros_like(position))
    torch.testing.assert_close(pose_gradient, torch.ones_like(pose_block))
    assert torch.autograd.gradcheck(
        lambda x, z: scalar_contraction(x, z, 0), (position, pose_block)
    )
    assert torch.autograd.gradcheck(
        lambda x, z: vector_covariant(x, z, 0), (position, pose_block)
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_same_edge_batches_and_empty_batches(dtype: torch.dtype) -> None:
    """Preserve exact batch alignment including an empty edge set."""
    position = torch.randn((5, 3), dtype=dtype)
    pose_block = torch.randn((5, 5, 2), dtype=dtype)
    assert scalar_contraction(position, pose_block, 2).shape == (5, 2)
    assert vector_covariant(position, pose_block, 2).shape == (5, 2, 3)

    empty_position = torch.empty((0, 3), dtype=dtype)
    empty_pose = torch.empty((0, 5, 2), dtype=dtype)
    assert scalar_contraction(empty_position, empty_pose, 2).shape == (0, 2)
    assert vector_covariant(empty_position, empty_pose, 2).shape == (0, 2, 3)


def test_derivative_masters_and_casts_are_cached() -> None:
    """Reuse canonical derivative constants across repeated calls."""
    master = covariants_module._symmetric_derivatives_master(3)
    assert master.dtype == torch.float64
    assert master.device.type == "cpu"
    assert master is covariants_module._symmetric_derivatives_master(3)

    first = covariants_module._symmetric_derivatives(
        3, torch.float32, torch.device("cpu")
    )
    second = covariants_module._symmetric_derivatives(
        3, torch.float32, torch.device("cpu")
    )
    assert first is second
    torch.testing.assert_close(first, master.float(), atol=0.0, rtol=0.0)
