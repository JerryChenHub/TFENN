"""Validate STF spaces, representations, products, and gradients."""

from __future__ import annotations

import math

import pytest
import torch

from TFENN.tensor_math import (
    dense_to_stf,
    project_symmetric,
    stf_basis,
    stf_dimension,
    stf_power_components,
    stf_representation,
    stf_symmetric_product,
    stf_tensor_coupling,
    stf_to_dense,
    stf_to_symmetric,
    symmetric_dimension,
    symmetric_power_components,
    symmetric_representation,
    symmetric_to_stf,
    trace_matrix,
)
from TFENN.tensor_math.stf_space import (
    MAX_STF_RANK,
    STF_BASIS_VERSION,
    TENSOR_CONVENTION_VERSION,
    _stf_tensor_coupling_reference,
)

from ._groups import ATOL, DTYPE, RTOL, rotation, rotation_from_rotvec


_ALL_COUPLING_CHANNELS = tuple(
    (rank_left, rank_right, output_rank)
    for rank_left in range(7)
    for rank_right in range(7)
    for output_rank in range(7)
    if abs(rank_left - rank_right) <= output_rank <= rank_left + rank_right
)

_FLOAT32_COUPLING_CHANNELS = (
    (0, 0, 0),
    (0, 6, 6),
    (1, 1, 0),
    (1, 1, 1),
    (1, 1, 2),
    (2, 3, 1),
    (2, 3, 2),
    (2, 3, 3),
    (2, 3, 4),
    (2, 3, 5),
    (3, 3, 0),
    (3, 3, 3),
    (3, 3, 6),
    (4, 2, 2),
    (4, 2, 4),
    (4, 2, 6),
    (6, 6, 0),
    (6, 6, 6),
)


def _numeric_tolerance(dtype: torch.dtype) -> float:
    """Return a test tolerance appropriate for one precision."""
    return 4e-5 if dtype == torch.float32 else ATOL


@pytest.mark.parametrize("rank", (0, 1, 2, 6))
def test_stf_basis_is_an_exact_cast_of_the_master(rank: int) -> None:
    """Check every CPU precision uses one canonical master basis."""
    master = stf_basis(rank, dtype=torch.float64, device="cpu")
    single = stf_basis(rank, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(single, master.float(), atol=0.0, rtol=0.0)
    assert STF_BASIS_VERSION == TENSOR_CONVENTION_VERSION
    assert TENSOR_CONVENTION_VERSION == "tfenn_tensor_convention_v2"


def test_default_dtype_does_not_change_stf_coordinates() -> None:
    """Check the default dtype only controls the returned storage type."""
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        single = stf_basis(6)
        torch.set_default_dtype(torch.float64)
        double = stf_basis(6)
    finally:
        torch.set_default_dtype(original)
    torch.testing.assert_close(single, double.float(), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("invalid_dtype", (False, 0))
def test_explicit_invalid_dtype_never_uses_the_default(invalid_dtype: object) -> None:
    """Check false values cannot masquerade as an omitted dtype."""
    with pytest.raises(TypeError, match="dtype"):
        stf_basis(2, dtype=invalid_dtype)


def test_rank_zero_runtime_outputs_keep_zero_derivative_graphs() -> None:
    """Check constant rank zero formulas remain connected to their inputs."""
    generator = torch.Generator().manual_seed(19)
    vector = torch.randn(
        (3,), dtype=DTYPE, generator=generator, requires_grad=True
    )
    matrix = torch.randn(
        (3, 3), dtype=DTYPE, generator=generator, requires_grad=True
    )
    symmetric_power = symmetric_power_components(vector, 0)
    stf_power = stf_power_components(vector, 0)
    symmetric_action = symmetric_representation(matrix, 0)
    stf_action = stf_representation(matrix, 0)
    for value in (symmetric_power, stf_power, symmetric_action, stf_action):
        assert value.requires_grad

    vector_gradient = torch.autograd.grad(
        symmetric_power.sum() + stf_power.sum(),
        vector,
    )[0]
    matrix_gradient = torch.autograd.grad(
        symmetric_action.sum() + stf_action.sum(),
        matrix,
    )[0]
    torch.testing.assert_close(vector_gradient, torch.zeros_like(vector))
    torch.testing.assert_close(matrix_gradient, torch.zeros_like(matrix))
    assert torch.autograd.gradcheck(
        lambda value: stf_power_components(value, 0),
        (vector,),
    )
    assert torch.autograd.gradcheck(
        lambda value: stf_representation(value, 0),
        (matrix,),
    )


def test_public_stf_basis_cannot_mutate_the_master() -> None:
    """Check public basis tensors do not expose cached master storage."""
    expected = stf_basis(2, dtype=torch.float64)
    changed = stf_basis(2, dtype=torch.float64)
    changed.zero_()
    torch.testing.assert_close(
        stf_basis(2, dtype=torch.float64), expected, atol=0.0, rtol=0.0
    )


def test_rank_two_equal_pivot_keeps_one_phase_across_precisions() -> None:
    """Cover the former equal magnitude pivot sign reversal."""
    double = stf_basis(2, dtype=torch.float64)
    single = stf_basis(2, dtype=torch.float32)
    assert double[3, 3] > 0.0
    assert single[3, 3] > 0.0
    torch.testing.assert_close(single, double.float(), atol=0.0, rtol=0.0)


def test_rank_two_basis_matches_the_versioned_golden() -> None:
    """Lock the rank two phase and column order independently."""
    expected = torch.tensor(
        (
            (math.sqrt(2.0 / 3.0), 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0),
            (-1.0 / math.sqrt(6.0), 0.0, 0.0, 1.0 / math.sqrt(2.0), 0.0),
            (0.0, 0.0, 0.0, 0.0, 1.0),
            (-1.0 / math.sqrt(6.0), 0.0, 0.0, -1.0 / math.sqrt(2.0), 0.0),
        ),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        stf_basis(2, dtype=torch.float64), expected, atol=2e-15, rtol=2e-15
    )


def test_float_coordinate_cast_preserves_the_physical_tensor() -> None:
    """Check a coordinate cast retains its represented dense tensor."""
    generator = torch.Generator().manual_seed(23)
    coordinates = torch.randn(13, dtype=torch.float32, generator=generator)
    dense_single = stf_to_dense(coordinates, 6).double()
    dense_double = stf_to_dense(coordinates.double(), 6)
    torch.testing.assert_close(dense_single, dense_double, atol=2e-7, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rank", (0, 1, 2, 6))
def test_cuda_stf_basis_is_an_exact_master_cast(rank: int) -> None:
    """Check CUDA coordinates use a direct master basis cast."""
    master = stf_basis(rank, dtype=torch.float64, device="cpu")
    actual = stf_basis(rank, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(actual, master.float().cuda(), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("rank", range(7))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_stf_basis_is_orthonormal_and_trace_free(rank: int, dtype: torch.dtype) -> None:
    """Check the defining STF space properties at representative ranks."""
    tolerance = _numeric_tolerance(dtype)
    basis = stf_basis(rank, dtype=dtype)
    assert basis.shape == (symmetric_dimension(rank), stf_dimension(rank))
    torch.testing.assert_close(
        basis.T @ basis,
        torch.eye(stf_dimension(rank), dtype=dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        trace_matrix(rank, dtype=dtype) @ basis,
        torch.zeros(
            (math.comb(rank, 2) if rank >= 2 else 0, stf_dimension(rank)),
            dtype=dtype,
        ),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("rank", range(7))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_stf_dense_coordinates_round_trip(rank: int, dtype: torch.dtype) -> None:
    """Check dense and STF coordinate transformations agree."""
    tolerance = _numeric_tolerance(dtype)
    generator = torch.Generator().manual_seed(31 + rank)
    coordinates = torch.randn(
        (4, stf_dimension(rank)), dtype=dtype, generator=generator
    )
    dense = stf_to_dense(coordinates, rank)
    torch.testing.assert_close(
        dense_to_stf(dense, rank),
        coordinates,
        atol=tolerance,
        rtol=tolerance,
    )
    symmetric_coordinates = stf_to_symmetric(coordinates, rank)
    torch.testing.assert_close(
        symmetric_to_stf(symmetric_coordinates, rank),
        coordinates,
        atol=tolerance,
        rtol=tolerance,
    )

    symmetric = torch.randn(
        (4, symmetric_dimension(rank)), dtype=dtype, generator=generator
    )
    projected = project_symmetric(symmetric, rank)
    torch.testing.assert_close(
        project_symmetric(projected, rank),
        projected,
        atol=tolerance,
        rtol=tolerance,
    )
    if rank >= 2:
        torch.testing.assert_close(
            trace_matrix(rank, dtype=dtype) @ projected.T,
            torch.zeros((symmetric_dimension(rank - 2), len(projected)), dtype=dtype),
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("rank", range(7))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_stf_representation_is_orthogonal_and_multiplicative(
    rank: int, dtype: torch.dtype
) -> None:
    """Check identity, inverse, orthogonality, and multiplication."""
    tolerance = _numeric_tolerance(dtype)
    first = rotation((0.23, -0.31, 0.17)).to(dtype)
    second = rotation((-0.11, 0.29, 0.37)).to(dtype)
    represented_identity = stf_representation(torch.eye(3, dtype=dtype), rank)
    represented_first = stf_representation(first, rank)
    represented_second = stf_representation(second, rank)
    represented_product = stf_representation(first @ second, rank)
    represented_inverse = stf_representation(first.mT, rank)
    torch.testing.assert_close(
        represented_identity,
        torch.eye(stf_dimension(rank), dtype=dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        represented_product,
        represented_first @ represented_second,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        represented_first.T @ represented_first,
        torch.eye(stf_dimension(rank), dtype=dtype),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        represented_inverse,
        represented_first.T,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("rank", range(7))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_stf_representation_matches_dense_cartesian_action(
    rank: int, dtype: torch.dtype
) -> None:
    """Compare STF coordinates with an independent dense index action."""
    generator = torch.Generator().manual_seed(181 + rank)
    coordinates = torch.randn(stf_dimension(rank), dtype=dtype, generator=generator)
    action = rotation((0.17, -0.29, 0.13)).to(dtype)
    transformed = stf_to_dense(stf_representation(action, rank) @ coordinates, rank)
    physical = stf_to_dense(coordinates, rank)
    for axis in range(rank):
        physical = torch.tensordot(action, physical, dims=((1,), (axis,)))
        physical = physical.movedim(0, axis)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-9
    torch.testing.assert_close(
        transformed,
        physical,
        atol=tolerance,
        rtol=tolerance,
    )


def test_symmetric_representation_accepts_general_finite_matrices() -> None:
    """Check the full symmetric action remains defined beyond rotations."""
    first = torch.tensor(
        ((1.2, 0.1, 0.0), (0.0, 0.8, 0.2), (0.3, 0.0, 1.1)), dtype=DTYPE
    )
    second = torch.tensor(
        ((0.9, 0.0, 0.1), (0.2, 1.3, 0.0), (0.0, 0.1, 0.7)), dtype=DTYPE
    )
    torch.testing.assert_close(
        symmetric_representation(first @ second, 3),
        symmetric_representation(first, 3) @ symmetric_representation(second, 3),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (torch.diag(torch.tensor((1.1, 1.0, 1.0), dtype=DTYPE)), "orthogonal"),
        (torch.diag(torch.tensor((-1.0, 1.0, 1.0), dtype=DTYPE)), "proper"),
    ),
)
def test_stf_representation_rejects_unsupported_matrix_actions(
    matrix: torch.Tensor, message: str
) -> None:
    """Check only proper orthogonal actions enter the STF representation."""
    with pytest.raises(ValueError, match=message):
        stf_representation(matrix, 2, validate=True)


@pytest.mark.parametrize("invalid", (float("nan"), float("inf")))
def test_stf_representation_rejects_nonfinite_matrices(invalid: float) -> None:
    """Check nonfinite rotation entries fail clearly."""
    matrix = torch.eye(3, dtype=DTYPE)
    matrix[0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        stf_representation(matrix, 2, validate=True)


def test_rotation_validation_tolerance_is_explicit() -> None:
    """Check callers can pass an independent rotation tolerance."""
    matrix = torch.eye(3, dtype=DTYPE)
    matrix[0, 0] += 2e-8
    with pytest.raises(ValueError, match="orthogonal"):
        stf_representation(matrix, 2, validate=True)
    represented = stf_representation(
        matrix,
        2,
        validate=True,
        rotation_atol=5e-8,
        rotation_rtol=5e-8,
    )
    assert torch.isfinite(represented).all()


def test_stf_symmetric_product_is_covariant() -> None:
    """Check STF product commutes with a simultaneous rotation."""
    generator = torch.Generator().manual_seed(47)
    left = torch.randn(stf_dimension(2), dtype=DTYPE, generator=generator)
    right = torch.randn(stf_dimension(3), dtype=DTYPE, generator=generator)
    action = rotation((0.13, 0.27, -0.21))
    product = stf_symmetric_product(left, right, 2, 3)
    transformed = stf_symmetric_product(
        stf_representation(action, 2) @ left,
        stf_representation(action, 3) @ right,
        2,
        3,
    )
    torch.testing.assert_close(
        transformed,
        stf_representation(action, 5) @ product,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_rank_one_coupling_has_cartesian_golden_normalization(
    dtype: torch.dtype,
) -> None:
    """Lock scalar and vector channels to dot and right handed cross products."""
    left = torch.tensor((1.0, 2.0, 3.0), dtype=dtype)
    right = torch.tensor((4.0, 5.0, 6.0), dtype=dtype)
    scalar = stf_tensor_coupling(left, right, 1, 1, 0)
    vector = stf_tensor_coupling(left, right, 1, 1, 1)
    tolerance = _numeric_tolerance(dtype)
    torch.testing.assert_close(
        scalar,
        torch.dot(left, right).reshape(1),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        vector,
        torch.cross(left, right, dim=0),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    ("rank_left", "rank_right", "output_rank"),
    ((1, 1, 0), (1, 1, 1), (2, 3, 1), (2, 3, 4), (3, 4, 5)),
)
def test_coupling_swap_phase_is_frozen(
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> None:
    """Check the documented exchange phase for representative channels."""
    generator = torch.Generator().manual_seed(
        307 + 31 * rank_left + 7 * rank_right + output_rank
    )
    left = torch.randn(
        stf_dimension(rank_left), dtype=DTYPE, generator=generator
    )
    right = torch.randn(
        stf_dimension(rank_right), dtype=DTYPE, generator=generator
    )
    forward = stf_tensor_coupling(
        left, right, rank_left, rank_right, output_rank
    )
    reverse = stf_tensor_coupling(
        right, left, rank_right, rank_left, output_rank
    )
    phase = (-1.0) ** (rank_left + rank_right - output_rank)
    torch.testing.assert_close(forward, phase * reverse, atol=ATOL, rtol=RTOL)


def test_project_rank_limit_fails_before_dense_allocation() -> None:
    """Check unsupported ranks fail before an exponential dense tensor exists."""
    unsupported = MAX_STF_RANK + 1
    with pytest.raises(ValueError, match="before allocation"):
        stf_basis(unsupported)
    with pytest.raises(ValueError, match="before allocation"):
        stf_to_dense(
            torch.empty(2 * unsupported + 1, dtype=DTYPE),
            unsupported,
        )


@pytest.mark.parametrize(
    ("rank_left", "rank_right", "output_rank"),
    _ALL_COUPLING_CHANNELS,
)
def test_cached_coupling_matches_the_dense_reference(
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> None:
    """Check every rank through six preserves normalization and phase."""
    dtype = torch.float64
    generator = torch.Generator().manual_seed(
        101 + 31 * rank_left + 7 * rank_right + output_rank
    )
    left = torch.randn(stf_dimension(rank_left), dtype=dtype, generator=generator)
    right = torch.randn(stf_dimension(rank_right), dtype=dtype, generator=generator)
    actual = stf_tensor_coupling(left, right, rank_left, rank_right, output_rank)
    expected = _stf_tensor_coupling_reference(
        left, right, rank_left, rank_right, output_rank
    )
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("rank_left", "rank_right", "output_rank"),
    _FLOAT32_COUPLING_CHANNELS,
)
def test_float32_cached_coupling_matches_the_dense_reference(
    rank_left: int,
    rank_right: int,
    output_rank: int,
) -> None:
    """Check representative single precision coefficient casts."""
    generator = torch.Generator().manual_seed(
        211 + 31 * rank_left + 7 * rank_right + output_rank
    )
    left = torch.randn(
        stf_dimension(rank_left), dtype=torch.float32, generator=generator
    )
    right = torch.randn(
        stf_dimension(rank_right), dtype=torch.float32, generator=generator
    )
    actual = stf_tensor_coupling(left, right, rank_left, rank_right, output_rank)
    expected = _stf_tensor_coupling_reference(
        left, right, rank_left, rank_right, output_rank
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_stf_tensor_coupling_broadcasts_batches() -> None:
    """Check coefficient contraction follows tensor batch broadcasting."""
    generator = torch.Generator().manual_seed(137)
    left = torch.randn((2, 1, 5), dtype=DTYPE, generator=generator)
    right = torch.randn((1, 3, 7), dtype=DTYPE, generator=generator)
    actual = stf_tensor_coupling(left, right, 2, 3, 4)
    expected = _stf_tensor_coupling_reference(left, right, 2, 3, 4)
    assert actual.shape == (2, 3, 9)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_stf_tensor_coupling_gradient() -> None:
    """Run gradcheck on both coefficient contraction inputs."""
    generator = torch.Generator().manual_seed(149)
    left = torch.randn(5, dtype=DTYPE, generator=generator, requires_grad=True)
    right = torch.randn(7, dtype=DTYPE, generator=generator, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda first, second: stf_tensor_coupling(first, second, 2, 3, 2),
        (left, right),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_coupling_matches_cpu_coefficients() -> None:
    """Check CUDA coupling is a device cast of canonical coefficients."""
    generator = torch.Generator().manual_seed(151)
    left = torch.randn(5, dtype=torch.float32, generator=generator)
    right = torch.randn(7, dtype=torch.float32, generator=generator)
    expected = stf_tensor_coupling(left, right, 2, 3, 4)
    actual = stf_tensor_coupling(left.cuda(), right.cuda(), 2, 3, 4).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("rank", (2, 6))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_roundtrip_and_representation_match_cpu(
    rank: int, dtype: torch.dtype
) -> None:
    """Check CUDA runtime paths use canonical CPU constants by cast."""
    generator = torch.Generator().manual_seed(163 + rank)
    coordinates = torch.randn(stf_dimension(rank), dtype=dtype, generator=generator)
    action = rotation((0.11, -0.19, 0.23)).to(dtype)
    expected_representation = stf_representation(action, rank)
    actual_representation = stf_representation(action.cuda(), rank).cpu()
    tolerance = _numeric_tolerance(dtype)
    torch.testing.assert_close(
        actual_representation,
        expected_representation,
        atol=tolerance,
        rtol=tolerance,
    )
    dense = stf_to_dense(coordinates.cuda(), rank)
    recovered = dense_to_stf(dense, rank).cpu()
    torch.testing.assert_close(
        recovered,
        coordinates,
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("output_rank", (1, 2, 3, 4, 5))
def test_every_stf_tensor_coupling_channel_is_covariant(output_rank: int) -> None:
    """Check delta and epsilon coupling channels commute with rotations."""
    generator = torch.Generator().manual_seed(53 + output_rank)
    left = torch.randn(stf_dimension(2), dtype=DTYPE, generator=generator)
    right = torch.randn(stf_dimension(3), dtype=DTYPE, generator=generator)
    action = rotation((-0.17, 0.29, 0.21))
    coupled = stf_tensor_coupling(left, right, 2, 3, output_rank)
    transformed = stf_tensor_coupling(
        stf_representation(action, 2) @ left,
        stf_representation(action, 3) @ right,
        2,
        3,
        output_rank,
    )
    torch.testing.assert_close(
        transformed,
        stf_representation(action, output_rank) @ coupled,
        atol=ATOL,
        rtol=RTOL,
    )


def test_stf_tensor_coupling_preserves_an_empty_batch() -> None:
    """Check an empty broadcast batch returns the expected shape."""
    left = torch.empty((0, stf_dimension(2)), dtype=DTYPE)
    right = torch.ones(stf_dimension(3), dtype=DTYPE)
    result = stf_tensor_coupling(left, right, 2, 3, 2)
    assert result.shape == (0, stf_dimension(2))


@pytest.mark.parametrize("rank", (2, 6))
def test_stf_representation_gradient(rank: int) -> None:
    """Run gradcheck on D_l along an SO(3) rotation vector."""
    tangent = torch.tensor((0.17, -0.23, 0.11), dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: stf_representation(rotation_from_rotvec(value), rank),
        (tangent,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )
