from __future__ import annotations

import math

import pytest
import torch

from TFENN.stf_encoding import (
    STFEncoder,
    d6_benzene_encoder,
    invariant_anchors,
    rotation_from_rotvec,
    stabilizer_certificate,
    stf_basis,
    stf_representation,
    trace_matrix,
)


DTYPE = torch.float64
ATOL = 2e-10
RTOL = 2e-10


def _d6_generators() -> torch.Tensor:
    return torch.stack(
        (
            rotation_from_rotvec(
                torch.tensor((0.0, 0.0, math.pi / 3.0), dtype=DTYPE)
            ),
            rotation_from_rotvec(
                torch.tensor((math.pi, 0.0, 0.0), dtype=DTYPE)
            ),
        )
    )


def _d6_elements() -> torch.Tensor:
    generators = _d6_generators()
    axial_turn, half_turn = generators.unbind()
    identity = torch.eye(3, dtype=DTYPE)
    powers = [identity]
    for _ in range(5):
        powers.append(powers[-1] @ axial_turn)
    return torch.stack((*powers, *(power @ half_turn for power in powers)))


def _rotation(vector: tuple[float, float, float]) -> torch.Tensor:
    return rotation_from_rotvec(torch.tensor(vector, dtype=DTYPE))


def _polyhedral_generators() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    cycle = torch.tensor(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=DTYPE,
    )
    tetrahedral = torch.stack((_rotation((math.pi, 0.0, 0.0)), cycle))
    octahedral = torch.stack(
        (_rotation((math.pi / 2.0, 0.0, 0.0)), _rotation((0.0, 0.0, math.pi / 2.0)))
    )
    icosahedral_half_turn = 0.5 * torch.tensor(
        (
            (-1.0, phi, 1.0 / phi),
            (phi, 1.0 / phi, 1.0),
            (1.0 / phi, 1.0, -phi),
        ),
        dtype=DTYPE,
    )
    icosahedral = torch.stack((icosahedral_half_turn, cycle))
    return tetrahedral, octahedral, icosahedral


@pytest.mark.parametrize(("rank", "dimension"), ((2, 5), (6, 13)))
def test_stf_basis_has_expected_dimension_trace_kernel_and_orthogonality(
    rank: int,
    dimension: int,
) -> None:
    basis = stf_basis(rank, dtype=DTYPE)
    symmetric_dimension = math.comb(rank + 2, 2)

    assert basis.shape == (symmetric_dimension, dimension)
    torch.testing.assert_close(
        trace_matrix(rank, dtype=DTYPE) @ basis,
        torch.zeros(math.comb(rank, 2), dimension, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        basis.T @ basis,
        torch.eye(dimension, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("rank", (2, 6))
def test_stf_representation_is_orthogonal_and_respects_multiplication(
    rank: int,
) -> None:
    first = _rotation((0.23, -0.31, 0.17))
    second = _rotation((-0.11, 0.29, 0.37))
    first_action = stf_representation(first, rank)
    second_action = stf_representation(second, rank)
    product_action = stf_representation(first @ second, rank)

    torch.testing.assert_close(
        product_action,
        first_action @ second_action,
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        first_action.T @ first_action,
        torch.eye(2 * rank + 1, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )


def test_generator_anchors_satisfy_every_invariance_constraint() -> None:
    generators = _d6_generators()
    anchors = invariant_anchors(generators, ranks=(2, 6))

    assert set(anchors) == {2, 6}
    assert anchors[2].shape == (5, 1)
    assert anchors[6].shape == (13, 2)
    for rank, anchor in anchors.items():
        torch.testing.assert_close(
            anchor.T @ anchor,
            torch.eye(anchor.shape[1], dtype=DTYPE),
            atol=ATOL,
            rtol=RTOL,
        )
        for generator in generators:
            residual = stf_representation(generator, rank) @ anchor - anchor
            torch.testing.assert_close(
                residual,
                torch.zeros_like(residual),
                atol=ATOL,
                rtol=RTOL,
            )


def test_d6_encoder_has_the_analytic_eighteen_component_code() -> None:
    encoder = d6_benzene_encoder(dtype=DTYPE)
    rotation = _rotation((0.19, -0.27, 0.41))
    position = torch.tensor((1.2, -0.8, 0.3), dtype=DTYPE)

    rotation_code = encoder.encode_rotation(rotation)
    full_code = encoder.encode(position, rotation)

    assert encoder.ranks == (2, 6)
    assert encoder.encoding_dimension == 18
    assert rotation_code.shape == (18,)
    assert full_code.shape == (21,)
    torch.testing.assert_close(full_code[:3], position, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        full_code[3:],
        rotation_code,
        atol=0.0,
        rtol=0.0,
    )


def test_d6_code_is_right_invariant_and_left_covariant() -> None:
    encoder = d6_benzene_encoder(dtype=DTYPE)
    rotation = _rotation((0.19, -0.27, 0.41))
    left = _rotation((-0.13, 0.38, 0.22))
    reference = encoder.encode_rotation(rotation)

    for right in _d6_elements():
        torch.testing.assert_close(
            encoder.encode_rotation(rotation @ right),
            reference,
            atol=ATOL,
            rtol=RTOL,
        )

    expected = torch.cat(
        (
            stf_representation(left, 2) @ reference[:5],
            stf_representation(left, 6) @ reference[5:],
        )
    )
    torch.testing.assert_close(
        encoder.encode_rotation(left @ rotation),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_generic_encoder_from_generators_preserves_generator_constraints() -> None:
    generators = _d6_generators()
    encoder = STFEncoder.from_generators(generators)
    rotation = _rotation((0.29, 0.07, -0.34))
    reference = encoder.encode_rotation(rotation)

    assert encoder.ranks == (2, 6)
    assert encoder.encoding_dimension == 31
    assert encoder.certificate.exact is True
    assert encoder.certificate.certifying_ranks == (2, 6)
    for generator in generators:
        torch.testing.assert_close(
            encoder.encode_rotation(rotation @ generator),
            reference,
            atol=ATOL,
            rtol=RTOL,
        )


def test_global_certificate_rejects_an_incomplete_rank_selection() -> None:
    with pytest.raises(ValueError, match="missing required rank 6"):
        STFEncoder.from_generators(_d6_generators(), ranks=(2,))


def test_global_certificate_tracks_extra_complete_blocks() -> None:
    encoder = STFEncoder.from_generators(_d6_generators(), ranks=(2, 4, 6))

    assert encoder.certificate.exact is True
    assert encoder.certificate.ranks == (2, 4, 6)
    assert encoder.certificate.certifying_ranks == (2, 6)
    assert encoder.certificate.encoding_dimension == encoder.encoding_dimension


def test_float32_generators_use_dtype_adaptive_tolerance() -> None:
    encoder = STFEncoder.from_generators(_d6_generators().to(torch.float32))

    assert encoder.certificate.exact is True
    assert encoder.certificate.group_name == "D6"


def test_certificate_rejects_nonfinite_anchor_blocks() -> None:
    generators = _d6_generators()
    anchors = invariant_anchors(generators, ranks=(2, 6))
    anchors[2][0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite values"):
        stabilizer_certificate(generators, anchors)


def test_all_finite_so3_families_receive_exact_global_certificates() -> None:
    tetrahedral, octahedral, icosahedral = _polyhedral_generators()
    cases = (
        (_rotation((0.0, 0.0, math.pi / 2.0))[None], "C4", (1, 4)),
        (torch.stack((_rotation((math.pi, 0.0, 0.0)), _rotation((0.0, math.pi, 0.0)))), "D2", (2,)),
        (tetrahedral, "T", (3,)),
        (octahedral, "O", (4,)),
        (icosahedral, "I", (6,)),
    )
    rotation = _rotation((0.17, -0.21, 0.36))

    for generators, label, ranks in cases:
        encoder = STFEncoder.from_generators(generators)
        reference = encoder.encode_rotation(rotation)
        assert encoder.certificate.group_name == label
        assert encoder.certificate.exact is True
        assert encoder.ranks == ranks
        for generator in generators:
            torch.testing.assert_close(
                encoder.encode_rotation(rotation @ generator),
                reference,
                atol=ATOL,
                rtol=RTOL,
            )


def test_d6_encoder_exposes_an_exact_global_certificate() -> None:
    certificate = d6_benzene_encoder(dtype=DTYPE).certificate

    assert certificate.exact is True
    assert certificate.group_order == 12
    assert certificate.encoding_dimension == 18


def test_numerical_inverse_fiber_contains_the_original_rotation() -> None:
    encoder = d6_benzene_encoder(dtype=DTYPE)
    rotation = _rotation((0.24, -0.32, 0.17))
    code = encoder.encode_rotation(rotation)
    result = encoder.inverse_fiber(
        code,
        reference_rotation=rotation,
        num_starts=8,
        adam_steps=80,
    )

    assert result.representative.shape == (3, 3)
    assert result.rotations.shape == (12, 3, 3)
    assert result.code_residual < 1e-8
    assert result.inclusion_error < 1e-8
    torch.testing.assert_close(
        encoder.encode_rotation(result.rotations),
        code.expand(12, 18),
        atol=1e-8,
        rtol=1e-8,
    )
    distance = (result.rotations - rotation).abs().amax(dim=(-2, -1))
    assert distance.min().item() < 1e-8


def test_inverse_fiber_accepts_code_with_a_different_dtype() -> None:
    encoder = d6_benzene_encoder(dtype=DTYPE)
    rotation = _rotation((0.24, -0.32, 0.17)).to(torch.float32)
    code = encoder.encode_rotation(rotation)
    result = encoder.inverse_fiber(code, reference_rotation=rotation)

    assert result.rotations.dtype == torch.float32
    assert result.code_residual < 1e-4
    assert result.inclusion_error < 1e-4


def test_jacobian_pseudoinverse_recovers_the_three_dimensional_tangent() -> None:
    encoder = d6_benzene_encoder(dtype=DTYPE)
    rotation = _rotation((0.21, -0.16, 0.37))
    jacobian = encoder.jacobian(rotation)
    pseudoinverse = encoder.jacobian_pseudoinverse(rotation)

    assert jacobian.shape == (18, 3)
    assert pseudoinverse.shape == (3, 18)
    assert torch.linalg.matrix_rank(jacobian).item() == 3
    torch.testing.assert_close(
        pseudoinverse @ jacobian,
        torch.eye(3, dtype=DTYPE),
        atol=2e-9,
        rtol=2e-9,
    )
    torch.testing.assert_close(
        jacobian @ pseudoinverse @ jacobian,
        jacobian,
        atol=2e-9,
        rtol=2e-9,
    )


def test_rotation_encoding_passes_gradcheck() -> None:
    encoder = d6_benzene_encoder(dtype=DTYPE)
    base = _rotation((0.12, -0.23, 0.31))
    tangent = torch.tensor(
        (0.017, -0.013, 0.019),
        dtype=DTYPE,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        lambda value: encoder.encode_rotation(rotation_from_rotvec(value) @ base),
        (tangent,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )
    assert encoder.verify_gradients(base) is True
