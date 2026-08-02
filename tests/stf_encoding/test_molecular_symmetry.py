"""Test the proper rotation groups I of C60 and T of methane."""

from __future__ import annotations

import itertools
import math

import torch

from TFENN.stf_encoding import STFEncoder, rotation_from_rotvec


DTYPE = torch.float64
ATOL = 3e-9
RTOL = 3e-9


def _c60_vertices() -> torch.Tensor:
    """Return the sixty vertices of an ideal truncated icosahedron."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    seeds = (
        (0.0, 1.0, 3.0 * phi),
        (1.0, 2.0 + phi, 2.0 * phi),
        (phi, 2.0, 1.0 + 2.0 * phi),
    )
    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    vertices = []
    for seed in seeds:
        for permutation in permutations:
            permuted = tuple(seed[index] for index in permutation)
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                vertices.append(
                    torch.tensor(
                        tuple(sign * value for sign, value in zip(signs, permuted)),
                        dtype=DTYPE,
                    )
                )
    return torch.unique(torch.stack(vertices), dim=0)


def _c60_rotation_generators() -> torch.Tensor:
    """Return order five and order three generators of rotational I."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    fivefold_axis = torch.tensor((0.0, 1.0, phi), dtype=DTYPE)
    fivefold_axis = fivefold_axis / torch.linalg.vector_norm(fivefold_axis)
    fivefold_turn = rotation_from_rotvec(
        fivefold_axis * (2.0 * math.pi / 5.0)
    )
    coordinate_cycle = torch.tensor(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=DTYPE,
    )
    return torch.stack((fivefold_turn, coordinate_cycle))


def _methane_hydrogen_vertices() -> torch.Tensor:
    """Return four unit bond directions of ideal methane."""
    return torch.tensor(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ),
        dtype=DTYPE,
    ) / math.sqrt(3.0)


def _methane_rotation_generators() -> torch.Tensor:
    """Return order two and order three generators of rotational T."""
    half_turn = rotation_from_rotvec(
        torch.tensor((math.pi, 0.0, 0.0), dtype=DTYPE)
    )
    coordinate_cycle = torch.tensor(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=DTYPE,
    )
    return torch.stack((half_turn, coordinate_cycle))


def _assert_point_permutations(
    points: torch.Tensor, rotations: torch.Tensor
) -> None:
    """Assert every rotation bijectively permutes a point cloud."""
    transformed = torch.einsum("gij,nj->gni", rotations, points)
    differences = transformed[:, :, None, :] - points[None, None, :, :]
    distances = torch.linalg.vector_norm(differences, dim=-1)
    errors, assignments = distances.min(dim=-1)
    assert errors.max().item() < ATOL
    expected = torch.arange(len(points), device=points.device).expand_as(assignments)
    assert torch.equal(assignments.sort(dim=-1).values, expected)


def _assert_so3_generator_orders(
    generators: torch.Tensor, orders: tuple[int, ...]
) -> None:
    """Assert generators lie in SO(3) and have the requested orders."""
    identity = torch.eye(3, dtype=generators.dtype, device=generators.device)
    torch.testing.assert_close(
        generators.transpose(-1, -2) @ generators,
        identity.expand_as(generators),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        torch.linalg.det(generators),
        torch.ones(len(generators), dtype=generators.dtype),
        atol=ATOL,
        rtol=RTOL,
    )
    for generator, order in zip(generators, orders):
        torch.testing.assert_close(
            torch.linalg.matrix_power(generator, order),
            identity,
            atol=ATOL,
            rtol=RTOL,
        )


def _assert_full_encoded_orbit(
    encoder: STFEncoder, rotation: torch.Tensor
) -> None:
    """Assert the code is constant on the complete right group orbit."""
    orbit = rotation @ encoder.group_elements
    expected = encoder.encode_rotation(rotation).expand(len(orbit), -1)
    assert torch.linalg.matrix_norm(
        orbit - rotation, dim=(-2, -1)
    ).min().item() < ATOL
    torch.testing.assert_close(
        encoder.encode_rotation(orbit),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_c60_uses_the_exact_rotational_icosahedral_stabilizer() -> None:
    """Verify ideal C60 geometry and its globally certified STF encoder."""
    vertices = _c60_vertices()
    generators = _c60_rotation_generators()
    encoder = STFEncoder.from_generators(generators)

    assert vertices.shape == (60, 3)
    radii = torch.linalg.vector_norm(vertices, dim=-1)
    torch.testing.assert_close(
        radii,
        radii[0].expand_as(radii),
        atol=ATOL,
        rtol=RTOL,
    )
    pairwise = torch.linalg.vector_norm(
        vertices[:, None, :] - vertices[None, :, :], dim=-1
    )
    edges = torch.isclose(pairwise, torch.tensor(2.0, dtype=DTYPE), atol=ATOL)
    assert bool((edges.sum(dim=-1) == 3).all())
    assert edges.sum().item() // 2 == 90
    _assert_so3_generator_orders(generators, (5, 3))
    _assert_point_permutations(vertices, encoder.group_elements)
    assert encoder.certificate.group_name == "I"
    assert encoder.certificate.group_order == 60
    assert encoder.certificate.exact is True
    assert encoder.certificate.complete_fixed_spaces is True
    assert encoder.ranks == (6,)
    assert encoder.anchors[6].shape == (13, 1)
    assert encoder.encoding_dimension == 13
    assert encoder.certificate.generator_constraint_residual < ATOL

    rotation = rotation_from_rotvec(
        torch.tensor((0.23, -0.31, 0.19), dtype=DTYPE)
    )
    _assert_full_encoded_orbit(encoder, rotation)
    jacobian = encoder.jacobian(rotation)
    assert torch.linalg.matrix_rank(jacobian).item() == 3
    torch.testing.assert_close(
        encoder.jacobian_pseudoinverse(rotation) @ jacobian,
        torch.eye(3, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )


def test_methane_uses_the_exact_rotational_tetrahedral_stabilizer() -> None:
    """Verify ideal methane geometry and its globally certified STF encoder."""
    hydrogens = _methane_hydrogen_vertices()
    generators = _methane_rotation_generators()
    encoder = STFEncoder.from_generators(generators)

    gram = hydrogens @ hydrogens.T
    expected_gram = torch.full((4, 4), -1.0 / 3.0, dtype=DTYPE)
    expected_gram.fill_diagonal_(1.0)
    torch.testing.assert_close(gram, expected_gram, atol=ATOL, rtol=RTOL)
    _assert_so3_generator_orders(generators, (2, 3))
    _assert_point_permutations(hydrogens, encoder.group_elements)
    assert encoder.certificate.group_name == "T"
    assert encoder.certificate.group_order == 12
    assert encoder.certificate.exact is True
    assert encoder.certificate.complete_fixed_spaces is True
    assert encoder.ranks == (3,)
    assert encoder.anchors[3].shape == (7, 1)
    assert encoder.encoding_dimension == 7
    assert encoder.certificate.generator_constraint_residual < ATOL

    rotation = rotation_from_rotvec(
        torch.tensor((-0.17, 0.29, 0.37), dtype=DTYPE)
    )
    _assert_full_encoded_orbit(encoder, rotation)
    assert encoder.verify_gradients(rotation) is True
