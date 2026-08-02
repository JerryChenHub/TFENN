"""Validate generator inputs using ideal C60 and methane geometry."""

from __future__ import annotations

import torch
from torch import Tensor

from TFENN.tensor_math import PoseEncoder, compile_anchors

from ._groups import (
    ATOL,
    DTYPE,
    RTOL,
    c60_generators,
    c60_vertices,
    methane_generators,
    methane_vertices,
    rotation,
)


def _assert_generator_orders(generators: Tensor, orders: tuple[int, ...]) -> None:
    """Check proper rotations and their stated finite orders."""
    identity = torch.eye(3, dtype=DTYPE)
    torch.testing.assert_close(
        generators.mT @ generators,
        identity.expand_as(generators),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        torch.linalg.det(generators),
        torch.ones(len(generators), dtype=DTYPE),
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


def _assert_point_permutations(points: Tensor, generators: Tensor) -> None:
    """Check every generator bijectively permutes the molecular sites."""
    transformed = torch.einsum("gij,nj->gni", generators, points)
    distances = torch.linalg.vector_norm(
        transformed[:, :, None] - points[None, None], dim=-1
    )
    errors, assignments = distances.min(dim=-1)
    assert errors.max().item() < ATOL
    expected = torch.arange(len(points)).expand_as(assignments)
    assert torch.equal(assignments.sort(dim=-1).values, expected)


def _assert_pose_law(generators: Tensor, rank: int, dimension: int) -> None:
    """Check the molecular generators compile the requested pose law."""
    compilation = compile_anchors(generators, ranks=(rank,))
    encoder = PoseEncoder(compilation)
    pose = rotation((0.19, -0.31, 0.23))
    left = generators[1] @ generators[0]
    right = generators[0] @ generators[1]
    assert compilation.encoding_dimension == dimension
    torch.testing.assert_close(
        encoder.encode(left @ pose @ right),
        encoder.representation(left) @ encoder.encode(pose),
        atol=ATOL,
        rtol=RTOL,
    )


def test_c60_generators_match_truncated_icosahedron_geometry() -> None:
    """Check C60 geometry and its unique primitive rank six anchor."""
    vertices = c60_vertices()
    generators = c60_generators()
    assert vertices.shape == (60, 3)
    pairwise = torch.linalg.vector_norm(vertices[:, None] - vertices[None, :], dim=-1)
    edges = torch.isclose(
        pairwise,
        torch.tensor(2.0, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )
    assert bool((edges.sum(dim=-1) == 3).all())
    assert edges.sum().item() // 2 == 90
    _assert_generator_orders(generators, (5, 3))
    _assert_point_permutations(vertices, generators)
    _assert_pose_law(generators, rank=6, dimension=13)


def test_methane_generators_match_tetrahedral_geometry() -> None:
    """Check methane geometry and its unique primitive rank three anchor."""
    vertices = methane_vertices()
    generators = methane_generators()
    expected = torch.full((4, 4), -1.0 / 3.0, dtype=DTYPE)
    expected.fill_diagonal_(1.0)
    torch.testing.assert_close(vertices @ vertices.T, expected, atol=ATOL, rtol=RTOL)
    _assert_generator_orders(generators, (2, 3))
    _assert_point_permutations(vertices, generators)
    _assert_pose_law(generators, rank=3, dimension=7)
