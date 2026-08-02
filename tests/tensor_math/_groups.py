"""Generator fixtures used only by tensor math tests."""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor


DTYPE = torch.float64
ATOL = 3e-9
RTOL = 3e-9


def skew(vector: Tensor) -> Tensor:
    """Return the skew matrix of a three vector."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def rotation_from_rotvec(vector: Tensor) -> Tensor:
    """Map a rotation vector to SO(3)."""
    return torch.matrix_exp(skew(vector))


def rotation(values: tuple[float, float, float]) -> Tensor:
    """Return one double precision rotation from numeric coordinates."""
    return rotation_from_rotvec(torch.tensor(values, dtype=DTYPE))


def benzene_generators() -> Tensor:
    """Return proper D6 generators for ideal benzene."""
    return torch.stack(
        (
            rotation((0.0, 0.0, math.pi / 3.0)),
            rotation((math.pi, 0.0, 0.0)),
        )
    )


def c60_generators() -> Tensor:
    """Return order five and order three generators of rotational I."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    axis = torch.tensor((0.0, 1.0, phi), dtype=DTYPE)
    axis = axis / torch.linalg.vector_norm(axis)
    fivefold = rotation_from_rotvec(axis * (2.0 * math.pi / 5.0))
    threefold = torch.tensor(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=DTYPE,
    )
    return torch.stack((fivefold, threefold))


def c60_vertices() -> Tensor:
    """Return analytic vertices of an ideal truncated icosahedron."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    seeds = (
        (0.0, 1.0, 3.0 * phi),
        (1.0, 2.0 + phi, 2.0 * phi),
        (phi, 2.0, 1.0 + 2.0 * phi),
    )
    permutations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    vertices = [
        tuple(sign * seed[index] for sign, index in zip(signs, permutation))
        for seed in seeds
        for permutation in permutations
        for signs in itertools.product((-1.0, 1.0), repeat=3)
    ]
    return torch.unique(torch.tensor(vertices, dtype=DTYPE), dim=0)


def methane_generators() -> Tensor:
    """Return order two and order three generators of rotational T."""
    return torch.stack(
        (
            rotation((math.pi, 0.0, 0.0)),
            torch.tensor(
                ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                dtype=DTYPE,
            ),
        )
    )


def methane_vertices() -> Tensor:
    """Return the four ideal methane bond directions."""
    return torch.tensor(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ),
        dtype=DTYPE,
    ) / math.sqrt(3.0)


def octahedral_generators() -> Tensor:
    """Return two quarter turns generating rotational octahedral symmetry."""
    return torch.stack(
        (
            rotation((math.pi / 2.0, 0.0, 0.0)),
            rotation((0.0, 0.0, math.pi / 2.0)),
        )
    )
