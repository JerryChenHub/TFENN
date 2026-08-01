"""Analytic STF anchors for the proper benzene rotation group."""

from __future__ import annotations

import math
from fractions import Fraction

import torch

from .basis import stf_basis, symmetric_multi_indices


def d6_generators(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the analytic generators Rz(pi/3) and Rx(pi)."""
    half = torch.as_tensor(0.5, dtype=dtype, device=device)
    sine = torch.as_tensor(math.sqrt(3.0) / 2.0, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    a = torch.stack(
        (
            half,
            -sine,
            zero,
            sine,
            half,
            zero,
            zero,
            zero,
            one,
        )
    ).reshape(3, 3)
    b = torch.diag(torch.tensor((1.0, -1.0, -1.0), dtype=dtype, device=device))
    return torch.stack((a, b))


def d6_group_elements(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the twelve matrices a^k and a^k b by their closed formulas."""
    angles = torch.arange(6, dtype=dtype, device=device) * (math.pi / 3.0)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    rotations = torch.stack(
        (
            cosine,
            -sine,
            zero,
            sine,
            cosine,
            zero,
            zero,
            zero,
            one,
        ),
        dim=-1,
    ).reshape(6, 3, 3)
    flip = torch.diag(torch.tensor((1.0, -1.0, -1.0), dtype=dtype, device=device))
    return torch.cat((rotations, rotations @ flip), dim=0)


def _radial_power_terms(power: int) -> dict[tuple[int, int, int], Fraction]:
    """Expand (x squared plus y squared plus z squared) to an integer power."""
    terms: dict[tuple[int, int, int], Fraction] = {}
    factorial = math.factorial(power)
    for a in range(power + 1):
        for b in range(power - a + 1):
            c = power - a - b
            coefficient = factorial // (
                math.factorial(a) * math.factorial(b) * math.factorial(c)
            )
            terms[(2 * a, 2 * b, 2 * c)] = Fraction(coefficient)
    return terms


def _analytic_k6_polynomial() -> dict[tuple[int, int, int], Fraction]:
    """Return coefficients of the analytic hexagonal harmonic polynomial."""
    terms = {
        (6, 0, 0): Fraction(1, 32),
        (4, 2, 0): Fraction(-15, 32),
        (2, 4, 0): Fraction(15, 32),
        (0, 6, 0): Fraction(-1, 32),
    }
    legendre_terms = ((6, 231), (4, -315), (2, 105), (0, -5))
    scale = Fraction(-5, 231 * 16)
    for z_power, coefficient in legendre_terms:
        radial_power = (6 - z_power) // 2
        for alpha, radial_coefficient in _radial_power_terms(radial_power).items():
            shifted = (alpha[0], alpha[1], alpha[2] + z_power)
            terms[shifted] = terms.get(shifted, Fraction()) + (
                scale * coefficient * radial_coefficient
            )
    return terms


def _polynomial_to_symmetric_components(
    coefficients: dict[tuple[int, int, int], Fraction],
    rank: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Convert contraction polynomial coefficients to normalized components."""
    values = []
    for alpha in symmetric_multi_indices(rank):
        multiplicity = math.factorial(rank)
        for count in alpha:
            multiplicity //= math.factorial(count)
        coefficient = float(coefficients.get(alpha, Fraction()))
        values.append(coefficient / math.sqrt(multiplicity))
    return torch.tensor(values, dtype=dtype, device=device)


def d6_analytic_anchors(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[int, torch.Tensor]:
    """Return the analytic rank two and rank six STF anchor columns."""
    k2_symmetric = torch.tensor(
        (1.0 / 6.0, 0.0, 0.0, 1.0 / 6.0, 0.0, -1.0 / 3.0),
        dtype=dtype,
        device=device,
    )
    k6_symmetric = _polynomial_to_symmetric_components(
        _analytic_k6_polynomial(), 6, dtype=dtype, device=device
    )
    k2 = stf_basis(2, dtype=dtype, device=device).T @ k2_symmetric
    k6 = stf_basis(6, dtype=dtype, device=device).T @ k6_symmetric
    return {2: k2[:, None], 6: k6[:, None]}


__all__ = [
    "d6_analytic_anchors",
    "d6_generators",
    "d6_group_elements",
]
