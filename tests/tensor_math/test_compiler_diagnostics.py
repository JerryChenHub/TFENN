"""Validate deterministic compiler rank decisions and diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from TFENN.tensor_math import PoseEncoder, a_representation, compile_anchors
from TFENN.tensor_math.intertwiner_compiler import (
    IntertwinerCompilation,
    compile_intertwiners,
)
from TFENN.tensor_math.stf_space import STF_BASIS_VERSION

from ._groups import benzene_generators


def _assert_invalid_tolerance(value: object, exception: type[Exception]) -> None:
    """Check both compilers reject one invalid tolerance value."""
    identity = torch.eye(3, dtype=torch.float64)[None]
    with pytest.raises(exception):
        compile_intertwiners(identity, identity, tolerance=value)
    for name in (
        "atol",
        "nullspace_atol",
        "nullspace_rtol",
        "rotation_atol",
        "rotation_rtol",
    ):
        with pytest.raises(exception):
            compile_anchors(identity, ranks=(2,), **{name: value})


@pytest.mark.parametrize("value", (False, True))
def test_compilers_reject_bool_tolerances(value: bool) -> None:
    """Check bool is never interpreted as a numerical tolerance."""
    _assert_invalid_tolerance(value, TypeError)


@pytest.mark.parametrize(
    "value",
    (0.0, -1.0, float("nan"), float("inf"), -float("inf")),
)
def test_compilers_reject_nonpositive_or_nonfinite_tolerances(
    value: float,
) -> None:
    """Check every explicit numerical tolerance is finite and positive."""
    _assert_invalid_tolerance(value, ValueError)


def test_anchor_rotation_and_nullspace_tolerances_are_independent() -> None:
    """Check a loose rank threshold cannot disable rotation validation."""
    invalid = 1.001 * torch.eye(3, dtype=torch.float64)[None]
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        compile_anchors(invalid, ranks=(2,), nullspace_atol=1.0)


def test_compiler_threshold_has_absolute_and_relative_terms() -> None:
    """Check a small scale constraint is not compared with unit scale."""
    rho_in = torch.tensor([[[1e-12]]], dtype=torch.float64)
    rho_out = torch.zeros_like(rho_in)
    compilation = compile_intertwiners(
        rho_in,
        rho_out,
        atol=1e-20,
        rtol=1e-3,
        return_compilation=True,
    )
    assert isinstance(compilation, IntertwinerCompilation)
    expected = 1e-20 + 1e-3 * float(compilation.singular_values[0])
    assert compilation.threshold == pytest.approx(expected)
    assert compilation.dimension == 0
    assert compilation.residual == 0.0
    assert compilation.singular_value_gap > compilation.threshold


def test_d6_anchor_and_intertwiner_golden_dimensions() -> None:
    """Check D6 fixed spaces, primitive spaces, and all linear Hom spaces."""
    generators = benzene_generators()
    anchors = compile_anchors(generators, ranks=(6,))
    assert [anchors.blocks[rank].dimensions.fixed for rank in range(1, 7)] == [
        0,
        1,
        0,
        1,
        0,
        2,
    ]
    assert [anchors.blocks[rank].dimensions.primitive for rank in range(1, 7)] == [
        0,
        1,
        0,
        0,
        0,
        1,
    ]
    assert anchors.generators.device.type == "cpu"
    assert anchors.generators.dtype == torch.float64
    assert anchors.basis_version == STF_BASIS_VERSION
    for block in anchors.blocks.values():
        sigma_max = float(block.singular_values[0])
        expected_threshold = anchors.nullspace_atol + anchors.nullspace_rtol * sigma_max
        assert block.nullspace_threshold == pytest.approx(expected_threshold)
        assert block.singular_value_gap > 0.5

    encoder = PoseEncoder(anchors)
    rho_a = a_representation(generators)
    rho_b = encoder.representation(generators)
    spaces = {
        "AA": (rho_a, rho_a, 2),
        "AB": (rho_a, rho_b, 4),
        "BA": (rho_b, rho_a, 4),
        "BB": (rho_b, rho_b, 30),
    }
    results: dict[str, IntertwinerCompilation] = {}
    for name, (rho_in, rho_out, expected_dimension) in spaces.items():
        result = compile_intertwiners(
            rho_in,
            rho_out,
            return_compilation=True,
        )
        assert isinstance(result, IntertwinerCompilation)
        assert result.dimension == expected_dimension
        assert result.basis.shape[0] == expected_dimension
        assert result.basis.device.type == "cpu"
        assert result.basis.dtype == torch.float64
        assert result.residual < 5e-13
        assert result.singular_value_gap > 0.5
        assert result.basis_version == STF_BASIS_VERSION
        flattened = result.basis.reshape(expected_dimension, -1)
        torch.testing.assert_close(
            flattened @ flattened.T,
            torch.eye(expected_dimension, dtype=torch.float64),
            atol=2e-12,
            rtol=2e-12,
        )
        results[name] = result

    plain = compile_intertwiners(rho_a, rho_a)
    assert isinstance(plain, torch.Tensor)
    torch.testing.assert_close(plain, results["AA"].basis)

    reversed_result = compile_intertwiners(
        rho_a.flip(0),
        rho_a.flip(0),
        return_compilation=True,
    )
    assert isinstance(reversed_result, IntertwinerCompilation)
    torch.testing.assert_close(
        reversed_result.basis,
        results["AA"].basis,
        atol=2e-12,
        rtol=2e-12,
    )


def test_float32_inputs_compile_on_cpu_float64() -> None:
    """Check source precision does not control compiler arithmetic."""
    generators = benzene_generators().to(torch.float32)
    anchors = compile_anchors(generators, ranks=(6,))
    assert anchors.generators.device.type == "cpu"
    assert anchors.generators.dtype == torch.float64
    assert anchors.blocks[6].primitive_basis.dtype == torch.float64
    assert anchors.blocks[6].dimensions.primitive == 1

    result = compile_intertwiners(
        generators,
        generators,
        return_compilation=True,
    )
    assert isinstance(result, IntertwinerCompilation)
    assert result.dimension == 2
    assert result.basis.device.type == "cpu"
    assert result.basis.dtype == torch.float64
    assert math.isfinite(result.singular_value_gap)
