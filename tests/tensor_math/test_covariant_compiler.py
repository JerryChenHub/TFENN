"""Validate symmetric lifts and complete generator constrained Hom bases."""

from __future__ import annotations

import math

import pytest
import torch

import TFENN.tensor_math.covariant_registry as registry_module
from TFENN.tensor_math import (
    CSignature,
    CSlot,
    TRIVIAL_SCALAR,
    TypeKey,
    compile_covariant_basis,
    normalized_symmetric_power_basis,
    normalized_symmetric_power_representation,
    symmetric_representation,
    tensor_product_representation,
)

from ._covariant_cases import benzene_covariant_case, methane_covariant_case
from ._groups import ATOL, DTYPE, RTOL, benzene_generators


A = TypeKey("A")
B0 = TypeKey("B", 0)
B1 = TypeKey("B", 1)


def distinct(name: str, key: TypeKey) -> CSlot:
    """Return one independent runtime slot."""
    return CSlot(name, key)


def symmetric_square(name: str, key: TypeKey) -> CSlot:
    """Return one genuinely repeated runtime slot."""
    return CSlot(name, key, 2, "symmetric_power")


def test_normalized_symmetric_power_basis_has_frozen_golden_order() -> None:
    """Check occupation order, normalization, arbitrary dimension, and clones."""
    basis = normalized_symmetric_power_basis(2, 2)
    root_two = math.sqrt(2.0)
    expected = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0 / root_two, 0.0),
            (0.0, 1.0 / root_two, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=DTYPE,
    )
    torch.testing.assert_close(basis, expected, atol=0.0, rtol=0.0)
    for dimension in (2, 3, 5):
        lift = normalized_symmetric_power_basis(dimension, 2)
        assert lift.shape == (
            dimension**2,
            math.comb(dimension + 1, 2),
        )
        torch.testing.assert_close(
            lift.T @ lift,
            torch.eye(lift.shape[1], dtype=DTYPE),
            atol=2e-15,
            rtol=2e-15,
        )
    basis.zero_()
    torch.testing.assert_close(
        normalized_symmetric_power_basis(2, 2),
        expected,
        atol=0.0,
        rtol=0.0,
    )


def test_symmetric_power_action_matches_existing_normalized_action() -> None:
    """Check the general d equals three construction against the frozen oracle."""
    generators = benzene_generators()
    actual = normalized_symmetric_power_representation(generators, 2)
    expected = symmetric_representation(generators, 2)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    first, second = actual
    composed = normalized_symmetric_power_representation(
        (generators[0] @ generators[1]).unsqueeze(0),
        2,
    )[0]
    torch.testing.assert_close(composed, first @ second, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "case_factory,signature,expected",
    (
        (
            benzene_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", B0))),
            5,
        ),
        (
            benzene_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", B1))),
            11,
        ),
        (
            benzene_covariant_case,
            CSignature(B0, (symmetric_square("a", A),)),
            4,
        ),
        (
            benzene_covariant_case,
            CSignature(B0, (distinct("left", A), distinct("right", A))),
            5,
        ),
        (
            benzene_covariant_case,
            CSignature(B0, (distinct("left", B0), distinct("right", B0))),
            11,
        ),
        (
            benzene_covariant_case,
            CSignature(B1, (distinct("left", B0), distinct("right", B0))),
            28,
        ),
        (
            benzene_covariant_case,
            CSignature(TRIVIAL_SCALAR, (symmetric_square("a", A),)),
            2,
        ),
        (
            benzene_covariant_case,
            CSignature(A, (CSlot("a", A, 3, "symmetric_power"),)),
            4,
        ),
        (
            methane_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", B0))),
            5,
        ),
        (
            methane_covariant_case,
            CSignature(B0, (symmetric_square("a", A),)),
            3,
        ),
        (
            methane_covariant_case,
            CSignature(B0, (distinct("left", B0), distinct("right", B0))),
            29,
        ),
    ),
)
def test_complete_hom_dimensions_and_diagnostics(
    case_factory,
    signature: CSignature,
    expected: int,
) -> None:
    """Lock complete finite group Hom dimensions for two molecules."""
    case = case_factory()
    artifact = compile_covariant_basis(case.catalog, signature)
    assert artifact.basis_dimension == expected
    assert artifact.basis.shape == (
        expected,
        artifact.output_dimension,
        artifact.effective_input_dimension,
    )
    flattened = artifact.basis.reshape(expected, -1)
    torch.testing.assert_close(
        flattened @ flattened.T,
        torch.eye(expected, dtype=DTYPE),
        atol=3e-12,
        rtol=3e-12,
    )
    assert artifact.residual <= artifact.threshold
    assert artifact.cost.full_basis_dimension == expected
    assert artifact.cost.active_basis_dimension == expected
    assert artifact.cost.coefficient_count == expected
    assert artifact.cost.reduction_ratio == 1.0

    source_actions = []
    for slot in signature.inputs:
        block = case.catalog.resolve(slot.type_key)
        actions = block.actions
        if slot.mode == "symmetric_power":
            actions = normalized_symmetric_power_representation(actions, slot.power)
        source_actions.append(actions)
    source = source_actions[0]
    for actions in source_actions[1:]:
        source = tensor_product_representation(source, actions)
    if signature.output == TRIVIAL_SCALAR:
        target = torch.ones((source.shape[0], 1, 1), dtype=DTYPE)
    else:
        target = case.catalog.resolve(signature.output).actions
    differences = torch.einsum("boi,qij->qboj", artifact.basis, source)
    differences = differences - torch.einsum(
        "qop,bpi->qboi",
        target,
        artifact.basis,
    )
    residual = (
        float(torch.linalg.vector_norm(differences, dim=(-2, -1)).amax())
        if artifact.basis_dimension
        else 0.0
    )
    assert residual == pytest.approx(artifact.residual, abs=5e-14)
    assert residual <= artifact.threshold


def test_distinct_equal_type_slots_are_not_a_symmetric_power() -> None:
    """Confirm antisymmetric source coordinates are absent only when requested."""
    case = benzene_covariant_case()
    distinct_signature = CSignature(
        B0,
        (distinct("a_i", A), distinct("a_j", A)),
    )
    repeated_signature = CSignature(B0, (symmetric_square("a", A),))
    distinct_artifact = compile_covariant_basis(case.catalog, distinct_signature)
    repeated_artifact = compile_covariant_basis(case.catalog, repeated_signature)
    assert distinct_artifact.effective_input_dimension == 9
    assert repeated_artifact.effective_input_dimension == 6
    assert distinct_artifact.basis_dimension == 5
    assert repeated_artifact.basis_dimension == 4


def test_zero_hom_space_preserves_shape_and_spectral_diagnostics() -> None:
    """Check a zero basis remains a complete compiler result rather than evidence loss."""
    case = methane_covariant_case()
    signature = CSignature(TRIVIAL_SCALAR, (distinct("a", A),))
    artifact = compile_covariant_basis(case.catalog, signature)
    assert artifact.basis_dimension == 0
    assert artifact.basis.shape == (0, 1, 3)
    assert artifact.singular_values.numel() > 0
    assert artifact.residual == 0.0
    assert artifact.cost.reduction_ratio == 1.0


def test_compiler_calls_existing_backend_once_and_preserves_basis(monkeypatch) -> None:
    """Ensure no group expansion, duplicated solver, or post compilation rotation."""
    case = benzene_covariant_case()
    signature = CSignature(A, (distinct("a", A), distinct("b", B0)))
    original = registry_module.compile_intertwiners
    calls = []

    def wrapped(source, target, **kwargs):
        result = original(source, target, **kwargs)
        calls.append((source.clone(), target.clone(), result))
        return result

    monkeypatch.setattr(registry_module, "compile_intertwiners", wrapped)
    artifact = compile_covariant_basis(case.catalog, signature)
    assert len(calls) == 1
    source, target, backend = calls[0]
    assert source.shape[0] == target.shape[0] == 2
    torch.testing.assert_close(artifact.basis, backend.basis, atol=0.0, rtol=0.0)
    assert artifact.basis_dimension == backend.dimension
    assert artifact.threshold == backend.threshold
    assert artifact.singular_value_gap == backend.singular_value_gap


def test_dense_guard_runs_before_tensor_actions_or_solver(monkeypatch) -> None:
    """Reject predicted allocations before constructing any dense Kronecker value."""
    case = benzene_covariant_case()
    signature = CSignature(B0, (symmetric_square("a", A),))

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation happened before the guard")

    monkeypatch.setattr(
        registry_module,
        "normalized_symmetric_power_representation",
        forbidden,
    )
    monkeypatch.setattr(registry_module, "compile_intertwiners", forbidden)
    with pytest.raises(ValueError, match="allocation guard"):
        compile_covariant_basis(
            case.catalog,
            signature,
            max_constraint_entries=1,
        )


def test_signature_identity_and_bounded_scope_error_contracts() -> None:
    """Bind slot name, order, mode, and degree without silent reinterpretation."""
    case = benzene_covariant_case()
    first = CSignature(A, (distinct("a", A), distinct("b", B0)))
    renamed = CSignature(A, (distinct("x", A), distinct("b", B0)))
    swapped = CSignature(A, (distinct("b", B0), distinct("a", A)))
    artifacts = tuple(
        compile_covariant_basis(case.catalog, signature)
        for signature in (first, renamed, swapped)
    )
    assert len({item.artifact_fingerprint for item in artifacts}) == 3
    repeated = compile_covariant_basis(case.catalog, first)
    assert repeated.artifact_fingerprint == artifacts[0].artifact_fingerprint
    torch.testing.assert_close(
        repeated.basis,
        artifacts[0].basis,
        atol=0.0,
        rtol=0.0,
    )
    with pytest.raises(ValueError):
        CSlot("a", A, 2, "distinct")
    with pytest.raises(ValueError):
        CSlot("a", A, 1, "symmetric_power")
    with pytest.raises(ValueError, match="total degree"):
        compile_covariant_basis(
            case.catalog,
            CSignature(
                B0,
                (symmetric_square("a", A), symmetric_square("b", B0)),
            ),
        )
    with pytest.raises(ValueError, match="solver"):
        compile_covariant_basis(case.catalog, first, solver="matrix_free")
