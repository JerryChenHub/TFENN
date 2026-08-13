"""Test every bounded C family on Benzene and C60 generator systems.

The catalog covers linear maps, pure A symmetric powers, mixed A and B maps,
independent B slots, repeated B symmetric powers, typed outputs, and trivial
scalar outputs.  Every compiled artifact keeps its complete Hom basis.  Group
elements used below are finite words in the supplied generators.  No group
closure, averaging, irrep table, or Clebsch Gordon data is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from TFENN.tensor_math import (
    CSignature,
    CSlot,
    CovariantCompilation,
    RegisteredCovariant,
    TRIVIAL_SCALAR,
    TypeKey,
    compile_anchors,
    compile_covariant_basis,
    normalized_symmetric_power_representation,
    tensor_product_representation,
)
from TFENN.tensor_math._compiler_utils import canonical_nullspace

from ._covariant_cases import (
    CovariantCase,
    benzene_covariant_case,
    c60_covariant_case,
)
from ._groups import c60_generators


DTYPE = torch.float64
MAX_CONSTRAINT_ENTRIES = 10_000_000
A = TypeKey("A")


@dataclass(frozen=True)
class CompiledFamily:
    """Bind one diagnostic label to a complete C artifact and runtime."""

    label: str
    signature: CSignature
    artifact: CovariantCompilation
    runtime: RegisteredCovariant


@dataclass(frozen=True)
class CompiledMolecule:
    """Store all bounded C families for one generator defined molecule."""

    case: CovariantCase
    families: tuple[CompiledFamily, ...]


def distinct(name: str, key: TypeKey) -> CSlot:
    """Create one independent runtime slot."""
    return CSlot(name, key)


def symmetric(name: str, key: TypeKey, power: int) -> CSlot:
    """Create one repeated runtime variable."""
    return CSlot(name, key, power, "symmetric_power")


def family_signatures(case: CovariantCase) -> tuple[tuple[str, CSignature], ...]:
    """Instantiate every bounded source topology and output kind."""
    result: list[tuple[str, CSignature]] = []
    b_keys = tuple(item.key for item in case.manifest)

    result.extend(
        (
            ("linear_A_to_A", CSignature(A, (distinct("a", A),))),
            (
                "pure_Sym2A_to_A",
                CSignature(A, (symmetric("a", A, 2),)),
            ),
            (
                "pure_Sym3A_to_A",
                CSignature(A, (symmetric("a", A, 3),)),
            ),
            (
                "pure_Sym2A_to_scalar",
                CSignature(TRIVIAL_SCALAR, (symmetric("a", A, 2),)),
            ),
        )
    )

    for b_index, b_key in enumerate(b_keys):
        prefix = f"B{b_index}"
        result.extend(
            (
                (
                    f"linear_A_to_{prefix}",
                    CSignature(b_key, (distinct("a", A),)),
                ),
                (
                    f"linear_{prefix}_to_A",
                    CSignature(A, (distinct("b", b_key),)),
                ),
                (
                    f"linear_{prefix}_to_{prefix}",
                    CSignature(b_key, (distinct("b", b_key),)),
                ),
                (
                    f"pure_Sym2A_to_{prefix}",
                    CSignature(b_key, (symmetric("a", A, 2),)),
                ),
                (
                    f"mixed_A_{prefix}_to_A",
                    CSignature(
                        A,
                        (distinct("a", A), distinct("b", b_key)),
                    ),
                ),
                (
                    f"mixed_A_{prefix}_to_{prefix}",
                    CSignature(
                        b_key,
                        (distinct("a", A), distinct("b", b_key)),
                    ),
                ),
                (
                    f"mixed_A_{prefix}_to_scalar",
                    CSignature(
                        TRIVIAL_SCALAR,
                        (distinct("a", A), distinct("b", b_key)),
                    ),
                ),
                (
                    f"mixed_Sym2A_{prefix}_to_A",
                    CSignature(
                        A,
                        (symmetric("a", A, 2), distinct("b", b_key)),
                    ),
                ),
                (
                    f"mixed_A_Sym2{prefix}_to_scalar",
                    CSignature(
                        TRIVIAL_SCALAR,
                        (distinct("a", A), symmetric("b", b_key, 2)),
                    ),
                ),
                (
                    f"independent_{prefix}_{prefix}_to_A",
                    CSignature(
                        A,
                        (distinct("left", b_key), distinct("right", b_key)),
                    ),
                ),
                (
                    f"independent_{prefix}_{prefix}_to_{prefix}",
                    CSignature(
                        b_key,
                        (distinct("left", b_key), distinct("right", b_key)),
                    ),
                ),
                (
                    f"independent_{prefix}_{prefix}_to_scalar",
                    CSignature(
                        TRIVIAL_SCALAR,
                        (distinct("left", b_key), distinct("right", b_key)),
                    ),
                ),
                (
                    f"self_Sym2{prefix}_to_A",
                    CSignature(A, (symmetric("b", b_key, 2),)),
                ),
                (
                    f"self_Sym2{prefix}_to_{prefix}",
                    CSignature(b_key, (symmetric("b", b_key, 2),)),
                ),
                (
                    f"self_Sym2{prefix}_to_scalar",
                    CSignature(TRIVIAL_SCALAR, (symmetric("b", b_key, 2),)),
                ),
                (
                    f"self_Sym3{prefix}_to_scalar",
                    CSignature(TRIVIAL_SCALAR, (symmetric("b", b_key, 3),)),
                ),
            )
        )

    if len(b_keys) == 2:
        first, second = b_keys
        result.extend(
            (
                (
                    "linear_B0_to_B1",
                    CSignature(second, (distinct("b", first),)),
                ),
                (
                    "linear_B1_to_B0",
                    CSignature(first, (distinct("b", second),)),
                ),
                (
                    "independent_B0_B1_to_B0",
                    CSignature(
                        first,
                        (distinct("left", first), distinct("right", second)),
                    ),
                ),
                (
                    "independent_B1_B0_to_B0",
                    CSignature(
                        first,
                        (distinct("left", second), distinct("right", first)),
                    ),
                ),
                (
                    "independent_B0_B1_to_B1",
                    CSignature(
                        second,
                        (distinct("left", first), distinct("right", second)),
                    ),
                ),
                (
                    "independent_B1_B0_to_B1",
                    CSignature(
                        second,
                        (distinct("left", second), distinct("right", first)),
                    ),
                ),
                (
                    "independent_B1_B1_to_B0",
                    CSignature(
                        first,
                        (distinct("left", second), distinct("right", second)),
                    ),
                ),
                (
                    "self_Sym2B0_to_B1",
                    CSignature(second, (symmetric("b", first, 2),)),
                ),
            )
        )
    labels = tuple(label for label, _ in result)
    assert len(labels) == len(set(labels))
    return tuple(result)


@pytest.fixture(scope="module", params=("benzene", "c60"))
def compiled_molecule(request: pytest.FixtureRequest) -> CompiledMolecule:
    """Compile every selected artifact once using only ordered generators."""
    case = (
        benzene_covariant_case()
        if request.param == "benzene"
        else c60_covariant_case()
    )
    families = []
    for label, signature in family_signatures(case):
        try:
            artifact = compile_covariant_basis(
                case.catalog,
                signature,
                max_constraint_entries=MAX_CONSTRAINT_ENTRIES,
            )
        except Exception as error:
            raise RuntimeError(f"failed to compile {case.name} signature {label}") from error
        families.append(
            CompiledFamily(label, signature, artifact, RegisteredCovariant(artifact))
        )
    return CompiledMolecule(case, tuple(families))


def slot_actions(case: CovariantCase, signature: CSignature) -> Tensor:
    """Build the source action in signature order from public operations."""
    actions = []
    for slot in signature.inputs:
        block_actions = case.catalog.resolve(slot.type_key).actions
        actions.append(
            block_actions
            if slot.mode == "distinct"
            else normalized_symmetric_power_representation(block_actions, slot.power)
        )
    source = actions[0]
    for action in actions[1:]:
        source = tensor_product_representation(source, action)
    return source


def target_actions(case: CovariantCase, signature: CSignature) -> Tensor:
    """Return typed target actions or the trivial scalar action."""
    generator_count = len(case.catalog.generator_system.names)
    if signature.output == TRIVIAL_SCALAR:
        return torch.ones((generator_count, 1, 1), dtype=DTYPE)
    return case.catalog.resolve(signature.output).actions


def word_action(actions: Tensor, word: tuple[int, ...]) -> Tensor:
    """Multiply one ordered generator word without constructing group closure."""
    result = torch.eye(actions.shape[-1], dtype=actions.dtype)
    for index in word:
        result = actions[index] @ result
    return result


def live_inputs(
    case: CovariantCase,
    signature: CSignature,
    seed: int,
    *,
    requires_grad: bool,
) -> dict[str, Tensor]:
    """Create one finite input for every explicitly named runtime slot."""
    generator = torch.Generator().manual_seed(seed)
    return {
        slot.name: torch.randn(
            case.catalog.resolve(slot.type_key).representation_dim,
            dtype=DTYPE,
            generator=generator,
            requires_grad=requires_grad,
        )
        for slot in signature.inputs
    }


def test_all_complete_basis_elements_obey_generator_constraints(
    compiled_molecule: CompiledMolecule,
) -> None:
    """Check every alpha against every supplied generator equation."""
    maximum = 0.0
    dimensions = {}
    for family in compiled_molecule.families:
        artifact = family.artifact
        source = slot_actions(compiled_molecule.case, family.signature)
        target = target_actions(compiled_molecule.case, family.signature)
        basis = artifact.basis
        left = torch.einsum("boi,qij->qboj", basis, source)
        right = torch.einsum("qok,bki->qboi", target, basis)
        residual = float((left - right).abs().max()) if basis.numel() else 0.0
        maximum = max(maximum, residual)
        dimensions[family.label] = artifact.basis_dimension
        assert residual < 2.0e-8, (
            f"group residual failure molecule={compiled_molecule.case.name} "
            f"signature={family.label} dtype={basis.dtype} residual={residual}"
        )
    print(
        {
            "molecule": compiled_molecule.case.name,
            "signature_count": len(compiled_molecule.families),
            "maximum_generator_residual": maximum,
            "hom_dimensions": dimensions,
        }
    )


def test_all_complete_basis_elements_are_functionally_covariant(
    compiled_molecule: CompiledMolecule,
) -> None:
    """Transform random live inputs by finite generator words."""
    words = ((), (0,), (1,), (0, 1, 0), (1, 0, 1, 0, 0))
    maximum = 0.0
    for family_index, family in enumerate(compiled_molecule.families):
        inputs = live_inputs(
            compiled_molecule.case,
            family.signature,
            7100 + family_index,
            requires_grad=False,
        )
        reference = family.runtime.evaluate_basis(inputs)
        for word in words:
            transformed = {}
            for slot in family.signature.inputs:
                actions = compiled_molecule.case.catalog.resolve(slot.type_key).actions
                transformed[slot.name] = inputs[slot.name] @ word_action(actions, word).T
            actual = family.runtime.evaluate_basis(transformed)
            output_action = word_action(
                target_actions(compiled_molecule.case, family.signature),
                word,
            )
            expected = reference @ output_action.T
            residual = float((actual - expected).abs().max()) if actual.numel() else 0.0
            maximum = max(maximum, residual)
            assert residual < 3.0e-8, (
                f"functional residual failure molecule={compiled_molecule.case.name} "
                f"signature={family.label} word={word} dtype={actual.dtype} "
                f"residual={residual}"
            )
    print(
        {
            "molecule": compiled_molecule.case.name,
            "maximum_word_residual": maximum,
        }
    )


def test_every_runtime_slot_has_finite_full_basis_gradients(
    compiled_molecule: CompiledMolecule,
) -> None:
    """Use every basis element in one scalar float64 gradcheck objective."""
    maximum_gradient = 0.0
    for family_index, family in enumerate(compiled_molecule.families):
        inputs = live_inputs(
            compiled_molecule.case,
            family.signature,
            8100 + family_index,
            requires_grad=True,
        )
        names = tuple(slot.name for slot in family.signature.inputs)
        values = tuple(inputs[name] for name in names)
        basis_dimension = family.runtime.basis.shape[0]
        output_dimension = family.runtime.basis.shape[1]

        if basis_dimension == 0:
            coefficients = torch.empty(0, dtype=DTYPE)
            zero = family.runtime.apply_coefficients(inputs, coefficients).sum()
            gradients = torch.autograd.grad(zero, values)
            assert all(bool(torch.isfinite(value).all()) for value in gradients)
            assert all(float(value.abs().max()) == 0.0 for value in gradients)
            continue

        weights = torch.linspace(
            0.25,
            1.25,
            basis_dimension * output_dimension,
            dtype=DTYPE,
        ).reshape(basis_dimension, output_dimension)

        def objective(*arguments: Tensor) -> Tensor:
            current = dict(zip(names, arguments))
            return (family.runtime.evaluate_basis(current) * weights).sum()

        assert torch.autograd.gradcheck(
            objective,
            values,
            eps=1.0e-6,
            atol=4.0e-5,
            rtol=4.0e-4,
        ), (
            f"gradcheck failure molecule={compiled_molecule.case.name} "
            f"signature={family.label} dtype={DTYPE}"
        )
        gradients = torch.autograd.grad(objective(*values), values)
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        for name, gradient in zip(names, gradients):
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            assert 0.0 < gradient_norm < 1.0e8, (
                f"gradient norm failure molecule={compiled_molecule.case.name} "
                f"signature={family.label} slot={name} dtype={DTYPE} "
                f"gradient_norm={gradient_norm}"
            )
        maximum_gradient = max(
            maximum_gradient,
            max(float(value.abs().max()) for value in gradients),
        )
    assert 0.0 < maximum_gradient < 1.0e8
    print(
        {
            "molecule": compiled_molecule.case.name,
            "maximum_absolute_gradient": maximum_gradient,
        }
    )


def test_c60_rank_ten_is_generated_and_not_a_primitive_pose_block() -> None:
    """Keep the C60 manifest aligned with generator compiled anchor provenance."""
    compilation = compile_anchors(c60_generators(), output_ranks=(6, 10))
    rank_six = compilation.blocks[6].dimensions
    rank_ten = compilation.blocks[10].dimensions
    assert (rank_six.fixed, rank_six.generated, rank_six.primitive) == (1, 0, 1)
    assert (rank_ten.fixed, rank_ten.generated, rank_ten.primitive) == (1, 1, 0)
    assert compilation.output_ranks == (6,)


def test_nullspace_fallback_keeps_the_original_rank_rule(monkeypatch) -> None:
    """Force the robust path without changing tolerance or canonical order."""
    matrix = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=DTYPE,
    )
    expected = canonical_nullspace(matrix, 1.0e-10, 1.0e-12)

    def fail_singular_vectors(*args, **kwargs):
        raise torch.linalg.LinAlgError("forced singular vector failure")

    monkeypatch.setattr(torch.linalg, "svd", fail_singular_vectors)
    actual = canonical_nullspace(matrix, 1.0e-10, 1.0e-12)
    assert actual.dimension == expected.dimension == 1
    assert actual.numerical_rank == expected.numerical_rank == 2
    torch.testing.assert_close(actual.singular_values, expected.singular_values)
    torch.testing.assert_close(actual.basis, expected.basis, atol=1.0e-14, rtol=0.0)

    near_threshold = torch.diag(
        torch.tensor((1.0, 2.0e-10, 5.0e-11), dtype=DTYPE)
    )
    classified = canonical_nullspace(near_threshold, 1.0e-10, 1.0e-12)
    assert classified.numerical_rank == 2
    assert classified.dimension == 1
    torch.testing.assert_close(
        classified.basis,
        torch.tensor(((0.0,), (0.0,), (1.0,)), dtype=DTYPE),
        atol=1.0e-14,
        rtol=0.0,
    )
