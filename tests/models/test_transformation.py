"""Test invariant controlled typed pathway transformations."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from TFENN.models import InvariantGate
from TFENN.tensor_math import (
    BBlockManifest,
    CSignature,
    CSlot,
    GeneratorSystem,
    RegisteredCovariant,
    TypeKey,
    build_type_catalog,
    compile_covariant_basis,
)


DTYPE = torch.float64
A = TypeKey("A")
B0 = TypeKey("B", 0)


def benzene_catalog():
    """Build the proper rotational benzene catalog used by every test."""
    angle = math.pi / 3.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    generators = torch.tensor(
        (
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
        ),
        dtype=DTYPE,
    )
    system = GeneratorSystem(("sixfold", "twofold"), generators)
    manifest = (BBlockManifest(0, 2, (0,), 1, "benzene_b_rank_2"),)
    return build_type_catalog(system, manifest)


@pytest.fixture(scope="module")
def paths():
    """Compile the three representative pathway artifacts once."""
    catalog = benzene_catalog()
    signatures = {
        "a_to_a": CSignature(A, (CSlot("a", A),)),
        "a_to_b": CSignature(B0, (CSlot("a", A),)),
        "bb_to_b": CSignature(
            B0,
            (CSlot("left", B0), CSlot("right", B0)),
        ),
    }
    artifacts = {
        name: compile_covariant_basis(catalog, signature)
        for name, signature in signatures.items()
    }
    return catalog, signatures, artifacts


def test_default_mlp_and_representative_path_shapes(paths) -> None:
    """Cover A to A, A to B, and two B inputs to B."""
    _, signatures, artifacts = paths
    torch.manual_seed(11)
    cases = (
        (
            "a_to_a",
            {"a": 3},
            {"a": torch.randn(2, 3, 3, dtype=DTYPE)},
            (2, 4, 3),
        ),
        (
            "a_to_b",
            {"a": 3},
            {"a": torch.randn(2, 3, 3, dtype=DTYPE)},
            (2, 4, 5),
        ),
        (
            "bb_to_b",
            {"left": 2, "right": 3},
            {
                "left": torch.randn(2, 2, 5, dtype=DTYPE),
                "right": torch.randn(2, 3, 5, dtype=DTYPE),
            },
            (2, 4, 5),
        ),
    )
    for name, channels, inputs, expected_shape in cases:
        gate = InvariantGate(
            signatures[name],
            artifacts[name],
            channels,
            4,
            6,
        )
        invariants = torch.randn(2, 6, dtype=DTYPE)
        coefficients = gate.coefficients(invariants)
        expected_coefficients = (
            2,
            4,
            *(channels[slot.name] for slot in artifacts[name].signature.inputs),
            artifacts[name].basis_dimension,
        )
        assert coefficients.shape == expected_coefficients
        assert gate(inputs, invariants).shape == expected_shape
        assert tuple(gate.covariant.parameters()) == ()

    default_gate = InvariantGate(
        signatures["a_to_a"],
        artifacts["a_to_a"],
        {"a": 3},
        4,
        6,
    )
    assert [type(module) for module in default_gate.invariant_mlp] == [
        nn.Linear,
        nn.SiLU,
        nn.Linear,
    ]
    first = default_gate.invariant_mlp[0]
    assert isinstance(first, nn.Linear)
    assert (first.in_features, first.out_features) == (6, 64)


def test_bilinear_channel_contraction_matches_manual_basis_sum(paths) -> None:
    """Lock the output, source channel, basis, and representation axis order."""
    _, signatures, artifacts = paths
    artifact = artifacts["bb_to_b"]
    gate = InvariantGate(
        signatures["bb_to_b"],
        artifact,
        {"left": 2, "right": 3},
        4,
        5,
        hidden_channels=7,
        num_linear_layers=3,
        activation="tanh",
    )
    generator = torch.Generator().manual_seed(12)
    left = torch.randn(2, 2, 5, dtype=DTYPE, generator=generator)
    right = torch.randn(2, 3, 5, dtype=DTYPE, generator=generator)
    invariants = torch.randn(2, 5, dtype=DTYPE, generator=generator)

    primitive = RegisteredCovariant(artifact).evaluate_basis(
        {
            "left": left[:, :, None, :],
            "right": right[:, None, :, :],
        }
    )
    coefficients = gate.coefficients(invariants)
    expected = torch.einsum(
        "...oijb,...ijbd->...od",
        coefficients,
        primitive,
    )
    torch.testing.assert_close(gate({"left": left, "right": right}, invariants), expected)


def test_symmetric_power_uses_one_learned_channel_axis(paths) -> None:
    """Leave repeated representation powers to the fixed slot lift."""
    catalog, _, _ = paths
    signature = CSignature(
        B0,
        (CSlot("a", A, power=2, mode="symmetric_power"),),
    )
    artifact = compile_covariant_basis(catalog, signature)
    gate = InvariantGate(
        signature,
        artifact,
        {"a": 3},
        4,
        5,
    )
    value = torch.randn(2, 3, 3, dtype=DTYPE)
    invariants = torch.randn(2, 5, dtype=DTYPE)
    assert gate.coefficients(invariants).shape == (
        2,
        4,
        3,
        artifact.basis_dimension,
    )
    assert gate({"a": value}, invariants).shape == (2, 4, 5)


@pytest.mark.parametrize("name", ("a_to_a", "a_to_b", "bb_to_b"))
def test_every_representative_path_is_equivariant(paths, name: str) -> None:
    """Keep invariant gates fixed while every typed input transforms."""
    catalog, signatures, artifacts = paths
    signature = signatures[name]
    artifact = artifacts[name]
    channels = {slot.name: index + 2 for index, slot in enumerate(signature.inputs)}
    gate = InvariantGate(signature, artifact, channels, 3, 4)
    generator = torch.Generator().manual_seed(20 + len(signature.inputs))
    inputs = {
        slot.name: torch.randn(
            2,
            channels[slot.name],
            artifact.input_dimensions[index],
            dtype=DTYPE,
            generator=generator,
        )
        for index, slot in enumerate(signature.inputs)
    }
    invariants = torch.randn(2, 4, dtype=DTYPE, generator=generator)
    transformed = {}
    for slot in signature.inputs:
        action = catalog.resolve(slot.type_key).actions[0]
        transformed[slot.name] = torch.einsum(
            "ab,...cb->...ca",
            action,
            inputs[slot.name],
        )
    output_action = catalog.resolve(signature.output).actions[0]
    expected = torch.einsum(
        "ab,...cb->...ca",
        output_action,
        gate(inputs, invariants),
    )
    actual = gate(transformed, invariants)
    torch.testing.assert_close(actual, expected, atol=2.0e-10, rtol=2.0e-10)


def test_signature_check_is_enabled_by_default_and_can_be_disabled(paths) -> None:
    """Compare the declared Sigma with the signature embedded in C."""
    _, signatures, artifacts = paths
    mismatch = CSignature(B0, (CSlot("a", A),))
    with pytest.raises(ValueError, match="does not match"):
        InvariantGate(
            mismatch,
            artifacts["a_to_a"],
            {"a": 2},
            3,
            4,
        )
    unchecked = InvariantGate(
        mismatch,
        artifacts["a_to_a"],
        {"a": 2},
        3,
        4,
        check_signature=False,
    )
    assert unchecked.signature == artifacts["a_to_a"].signature
    assert unchecked.check_signature is False
    output = unchecked(
        {"a": torch.randn(2, 2, 3, dtype=DTYPE)},
        torch.randn(2, 4, dtype=DTYPE),
    )
    assert output.shape == (2, 3, 3)
    with pytest.raises(TypeError, match="must be bool"):
        InvariantGate(
            signatures["a_to_a"],
            artifacts["a_to_a"],
            {"a": 2},
            3,
            4,
            check_signature=1,
        )


def test_shapes_dtype_empty_batches_and_configurability(paths) -> None:
    """Reject malformed tensors and preserve general prefix axes."""
    _, signatures, artifacts = paths
    gate = InvariantGate(
        signatures["bb_to_b"],
        artifacts["bb_to_b"],
        {"left": 2, "right": 3},
        4,
        5,
        hidden_channels=9,
        num_linear_layers=4,
        activation="softplus",
        output_activation="tanh",
        use_bias=False,
    )
    left = torch.randn(2, 3, 2, 5, dtype=DTYPE)
    right = torch.randn(2, 3, 3, 5, dtype=DTYPE)
    invariants = torch.randn(2, 3, 5, dtype=DTYPE)
    output = gate({"left": left, "right": right}, invariants)
    assert output.shape == (2, 3, 4, 5)
    assert bool((gate.coefficients(invariants).abs() <= 1.0).all())
    assert sum(isinstance(module, nn.Linear) for module in gate.invariant_mlp) == 4
    assert all(
        module.bias is None
        for module in gate.invariant_mlp
        if isinstance(module, nn.Linear)
    )

    empty = gate(
        {
            "left": torch.empty(0, 2, 5, dtype=DTYPE),
            "right": torch.empty(0, 3, 5, dtype=DTYPE),
        },
        torch.empty(0, 5, dtype=DTYPE),
    )
    assert empty.shape == (0, 4, 5)
    with pytest.raises(ValueError, match="end with shape"):
        gate(
            {
                "left": torch.randn(2, 3, 5, dtype=DTYPE),
                "right": torch.randn(2, 3, 5, dtype=DTYPE),
            },
            torch.randn(2, 5, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="prefix axes"):
        gate(
            {
                "left": torch.randn(2, 2, 5, dtype=DTYPE),
                "right": torch.randn(3, 3, 5, dtype=DTYPE),
            },
            torch.randn(2, 5, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="invariant prefix"):
        gate(
            {
                "left": torch.randn(2, 2, 5, dtype=DTYPE),
                "right": torch.randn(2, 3, 5, dtype=DTYPE),
            },
            torch.randn(1, 5, dtype=DTYPE),
        )
    with pytest.raises(TypeError, match="dtype"):
        gate(
            {
                "left": torch.randn(2, 2, 5, dtype=torch.float32),
                "right": torch.randn(2, 3, 5, dtype=DTYPE),
            },
            torch.randn(2, 5, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="must be one of"):
        InvariantGate(
            signatures["a_to_a"],
            artifacts["a_to_a"],
            {"a": 2},
            3,
            4,
            activation="relu",
        )


def test_inputs_invariants_and_parameters_receive_finite_gradients(paths) -> None:
    """Differentiate through both live slots, invariants, and the MLP."""
    _, signatures, artifacts = paths
    gate = InvariantGate(
        signatures["bb_to_b"],
        artifacts["bb_to_b"],
        {"left": 2, "right": 2},
        2,
        3,
        hidden_channels=5,
    )
    left = torch.randn(2, 2, 5, dtype=DTYPE, requires_grad=True)
    right = torch.randn(2, 2, 5, dtype=DTYPE, requires_grad=True)
    invariants = torch.randn(2, 3, dtype=DTYPE, requires_grad=True)
    loss = gate({"left": left, "right": right}, invariants).square().sum()
    loss.backward()
    for value in (left, right, invariants):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
        assert bool(value.grad.abs().sum() > 0.0)
    for parameter in gate.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_dtype_migration_and_state_round_trip(paths) -> None:
    """Move learned and fixed pathway tensors together and restore a checkpoint."""
    _, signatures, artifacts = paths
    first = InvariantGate(
        signatures["a_to_b"],
        artifacts["a_to_b"],
        {"a": 2},
        3,
        4,
    ).float()
    second = InvariantGate(
        signatures["a_to_b"],
        artifacts["a_to_b"],
        {"a": 2},
        3,
        4,
    ).float()
    second.load_state_dict(first.state_dict())
    inputs = {"a": torch.randn(5, 2, 3, dtype=torch.float32)}
    invariants = torch.randn(5, 4, dtype=torch.float32)
    torch.testing.assert_close(first(inputs, invariants), second(inputs, invariants))
    assert all(value.dtype == torch.float32 for value in first.parameters())
    assert all(value.dtype == torch.float32 for value in first.buffers())


def test_checkpoint_rejects_an_aliased_channel_axis_layout(paths) -> None:
    """Bind flattened coefficient rows to their exact logical axes."""
    _, signatures, artifacts = paths
    source = InvariantGate(
        signatures["bb_to_b"],
        artifacts["bb_to_b"],
        {"left": 2, "right": 3},
        4,
        5,
    )
    target = InvariantGate(
        signatures["bb_to_b"],
        artifacts["bb_to_b"],
        {"left": 2, "right": 2},
        6,
        5,
    )
    assert source.coefficient_count == target.coefficient_count
    with pytest.raises(RuntimeError, match="configuration does not match"):
        target.load_state_dict(source.state_dict())

    tampered = source.state_dict()
    metadata = dict(tampered["_extra_state"])
    metadata["schema_version"] = True
    tampered["_extra_state"] = metadata
    with pytest.raises(RuntimeError, match="configuration does not match"):
        source.load_state_dict(tampered)


def test_cpu_autocast_preserves_the_fixed_c_dtype_contract(paths) -> None:
    """Cast learned coefficients back before the fixed C contraction."""
    _, signatures, artifacts = paths
    gate = InvariantGate(
        signatures["a_to_b"],
        artifacts["a_to_b"],
        {"a": 2},
        3,
        4,
    ).float()
    inputs = {"a": torch.randn(5, 2, 3, dtype=torch.float32)}
    invariants = torch.randn(5, 4, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = gate(inputs, invariants)
    assert output.shape == (5, 3, 5)
    assert output.dtype == torch.float32
    assert bool(torch.isfinite(output).all())


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch compile is unavailable")
def test_full_graph_compile_matches_eager(paths) -> None:
    """Keep the ordinary runtime available to full graph compilation."""
    _, signatures, artifacts = paths
    gate = InvariantGate(
        signatures["a_to_a"],
        artifacts["a_to_a"],
        {"a": 2},
        3,
        4,
    )
    inputs = {"a": torch.randn(5, 2, 3, dtype=DTYPE)}
    invariants = torch.randn(5, 4, dtype=DTYPE)
    expected = gate(inputs, invariants)
    compiled = torch.compile(gate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(inputs, invariants), expected)


def test_bilinear_inputs_and_invariants_pass_gradcheck(paths) -> None:
    """Check the complete smooth runtime against finite differences."""
    _, signatures, artifacts = paths
    gate = InvariantGate(
        signatures["bb_to_b"],
        artifacts["bb_to_b"],
        {"left": 1, "right": 1},
        1,
        2,
        hidden_channels=3,
    )
    left = torch.randn(1, 1, 5, dtype=DTYPE, requires_grad=True)
    right = torch.randn(1, 1, 5, dtype=DTYPE, requires_grad=True)
    invariants = torch.randn(1, 2, dtype=DTYPE, requires_grad=True)

    def function(first: Tensor, second: Tensor, scalar: Tensor) -> Tensor:
        return gate({"left": first, "right": second}, scalar)

    assert torch.autograd.gradcheck(
        function,
        (left, right, invariants),
        eps=1.0e-6,
        atol=4.0e-5,
        rtol=4.0e-4,
    )
