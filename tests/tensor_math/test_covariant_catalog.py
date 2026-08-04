"""Validate typed catalog provenance, manifests, and Pose adaptation."""

from __future__ import annotations

from types import MappingProxyType

import pytest
import torch

from TFENN.tensor_math import (
    BBlockManifest,
    CSignature,
    CSlot,
    GeneratorSystem,
    PoseEncoder,
    PoseTypedBlockAdapter,
    TypeBlock,
    TypeCatalog,
    TypeKey,
    build_type_catalog,
    compile_covariant_basis,
    compile_anchors,
    encode_typed_blocks,
    stf_representation,
)

from ._covariant_cases import benzene_covariant_case, methane_covariant_case
from ._groups import DTYPE, benzene_generators, methane_generators, rotation


def test_generator_system_fingerprint_is_ordered_and_mutation_resistant() -> None:
    """Check names, order, matrices, and convention bind the fingerprint."""
    generators = benzene_generators()
    system = GeneratorSystem(("r", "s"), generators)
    clone = GeneratorSystem(("r", "s"), generators.clone())
    assert system.fingerprint == clone.fingerprint
    assert system.matrices.dtype == torch.float64
    assert system.matrices.device.type == "cpu"

    exposed = system.matrices
    exposed.zero_()
    torch.testing.assert_close(system.matrices, generators, atol=0.0, rtol=0.0)
    assert GeneratorSystem(("x", "s"), generators).fingerprint != system.fingerprint
    assert (
        GeneratorSystem(("s", "r"), generators.flip(0)).fingerprint
        != system.fingerprint
    )


def test_type_catalog_requires_exact_generator_provenance_and_manifest_action() -> None:
    """Reject equal generator counts with different ordered fingerprints."""
    case = benzene_covariant_case()
    catalog = case.catalog
    a_key = TypeKey("A")
    b_key = TypeKey("B", 0)
    assert isinstance(catalog.blocks, MappingProxyType)
    torch.testing.assert_close(
        catalog.resolve(a_key).actions,
        benzene_generators(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        catalog.resolve(b_key).actions,
        stf_representation(benzene_generators(), 2),
        atol=0.0,
        rtol=0.0,
    )
    assert catalog.resolve(b_key).representation_dim == 5
    assert catalog.resolve(b_key).metadata["anchor_multiplicity"] == 1

    other = GeneratorSystem(("a", "b"), methane_generators())
    foreign_a = TypeBlock(
        a_key,
        other.matrices,
        3,
        other.fingerprint,
        "active_column_cartesian_xyz_v1",
        {"stable_component_id": "A", "layout": "final_representation_axis"},
    )
    with pytest.raises(ValueError, match="fingerprint"):
        TypeCatalog(catalog.generator_system, {a_key: foreign_a})

    wrong_b = TypeBlock(
        b_key,
        torch.eye(5, dtype=DTYPE).expand(2, 5, 5),
        5,
        catalog.generator_system.fingerprint,
        "wrong",
        catalog.resolve(b_key).metadata,
    )
    with pytest.raises(ValueError, match="manifest rank"):
        TypeCatalog(
            catalog.generator_system,
            {a_key: catalog.resolve(a_key), b_key: wrong_b},
        )


def test_catalog_fingerprint_is_mapping_order_independent() -> None:
    """Check explicit type identities rather than insertion order define catalog."""
    case = benzene_covariant_case()
    reversed_blocks = dict(reversed(tuple(case.catalog.blocks.items())))
    rebuilt = TypeCatalog(case.catalog.generator_system, reversed_blocks)
    assert rebuilt.fingerprint == case.catalog.fingerprint
    exposed = rebuilt.resolve(TypeKey("B", 0)).actions
    exposed.add_(10.0)
    assert rebuilt.fingerprint == case.catalog.fingerprint


@pytest.mark.parametrize(
    "case_factory,generators,ranks",
    (
        (benzene_covariant_case, benzene_generators, (2, 6)),
        (methane_covariant_case, methane_generators, (3,)),
    ),
)
def test_pose_typed_adapter_is_noninvasive_and_uses_channel_axis(
    case_factory,
    generators,
    ranks: tuple[int, ...],
) -> None:
    """Move only STF and anchor axes while preserving original Pose blocks."""
    case = case_factory()
    encoder = PoseEncoder(compile_anchors(generators(), output_ranks=ranks))
    rotations = torch.stack(
        (
            torch.eye(3, dtype=DTYPE),
            rotation((0.2, -0.3, 0.1)),
        )
    )
    before = encoder.encode_blocks(rotations)
    typed = encode_typed_blocks(encoder, rotations, case.manifest)
    wrapped = PoseTypedBlockAdapter(encoder).encode_typed_blocks(
        rotations,
        case.manifest,
    )
    after = encoder.encode_blocks(rotations)
    assert tuple(typed) == tuple(item.key for item in case.manifest)
    for item in case.manifest:
        expected = before[item.stf_rank].movedim(-2, -1)[
            ..., item.anchor_columns, :
        ]
        torch.testing.assert_close(typed[item.key], expected)
        torch.testing.assert_close(wrapped[item.key], expected)
        assert typed[item.key].shape[-2:] == (
            len(item.anchor_columns),
            2 * item.stf_rank + 1,
        )
        torch.testing.assert_close(after[item.stf_rank], before[item.stf_rank])

        sender = generators()[0]
        receiver = generators()[1]
        right = encode_typed_blocks(
            encoder,
            rotations @ sender,
            case.manifest,
        )[item.key]
        torch.testing.assert_close(right, typed[item.key], atol=3e-9, rtol=3e-9)
        left = encode_typed_blocks(
            encoder,
            receiver @ rotations,
            case.manifest,
        )[item.key]
        action = stf_representation(receiver, item.stf_rank)
        torch.testing.assert_close(
            left,
            typed[item.key] @ action.T,
            atol=3e-9,
            rtol=3e-9,
        )


def test_manifest_and_type_error_contracts_are_explicit() -> None:
    """Reject malformed component identities and anchor channel declarations."""
    with pytest.raises(ValueError):
        TypeKey("A", 0)
    with pytest.raises(ValueError):
        TypeKey("B", None)
    with pytest.raises(ValueError):
        BBlockManifest(0, 2, (1,), 1, "bad")
    with pytest.raises(ValueError):
        BBlockManifest(0, 0, (0,), 1, "bad")
    with pytest.raises(ValueError):
        build_type_catalog(
            benzene_covariant_case().catalog.generator_system,
            (
                BBlockManifest(0, 2, (0,), 1, "one"),
                BBlockManifest(0, 6, (0,), 1, "two"),
            ),
        )


def test_private_generator_and_catalog_tensor_mutation_is_detected() -> None:
    """Reject altered provenance before resolving or compiling typed maps."""
    system = GeneratorSystem(("r", "s"), benzene_generators())
    system._matrices[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="fingerprint"):
        _ = system.matrices

    case = benzene_covariant_case()
    block = case.catalog.resolve(TypeKey("B", 0))
    block._actions.zero_()
    with pytest.raises(RuntimeError, match="fingerprint"):
        case.catalog.resolve(TypeKey("B", 0))
    with pytest.raises(RuntimeError, match="fingerprint"):
        compile_covariant_basis(
            case.catalog,
            CSignature(
                TypeKey("A"),
                (
                    CSlot("a", TypeKey("A")),
                    CSlot("b", TypeKey("B", 0)),
                ),
            ),
        )
