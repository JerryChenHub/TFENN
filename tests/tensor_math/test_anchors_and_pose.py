"""Validate primitive anchors and generator conditioned pose encoding."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from TFENN.tensor_math import (
    AnchorCompilation,
    PoseEncoder,
    compile_anchors,
    stf_representation,
    stf_symmetric_product,
    stf_tensor_coupling,
    symmetric_dimension,
    symmetric_multi_indices,
    symmetric_to_stf,
)

from ._groups import (
    ATOL,
    DTYPE,
    RTOL,
    benzene_generators,
    c60_generators,
    octahedral_generators,
    rotation,
    rotation_from_rotvec,
)


def _projector(vector: Tensor) -> Tensor:
    """Return the projector of one nonzero vector."""
    normalized = vector / torch.linalg.vector_norm(vector)
    return normalized[:, None] @ normalized[None, :]


def _analytic_benzene_k2() -> Tensor:
    """Return the normalized uniaxial rank two reference for tests."""
    symmetric = torch.zeros(symmetric_dimension(2), dtype=DTYPE)
    diagonal = {
        (2, 0, 0): -1.0 / math.sqrt(6.0),
        (0, 2, 0): -1.0 / math.sqrt(6.0),
        (0, 0, 2): 2.0 / math.sqrt(6.0),
    }
    for index, alpha in enumerate(symmetric_multi_indices(2)):
        symmetric[index] = diagonal.get(alpha, 0.0)
    return symmetric_to_stf(symmetric, 2)


def _analytic_benzene_k6() -> Tensor:
    """Return the normalized planar sixfold reference for tests."""
    symmetric = torch.zeros(symmetric_dimension(6), dtype=DTYPE)
    for index, alpha in enumerate(symmetric_multi_indices(6)):
        x_power, y_power, z_power = alpha
        if z_power or y_power % 2:
            continue
        multiplicity = math.factorial(6) // (
            math.factorial(x_power) * math.factorial(y_power)
        )
        symmetric[index] = (
            (-1.0) ** (y_power // 2) * math.sqrt(multiplicity) / math.sqrt(32.0)
        )
    return symmetric_to_stf(symmetric, 6)


def _assert_anchor_constraints(generators: Tensor, anchors: dict[int, Tensor]) -> None:
    """Check every retained anchor is fixed by every generator."""
    for rank, block in anchors.items():
        represented = stf_representation(generators, rank)
        residual = represented @ block - block
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), atol=ATOL, rtol=RTOL
        )


def _replace_scanned_block(
    compilation: AnchorCompilation,
    rank: int,
    block: object,
) -> AnchorCompilation:
    """Return a compilation with one diagnostic block replaced."""
    return replace(
        compilation,
        scanned_blocks=tuple(
            block if candidate.rank == rank else candidate
            for candidate in compilation.scanned_blocks
        ),
    )


def test_benzene_removes_the_rank_six_descendant_of_k2() -> None:
    """Check the generic primitive rule produces the desired 18 dimensions."""
    compilation = compile_anchors(benzene_generators(), output_ranks=(2, 6))
    rank_two = compilation.blocks[2]
    rank_six = compilation.blocks[6]

    assert rank_two.dimensions.fixed == 1
    assert rank_two.dimensions.generated == 0
    assert rank_two.dimensions.primitive == 1
    assert rank_six.dimensions.fixed == 2
    assert rank_six.dimensions.generated == 1
    assert rank_six.dimensions.primitive == 1
    assert compilation.encoding_dimension == 18
    _assert_anchor_constraints(benzene_generators(), compilation.primitive_anchors)

    compiled_k2 = rank_two.primitive_basis[:, 0]
    torch.testing.assert_close(
        _projector(compiled_k2),
        _projector(_analytic_benzene_k2()),
        atol=ATOL,
        rtol=RTOL,
    )
    rank_four_descendant = stf_symmetric_product(compiled_k2, compiled_k2, 2, 2)
    rank_six_descendant = stf_symmetric_product(rank_four_descendant, compiled_k2, 4, 2)
    torch.testing.assert_close(
        rank_six.generated_basis @ rank_six.generated_basis.T,
        _projector(rank_six_descendant),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        rank_six.primitive_basis @ rank_six.primitive_basis.T,
        _projector(_analytic_benzene_k6()),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        rank_six.generated_basis.T @ rank_six.primitive_basis,
        torch.zeros((1, 1), dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )


def test_fixed_generated_and_primitive_spaces_are_orthogonal() -> None:
    """Check every scanned decomposition is complete and orthogonal."""
    compilation = compile_anchors(
        benzene_generators(), output_ranks=(2, 6)
    )
    for block in compilation.scanned_blocks:
        fixed = block.fixed_basis
        generated = block.generated_basis
        primitive = block.primitive_basis
        torch.testing.assert_close(
            fixed.T @ fixed,
            torch.eye(fixed.shape[1], dtype=DTYPE),
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            generated.T @ generated,
            torch.eye(generated.shape[1], dtype=DTYPE),
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            primitive.T @ primitive,
            torch.eye(primitive.shape[1], dtype=DTYPE),
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            generated.T @ primitive,
            torch.zeros(
                (generated.shape[1], primitive.shape[1]), dtype=DTYPE
            ),
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            generated @ generated.T + primitive @ primitive.T,
            fixed @ fixed.T,
            atol=ATOL,
            rtol=RTOL,
        )


def test_generated_only_intermediate_ranks_are_not_encoded() -> None:
    """Check descendant ranks remain available for diagnostics but leave B."""
    compilation = compile_anchors(
        benzene_generators().to(torch.float32), output_ranks=(2, 4, 6)
    )
    assert compilation.blocks[4].dimensions.fixed == 1
    assert compilation.blocks[4].dimensions.generated == 1
    assert compilation.blocks[4].dimensions.primitive == 0
    assert tuple(compilation.primitive_anchors) == (2, 6)
    assert compilation.encoding_dimension == 18


def test_lower_primitive_ranks_are_discovered_automatically() -> None:
    """Check scanned lower primitives do not silently become outputs."""
    compilation = compile_anchors(benzene_generators(), output_ranks=(6,))
    assert compilation.requested_output_ranks == (6,)
    assert compilation.scanned_ranks == tuple(range(1, 7))
    assert compilation.output_ranks == (6,)
    assert tuple(compilation.primitive_anchors) == (6,)
    assert compilation.blocks[2].dimensions.primitive == 1
    assert compilation.blocks[6].dimensions.fixed == 2
    assert compilation.blocks[6].dimensions.generated == 1
    assert compilation.blocks[6].dimensions.primitive == 1
    assert compilation.encoding_dimension == 13


def test_primitive_removal_applies_beyond_the_molecular_examples() -> None:
    """Check lower products span all higher invariants of the identity group."""
    compilation = compile_anchors(
        torch.eye(3, dtype=DTYPE)[None], output_ranks=(3,)
    )
    assert compilation.blocks[1].dimensions.fixed == 3
    assert compilation.blocks[1].dimensions.primitive == 3
    assert compilation.blocks[2].dimensions.generated == 5
    assert compilation.blocks[2].dimensions.primitive == 0
    assert compilation.blocks[3].dimensions.generated == 7
    assert compilation.blocks[3].dimensions.primitive == 0
    assert compilation.requested_output_ranks == (3,)
    assert compilation.output_ranks == ()
    assert tuple(compilation.primitive_anchors) == ()
    assert compilation.encoding_dimension == 0


def test_contracted_lower_anchor_direction_is_removed_generically() -> None:
    """Check the rank six octahedral descendant of rank four is removed."""
    compilation = compile_anchors(octahedral_generators(), output_ranks=(6,))
    rank_four = compilation.blocks[4]
    rank_six = compilation.blocks[6]
    assert rank_four.dimensions.fixed == 1
    assert rank_four.dimensions.primitive == 1
    assert rank_six.dimensions.fixed == 1
    assert rank_six.dimensions.generated == 1
    assert rank_six.dimensions.primitive == 0
    assert compilation.output_ranks == ()
    assert tuple(compilation.primitive_anchors) == ()
    assert compilation.encoding_dimension == 0

    descendant = stf_tensor_coupling(
        rank_four.primitive_basis[:, 0],
        rank_four.primitive_basis[:, 0],
        4,
        4,
        6,
    )
    torch.testing.assert_close(
        rank_six.generated_basis @ rank_six.generated_basis.T,
        _projector(descendant),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("invalid_tolerance", (float("nan"), float("inf")))
def test_anchor_compiler_rejects_nonfinite_tolerance(
    invalid_tolerance: float,
) -> None:
    """Check tolerance cannot disable rotation validation."""
    invalid = torch.zeros((1, 3, 3), dtype=DTYPE)
    with pytest.raises(ValueError, match="finite and positive"):
        compile_anchors(
            invalid,
            output_ranks=(2,),
            nullspace_atol=invalid_tolerance,
        )


def test_c60_retains_its_unique_rank_six_fixed_direction() -> None:
    """Check rotational I has one primitive rank six anchor."""
    generators = c60_generators()
    compilation = compile_anchors(generators, output_ranks=(6,))
    dimensions = compilation.blocks[6].dimensions
    assert dimensions.fixed == 1
    assert dimensions.generated == 0
    assert dimensions.primitive == 1
    assert compilation.encoding_dimension == 13
    _assert_anchor_constraints(generators, compilation.primitive_anchors)


@pytest.mark.parametrize(
    ("generators", "ranks", "expected_dimension"),
    (
        (benzene_generators(), (2, 6), 18),
        (c60_generators(), (6,), 13),
    ),
)
def test_pose_encoding_removes_the_right_action_and_covaries_on_the_left(
    generators: Tensor,
    ranks: tuple[int, ...],
    expected_dimension: int,
) -> None:
    """Check Z(g1 R g2) equals rho_B(g1) Z(R)."""
    compilation = compile_anchors(generators, ranks)
    encoder = PoseEncoder(compilation)
    pose = rotation((0.19, -0.27, 0.41))
    left, right = generators.unbind()
    encoded = encoder.encode(pose)

    assert encoder.encoding_dimension == expected_dimension
    torch.testing.assert_close(
        encoder.encode(pose @ right), encoded, atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(
        encoder.encode(left @ pose @ right),
        encoder.representation(left) @ encoded,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    ("generators", "ranks"),
    ((benzene_generators(), (2, 6)), (c60_generators(), (6,))),
)
def test_pose_encoding_gradient(generators: Tensor, ranks: tuple[int, ...]) -> None:
    """Run gradcheck on Z through the rotation matrix."""
    compilation = compile_anchors(generators, ranks)
    encoder = PoseEncoder(compilation)
    tangent = torch.tensor((0.12, -0.23, 0.31), dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: encoder.encode(rotation_from_rotvec(value)),
        (tangent,),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )


def test_pose_encoder_registers_versioned_verified_buffers() -> None:
    """Check anchors and generators persist with canonical metadata."""
    compilation = compile_anchors(benzene_generators(), output_ranks=(2, 6))
    encoder = PoseEncoder(compilation)
    state = encoder.state_dict()
    assert isinstance(encoder, torch.nn.Module)
    assert "_generators" in state
    assert "_anchor_rank_2" in state
    assert "_anchor_rank_6" in state
    assert (
        state["_extra_state"]["convention_version"]
        == encoder.get_extra_state()["convention_version"]
    )

    with pytest.raises(TypeError, match="AnchorCompilation"):
        PoseEncoder(compilation.primitive_anchors)

    obsolete = replace(compilation, convention_version="obsolete")
    with pytest.raises(ValueError, match="recompiled"):
        PoseEncoder(obsolete)


def test_pose_state_roundtrip_and_public_property_mutation_resistance() -> None:
    """Check versioned state and cloned properties cannot alter buffers."""
    compilation = compile_anchors(
        benzene_generators(), output_ranks=(2, 6)
    )
    source = PoseEncoder(compilation)
    target = PoseEncoder(compilation)
    pose = rotation((0.17, -0.29, 0.31))
    reference = source(pose)
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(target(pose), reference, atol=ATOL, rtol=RTOL)

    exposed_anchors = source.anchors
    exposed_generators = source.generators
    exposed_anchors[2].zero_()
    exposed_generators.zero_()
    torch.testing.assert_close(source(pose), reference, atol=ATOL, rtol=RTOL)
    assert bool((source.generators != 0).any())

    obsolete = source.state_dict()
    obsolete["_extra_state"] = dict(obsolete["_extra_state"])
    obsolete["_extra_state"]["convention_version"] = "obsolete"
    with pytest.raises(RuntimeError, match="obsolete convention"):
        target.load_state_dict(obsolete)


def test_pose_flat_order_matches_anchor_major_kronecker_representation() -> None:
    """Check flat coordinates and representation use one exact order."""
    identity_group = torch.eye(3, dtype=DTYPE)[None]
    encoder = PoseEncoder(
        compile_anchors(identity_group, output_ranks=(1,))
    )
    pose = rotation((0.23, -0.11, 0.37))
    block = encoder.encode_blocks(pose)[1]
    expected_flat = block.T.reshape(-1)
    torch.testing.assert_close(encoder(pose), expected_flat, atol=ATOL, rtol=RTOL)
    represented = stf_representation(pose, 1)
    expected_representation = torch.block_diag(
        represented,
        represented,
        represented,
    )
    assert encoder.multiplicities == {1: 3}
    torch.testing.assert_close(
        encoder.representation(pose),
        expected_representation,
        atol=ATOL,
        rtol=RTOL,
    )
    with pytest.raises(TypeError, match="torch.Tensor"):
        encoder.encode(None)
    with pytest.raises(ValueError, match="trailing shape"):
        encoder.encode(torch.eye(2, dtype=DTYPE))


def test_pose_encoder_rejects_nonfinite_and_noninvariant_state() -> None:
    """Check malformed persisted anchors cannot bypass provenance validation."""
    compilation = compile_anchors(benzene_generators(), output_ranks=(2, 6))
    blocks = dict(compilation.blocks)
    rank_two = blocks[2]
    nonfinite = rank_two.primitive_basis.clone()
    nonfinite[0, 0] = float("nan")
    blocks[2] = replace(rank_two, primitive_basis=nonfinite)
    with pytest.raises(ValueError, match="finite"):
        PoseEncoder(_replace_scanned_block(compilation, 2, blocks[2]))

    encoder = PoseEncoder(compilation)
    original = {
        name: value.clone()
        for name, value in encoder.state_dict().items()
        if isinstance(value, Tensor)
    }
    state = encoder.state_dict()
    random_anchor = torch.randn_like(state["_anchor_rank_2"])
    state["_anchor_rank_2"] = random_anchor / torch.linalg.vector_norm(
        random_anchor, dim=0
    )
    with pytest.raises(ValueError, match="invariance"):
        encoder.load_state_dict(state)
    for name, expected in original.items():
        torch.testing.assert_close(encoder.state_dict()[name], expected)

    state = encoder.state_dict()
    state["_anchor_rank_2"] = torch.zeros_like(state["_anchor_rank_2"])
    with pytest.raises(ValueError, match="orthonormal"):
        encoder.load_state_dict(state)
    for name, expected in original.items():
        torch.testing.assert_close(encoder.state_dict()[name], expected)

    state = encoder.state_dict()
    state["_anchor_rank_6"] = compilation.blocks[6].fixed_basis
    with pytest.raises(ValueError, match="shape"):
        encoder.load_state_dict(state)
    for name, expected in original.items():
        torch.testing.assert_close(encoder.state_dict()[name], expected)

    state = encoder.state_dict()
    state["_anchor_rank_2"] = -state["_anchor_rank_2"]
    with pytest.raises(ValueError, match="canonical compilation"):
        encoder.load_state_dict(state)
    for name, expected in original.items():
        torch.testing.assert_close(encoder.state_dict()[name], expected)

    state = encoder.state_dict()
    state["_anchor_rank_6"] = compilation.blocks[6].generated_basis
    with pytest.raises(ValueError, match="canonical compilation"):
        encoder.load_state_dict(state)
    for name, expected in original.items():
        torch.testing.assert_close(encoder.state_dict()[name], expected)


@pytest.mark.parametrize("invalid", (0.0, float("nan"), float("inf")))
def test_pose_encoder_rejects_invalid_compilation_metadata(invalid: float) -> None:
    """Check corrupt compiler diagnostics cannot weaken validation."""
    compilation = compile_anchors(benzene_generators(), output_ranks=(2, 6))
    with pytest.raises(ValueError, match="finite and positive"):
        PoseEncoder(replace(compilation, nullspace_atol=invalid))

    blocks = dict(compilation.blocks)
    rank_two = blocks[2]
    random_anchor = torch.randn_like(rank_two.primitive_basis)
    random_anchor = random_anchor / torch.linalg.vector_norm(random_anchor, dim=0)
    blocks[2] = replace(
        rank_two,
        primitive_basis=random_anchor,
    )
    loose = replace(
        _replace_scanned_block(compilation, 2, blocks[2]),
        nullspace_atol=1.0,
    )
    with pytest.raises(ValueError, match="invariance"):
        PoseEncoder(loose)

    blocks = dict(compilation.blocks)
    rank_six = blocks[6]
    blocks[6] = replace(
        rank_six,
        generated_basis=rank_six.primitive_basis,
        primitive_basis=rank_six.generated_basis,
    )
    with pytest.raises(ValueError, match="subspace"):
        PoseEncoder(_replace_scanned_block(compilation, 6, blocks[6]))

    blocks = dict(compilation.blocks)
    rank_six = blocks[6]
    blocks[6] = replace(
        rank_six,
        fixed_basis=rank_six.generated_basis,
        dimensions=replace(
            rank_six.dimensions,
            fixed=1,
            generated=1,
            primitive=1,
        ),
    )
    with pytest.raises(ValueError, match="dimensions"):
        PoseEncoder(_replace_scanned_block(compilation, 6, blocks[6]))

    with pytest.raises(ValueError, match="generator count"):
        PoseEncoder(replace(compilation, generator_count=3))

    blocks = dict(compilation.blocks)
    rank_two = blocks[2]
    blocks[2] = replace(
        rank_two,
        primitive_basis=torch.zeros_like(rank_two.primitive_basis),
    )
    with pytest.raises(ValueError, match="orthonormal"):
        PoseEncoder(_replace_scanned_block(compilation, 2, blocks[2]))


def test_float32_generators_compile_into_a_verified_pose_encoder() -> None:
    """Check source precision is retained in validation tolerances."""
    compilation = compile_anchors(
        benzene_generators().float(), output_ranks=(6,)
    )
    encoder = PoseEncoder(compilation)
    for residual in encoder.gauge_residuals().values():
        assert residual.item() < 2e-6


def test_nested_pose_encoder_load_rechecks_version_and_invariance() -> None:
    """Check parent module loading cannot bypass pose state validation."""

    class Container(torch.nn.Module):
        """Hold one pose encoder as a nested module."""

        def __init__(self) -> None:
            super().__init__()
            compilation = compile_anchors(
                benzene_generators(), output_ranks=(2, 6)
            )
            self.encoder = PoseEncoder(compilation)

    container = Container()
    original = {
        name: value.clone()
        for name, value in container.state_dict().items()
        if isinstance(value, Tensor)
    }
    noninvariant = container.state_dict()
    random_anchor = torch.randn_like(noninvariant["encoder._anchor_rank_2"])
    noninvariant["encoder._anchor_rank_2"] = random_anchor / torch.linalg.vector_norm(
        random_anchor, dim=0
    )
    with pytest.raises(ValueError, match="invariance"):
        container.load_state_dict(noninvariant)
    for name, expected in original.items():
        torch.testing.assert_close(container.state_dict()[name], expected)

    zero = container.state_dict()
    zero["encoder._anchor_rank_2"] = torch.zeros_like(
        zero["encoder._anchor_rank_2"]
    )
    with pytest.raises(ValueError, match="orthonormal"):
        container.load_state_dict(zero)
    for name, expected in original.items():
        torch.testing.assert_close(container.state_dict()[name], expected)

    generated = container.state_dict()
    compilation = compile_anchors(benzene_generators(), output_ranks=(2, 6))
    generated["encoder._anchor_rank_6"] = compilation.blocks[6].generated_basis
    with pytest.raises(ValueError, match="canonical compilation"):
        container.load_state_dict(generated)
    for name, expected in original.items():
        torch.testing.assert_close(container.state_dict()[name], expected)

    container = Container()
    unversioned = container.state_dict()
    del unversioned["encoder._extra_state"]
    with pytest.raises(RuntimeError, match="unversioned"):
        container.load_state_dict(unversioned, strict=False)


@pytest.mark.parametrize("batch_shape", ((0,), (2, 0)))
def test_pose_encoder_supports_empty_batches(batch_shape: tuple[int, ...]) -> None:
    """Check known block sizes make empty flattening unambiguous."""
    encoder = PoseEncoder(
        compile_anchors(benzene_generators(), output_ranks=(2, 6))
    )
    poses = torch.empty(batch_shape + (3, 3), dtype=DTYPE)
    encoded = encoder.encode(poses)
    assert encoded.shape == batch_shape + (18,)


def test_pose_encoder_independent_receiver_and_sender_gauges() -> None:
    """Check independent D6 receiver and sender generator actions."""
    generators = benzene_generators()
    encoder = PoseEncoder(compile_anchors(generators, output_ranks=(2, 6)))
    pose = rotation((0.21, -0.17, 0.29))
    reference = encoder.encode_blocks(pose)

    for receiver in generators:
        for sender in generators:
            sender_blocks = encoder.encode_blocks(pose @ sender)
            transformed = encoder.encode_blocks(receiver.T @ pose @ sender)
            for rank, block in reference.items():
                torch.testing.assert_close(
                    sender_blocks[rank], block, atol=ATOL, rtol=RTOL
                )
                torch.testing.assert_close(
                    transformed[rank],
                    stf_representation(receiver.T, rank) @ block,
                    atol=ATOL,
                    rtol=RTOL,
                )


def test_float64_compilation_supports_float32_pose_runtime() -> None:
    """Check module conversion preserves rank two and rank six gauge accuracy."""
    compilation = compile_anchors(benzene_generators(), output_ranks=(2, 6))
    encoder64 = PoseEncoder(compilation)
    encoder = PoseEncoder(compilation).float()
    pose64 = rotation((0.13, 0.19, -0.23))
    pose = pose64.float().requires_grad_()
    blocks = encoder.encode_blocks(pose)
    assert blocks[2].dtype == torch.float32
    assert blocks[6].dtype == torch.float32
    torch.testing.assert_close(
        encoder.encode(pose).double(),
        encoder64.encode(pose64),
        atol=5e-5,
        rtol=5e-5,
    )
    for residual in encoder.gauge_residuals().values():
        assert residual.item() < 5e-5
    generators = benzene_generators().float()
    reference = encoder.encode(pose)
    for receiver in generators:
        for sender in generators:
            torch.testing.assert_close(
                encoder.encode(receiver.T @ pose @ sender),
                encoder.representation(receiver.T) @ reference,
                atol=5e-5,
                rtol=5e-5,
            )
    encoder.encode(pose).square().sum().backward()
    assert pose.grad is not None
    assert bool(torch.isfinite(pose.grad).all())
    with pytest.raises(TypeError, match="dtype"):
        encoder.encode(pose.double())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pose_encoder_preserves_gauge_and_gradient() -> None:
    """Check the complete pose runtime migrates through module buffers."""
    generators = benzene_generators().float().cuda()
    encoder = (
        PoseEncoder(
            compile_anchors(benzene_generators(), output_ranks=(2, 6))
        ).float().cuda()
    )
    pose = rotation((0.15, -0.21, 0.27)).float().cuda().requires_grad_()
    reference = encoder.encode(pose)
    receiver, sender = generators.unbind()
    torch.testing.assert_close(
        encoder.encode(receiver.T @ pose @ sender),
        encoder.representation(receiver.T) @ reference,
        atol=5e-5,
        rtol=5e-5,
    )
    reference.square().sum().backward()
    assert pose.grad is not None
    assert bool(torch.isfinite(pose.grad).all())
