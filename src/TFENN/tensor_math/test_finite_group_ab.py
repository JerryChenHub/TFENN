"""Independent algebra and tiny-edge tests for tensor_math.

Place this file inside the tensor_math package and run it with pytest.
Every group, point set, anchor, representation and Lift basis is built at
test time; there are no reference tensor files.

Scope: finite proper rotation groups G < SO(3).  The C60 case therefore uses
the order-60 proper icosahedral group I, not the inversion-containing I_h.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from .anchor_compiler import AnchorCompilation, compile_anchors
from .covariants import scalar_contraction, vector_covariant
from .intertwiner_compiler import (
    IntertwinerCompilation,
    compile_intertwiners,
    intertwiner_residual,
)
from .pose_encoding import PoseEncoder
from .stf_rep import stf_representation
from .stf_space import (
    dense_to_stf,
    stf_basis,
    stf_tensor_coupling,
    stf_to_dense,
    trace_matrix,
)


DTYPE = torch.float64
ATOL = 2.0e-8
RTOL = 2.0e-8
GROUP_MATCH_ATOL = 5.0e-8
RUN_SLOW = os.environ.get("TENSOR_MATH_RUN_SLOW") == "1"


def _skew(vector: Tensor) -> Tensor:
    """Return the cross-product matrix, with batch support."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def _rotation(axis: Tensor | tuple[float, float, float], angle: float) -> Tensor:
    """Construct one proper rotation by Rodrigues' formula."""
    axis_tensor = torch.as_tensor(axis, dtype=DTYPE)
    axis_tensor = axis_tensor / torch.linalg.vector_norm(axis_tensor)
    cross = _skew(axis_tensor)
    identity = torch.eye(3, dtype=DTYPE)
    return (
        identity
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def _group_closure(generators: Tensor, *, max_size: int = 256) -> Tensor:
    """Enumerate a finite group from generators, only for an independent oracle."""
    elements = [torch.eye(3, dtype=generators.dtype)]
    cursor = 0
    while cursor < len(elements):
        for generator in generators:
            candidate = elements[cursor] @ generator
            if not any(
                torch.allclose(
                    candidate,
                    known,
                    atol=GROUP_MATCH_ATOL,
                    rtol=0.0,
                )
                for known in elements
            ):
                elements.append(candidate)
                if len(elements) > max_size:
                    raise RuntimeError("closure exceeded max_size; group may be infinite")
        cursor += 1
    return torch.stack(elements)


def _benzene_generators() -> Tensor:
    """D6 as proper 3D rotations: C6 about z and C2 about x."""
    return torch.stack(
        (
            _rotation((0.0, 0.0, 1.0), math.pi / 3.0),
            _rotation((1.0, 0.0, 0.0), math.pi),
        )
    )


def _benzene_points() -> Tensor:
    angles = torch.arange(6, dtype=DTYPE) * (math.pi / 3.0)
    return torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)),
        dim=-1,
    )


def _icosahedron_vertices() -> Tensor:
    """Generate the twelve icosahedron vertices without a stored table."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    points: list[tuple[float, float, float]] = []
    for sign_a in (-1.0, 1.0):
        for sign_b in (-1.0, 1.0):
            points.extend(
                (
                    (0.0, sign_a, sign_b * phi),
                    (sign_a, sign_b * phi, 0.0),
                    (sign_b * phi, 0.0, sign_a),
                )
            )
    return torch.tensor(points, dtype=DTYPE)


def _c60_points() -> Tensor:
    """Truncate every directed icosahedron edge to obtain 60 C60 sites."""
    vertices = _icosahedron_vertices()
    distances = torch.cdist(vertices, vertices)
    positive = distances[distances > 1.0e-9]
    edge_length = positive.min()
    directed_edges = torch.nonzero(
        torch.isclose(distances, edge_length, atol=1.0e-9, rtol=1.0e-9),
        as_tuple=False,
    )
    points = torch.stack(
        [
            (2.0 * vertices[source] + vertices[target]) / 3.0
            for source, target in directed_edges.tolist()
        ]
    )
    if points.shape != (60, 3):
        raise RuntimeError("procedural C60 construction did not yield 60 sites")
    return points


def _icosahedral_generators() -> Tensor:
    """Order-5 vertex rotation and order-3 face rotation for proper I."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    v0 = torch.tensor((0.0, 1.0, phi), dtype=DTYPE)
    v1 = torch.tensor((-1.0, phi, 0.0), dtype=DTYPE)
    v2 = torch.tensor((1.0, phi, 0.0), dtype=DTYPE)
    return torch.stack(
        (
            _rotation(v0, 2.0 * math.pi / 5.0),
            _rotation(v0 + v1 + v2, 2.0 * math.pi / 3.0),
        )
    )


@dataclass(frozen=True)
class GroupCase:
    name: str
    generators: Tensor
    group: Tensor
    molecular_points: Tensor
    order: int
    expected_fixed: tuple[int, ...]
    expected_primitive: tuple[int, ...]
    expected_lifts: dict[str, int]


def _make_group_case(name: str) -> GroupCase:
    if name == "benzene":
        generators = _benzene_generators()
        return GroupCase(
            name=name,
            generators=generators,
            group=_group_closure(generators),
            molecular_points=_benzene_points(),
            order=12,
            expected_fixed=(0, 1, 0, 1, 0, 2),
            expected_primitive=(0, 1, 0, 0, 0, 1),
            expected_lifts={"aa": 2, "ab": 4, "ba": 4, "bb": 30},
        )
    if name == "c60":
        generators = _icosahedral_generators()
        return GroupCase(
            name=name,
            generators=generators,
            group=_group_closure(generators),
            molecular_points=_c60_points(),
            order=60,
            expected_fixed=(0, 0, 0, 0, 0, 1),
            expected_primitive=(0, 0, 0, 0, 0, 1),
            expected_lifts={"aa": 1, "ab": 1, "ba": 1, "bb": 4},
        )
    raise ValueError(f"unknown group case: {name}")


@pytest.fixture(scope="module", params=("benzene", "c60"))
def group_case(request: pytest.FixtureRequest) -> GroupCase:
    return _make_group_case(str(request.param))


@dataclass
class CompiledCase:
    group: GroupCase
    compilation: AnchorCompilation
    encoder: PoseEncoder
    rho_a_generators: Tensor
    rho_b_generators: Tensor
    lifts: dict[str, IntertwinerCompilation]


def _compile_case(case: GroupCase) -> CompiledCase:
    compilation = compile_anchors(case.generators, output_ranks=(2, 6))
    encoder = PoseEncoder(compilation)
    rho_a = case.generators
    rho_b = encoder.representation(case.generators)
    lift_specs = {
        "aa": (rho_a, rho_a),
        "ab": (rho_a, rho_b),
        "ba": (rho_b, rho_a),
        "bb": (rho_b, rho_b),
    }
    lifts = {
        name: compile_intertwiners(
            rho_in,
            rho_out,
        )
        for name, (rho_in, rho_out) in lift_specs.items()
    }
    return CompiledCase(
        group=case,
        compilation=compilation,
        encoder=encoder,
        rho_a_generators=rho_a,
        rho_b_generators=rho_b,
        lifts=lifts,
    )


@pytest.fixture(scope="module")
def compiled_case(group_case: GroupCase) -> CompiledCase:
    return _compile_case(group_case)


def _character_hom_dimension(rho_in: Tensor, rho_out: Tensor) -> int:
    """Independent finite-group character inner product for real reps."""
    trace_in = rho_in.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_out = rho_out.diagonal(dim1=-2, dim2=-1).sum(-1)
    value = (trace_in * trace_out).mean()
    rounded = int(round(float(value)))
    assert abs(float(value) - rounded) < 2.0e-7
    return rounded


def _lift_all(basis: Tensor, value: Tensor) -> Tensor:
    """Apply every map in [maps, out, in] to row-batched input."""
    return torch.einsum("koi,bi->bko", basis, value)


def _lift_mix(basis: Tensor, coefficients: Tensor, value: Tensor) -> Tensor:
    """Apply a learned scalar mixture of an intertwiner basis."""
    return torch.einsum("koi,k,bi->bo", basis, coefficients, value)


def _random_rotation(seed: int, *, scale: float = 0.8) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    omega = scale * torch.randn(3, dtype=DTYPE, generator=generator)
    return torch.matrix_exp(_skew(omega))


def test_runtime_groups_and_molecular_point_sets(group_case: GroupCase) -> None:
    """Generators close to the intended finite group and preserve the sites."""
    case = group_case
    assert case.group.shape == (case.order, 3, 3)

    identity = torch.eye(3, dtype=DTYPE)
    gram = case.group.mT @ case.group
    torch.testing.assert_close(
        gram,
        identity.expand_as(gram),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        torch.linalg.det(case.group),
        torch.ones(case.order, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )

    transformed = torch.einsum(
        "gij,pj->gpi",
        case.group,
        case.molecular_points,
    )
    nearest_distance = torch.cdist(
        transformed,
        case.molecular_points.expand(case.order, -1, -1),
    ).amin(dim=-1)
    assert float(nearest_distance.amax()) < GROUP_MATCH_ATOL

    first, second = case.generators
    if case.name == "benzene":
        relations = ((first, 6), (second, 2), (second @ first, 2))
    else:
        relations = ((first, 5), (second, 3), (first @ second, 2))
    for matrix, exponent in relations:
        torch.testing.assert_close(
            torch.linalg.matrix_power(matrix, exponent),
            identity,
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("rank", range(7))
def test_canonical_stf_basis_trace_and_roundtrip(rank: int) -> None:
    """The coordinate basis is orthonormal, traceless and invertible on STF."""
    basis = stf_basis(rank, dtype=DTYPE)
    identity = torch.eye(2 * rank + 1, dtype=DTYPE)
    torch.testing.assert_close(
        basis.mT @ basis,
        identity,
        atol=ATOL,
        rtol=RTOL,
    )
    trace = trace_matrix(rank, dtype=DTYPE)
    torch.testing.assert_close(
        trace @ basis,
        torch.zeros(
            trace.shape[0],
            basis.shape[1],
            dtype=DTYPE,
        ),
        atol=ATOL,
        rtol=RTOL,
    )

    generator = torch.Generator().manual_seed(4100 + rank)
    coordinates = torch.randn(
        3,
        2 * rank + 1,
        dtype=DTYPE,
        generator=generator,
    )
    recovered = dense_to_stf(stf_to_dense(coordinates, rank), rank)
    torch.testing.assert_close(recovered, coordinates, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("rank", (0, 1, 2, 4, 6))
def test_stf_representation_group_law(
    group_case: GroupCase,
    rank: int,
) -> None:
    """D_l is an orthogonal representation, not merely generator-consistent."""
    group = group_case.group
    representation = stf_representation(group, rank)
    dimension = 2 * rank + 1
    identity = torch.eye(dimension, dtype=DTYPE)
    torch.testing.assert_close(
        representation.mT @ representation,
        identity.expand(group.shape[0], -1, -1),
        atol=ATOL,
        rtol=RTOL,
    )

    sample_pairs = (
        (0, 0),
        (1, group.shape[0] // 3),
        (group.shape[0] // 2, group.shape[0] - 1),
    )
    for left_index, right_index in sample_pairs:
        product = group[left_index] @ group[right_index]
        represented_product = stf_representation(product, rank)
        expected = representation[left_index] @ representation[right_index]
        torch.testing.assert_close(
            represented_product,
            expected,
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize(
    ("rank_left", "rank_right", "rank_out"),
    (
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 1),
        (2, 2, 4),
        (1, 6, 5),
        (1, 6, 6),
        (1, 6, 7),
    ),
)
def test_stf_tensor_coupling_is_so3_covariant(
    rank_left: int,
    rank_right: int,
    rank_out: int,
) -> None:
    """Each Clebsch-like STF coupling obeys SO(3), hence every subgroup."""
    generator = torch.Generator().manual_seed(
        6000 + 100 * rank_left + 10 * rank_right + rank_out
    )
    left = torch.randn(
        2,
        2 * rank_left + 1,
        dtype=DTYPE,
        generator=generator,
    )
    right = torch.randn(
        2,
        2 * rank_right + 1,
        dtype=DTYPE,
        generator=generator,
    )
    rotation = _random_rotation(7000 + rank_out)
    d_left = stf_representation(rotation, rank_left)
    d_right = stf_representation(rotation, rank_right)
    d_out = stf_representation(rotation, rank_out)

    coupled_after_action = stf_tensor_coupling(
        left @ d_left.mT,
        right @ d_right.mT,
        rank_left,
        rank_right,
        rank_out,
    )
    expected = (
        stf_tensor_coupling(
            left,
            right,
            rank_left,
            rank_right,
            rank_out,
        )
        @ d_out.mT
    )
    torch.testing.assert_close(
        coupled_after_action,
        expected,
        atol=5.0e-8,
        rtol=5.0e-8,
    )


def test_anchor_compiler_against_full_group_reynolds_oracle(
    compiled_case: CompiledCase,
) -> None:
    """Generator nullspaces equal full-group fixed spaces at every scanned rank."""
    case = compiled_case.group
    compilation = compiled_case.compilation
    actual_fixed = tuple(
        compilation.blocks[rank].dimensions.fixed for rank in range(1, 7)
    )
    actual_primitive = tuple(
        compilation.blocks[rank].dimensions.primitive for rank in range(1, 7)
    )
    assert actual_fixed == case.expected_fixed
    assert actual_primitive == case.expected_primitive

    for rank in range(1, 7):
        block = compilation.blocks[rank]
        represented_group = stf_representation(case.group, rank)
        reynolds = represented_group.mean(dim=0)
        reynolds = 0.5 * (reynolds + reynolds.mT)

        fixed_projector = block.fixed_basis @ block.fixed_basis.mT
        generated_projector = block.generated_basis @ block.generated_basis.mT
        primitive_projector = block.primitive_basis @ block.primitive_basis.mT
        torch.testing.assert_close(
            fixed_projector,
            reynolds,
            atol=5.0e-8,
            rtol=5.0e-8,
        )
        torch.testing.assert_close(
            generated_projector + primitive_projector,
            fixed_projector,
            atol=5.0e-8,
            rtol=5.0e-8,
        )
        if block.fixed_basis.shape[1]:
            invariant_residual = (
                represented_group @ block.fixed_basis - block.fixed_basis
            )
            assert float(torch.linalg.vector_norm(invariant_residual, dim=-2).amax()) < (
                8.0e-8
            )

    if case.name == "benzene":
        primitive_rank_two = compilation.blocks[2].primitive_basis[:, 0]
        independently_generated_rank_four = stf_tensor_coupling(
            primitive_rank_two,
            primitive_rank_two,
            2,
            2,
            4,
        )
        generated_norm = torch.linalg.vector_norm(
            independently_generated_rank_four
        )
        assert float(generated_norm) > 1.0e-10
        generated_unit = independently_generated_rank_four / generated_norm
        torch.testing.assert_close(
            generated_unit[:, None] @ generated_unit[None, :],
            compilation.blocks[4].generated_basis
            @ compilation.blocks[4].generated_basis.mT,
            atol=8.0e-8,
            rtol=8.0e-8,
        )


@pytest.mark.skipif(
    not RUN_SLOW,
    reason="set TENSOR_MATH_RUN_SLOW=1 for the rank-10 C60 closure test",
)
def test_slow_c60_rank10_is_generated_from_rank6() -> None:
    """The rank-10 I-invariant exists but is not a new primitive anchor."""
    case = _make_group_case("c60")
    compilation = compile_anchors(case.generators, output_ranks=(6, 10))
    rank_ten = compilation.blocks[10]
    assert rank_ten.dimensions.fixed == 1
    assert rank_ten.dimensions.generated == 1
    assert rank_ten.dimensions.primitive == 0

    reynolds = stf_representation(case.group, 10).mean(dim=0)
    generated_projector = (
        rank_ten.generated_basis @ rank_ten.generated_basis.mT
    )
    primitive_rank_six = compilation.blocks[6].primitive_basis[:, 0]
    independently_generated = stf_tensor_coupling(
        primitive_rank_six,
        primitive_rank_six,
        6,
        6,
        10,
    )
    generated_norm = torch.linalg.vector_norm(independently_generated)
    assert float(generated_norm) > 1.0e-10
    generated_unit = independently_generated / generated_norm
    torch.testing.assert_close(
        generated_projector,
        generated_unit[:, None] @ generated_unit[None, :],
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    torch.testing.assert_close(
        generated_projector,
        0.5 * (reynolds + reynolds.mT),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


@pytest.mark.parametrize("order", (2, 3, 5, 7))
def test_generic_cyclic_groups_have_no_hardcoded_group_path(order: int) -> None:
    """A small family checks generator-only logic beyond named molecules."""
    generators = _rotation((0.0, 0.0, 1.0), 2.0 * math.pi / order).unsqueeze(0)
    group = _group_closure(generators)
    assert group.shape[0] == order

    compilation = compile_anchors(generators, output_ranks=(1, 2, 3))
    for rank, block in compilation.blocks.items():
        reynolds = stf_representation(group, rank).mean(dim=0)
        fixed_projector = block.fixed_basis @ block.fixed_basis.mT
        torch.testing.assert_close(
            fixed_projector,
            0.5 * (reynolds + reynolds.mT),
            atol=5.0e-8,
            rtol=5.0e-8,
        )

    encoder = PoseEncoder(compilation)
    rho_b_generators = encoder.representation(generators)
    rho_b_group = encoder.representation(group)
    lift = compile_intertwiners(
        generators,
        rho_b_generators,
    )
    assert lift.dimension == _character_hom_dimension(group, rho_b_group)
    assert lift.residual < 5.0e-8


def test_pose_encoding_has_independent_receiver_sender_gauge_law(
    compiled_case: CompiledCase,
) -> None:
    """B(g_i^T Q g_j) = rho_B(g_i^T) B(Q), independently for i and j."""
    case = compiled_case.group
    encoder = compiled_case.encoder
    relative_rotation = _random_rotation(8101)
    encoded = encoder(relative_rotation)

    receiver_indices = (0, 1, case.order // 2)
    sender_indices = (case.order - 1, case.order // 3, 1)
    for receiver_index, sender_index in zip(receiver_indices, sender_indices):
        receiver_gauge = case.group[receiver_index]
        sender_gauge = case.group[sender_index]
        transformed_rotation = (
            receiver_gauge.mT @ relative_rotation @ sender_gauge
        )
        actual = encoder(transformed_rotation)
        rho_receiver = encoder.representation(receiver_gauge.mT)
        expected = encoded @ rho_receiver.mT
        torch.testing.assert_close(actual, expected, atol=5.0e-8, rtol=5.0e-8)

        sender_only = encoder(relative_rotation @ sender_gauge)
        torch.testing.assert_close(
            sender_only,
            encoded,
            atol=5.0e-8,
            rtol=5.0e-8,
        )

    expected_dimension = 18 if case.name == "benzene" else 13
    assert encoder.encoding_dimension == expected_dimension
    assert encoded.shape == (expected_dimension,)


def test_pose_encoding_float32_runtime_law(
    compiled_case: CompiledCase,
) -> None:
    """The registered-buffer dtype path preserves the same law in float32."""
    case = compiled_case.group
    encoder = copy.deepcopy(compiled_case.encoder).to(dtype=torch.float32)
    relative_rotation = _random_rotation(8102).to(torch.float32)
    receiver_gauge = case.group[1].to(torch.float32)
    sender_gauge = case.group[-1].to(torch.float32)
    actual = encoder(receiver_gauge.mT @ relative_rotation @ sender_gauge)
    expected = encoder(relative_rotation) @ encoder.representation(
        receiver_gauge.mT
    ).mT
    torch.testing.assert_close(actual, expected, atol=8.0e-4, rtol=8.0e-4)


def test_all_aa_ab_ba_bb_lifts_are_complete_and_equivariant(
    compiled_case: CompiledCase,
) -> None:
    """Compile all four Lift types and check every basis map on the full group."""
    case = compiled_case.group
    encoder = compiled_case.encoder
    rho_a_group = case.group
    rho_b_group = encoder.representation(case.group)
    representation_specs = {
        "aa": (rho_a_group, rho_a_group),
        "ab": (rho_a_group, rho_b_group),
        "ba": (rho_b_group, rho_a_group),
        "bb": (rho_b_group, rho_b_group),
    }
    generator = torch.Generator().manual_seed(8200 + case.order)

    for name, compilation in compiled_case.lifts.items():
        rho_in, rho_out = representation_specs[name]
        basis = compilation.basis
        oracle_dimension = _character_hom_dimension(rho_in, rho_out)
        assert compilation.dimension == case.expected_lifts[name]
        assert compilation.dimension == oracle_dimension
        assert compilation.residual < 5.0e-8
        assert float(intertwiner_residual(basis, rho_in, rho_out)) < 8.0e-8

        flattened = basis.reshape(basis.shape[0], -1)
        torch.testing.assert_close(
            flattened @ flattened.mT,
            torch.eye(basis.shape[0], dtype=DTYPE),
            atol=ATOL,
            rtol=RTOL,
        )

        value = torch.randn(
            3,
            rho_in.shape[-1],
            dtype=DTYPE,
            generator=generator,
        )
        output = _lift_all(basis, value)
        for action_index in range(case.order):
            transformed_value = value @ rho_in[action_index].mT
            actual = _lift_all(basis, transformed_value)
            expected = output @ rho_out[action_index].mT
            torch.testing.assert_close(
                actual,
                expected,
                atol=8.0e-8,
                rtol=8.0e-8,
            )


def test_reference_covariants_match_autograd_and_transform_correctly(
    compiled_case: CompiledCase,
) -> None:
    """q is the exact position gradient and is covariant on every B block."""
    case = compiled_case.group
    encoder = compiled_case.encoder
    relative_rotation = _random_rotation(8301)
    blocks = encoder.encode_blocks(relative_rotation)

    for rank, block in blocks.items():
        generator = torch.Generator().manual_seed(8300 + rank)
        position = (
            0.35
            * torch.randn(3, dtype=DTYPE, generator=generator)
        ).requires_grad_(True)
        scalar = scalar_contraction(position, block, rank)
        automatic = torch.autograd.grad(scalar.sum(), position)[0]
        reference = vector_covariant(position.detach(), block, rank).sum(dim=-2)
        torch.testing.assert_close(
            automatic,
            reference,
            atol=8.0e-8,
            rtol=8.0e-8,
        )

        action = case.group[1]
        d_action = stf_representation(action, rank)
        transformed_position = action @ position.detach()
        transformed_block = d_action @ block
        transformed_scalar = scalar_contraction(
            transformed_position,
            transformed_block,
            rank,
        )
        transformed_vector = vector_covariant(
            transformed_position,
            transformed_block,
            rank,
        )
        torch.testing.assert_close(
            transformed_scalar,
            scalar.detach(),
            atol=8.0e-8,
            rtol=8.0e-8,
        )
        expected_vector = vector_covariant(
            position.detach(),
            block,
            rank,
        ) @ action.mT
        torch.testing.assert_close(
            transformed_vector,
            expected_vector,
            atol=1.0e-7,
            rtol=1.0e-7,
        )

        gradcheck_position = position.detach().clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda value: scalar_contraction(value, block, rank).sum(),
            (gradcheck_position,),
            eps=1.0e-6,
            atol=2.0e-5,
            rtol=2.0e-4,
        )


def test_covariant_gradients_are_finite_on_a_compact_edge_domain(
    compiled_case: CompiledCase,
) -> None:
    """Polynomial q need not be globally bounded, but must be stable on a cutoff."""
    encoder = compiled_case.encoder
    block_map = encoder.encode_blocks(_random_rotation(8401))
    directions = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (-1.0, 2.0, 0.5),
        ),
        dtype=DTYPE,
    )
    directions = directions / torch.linalg.vector_norm(
        directions,
        dim=-1,
        keepdim=True,
    )
    radii = torch.tensor((0.05, 0.5, 1.25), dtype=DTYPE).unsqueeze(-1)
    positions = (directions * radii).requires_grad_(True)

    for rank, block in block_map.items():
        values = vector_covariant(
            positions,
            block,
            rank,
            strict_same_edge=False,
        )
        assert bool(torch.isfinite(values).all())
        assert float(
            torch.linalg.vector_norm(values, dim=-1).amax().detach()
        ) < 1.0e4
        jacobian_probe = torch.autograd.grad(values.square().sum(), positions)[0]
        assert bool(torch.isfinite(jacobian_probe).all())
        assert float(torch.linalg.vector_norm(jacobian_probe)) < 1.0e6


def test_pose_lie_gradient_exists_and_passes_gradcheck(
    compiled_case: CompiledCase,
) -> None:
    """Differentiate B(exp([omega]_x)) in valid SO(3) coordinates."""
    encoder = compiled_case.encoder
    generator = torch.Generator().manual_seed(8500 + compiled_case.group.order)
    omega = (
        0.15 * torch.randn(3, dtype=DTYPE, generator=generator)
    ).requires_grad_(True)
    weights = torch.randn(
        encoder.encoding_dimension,
        dtype=DTYPE,
        generator=generator,
    )

    def objective(value: Tensor) -> Tensor:
        rotation = torch.matrix_exp(_skew(value))
        return (encoder(rotation) * weights).sum()

    assert torch.autograd.gradcheck(
        objective,
        (omega,),
        eps=1.0e-6,
        atol=3.0e-5,
        rtol=3.0e-4,
    )
    gradient = torch.autograd.grad(objective(omega), omega)[0]
    assert bool(torch.isfinite(gradient).all())
    assert 1.0e-10 < float(torch.linalg.vector_norm(gradient)) < 1.0e4


class TinyEdgeAB(nn.Module):
    """One typed A/B edge layer, typed sum pooling and an A-valued head."""

    def __init__(self, compiled: CompiledCase, *, seed: int = 9100) -> None:
        super().__init__()
        self.encoder = compiled.encoder
        for name, compilation in compiled.lifts.items():
            self.register_buffer(f"lift_{name}", compilation.basis.clone())

        generator = torch.Generator().manual_seed(seed + compiled.group.order)

        def parameter(count: int) -> nn.Parameter:
            return nn.Parameter(
                0.08
                * torch.randn(
                    count,
                    dtype=DTYPE,
                    generator=generator,
                )
            )

        self.edge_aa = parameter(compiled.lifts["aa"].dimension)
        self.edge_ab = parameter(compiled.lifts["ab"].dimension)
        self.edge_ba = parameter(compiled.lifts["ba"].dimension)
        self.edge_bb = parameter(compiled.lifts["bb"].dimension)
        self.head_aa = parameter(compiled.lifts["aa"].dimension)
        self.head_ba = parameter(compiled.lifts["ba"].dimension)
        self.radial_scale = nn.Parameter(torch.tensor(0.15, dtype=DTYPE))
        self.radial_bias = nn.Parameter(torch.tensor(-0.05, dtype=DTYPE))

    def _map(self, name: str, coefficients: Tensor, value: Tensor) -> Tensor:
        return _lift_mix(getattr(self, f"lift_{name}"), coefficients, value)

    def edge_features(
        self,
        centers: Tensor,
        frames: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return paired typed messages; A and B are never concatenated."""
        receiver, sender = edge_index
        displacement_world = centers[sender] - centers[receiver]
        receiver_frame = frames[receiver]
        sender_frame = frames[sender]
        position_a = torch.einsum(
            "eji,ej->ei",
            receiver_frame,
            displacement_world,
        )
        relative_rotation = receiver_frame.mT @ sender_frame
        pose_b = self.encoder(relative_rotation)

        radius_squared = position_a.square().sum(dim=-1, keepdim=True)
        invariant_gate = torch.sigmoid(
            self.radial_scale * radius_squared + self.radial_bias
        )

        message_a = invariant_gate * (
            self._map("aa", self.edge_aa, position_a)
            + self._map("ba", self.edge_ba, pose_b)
        )
        message_b = invariant_gate * (
            self._map("ab", self.edge_ab, position_a)
            + self._map("bb", self.edge_bb, pose_b)
        )
        return message_a, message_b

    def forward(
        self,
        centers: Tensor,
        frames: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        message_a, message_b = self.edge_features(centers, frames, edge_index)
        receiver = edge_index[0]
        pooled_a = centers.new_zeros((centers.shape[0], message_a.shape[-1]))
        pooled_b = centers.new_zeros((centers.shape[0], message_b.shape[-1]))
        pooled_a.index_add_(0, receiver, message_a)
        pooled_b.index_add_(0, receiver, message_b)

        body_vector = self._map("aa", self.head_aa, pooled_a) + self._map(
            "ba",
            self.head_ba,
            pooled_b,
        )
        return torch.einsum("nij,nj->ni", frames, body_vector)


def _sample_three_molecule_system() -> tuple[Tensor, Tensor, Tensor]:
    centers = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (1.2, -0.3, 0.5),
            (-0.4, 1.1, 0.8),
        ),
        dtype=DTYPE,
    )
    omegas = torch.tensor(
        (
            (0.1, -0.2, 0.05),
            (-0.3, 0.15, 0.2),
            (0.25, 0.1, -0.18),
        ),
        dtype=DTYPE,
    )
    frames = torch.matrix_exp(_skew(omegas))
    edge_index = torch.tensor(
        (
            (0, 0, 1, 1, 2, 2),
            (1, 2, 0, 2, 0, 1),
        ),
        dtype=torch.long,
    )
    return centers, frames, edge_index


def test_tiny_edge_ab_obeys_gauge_se3_permutation_and_edge_order(
    compiled_case: CompiledCase,
) -> None:
    """Exercise the actual A/B Lifts in one minimal multi-molecule forward."""
    case = compiled_case.group
    model = TinyEdgeAB(compiled_case)
    centers, frames, edge_index = _sample_three_molecule_system()
    baseline_edge_a, baseline_edge_b = model.edge_features(
        centers,
        frames,
        edge_index,
    )
    baseline = model(centers, frames, edge_index)

    gauge_indices = torch.tensor(
        (1, case.order // 3, case.order - 1),
        dtype=torch.long,
    )
    gauges = case.group[gauge_indices]
    gauged_frames = frames @ gauges
    gauged_edge_a, gauged_edge_b = model.edge_features(
        centers,
        gauged_frames,
        edge_index,
    )
    receiver = edge_index[0]
    expected_edge_a = torch.einsum(
        "eij,ej->ei",
        gauges[receiver].mT,
        baseline_edge_a,
    )
    rho_b_receiver = model.encoder.representation(gauges[receiver].mT)
    expected_edge_b = torch.einsum(
        "eij,ej->ei",
        rho_b_receiver,
        baseline_edge_b,
    )
    torch.testing.assert_close(
        gauged_edge_a,
        expected_edge_a,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    torch.testing.assert_close(
        gauged_edge_b,
        expected_edge_b,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    torch.testing.assert_close(
        model(centers, gauged_frames, edge_index),
        baseline,
        atol=1.0e-7,
        rtol=1.0e-7,
    )

    global_rotation = _random_rotation(9201)
    translation = torch.tensor((0.3, -0.7, 1.1), dtype=DTYPE)
    moved_centers = centers @ global_rotation.mT + translation
    moved_frames = global_rotation @ frames
    moved_output = model(moved_centers, moved_frames, edge_index)
    torch.testing.assert_close(
        moved_output,
        baseline @ global_rotation.mT,
        atol=1.0e-7,
        rtol=1.0e-7,
    )

    edge_permutation = torch.tensor((4, 0, 5, 2, 1, 3), dtype=torch.long)
    torch.testing.assert_close(
        model(centers, frames, edge_index[:, edge_permutation]),
        baseline,
        atol=1.0e-7,
        rtol=1.0e-7,
    )

    node_permutation = torch.tensor((2, 0, 1), dtype=torch.long)
    inverse = torch.empty_like(node_permutation)
    inverse[node_permutation] = torch.arange(3)
    permuted_edge_index = inverse[edge_index]
    permuted_output = model(
        centers[node_permutation],
        frames[node_permutation],
        permuted_edge_index,
    )
    torch.testing.assert_close(
        permuted_output,
        baseline[node_permutation],
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_tiny_edge_ab_has_finite_gradients_and_one_optimizer_step(
    compiled_case: CompiledCase,
) -> None:
    """Backpropagate through A/B and take one ordinary finite SGD step."""
    model = TinyEdgeAB(compiled_case, seed=9300)
    centers, base_frames, edge_index = _sample_three_molecule_system()
    centers = centers.clone().requires_grad_(True)
    omega_delta = torch.zeros(3, 3, dtype=DTYPE, requires_grad=True)
    frames = base_frames @ torch.matrix_exp(_skew(omega_delta))

    prediction = model(centers, frames, edge_index)
    generator = torch.Generator().manual_seed(9301 + compiled_case.group.order)
    target = prediction.detach() + 0.2 * torch.randn(
        prediction.shape,
        dtype=DTYPE,
        generator=generator,
    )
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()

    tensors_to_check = [centers.grad, omega_delta.grad]
    tensors_to_check.extend(parameter.grad for parameter in model.parameters())
    assert all(value is not None for value in tensors_to_check)
    assert all(bool(torch.isfinite(value).all()) for value in tensors_to_check)
    gradient_norm = torch.sqrt(
        sum(value.square().sum() for value in tensors_to_check if value is not None)
    )
    assert 0.0 < float(gradient_norm) < 1.0e6

    parameters = list(model.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    old_loss = float(loss.detach())
    parameter_gradient_norm = torch.sqrt(
        sum(
            parameter.grad.square().sum()
            for parameter in parameters
            if parameter.grad is not None
        )
    )
    assert 0.0 < float(parameter_gradient_norm) < 1.0e6

    candidate_learning_rate = 1.0e-3 / (
        1.0 + float(parameter_gradient_norm)
    )
    accepted_learning_rate: float | None = None
    with torch.no_grad():
        for _ in range(16):
            for parameter, initial in zip(parameters, before):
                if parameter.grad is not None:
                    parameter.copy_(
                        initial - candidate_learning_rate * parameter.grad
                    )
            trial_loss = torch.nn.functional.mse_loss(
                model(centers.detach(), base_frames, edge_index),
                target,
            )
            if bool(torch.isfinite(trial_loss)) and float(trial_loss) < old_loss:
                accepted_learning_rate = candidate_learning_rate
                break
            candidate_learning_rate *= 0.25
        for parameter, initial in zip(parameters, before):
            parameter.copy_(initial)
    assert accepted_learning_rate is not None

    optimizer = torch.optim.SGD(parameters, lr=accepted_learning_rate)
    optimizer.step()

    after = list(model.parameters())
    assert any(
        not torch.equal(old_parameter, new_parameter.detach())
        for old_parameter, new_parameter in zip(before, after)
    )
    new_prediction = model(
        centers.detach(),
        base_frames,
        edge_index,
    )
    new_loss = torch.nn.functional.mse_loss(new_prediction, target)
    assert bool(torch.isfinite(new_loss))
    assert float(new_loss.detach()) < old_loss
