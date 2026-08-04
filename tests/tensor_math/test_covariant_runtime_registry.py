"""Validate differentiable C runtime, views, oracles, and registry persistence."""

from __future__ import annotations

import json

import pytest
import torch

import TFENN.tensor_math.covariant_registry as registry_module
from TFENN.tensor_math import (
    CRegistry,
    CSignature,
    CSlot,
    CovariantBasisView,
    RegisteredCovariant,
    TRIVIAL_SCALAR,
    TypeKey,
    compile_covariant_basis,
    scalar_contraction,
    stf_basis,
    vector_covariant,
)

from ._covariant_cases import (
    benzene_covariant_case,
    methane_covariant_case,
    transformed_inputs,
)
from ._groups import ATOL, DTYPE, RTOL


A = TypeKey("A")
B0 = TypeKey("B", 0)


def distinct(name: str, key: TypeKey) -> CSlot:
    """Return one independent slot."""
    return CSlot(name, key)


def symmetric_square(name: str, key: TypeKey) -> CSlot:
    """Return one repeated variable slot."""
    return CSlot(name, key, 2, "symmetric_power")


def output_action(case, signature: CSignature, word: tuple[int, ...]) -> torch.Tensor:
    """Return the target action for one ordered generator word."""
    if signature.output == TRIVIAL_SCALAR:
        return torch.ones((1, 1), dtype=DTYPE)
    block = case.catalog.resolve(signature.output)
    action = torch.eye(block.representation_dim, dtype=DTYPE)
    for index in word:
        action = block.actions[index] @ action
    return action


@pytest.mark.parametrize(
    "case_factory,signature",
    (
        (
            benzene_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", B0))),
        ),
        (
            benzene_covariant_case,
            CSignature(B0, (symmetric_square("a", A),)),
        ),
        (
            benzene_covariant_case,
            CSignature(B0, (distinct("left", B0), distinct("right", B0))),
        ),
        (
            methane_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", B0))),
        ),
        (
            methane_covariant_case,
            CSignature(B0, (symmetric_square("a", A),)),
        ),
        (
            methane_covariant_case,
            CSignature(TRIVIAL_SCALAR, (symmetric_square("a", A),)),
        ),
    ),
)
def test_random_generator_words_act_covariantly(case_factory, signature) -> None:
    """Check complete basis covariance without closing or averaging the group."""
    case = case_factory()
    artifact = compile_covariant_basis(case.catalog, signature)
    module = RegisteredCovariant(artifact)
    generator = torch.Generator().manual_seed(1701)
    inputs = {
        slot.name: torch.randn(
            (4, case.catalog.resolve(slot.type_key).representation_dim),
            dtype=DTYPE,
            generator=generator,
        )
        for slot in signature.inputs
    }
    baseline = module.evaluate_basis(inputs)
    slot_keys = {slot.name: slot.type_key for slot in signature.inputs}
    for length in (0, 1, 2, 5, 9):
        word = tuple(
            torch.randint(
                len(case.catalog.generator_system.names),
                (),
                generator=generator,
            ).item()
            for _ in range(length)
        )
        transformed = transformed_inputs(inputs, slot_keys, case.catalog, word)
        actual = module.evaluate_basis(transformed)
        action = output_action(case, signature, word)
        expected = torch.einsum("op,bap->bao", action, baseline)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "case_factory,signature",
    (
        (
            benzene_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", B0))),
        ),
        (
            benzene_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", TypeKey("B", 1)))),
        ),
        (
            benzene_covariant_case,
            CSignature(B0, (symmetric_square("a", A),)),
        ),
        (
            benzene_covariant_case,
            CSignature(B0, (distinct("left", B0), distinct("right", B0))),
        ),
        (
            benzene_covariant_case,
            CSignature(
                TRIVIAL_SCALAR,
                (symmetric_square("a", A), distinct("b", B0)),
            ),
        ),
        (
            benzene_covariant_case,
            CSignature(A, (CSlot("a", A, 3, "symmetric_power"),)),
        ),
        (
            methane_covariant_case,
            CSignature(A, (distinct("a", A), distinct("b", B0))),
        ),
    ),
)
def test_runtime_gradcheck_for_every_live_slot_and_coefficients(
    case_factory,
    signature,
) -> None:
    """Check every slot and future invariant coefficient path remains differentiable."""
    case = case_factory()
    artifact = compile_covariant_basis(case.catalog, signature)
    module = RegisteredCovariant(artifact)
    generator = torch.Generator().manual_seed(1801)
    values = tuple(
        torch.randn(
            (2, case.catalog.resolve(slot.type_key).representation_dim),
            dtype=DTYPE,
            generator=generator,
            requires_grad=True,
        )
        for slot in signature.inputs
    )
    coefficients = torch.randn(
        (2, artifact.basis_dimension),
        dtype=DTYPE,
        generator=generator,
        requires_grad=True,
    )

    def function(*arguments):
        live = {
            slot.name: value
            for slot, value in zip(signature.inputs, arguments[:-1])
        }
        return module.apply_coefficients(live, arguments[-1])

    assert torch.autograd.gradcheck(
        function,
        (*values, coefficients),
        eps=1e-6,
        atol=4e-5,
        rtol=4e-4,
    )


def test_runtime_broadcasting_requires_explicit_channel_singletons() -> None:
    """Broadcast only caller supplied leading axes and preserve empty batches."""
    case = benzene_covariant_case()
    signature = CSignature(A, (distinct("a", A), distinct("b", B0)))
    artifact = compile_covariant_basis(case.catalog, signature)
    module = RegisteredCovariant(artifact)
    a = torch.randn((2, 3, 1, 3), dtype=DTYPE)
    b = torch.randn((2, 1, 4, 5), dtype=DTYPE)
    output = module.evaluate_basis({"a": a, "b": b})
    assert output.shape == (2, 3, 4, artifact.basis_dimension, 3)

    manual_source = torch.einsum("...i,...j->...ij", a, b).reshape(2, 3, 4, 15)
    expected = torch.einsum("boi,...i->...bo", artifact.basis, manual_source)
    torch.testing.assert_close(output, expected)

    empty = module.evaluate_basis(
        {
            "a": torch.empty((0, 3), dtype=DTYPE),
            "b": torch.empty((0, 5), dtype=DTYPE),
        }
    )
    assert empty.shape == (0, artifact.basis_dimension, 3)
    with pytest.raises(ValueError, match="exactly match"):
        module.evaluate_basis({"a": a, "extra": b})
    with pytest.raises(ValueError, match="final dimension"):
        module.evaluate_basis(
            {"a": torch.randn(2, 4, dtype=DTYPE), "b": torch.randn(2, 5, dtype=DTYPE)}
        )

    symmetric_artifact = compile_covariant_basis(
        case.catalog,
        CSignature(B0, (symmetric_square("a", A),)),
    )
    symmetric_empty = RegisteredCovariant(symmetric_artifact).evaluate_basis(
        {"a": torch.empty((0, 3), dtype=DTYPE)}
    )
    assert symmetric_empty.shape == (
        0,
        symmetric_artifact.basis_dimension,
        5,
    )


def test_zero_hom_runtime_has_valid_empty_basis_and_fused_output() -> None:
    """Preserve zero dimensional Hom shapes through all runtime operations."""
    case = methane_covariant_case()
    signature = CSignature(TRIVIAL_SCALAR, (distinct("a", A),))
    artifact = compile_covariant_basis(case.catalog, signature)
    module = RegisteredCovariant(artifact)
    value = torch.randn((5, 3), dtype=DTYPE, requires_grad=True)
    basis_output = module.evaluate_basis({"a": value})
    assert basis_output.shape == (5, 0, 1)
    coefficients = torch.empty((5, 0), dtype=DTYPE, requires_grad=True)
    fused = module.apply_coefficients({"a": value}, coefficients)
    assert fused.shape == (5, 1)
    torch.testing.assert_close(fused, torch.zeros_like(fused))


def test_registered_covariant_has_only_buffers_and_migrates_dtype() -> None:
    """Check frozen C assets migrate without becoming trainable parameters."""
    case = benzene_covariant_case()
    signature = CSignature(B0, (symmetric_square("a", A),))
    artifact = compile_covariant_basis(case.catalog, signature)
    module = RegisteredCovariant(artifact).float()
    exposed = artifact.basis
    exposed.zero_()
    assert bool(module.basis.abs().amax() > 0.0)
    assert tuple(module.parameters()) == ()
    names = dict(module.named_buffers())
    assert set(names) == {"_basis", "_slot_lift_0"}
    assert all(value.dtype == torch.float32 for value in names.values())
    value = torch.randn((3, 3), dtype=torch.float32, requires_grad=True)
    output = module.evaluate_basis({"a": value})
    output.square().sum().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())


def test_analytic_vector_and_scalar_kernels_lie_in_complete_basis() -> None:
    """Project existing rank two reference kernels into complete finite group spaces."""
    case = benzene_covariant_case()
    vector_signature = CSignature(A, (distinct("a", A), distinct("b", B0)))
    vector_artifact = compile_covariant_basis(case.catalog, vector_signature)
    vector_matrix = torch.empty((3, 3, 5), dtype=DTYPE)
    for input_index in range(3):
        for pose_index in range(5):
            position = torch.eye(3, dtype=DTYPE)[input_index]
            pose = torch.eye(5, dtype=DTYPE)[pose_index, :, None]
            vector_matrix[:, input_index, pose_index] = vector_covariant(
                position,
                pose,
                2,
            )[0]
    vector_flat = vector_matrix.reshape(3, 15)
    vector_coefficients = torch.einsum(
        "boi,oi->b",
        vector_artifact.basis,
        vector_flat,
    )
    reconstructed_vector = torch.einsum(
        "b,boi->oi",
        vector_coefficients,
        vector_artifact.basis,
    )
    torch.testing.assert_close(
        reconstructed_vector,
        vector_flat,
        atol=5e-12,
        rtol=5e-12,
    )

    scalar_signature = CSignature(
        TRIVIAL_SCALAR,
        (symmetric_square("a", A), distinct("b", B0)),
    )
    scalar_artifact = compile_covariant_basis(case.catalog, scalar_signature)
    scalar_flat = stf_basis(2, dtype=DTYPE).reshape(1, 30)
    scalar_coefficients = torch.einsum(
        "boi,oi->b",
        scalar_artifact.basis,
        scalar_flat,
    )
    reconstructed_scalar = torch.einsum(
        "b,boi->oi",
        scalar_coefficients,
        scalar_artifact.basis,
    )
    torch.testing.assert_close(
        reconstructed_scalar,
        scalar_flat,
        atol=5e-12,
        rtol=5e-12,
    )

    generator = torch.Generator().manual_seed(1901)
    position = torch.randn((6, 3), dtype=DTYPE, generator=generator)
    pose = torch.randn((6, 5), dtype=DTYPE, generator=generator)
    vector_module = RegisteredCovariant(vector_artifact)
    scalar_module = RegisteredCovariant(scalar_artifact)
    actual_vector = vector_module.apply_coefficients(
        {"a": position, "b": pose},
        vector_coefficients,
    )
    expected_vector = vector_covariant(position, pose[..., :, None], 2).squeeze(-2)
    torch.testing.assert_close(actual_vector, expected_vector, atol=5e-12, rtol=5e-12)
    actual_scalar = scalar_module.apply_coefficients(
        {"a": position, "b": pose},
        scalar_coefficients,
    )
    expected_scalar = scalar_contraction(
        position,
        pose[..., :, None],
        2,
    )
    torch.testing.assert_close(actual_scalar, expected_scalar, atol=5e-12, rtol=5e-12)


def test_basis_views_preserve_equivariance_and_require_opt_in() -> None:
    """Separate exact equivariance from single layer basis completeness."""
    case = benzene_covariant_case()
    signature = CSignature(A, (distinct("a", A), distinct("b", B0)))
    artifact = compile_covariant_basis(case.catalog, signature)
    identity_view = CovariantBasisView(
        artifact,
        torch.eye(artifact.basis_dimension, dtype=DTYPE),
    )
    incomplete_view = CovariantBasisView(
        artifact,
        torch.eye(artifact.basis_dimension, dtype=DTYPE)[:-1],
    )
    deficient = torch.eye(artifact.basis_dimension, dtype=DTYPE)
    deficient[-1] = deficient[0]
    deficient_view = CovariantBasisView(artifact, deficient)
    tall_view = CovariantBasisView(
        artifact,
        torch.cat(
            (
                torch.eye(artifact.basis_dimension, dtype=DTYPE),
                torch.zeros((1, artifact.basis_dimension), dtype=DTYPE),
            )
        ),
    )
    assert identity_view.is_complete
    assert tall_view.is_complete
    assert not incomplete_view.is_complete
    assert not deficient_view.is_complete
    assert incomplete_view.transform_rank == artifact.basis_dimension - 1
    assert incomplete_view.cost.reduction_ratio == pytest.approx(
        (artifact.basis_dimension - 1) / artifact.basis_dimension
    )
    module = RegisteredCovariant(artifact, (identity_view, incomplete_view))
    inputs = {
        "a": torch.randn((4, 3), dtype=DTYPE),
        "b": torch.randn((4, 5), dtype=DTYPE),
    }
    full = module.evaluate_basis(inputs)
    identity = module.evaluate_view(inputs, identity_view)
    torch.testing.assert_close(identity, full)
    with pytest.raises(ValueError, match="opt in"):
        module.evaluate_view(inputs, incomplete_view)
    reduced = module.evaluate_view(
        inputs,
        incomplete_view,
        allow_incomplete=True,
    )
    expected = torch.einsum("af,...fo->...ao", incomplete_view.transform, full)
    torch.testing.assert_close(reduced, expected)
    exposed_transform = incomplete_view.transform
    exposed_transform.zero_()
    repeated = module.evaluate_view(
        inputs,
        incomplete_view,
        allow_incomplete=True,
    )
    torch.testing.assert_close(repeated, expected)
    assert tuple(module.parameters()) == ()
    assert "_view_transform_0" in dict(module.named_buffers())


def test_registry_save_load_is_deterministic_and_validates_views(
    tmp_path,
    monkeypatch,
) -> None:
    """Check byte stability, catalog binding, and incomplete view protection."""
    case = benzene_covariant_case()
    signature = CSignature(A, (distinct("a", A), distinct("b", B0)))
    artifact = compile_covariant_basis(case.catalog, signature)
    view = CovariantBasisView(
        artifact,
        torch.eye(artifact.basis_dimension, dtype=DTYPE)[:-1],
    )
    registry = CRegistry()
    registry.register(artifact)
    registry.register(view)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    registry.save(first)
    registry.save(second)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(RuntimeError, match="incomplete"):
        CRegistry.load(first)

    def forbidden(*args, **kwargs):
        raise AssertionError("registry load attempted compilation")

    monkeypatch.setattr(registry_module, "compile_intertwiners", forbidden)
    loaded = CRegistry.load(first, require_complete=False)
    third = tmp_path / "third.json"
    loaded.save(third)
    assert first.read_bytes() == third.read_bytes()
    assert loaded.fingerprint == registry.fingerprint
    assert (
        loaded.resolve(signature, case.catalog.fingerprint).artifact_fingerprint
        == artifact.artifact_fingerprint
    )
    with pytest.raises(ValueError, match="require_complete"):
        loaded.resolve(
            signature,
            case.catalog.fingerprint,
            view_fingerprint=view.view_fingerprint,
        )
    resolved_view = loaded.resolve(
        signature,
        case.catalog.fingerprint,
        view_fingerprint=view.view_fingerprint,
        require_complete=False,
    )
    assert resolved_view.view_fingerprint == view.view_fingerprint
    with pytest.raises(KeyError):
        loaded.resolve(signature, methane_covariant_case().catalog.fingerprint)


def test_registry_bytes_do_not_depend_on_artifact_insertion_order(tmp_path) -> None:
    """Sort complete primary keys before deterministic persistence."""
    case = benzene_covariant_case()
    first_artifact = compile_covariant_basis(
        case.catalog,
        CSignature(A, (distinct("a", A), distinct("b", B0))),
    )
    second_artifact = compile_covariant_basis(
        case.catalog,
        CSignature(B0, (symmetric_square("a", A),)),
    )
    first_registry = CRegistry()
    first_registry.register(first_artifact)
    first_registry.register(second_artifact)
    second_registry = CRegistry()
    second_registry.register(second_artifact)
    second_registry.register(first_artifact)
    first_path = tmp_path / "ordered.json"
    second_path = tmp_path / "reversed.json"
    first_registry.save(first_path)
    second_registry.save(second_path)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_registry_tamper_and_runtime_recompilation_are_rejected(
    tmp_path,
    monkeypatch,
) -> None:
    """Ensure persistence and forward never repair or rebuild frozen artifacts."""
    case = benzene_covariant_case()
    signature = CSignature(B0, (symmetric_square("a", A),))
    artifact = compile_covariant_basis(case.catalog, signature)
    registry = CRegistry()
    registry.register(artifact)
    path = tmp_path / "registry.json"
    registry.save(path)
    payload = json.loads(path.read_text(encoding="utf8"))
    payload["artifacts"][0]["artifact"]["basis"]["data"] = "AAAA"
    path.write_text(json.dumps(payload), encoding="utf8")
    with pytest.raises(RuntimeError):
        CRegistry.load(path)

    module = RegisteredCovariant(artifact)
    corrupted = compile_covariant_basis(case.catalog, signature)
    corrupted_registry = CRegistry()
    corrupted_registry.register(corrupted)

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime attempted compilation")

    monkeypatch.setattr(registry_module, "compile_intertwiners", forbidden)
    output = module.evaluate_basis({"a": torch.randn((2, 3), dtype=DTYPE)})
    assert output.shape == (2, artifact.basis_dimension, 5)

    corrupted._basis[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="fingerprint"):
        corrupted_registry.resolve(signature, case.catalog.fingerprint)
    with pytest.raises(RuntimeError, match="fingerprint"):
        corrupted_registry.register(artifact)
    with pytest.raises(RuntimeError, match="fingerprint"):
        RegisteredCovariant(corrupted)

    view = CovariantBasisView(
        artifact,
        torch.eye(artifact.basis_dimension, dtype=DTYPE),
    )
    view_registry = CRegistry()
    view_registry.register(artifact)
    view_registry.register(view)
    artifact._basis[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="fingerprint"):
        _ = view_registry.views
    with pytest.raises(RuntimeError, match="fingerprint"):
        view_registry.register(view)


def test_registry_json_validation_distinguishes_numeric_types(tmp_path) -> None:
    """Reject bool schema values and float dimensions that compare equal in Python."""
    case = benzene_covariant_case()
    signature = CSignature(A, (distinct("a", A), distinct("b", B0)))
    artifact = compile_covariant_basis(case.catalog, signature)
    registry = CRegistry()
    registry.register(artifact)
    source = tmp_path / "source.json"
    registry.save(source)
    original = json.loads(source.read_text(encoding="utf8"))

    schema_payload = dict(original)
    schema_payload["schema_version"] = True
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema_payload), encoding="utf8")
    with pytest.raises(RuntimeError):
        CRegistry.load(schema_path)

    dimension_payload = json.loads(source.read_text(encoding="utf8"))
    dimension_payload["artifacts"][0]["artifact"]["input_dimensions"][0] = 3.0
    dimension_path = tmp_path / "dimension.json"
    dimension_path.write_text(json.dumps(dimension_payload), encoding="utf8")
    with pytest.raises(RuntimeError):
        CRegistry.load(dimension_path)

    fingerprint_payload = json.loads(source.read_text(encoding="utf8"))
    fingerprint_payload["artifacts"][0]["artifact"][
        "generator_fingerprint"
    ] = "x"
    fingerprint_path = tmp_path / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint_payload), encoding="utf8")
    with pytest.raises(RuntimeError):
        CRegistry.load(fingerprint_path)

    nonfinite_payload = json.loads(source.read_text(encoding="utf8"))
    nonfinite_payload["unexpected"] = float("nan")
    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(json.dumps(nonfinite_payload), encoding="utf8")
    with pytest.raises(RuntimeError):
        CRegistry.load(nonfinite_path)

    view = CovariantBasisView(
        artifact,
        torch.eye(artifact.basis_dimension, dtype=DTYPE),
    )
    view_registry = CRegistry()
    view_registry.register(artifact)
    view_registry.register(view)
    view_path = tmp_path / "view.json"
    view_registry.save(view_path)
    view_payload = json.loads(view_path.read_text(encoding="utf8"))
    view_payload["views"][0]["view"]["parent_artifact_fingerprint"] = []
    view_path.write_text(json.dumps(view_payload), encoding="utf8")
    with pytest.raises(RuntimeError):
        CRegistry.load(view_path)


def test_registered_state_dict_binds_exact_artifact_and_rolls_back() -> None:
    """Reject changed frozen buffers even when shapes and metadata still match."""
    case = benzene_covariant_case()
    signature = CSignature(A, (distinct("a", A), distinct("b", B0)))
    artifact = compile_covariant_basis(case.catalog, signature)
    source = RegisteredCovariant(artifact)
    target = RegisteredCovariant(artifact)
    state = source.state_dict()
    target.load_state_dict(state)
    before = target.basis
    tampered = dict(state)
    tampered["_basis"] = state["_basis"].clone()
    tampered["_basis"][0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="artifact"):
        target.load_state_dict(tampered, strict=False)
    torch.testing.assert_close(target.basis, before, atol=0.0, rtol=0.0)
    missing = dict(state)
    missing.pop("_extra_state")
    with pytest.raises(RuntimeError, match="unversioned"):
        target.load_state_dict(missing, strict=False)

    for mutate in (
        lambda extra: extra.update(state_schema_version=True),
        lambda extra: extra["signature"]["inputs"][0].update(power=1.0),
        lambda extra: extra.update(
            basis_shape=(float(extra["basis_shape"][0]), *extra["basis_shape"][1:])
        ),
    ):
        typed_alias = dict(state)
        typed_alias["_extra_state"] = json.loads(
            json.dumps(state["_extra_state"])
        )
        typed_alias["_extra_state"]["basis_shape"] = tuple(
            typed_alias["_extra_state"]["basis_shape"]
        )
        typed_alias["_extra_state"]["slot_descriptors"] = tuple(
            tuple(item) for item in typed_alias["_extra_state"]["slot_descriptors"]
        )
        typed_alias["_extra_state"]["view_descriptors"] = tuple(
            tuple(item) for item in typed_alias["_extra_state"]["view_descriptors"]
        )
        mutate(typed_alias["_extra_state"])
        with pytest.raises(RuntimeError, match="metadata"):
            target.load_state_dict(typed_alias, strict=False)

    source32 = RegisteredCovariant(artifact).float()
    target64 = RegisteredCovariant(artifact)
    target64.load_state_dict(source32.state_dict())
    torch.testing.assert_close(
        target64.basis,
        artifact.basis.float().double(),
        atol=0.0,
        rtol=0.0,
    )
    previous64 = RegisteredCovariant(artifact).basis
    invalid_cross_dtype = dict(source32.state_dict())
    invalid_cross_dtype["unexpected"] = torch.ones((), dtype=torch.float32)
    target64 = RegisteredCovariant(artifact)
    with pytest.raises(RuntimeError):
        target64.load_state_dict(invalid_cross_dtype)
    torch.testing.assert_close(
        target64.basis,
        previous64,
        atol=0.0,
        rtol=0.0,
    )


def test_registered_runtime_compiles_as_one_full_graph() -> None:
    """Keep compilation outside the differentiable tensor contraction graph."""
    case = benzene_covariant_case()
    artifact = compile_covariant_basis(
        case.catalog,
        CSignature(A, (distinct("a", A), distinct("b", B0))),
    )
    module = RegisteredCovariant(artifact)
    inputs = {
        "a": torch.randn((2, 3), dtype=DTYPE, requires_grad=True),
        "b": torch.randn((2, 5), dtype=DTYPE, requires_grad=True),
    }
    expected = module(inputs)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    actual = compiled(inputs)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert inputs["a"].grad is not None
    assert inputs["b"].grad is not None


def test_zero_parent_views_follow_exact_rank_completeness_definition() -> None:
    """Apply rank equals full dimension even with positive active zero maps."""
    case = methane_covariant_case()
    artifact = compile_covariant_basis(
        case.catalog,
        CSignature(TRIVIAL_SCALAR, (distinct("a", A),)),
    )
    view = CovariantBasisView(artifact, torch.empty((0, 0), dtype=DTYPE))
    assert view.is_complete
    assert view.transform_rank == 0
    assert view.cost.reduction_ratio == 1.0
    active = CovariantBasisView(artifact, torch.empty((2, 0), dtype=DTYPE))
    assert active.is_complete
    assert active.active_dimension == 2
    module = RegisteredCovariant(artifact, (active,))
    output = module.evaluate_view(
        {"a": torch.randn((3, 3), dtype=DTYPE)},
        active,
    )
    assert output.shape == (3, 2, 1)
    torch.testing.assert_close(output, torch.zeros_like(output))
