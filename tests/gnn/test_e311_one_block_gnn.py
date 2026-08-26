from __future__ import annotations

from dataclasses import asdict

import torch
from pytest import fixture

from TFENN.models import (
    E311MultibodyMessageBlockV1,
)
from experiments.gnn.e311_one_block_gnn import (
    E311OneBlockGNN,
    E311OneBlockGNNConfig,
    benzene_generators,
)


D6_ABSOLUTE_TOLERANCE = 5.0e-11
SO3_ABSOLUTE_TOLERANCE = 2.0e-10


@fixture(scope="module")
def model() -> E311OneBlockGNN:
    torch.manual_seed(20260826)
    result = E311OneBlockGNN(
        0.7,
        E311OneBlockGNNConfig(),
        dtype=torch.float64,
    )
    result.eval()
    return result


def _inputs(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(314159)
    centers = torch.randn(
        batch_size,
        2,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    vectors = torch.randn(
        batch_size,
        2,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    x, y, z = vectors.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(batch_size, 2, 3, 3)
    return centers, torch.matrix_exp(skew)


def _running_rms_buffers(model: E311OneBlockGNN) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if name.endswith(("mean_square", "sample_count"))
    }


def _assert_state_value_equal(expected: object, actual: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(expected, actual)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_state_value_equal(expected[key], actual[key])
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_state_value_equal(expected_item, actual_item)
        return
    assert expected == actual


def test_one_block_and_parameter_configuration(
    model: E311OneBlockGNN,
) -> None:
    assert isinstance(model.message_block, E311MultibodyMessageBlockV1)
    assert model.message_block_count == 1
    assert model.pair_count == 1
    assert model.config.molecule_count == 2
    assert model.config.hidden_b_channels == 1
    assert model.config.edge_a_channels == 1
    assert model.config.a_mid_channels == 1
    assert model.config.b_wide_channels == 2
    assert model.config.b_out_channels == 1
    assert model.config.gate_width == 8
    assert model.trainable_parameter_count == 37_267
    assert model.block_configuration() == {
        "message_block_count": 1,
        **asdict(model.config),
    }
    assert torch.equal(
        model.pair_index,
        torch.tensor(((0,), (1,)), dtype=torch.int64),
    )


def test_zero_initial_state_and_output_shapes(
    model: E311OneBlockGNN,
) -> None:
    model.eval()
    centers, rotations = _inputs()
    hidden = model.initial_hidden_b(centers)
    edge_a = model.initial_edge_a_world(centers)
    output = model.one_block_output(centers, rotations)
    assert all(torch.count_nonzero(value) == 0 for value in hidden.values())
    assert torch.count_nonzero(edge_a) == 0
    assert output.pair_force_world.shape == (3, 1, 3)
    assert output.node_force_world.shape == (3, 2, 3)
    assert output.edge_a_world.shape == (3, 1, 1, 3)
    assert output.trace is None
    for key in model.b_keys:
        representation_dim = model.catalog.resolve(key).representation_dim
        assert hidden[key].shape == (3, 2, 1, representation_dim)
        assert output.message_j_to_i_local[key].shape == (
            3,
            1,
            1,
            representation_dim,
        )
        assert output.message_i_to_j_local[key].shape == (
            3,
            1,
            1,
            representation_dim,
        )
        assert output.node_b_local[key].shape == (
            3,
            2,
            1,
            representation_dim,
        )


def test_strict_total_force_conservation(
    model: E311OneBlockGNN,
) -> None:
    model.eval()
    centers, rotations = _inputs(4)
    node_force, pair_force = model.normalized_forces_and_pairs_world(
        centers,
        rotations,
    )
    assert torch.equal(node_force[:, 0], pair_force[:, 0])
    assert torch.equal(node_force[:, 1], -pair_force[:, 0])
    assert torch.equal(
        node_force.sum(dim=1),
        torch.zeros_like(node_force[:, 0]),
    )


def test_finite_forward_and_backward_gradients(
    model: E311OneBlockGNN,
) -> None:
    model.reset_running_rms()
    model.train()
    model.zero_grad(set_to_none=True)
    centers, rotations = _inputs(2)
    centers = centers.detach().requires_grad_(True)
    rotations = rotations.detach().requires_grad_(True)
    _, pair_force = model.normalized_forces_and_pairs_world(centers, rotations)
    target = torch.tensor(
        (((0.3, -0.2, 0.1),), ((-0.1, 0.4, 0.2),)),
        dtype=torch.float64,
    )
    loss = (pair_force - target).square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert torch.isfinite(pair_force).all()
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert centers.grad is not None and torch.isfinite(centers.grad).all()
    assert rotations.grad is not None and torch.isfinite(rotations.grad).all()
    model.zero_grad(set_to_none=True)
    model.eval()


def test_single_benzene_sixty_degree_d6_invariance(
    model: E311OneBlockGNN,
) -> None:
    model.eval()
    centers, rotations = _inputs(1)
    reference = model.normalized_forces_world(centers, rotations)
    changed_rotations = rotations.clone()
    changed_rotations[:, 1] = (
        rotations[:, 1] @ benzene_generators(torch.float64)[0]
    )
    actual = model.normalized_forces_world(centers, changed_rotations)
    residual = float((actual - reference).detach().abs().max())
    assert residual <= D6_ABSOLUTE_TOLERANCE


def test_global_so3_equivariance(model: E311OneBlockGNN) -> None:
    model.eval()
    centers, rotations = _inputs(2)
    rotation_vector = torch.tensor(
        (0.31, -0.42, 0.17),
        dtype=torch.float64,
    )
    x, y, z = rotation_vector
    zero = rotation_vector.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(
        3,
        3,
    )
    global_rotation = torch.matrix_exp(skew)
    reference = model.normalized_forces_world(centers, rotations)
    actual = model.normalized_forces_world(
        centers @ global_rotation.mT,
        global_rotation @ rotations,
    )
    expected = reference @ global_rotation.mT
    residual = float((actual - expected).detach().abs().max())
    assert residual <= SO3_ABSOLUTE_TOLERANCE


def test_eval_preserves_running_rms_buffers(
    model: E311OneBlockGNN,
) -> None:
    centers, rotations = _inputs(2)
    model.reset_running_rms()
    model.train()
    with torch.no_grad():
        model.normalized_forces_world(centers, rotations)
    model.eval()
    before = _running_rms_buffers(model)
    assert before
    sample_counts = {
        name: value
        for name, value in before.items()
        if name.endswith("sample_count")
    }
    assert sample_counts
    assert all(int(value) > 0 for value in sample_counts.values())
    with torch.no_grad():
        model.normalized_forces_world(centers, rotations)
        model.normalized_forces_world(centers, rotations)
    after = _running_rms_buffers(model)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_checkpoint_config_reconstruction_and_strict_state_replay(
    model: E311OneBlockGNN,
) -> None:
    model.eval()
    centers, rotations = _inputs(2)
    checkpoint = {
        "model_state": model.state_dict(),
        "force_scale": float(model.force_scale.detach()),
        "model_config": asdict(model.config),
        "dtype": "float64",
    }
    rebuilt_config = E311OneBlockGNNConfig(**checkpoint["model_config"])
    rebuilt = E311OneBlockGNN(
        checkpoint["force_scale"],
        rebuilt_config,
        dtype=torch.float64,
    )
    incompatible = rebuilt.load_state_dict(checkpoint["model_state"], strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    rebuilt.eval()
    assert asdict(rebuilt.config) == checkpoint["model_config"]
    assert rebuilt.trainable_parameter_count == model.trainable_parameter_count
    rebuilt_state = rebuilt.state_dict()
    assert rebuilt_state.keys() == checkpoint["model_state"].keys()
    for name, value in checkpoint["model_state"].items():
        _assert_state_value_equal(value, rebuilt_state[name])
    expected = model.forward_world(centers, rotations)
    actual = rebuilt.forward_world(centers, rotations)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
