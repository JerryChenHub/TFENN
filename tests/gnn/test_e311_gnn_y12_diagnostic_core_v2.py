from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import Tensor

from experiments.gnn.e311_gnn_12_experiment_core_v1 import (
    BAggregationV1,
    build_experiment_core_v1,
    complete_pair_index_v1,
)
from experiments.gnn.e311_gnn_y12_diagnostic_core_v2 import (
    CURRENT_X01_STEPS,
    EMA_DECAY,
    LEGACY_5K_V3_STEPS,
    EMARunningRMSV2,
    Y_EXPERIMENT_SPECS_V2,
    assert_one_layer_stack_preflight_v2,
    build_y_diagnostic_core_v2,
    get_y_experiment_spec_v2,
)


@pytest.fixture(scope="module")
def two_node_geometry() -> tuple[Tensor, Tensor, Tensor]:
    centers = torch.tensor(
        (
            ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
            ((0.2, 0.1, 0.0), (0.0, 5.7, 0.3)),
        ),
        dtype=torch.float32,
    )
    frames = torch.eye(3, dtype=torch.float32).repeat(2, 2, 1, 1)
    return centers, frames, complete_pair_index_v1(2)


@pytest.fixture(scope="module")
def five_node_geometry() -> tuple[Tensor, Tensor, Tensor]:
    centers = torch.tensor(
        (
            (
                (0.0, 0.0, 0.0),
                (5.2, 0.2, 0.0),
                (0.3, 5.4, 0.1),
                (0.2, 0.4, 5.6),
                (4.1, 4.4, 4.7),
            ),
        ),
        dtype=torch.float32,
    )
    frames = torch.eye(3, dtype=torch.float32).repeat(1, 5, 1, 1)
    return centers, frames, complete_pair_index_v1(5)


@pytest.fixture(scope="module")
def x01_and_y01() -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(20260824)
    y01 = build_y_diagnostic_core_v2("Y01", 1.25)
    torch.manual_seed(20260824)
    x01 = build_experiment_core_v1("X01", 1.25)
    return x01, y01


@pytest.fixture(scope="module")
def paired_y03_y05() -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    torch.manual_seed(20260825)
    y03 = build_y_diagnostic_core_v2("Y03", 0.75)
    torch.manual_seed(20260825)
    y05 = build_y_diagnostic_core_v2("Y05", 0.75)
    with torch.no_grad():
        for model in (y03, y05):
            for block in model.message_blocks:
                block.pair_kernel._runtime_reference.zero_()
    y03_state = copy.deepcopy(y03.state_dict())
    y05_state = copy.deepcopy(y05.state_dict())
    return y03, y05, y03_state, y05_state


@pytest.fixture(scope="module")
def y09() -> torch.nn.Module:
    torch.manual_seed(20260826)
    return build_y_diagnostic_core_v2("Y09", 1.0)


@pytest.fixture(scope="module")
def y11() -> torch.nn.Module:
    torch.manual_seed(20260826)
    return build_y_diagnostic_core_v2("Y11", 1.0)


def _assert_nested_equal(first: Any, second: Any, location: str = "root") -> None:
    if isinstance(first, Tensor):
        assert isinstance(second, Tensor), location
        assert torch.equal(first, second), location
    elif isinstance(first, Mapping):
        assert isinstance(second, Mapping), location
        assert tuple(first) == tuple(second), location
        for key in first:
            _assert_nested_equal(first[key], second[key], f"{location}.{key}")
    elif isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        assert isinstance(second, Sequence), location
        assert len(first) == len(second), location
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            _assert_nested_equal(left, right, f"{location}.{index}")
    else:
        assert first == second, location


def test_registry_covers_all_twelve_diagnostics() -> None:
    assert tuple(spec.experiment_id for spec in Y_EXPERIMENT_SPECS_V2) == tuple(
        f"Y{index:02d}" for index in range(1, 13)
    )
    assert get_y_experiment_spec_v2("y01").optimizer_steps == CURRENT_X01_STEPS
    assert get_y_experiment_spec_v2("Y02").optimizer_steps == LEGACY_5K_V3_STEPS
    for experiment_id in ("Y03", "Y04", "Y05"):
        assert (
            get_y_experiment_spec_v2(experiment_id).optimizer_protocol
            == "legacy_5k_v3_protocol"
        )
    assert get_y_experiment_spec_v2("Y04").dataset == "pair_5k_legacy_v3"
    assert get_y_experiment_spec_v2("Y05").ema_layers == (0,)
    assert get_y_experiment_spec_v2("Y09").edge_alpha_init == 0.0
    assert get_y_experiment_spec_v2("Y10").aggregation is BAggregationV1.SUM
    assert get_y_experiment_spec_v2("Y11").aggregation_scale == 0.25
    assert get_y_experiment_spec_v2("Y12").ema_layers == (1,)


def test_y01_is_exactly_the_x01_one_layer_core(
    x01_and_y01: tuple[torch.nn.Module, torch.nn.Module],
    two_node_geometry: tuple[Tensor, Tensor, Tensor],
) -> None:
    x01, y01 = x01_and_y01
    centers, frames, pair_index = two_node_geometry
    x01.eval()
    y01.eval()
    assert_one_layer_stack_preflight_v2(y01, centers, frames, pair_index)
    actual = y01.normalized_forces_world(centers, frames, pair_index)
    expected = x01.normalized_forces_world(centers, frames, pair_index)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_paired_rms_conditions_have_identical_initial_state(
    paired_y03_y05: tuple[
        torch.nn.Module,
        torch.nn.Module,
        Mapping[str, Any],
        Mapping[str, Any],
    ],
) -> None:
    _, _, y03_state, y05_state = paired_y03_y05
    _assert_nested_equal(y03_state, y05_state)


def test_ema_updates_and_model_reset_are_explicit(
    paired_y03_y05: tuple[
        torch.nn.Module,
        torch.nn.Module,
        Mapping[str, Any],
        Mapping[str, Any],
    ],
) -> None:
    _, y05, _, _ = paired_y03_y05
    normalizers = [
        module
        for module in y05._block_at(0).pair_kernel.modules()
        if isinstance(module, EMARunningRMSV2)
    ]
    assert normalizers
    normalizer = normalizers[0]
    normalizer.train()
    normalizer(torch.tensor((3.0, 4.0)))
    assert float(normalizer.mean_square) == pytest.approx(12.5)
    assert int(normalizer.sample_count) == 2
    normalizer(torch.tensor((0.0, 2.0)))
    expected = EMA_DECAY * 12.5 + (1.0 - EMA_DECAY) * 2.0
    assert float(normalizer.mean_square) == pytest.approx(expected)
    assert int(normalizer.sample_count) == 4
    y05.reset_running_rms()
    assert all(int(module.sample_count) == 0 for module in normalizers)
    assert all(float(module.mean_square) == 1.0 for module in normalizers)


def test_architecture_record_exposes_every_y_intervention(
    paired_y03_y05: tuple[
        torch.nn.Module,
        torch.nn.Module,
        Mapping[str, Any],
        Mapping[str, Any],
    ],
    y09: torch.nn.Module,
) -> None:
    _, y05, _, _ = paired_y03_y05
    ema_record = dict(y05.architecture_record())
    assert ema_record["core_version"] == "e311_gnn_y12_diagnostic_core_v2"
    assert ema_record["aggregation_scale"] == 1.0
    assert ema_record["ema_layers"] == (0,)
    assert ema_record["ema_decay"] == EMA_DECAY
    assert ema_record["running_rms_policy_by_layer"] == ("ema",)
    identity_record = dict(y09.architecture_record())
    assert identity_record["edge_transition"] == "identity_interpolation"
    assert identity_record["edge_alpha_init"] == 0.0
    assert identity_record["edge_alpha_current"] == (0.0,)
    assert identity_record["edge_alpha_trainable"] is True


def test_identity_alpha_zero_preserves_the_first_layer_edge(
    y09: torch.nn.Module,
    five_node_geometry: tuple[Tensor, Tensor, Tensor],
) -> None:
    centers, frames, pair_index = five_node_geometry
    y09.eval()
    assert isinstance(y09.edge_alpha, torch.nn.Parameter)
    torch.testing.assert_close(
        y09.edge_alpha.detach(), torch.zeros_like(y09.edge_alpha), rtol=0.0, atol=0.0
    )
    output = y09.core_output(centers, frames, pair_index)
    torch.testing.assert_close(
        output.layer_outputs[1].edge_a_world,
        output.layer_outputs[0].edge_a_world,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.normalized_node_force_world.sum(dim=-2),
        torch.zeros_like(output.normalized_node_force_world.sum(dim=-2)),
        rtol=0.0,
        atol=2.0e-6,
    )


def test_quarter_sum_scales_the_receiver_bank_before_layer_two(
    y11: torch.nn.Module,
    five_node_geometry: tuple[Tensor, Tensor, Tensor],
) -> None:
    centers, frames, pair_index = five_node_geometry
    y11.eval()
    output = y11.core_output(centers, frames, pair_index)
    node_count = centers.shape[-2]
    raw_first_sum = y11.b_aggregator(
        output.layer_outputs[0], pair_index, node_count
    )
    expected_scaled = {
        key: value * 0.25 for key, value in raw_first_sum.items()
    }
    initial_hidden = y11._initial_hidden_b(centers)
    expected_hidden = y11._node_update_at(0)(initial_hidden, expected_scaled)
    for key in expected_hidden:
        torch.testing.assert_close(
            output.hidden_b_input_to_last_layer[key],
            expected_hidden[key],
            rtol=0.0,
            atol=0.0,
        )
    raw_final_sum = y11.b_aggregator(
        output.layer_outputs[-1], pair_index, node_count
    )
    for key in raw_final_sum:
        torch.testing.assert_close(
            output.final_aggregated_b_local[key],
            raw_final_sum[key] * 0.25,
            rtol=0.0,
            atol=0.0,
        )


def test_state_dict_strict_replay_restores_alpha_and_model_state(
    y09: torch.nn.Module,
) -> None:
    saved = copy.deepcopy(y09.state_dict())
    with torch.no_grad():
        y09.edge_alpha.fill_(0.75)
        next(y09.parameters()).add_(0.125)
    incompatible = y09.load_state_dict(saved, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    _assert_nested_equal(y09.state_dict(), saved)
