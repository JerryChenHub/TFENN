from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from experiments.gnn.e311_gnn_12_experiment_core_v1 import (
    build_experiment_core_v1,
)
from experiments.gnn.e311_gnn_12_experiment_runner_v1 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FIVE_BENZENE_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_SEEDS,
    DEFAULT_WEIGHT_DECAY,
    GraphBucketV1,
    RunnerConfigV1,
    _bucket_loaders,
    _evaluate_node_metrics,
    build_argument_parser_v1,
    build_seeded_experiment_model_v1,
    compute_node_metrics_v1,
    deterministic_group_split_v1,
    generate_three_body_chain_v1,
    prepare_experiment_data_v1,
    scatter_pair_forces_v1,
)


class _ZeroNodeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("force_scale", torch.tensor(1.0))

    def normalized_forces_world(
        self,
        centers: Tensor,
        _frames: Tensor,
        _pair_index: Tensor,
    ) -> Tensor:
        return torch.zeros_like(centers)


def _constant_node_bucket(name: str, sample_count: int, node_count: int) -> GraphBucketV1:
    centers = np.zeros((sample_count, node_count, 3), dtype=np.float64)
    frames = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (sample_count, node_count, 3, 3),
    ).copy()
    target = np.ones_like(centers)
    return GraphBucketV1(
        name=name,
        centers_world=centers,
        frames_body_to_world=frames,
        node_force_world=target,
        pair_index=np.asarray(((0,), (1,)), dtype=np.int64),
        pair_force_world=None,
        group_id=np.arange(sample_count, dtype=np.int64),
    )


def test_deterministic_group_split_keeps_matched_interventions_together() -> None:
    group_id = np.repeat(np.arange(20, dtype=np.int64), 3)
    first = deterministic_group_split_v1(group_id, 20260824)
    second = deterministic_group_split_v1(group_id, 20260824)
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
    split_groups = [
        set(group_id[indices].tolist())
        for indices in (first.train, first.validation, first.test)
    ]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    for group in range(20):
        membership = [group in values for values in split_groups]
        assert sum(membership) == 1


def test_signed_scatter_constructs_equal_and_opposite_node_forces() -> None:
    pair_index = np.asarray(((0, 0, 1), (1, 2, 2)), dtype=np.int64)
    pair_force = np.asarray(
        (
            (
                (1.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.0, 0.0, 3.0),
            ),
        ),
        dtype=np.float64,
    )
    node_force = scatter_pair_forces_v1(pair_force, pair_index, 3)
    assert node_force.shape == (1, 3, 3)
    assert np.allclose(node_force.sum(axis=1), 0.0)
    assert np.allclose(node_force[0, 0], (1.0, 2.0, 0.0))
    assert np.allclose(node_force[0, 1], (-1.0, 0.0, 3.0))
    assert np.allclose(node_force[0, 2], (0.0, -2.0, -3.0))


@pytest.mark.parametrize(
    ("experiment_id", "train_nodes", "test_nodes", "supervision"),
    (
        ("X01", (2,), (2,), "pair_force"),
        ("X03", (5,), (5,), "node_force"),
        ("X04", (5,), (5,), "pair_force"),
        ("X09", (3, 4), (5,), "node_force"),
        ("X10", (3, 4), (5,), "node_force"),
    ),
)
def test_data_routing_matches_registered_experiment(
    experiment_id: str,
    train_nodes: tuple[int, ...],
    test_nodes: tuple[int, ...],
    supervision: str,
) -> None:
    prepared = prepare_experiment_data_v1(experiment_id)
    assert tuple(bucket.node_count for bucket in prepared.train) == train_nodes
    assert tuple(bucket.node_count for bucket in prepared.test) == test_nodes
    assert prepared.spec.supervision.value == supervision
    assert prepared.force_scale is not None
    assert prepared.force_scale > 0.0
    for bucket in (*prepared.train, *prepared.validation, *prepared.test):
        assert np.allclose(bucket.node_force_world.sum(axis=1), 0.0, atol=1.0e-9)


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


@pytest.mark.parametrize(
    ("first_id", "second_id", "three_body"),
    (
        ("X03", "X04", False),
        ("X05", "X06", False),
        ("X09", "X10", False),
        ("X11", "X12", True),
    ),
)
def test_paired_conditions_use_identical_state_split_and_batch_order(
    first_id: str,
    second_id: str,
    three_body: bool,
) -> None:
    options = (
        {"three_body_base_count": 10, "three_body_interventions": 2}
        if three_body
        else {}
    )
    first = prepare_experiment_data_v1(first_id, **options)
    second = prepare_experiment_data_v1(second_id, **options)
    assert first.force_scale == second.force_scale
    assert first.force_scale is not None
    for first_split, second_split in zip(
        (first.train, first.validation, first.test),
        (second.train, second.validation, second.test),
        strict=True,
    ):
        assert len(first_split) == len(second_split)
        for first_bucket, second_bucket in zip(
            first_split, second_split, strict=True
        ):
            assert np.array_equal(first_bucket.group_id, second_bucket.group_id)
            assert np.array_equal(first_bucket.centers_world, second_bucket.centers_world)

    seed = DEFAULT_MODEL_SEEDS[0]
    config = RunnerConfigV1(device="cpu", comet_enabled=False)
    first_batch_size = config.batch_size_for(first_id)
    second_batch_size = config.batch_size_for(second_id)
    assert first_batch_size == second_batch_size
    first_model = build_seeded_experiment_model_v1(
        first_id, first.force_scale, seed
    )
    second_model = build_seeded_experiment_model_v1(
        second_id, second.force_scale, seed
    )
    _assert_nested_equal(first_model.state_dict(), second_model.state_dict())

    first_loaders = _bucket_loaders(
        first.train,
        first.spec.supervision,
        first.force_scale,
        first_batch_size,
        True,
        seed,
    )
    second_loaders = _bucket_loaders(
        second.train,
        second.spec.supervision,
        second.force_scale,
        second_batch_size,
        True,
        seed,
    )
    assert len(first_loaders) == len(second_loaders)
    for first_loader, second_loader in zip(
        first_loaders, second_loaders, strict=True
    ):
        assert first_loader.loader.batch_size == first_batch_size
        assert second_loader.loader.batch_size == second_batch_size
        first_order = torch.cat([batch[0] for batch in first_loader.loader])
        second_order = torch.cat([batch[0] for batch in second_loader.loader])
        assert torch.equal(first_order, second_order)


def test_x02_routes_three_graph_sizes_without_training_data() -> None:
    prepared = prepare_experiment_data_v1("X02")
    assert prepared.train == ()
    assert prepared.validation == ()
    assert prepared.force_scale is None
    assert tuple(bucket.node_count for bucket in prepared.test) == (3, 4, 5)
    assert all(bucket.sample_count == 100 for bucket in prepared.test)


def test_three_body_data_preserves_matched_inputs_and_conservation() -> None:
    bucket = generate_three_body_chain_v1(base_count=12, interventions=3, seed=81)
    assert bucket.sample_count == 36
    assert np.max(np.abs(bucket.node_force_world.sum(axis=1))) < 1.0e-10
    for group in range(12):
        indices = np.flatnonzero(bucket.group_id == group)
        assert len(indices) == 3
        assert np.allclose(
            bucket.centers_world[indices, :2],
            bucket.centers_world[indices[0], :2],
        )
        assert np.allclose(
            bucket.frames_body_to_world[indices, :2],
            bucket.frames_body_to_world[indices[0], :2],
        )
        assert not np.allclose(
            bucket.centers_world[indices, 2],
            bucket.centers_world[indices[0], 2],
        )


def test_node_metrics_use_component_mae_and_relative_frobenius_percent() -> None:
    target = torch.tensor(((((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),))
    prediction = 0.5 * target
    metrics = compute_node_metrics_v1(prediction, target)
    assert metrics.final_test_mae == pytest.approx(1.0 / 6.0)
    assert metrics.final_normal_f_difference == pytest.approx(50.0)
    assert metrics.residual_frobenius_norm == pytest.approx(2.0**0.5 / 2.0)
    assert metrics.target_frobenius_norm == pytest.approx(2.0**0.5)
    assert metrics.component_count == 6


def test_node_evaluation_combines_buckets_with_different_node_counts() -> None:
    buckets = (
        _constant_node_bucket("n3", sample_count=2, node_count=3),
        _constant_node_bucket("n4", sample_count=1, node_count=4),
        _constant_node_bucket("n5", sample_count=3, node_count=5),
    )
    metrics = _evaluate_node_metrics(
        _ZeroNodeModel(),
        buckets,
        batch_size=2,
        device=torch.device("cpu"),
    )
    component_count = (2 * 3 + 1 * 4 + 3 * 5) * 3
    assert metrics.final_test_mae == pytest.approx(1.0)
    assert metrics.final_normal_f_difference == pytest.approx(100.0)
    assert metrics.residual_frobenius_norm == pytest.approx(component_count**0.5)
    assert metrics.target_frobenius_norm == pytest.approx(component_count**0.5)
    assert metrics.component_count == component_count


def test_cli_defaults_match_frozen_training_protocol() -> None:
    parser = build_argument_parser_v1()
    run = parser.parse_args(("run", "X06"))
    assert run.epochs == DEFAULT_EPOCHS == 500
    assert run.batch_size == DEFAULT_BATCH_SIZE == 100
    assert (
        run.five_benzene_batch_size
        == DEFAULT_FIVE_BENZENE_BATCH_SIZE
        == 128
    )
    assert run.learning_rate == DEFAULT_LEARNING_RATE == 0.002
    assert run.weight_decay == DEFAULT_WEIGHT_DECAY == 1.0e-6
    assert run.seed == DEFAULT_MODEL_SEEDS[0]
    assert DEFAULT_MODEL_SEEDS == (20260824,)
    group = parser.parse_args(("run-group",))
    assert tuple(group.seeds) == DEFAULT_MODEL_SEEDS
    assert tuple(group.experiments) == tuple(f"X{index:02d}" for index in range(1, 13))


@pytest.mark.parametrize(
    ("experiment_id", "expected"),
    tuple((f"X{index:02d}", 128) for index in range(2, 11))
    + (("X01", 100), ("X11", 100), ("X12", 100)),
)
def test_experiment_batch_size_routing(
    experiment_id: str,
    expected: int,
) -> None:
    config = RunnerConfigV1(
        batch_size=100,
        five_benzene_batch_size=128,
        device="cpu",
        comet_enabled=False,
    )
    assert config.batch_size_for(experiment_id) == expected


def test_cloud_launch_arguments_parse_with_study_root_alias(tmp_path: Path) -> None:
    parser = build_argument_parser_v1()
    study_root = tmp_path / "study"
    parsed = parser.parse_args(
        (
            "run-group",
            "--experiments",
            "X01",
            "X02",
            "X03",
            "X04",
            "X05",
            "X06",
            "--seeds",
            "20260824",
            "--study-root",
            str(study_root),
            "--epochs",
            "500",
            "--batch-size",
            "100",
            "--five-benzene-batch-size",
            "128",
            "--device",
            "cuda:0",
            "--comet-project",
            "tfenn_e311_gnn_12_v1",
        )
    )
    assert parsed.output_root == study_root
    assert parsed.epochs == 500
    assert parsed.batch_size == 100
    assert parsed.five_benzene_batch_size == 128
    assert parsed.device == "cuda:0"
    assert parsed.comet_project == "tfenn_e311_gnn_12_v1"
    assert tuple(parsed.experiments) == ("X01", "X02", "X03", "X04", "X05", "X06")
    assert tuple(parsed.seeds) == DEFAULT_MODEL_SEEDS


def test_cloud_launch_uses_one_seed_and_covers_all_conditions() -> None:
    launch_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "gnn"
        / "launch_e311_12_cloud_v1.sh"
    )
    lines = launch_path.read_text(encoding="utf_8").splitlines()
    declarations = {
        line.split("=", maxsplit=1)[0]: line.split("(", maxsplit=1)[1].removesuffix(")").split()
        for line in lines
        if line.startswith(("SEEDS=(", "GROUP_A=(", "GROUP_B=("))
    }
    assert declarations["SEEDS"] == ["20260824"]
    group_a = declarations["GROUP_A"]
    group_b = declarations["GROUP_B"]
    assert set(group_a).isdisjoint(group_b)
    assert set((*group_a, *group_b)) == {
        f"X{index:02d}" for index in range(1, 13)
    }
    for first, second in (
        ("X01", "X02"),
        ("X03", "X04"),
        ("X05", "X06"),
        ("X09", "X10"),
        ("X11", "X12"),
    ):
        assert (first in group_a) == (second in group_a)


def test_small_three_body_batch_runs_through_x12_core() -> None:
    prepared = prepare_experiment_data_v1(
        "X12",
        three_body_base_count=10,
        three_body_interventions=2,
    )
    assert prepared.force_scale is not None
    model = build_experiment_core_v1("X12", prepared.force_scale, torch.float32)
    bucket = prepared.train[0]
    centers = torch.from_numpy(bucket.centers_world[:2].astype(np.float32))
    frames = torch.from_numpy(bucket.frames_body_to_world[:2].astype(np.float32))
    pair_index = torch.from_numpy(bucket.pair_index.copy())
    output = model.core_output(centers, frames, pair_index)
    assert output.normalized_node_force_world.shape == (2, 3, 3)
    assert output.normalized_pair_force_world.shape == (2, 2, 3)
    assert torch.isfinite(output.normalized_node_force_world).all()
    assert torch.allclose(
        output.normalized_node_force_world.sum(dim=1),
        torch.zeros((2, 3)),
        atol=2.0e-6,
        rtol=0.0,
    )
