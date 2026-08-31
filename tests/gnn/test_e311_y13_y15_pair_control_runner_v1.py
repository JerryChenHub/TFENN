from __future__ import annotations

import numpy as np

from experiments.gnn.e311_y13_y15_pair_control_core_v1 import (
    complete_pair_index_v1,
)
from experiments.gnn.e311_y13_y15_pair_control_runner_v1 import (
    _deterministic_group_split,
    _numpy_signed_scatter,
    build_argument_parser_v1,
)


def test_y15_split_occurs_at_configuration_level() -> None:
    group_id = np.repeat(np.arange(100, dtype=np.int64), 3)
    split = _deterministic_group_split(group_id, 20260821)
    assert split.counts() == {"train": 240, "validation": 30, "test": 30}
    train_groups = set(group_id[split.train].tolist())
    validation_groups = set(group_id[split.validation].tolist())
    test_groups = set(group_id[split.test].tolist())
    assert not train_groups & validation_groups
    assert not train_groups & test_groups
    assert not validation_groups & test_groups


def test_pair_target_contract_reaggregates_with_zero_total_force() -> None:
    pair_index = complete_pair_index_v1(5).cpu().numpy()
    pair_force = np.arange(60, dtype=np.float64).reshape(2, 10, 3)
    node_force = _numpy_signed_scatter(pair_force, pair_index, 5)
    np.testing.assert_array_equal(node_force.sum(axis=1), np.zeros((2, 3)))


def test_cli_has_three_separate_experiment_commands(tmp_path) -> None:
    parser = build_argument_parser_v1()
    y13 = parser.parse_args(
        ("y13", "--study-root", str(tmp_path / "e"), "--device", "cpu")
    )
    y14 = parser.parse_args(
        (
            "y14",
            "--e-study-root",
            str(tmp_path / "e"),
            "--output-directory",
            str(tmp_path / "y14"),
            "--device",
            "cpu",
        )
    )
    y15 = parser.parse_args(
        (
            "y15",
            "--csv",
            str(tmp_path / "five.csv"),
            "--pair-npz",
            str(tmp_path / "five_pair.npz"),
            "--output-directory",
            str(tmp_path / "y15"),
            "--device",
            "cpu",
        )
    )
    assert (y13.command, y14.command, y15.command) == ("y13", "y14", "y15")
