from __future__ import annotations

import json

import pytest
import torch

from experiments.benzene_pair.metrics import (
    deterministic_partition_indices,
    relative_force_norm_difference,
    summarize_relative_force_norm_difference,
)


def test_relative_force_norm_difference_uses_vector_magnitudes() -> None:
    target = torch.tensor(
        [
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    prediction = torch.tensor(
        [
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    actual = relative_force_norm_difference(
        prediction,
        target,
        epsilon=1.0e-12,
    )
    torch.testing.assert_close(actual, torch.tensor([0.2, 0.5]))


def test_summary_reports_required_distribution_statistics() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
    prediction = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    summary = summarize_relative_force_norm_difference(
        prediction,
        target,
        partition="validation",
        maximum_samples=None,
        seed=31,
    )
    assert summary["count"] == 4
    assert summary["total_count"] == 4
    assert summary["min"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(1.5)
    assert summary["mean"] == pytest.approx(1.5)
    assert summary["p90"] == pytest.approx(2.7)
    assert summary["p95"] == pytest.approx(2.85)
    assert summary["p99"] == pytest.approx(2.97)
    assert summary["near_zero_target_count"] == 0
    json.dumps(summary, allow_nan=False)


def test_partition_sampling_is_stable_and_partition_specific() -> None:
    first = deterministic_partition_indices(
        100,
        partition="train",
        maximum_samples=12,
        seed=7,
    )
    repeated = deterministic_partition_indices(
        100,
        partition="train",
        maximum_samples=12,
        seed=7,
    )
    validation = deterministic_partition_indices(
        100,
        partition="validation",
        maximum_samples=12,
        seed=7,
    )
    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, validation)
    assert bool((first[1:] > first[:-1]).all())


def test_automatic_epsilon_is_scale_consistent_and_counts_near_zero() -> None:
    target = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=torch.float64,
    )
    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [6.0, 8.0, 0.0],
        ],
        dtype=torch.float64,
    )
    normalized = summarize_relative_force_norm_difference(
        prediction,
        target,
        partition="test",
        maximum_samples=None,
    )
    physical = summarize_relative_force_norm_difference(
        prediction * 17.0,
        target * 17.0,
        partition="test",
        maximum_samples=None,
    )
    assert normalized["near_zero_target_count"] == 1
    assert normalized["near_zero_target_fraction"] == pytest.approx(0.5)
    for name in ("min", "max", "median", "mean", "p90", "p95", "p99"):
        assert physical[name] == pytest.approx(normalized[name])
    assert physical["epsilon"] == pytest.approx(17.0 * normalized["epsilon"])


def test_all_zero_targets_require_an_explicit_epsilon() -> None:
    prediction = torch.zeros(2, 3)
    target = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="epsilon is required"):
        summarize_relative_force_norm_difference(
            prediction,
            target,
            partition="test",
            maximum_samples=None,
        )
    summary = summarize_relative_force_norm_difference(
        prediction,
        target,
        partition="test",
        maximum_samples=None,
        epsilon=1.0e-9,
    )
    assert summary["near_zero_target_count"] == 2
    assert summary["max"] == 0.0


def test_small_epsilon_remains_finite_for_half_precision_inputs() -> None:
    prediction = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float16,
    )
    target = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float16,
    )
    summary = summarize_relative_force_norm_difference(
        prediction,
        target,
        partition="test",
        maximum_samples=None,
    )
    assert summary["near_zero_target_count"] == 1
    assert summary["max"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (torch.zeros(2, 2), torch.zeros(2, 2), "shape count by three"),
        (torch.zeros(2, 3), torch.zeros(3, 3), "shapes must match"),
        (
            torch.tensor([[float("nan"), 0.0, 0.0]]),
            torch.zeros(1, 3),
            "must be finite",
        ),
    ],
)
def test_invalid_force_inputs_are_rejected(
    prediction: torch.Tensor,
    target: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_relative_force_norm_difference(
            prediction,
            target,
            partition="test",
            maximum_samples=None,
            epsilon=1.0e-12,
        )
