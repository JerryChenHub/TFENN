from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor


DEFAULT_FORCE_NORM_SAMPLE_COUNT = 4096
DEFAULT_EPSILON_FRACTION = 1.0e-12


def deterministic_partition_indices(
    total_count: int,
    *,
    partition: str,
    maximum_samples: int | None = DEFAULT_FORCE_NORM_SAMPLE_COUNT,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    """Choose a stable sample for one named data partition."""
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    if not partition:
        raise ValueError("partition must not be empty")
    if maximum_samples is not None and maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive or None")

    selected_count = (
        total_count
        if maximum_samples is None
        else min(total_count, int(maximum_samples))
    )
    if selected_count == total_count:
        indices = torch.arange(total_count, dtype=torch.int64)
    else:
        key = f"{int(seed)}:{partition}".encode("utf_8")
        stable_seed = int.from_bytes(
            hashlib.sha256(key).digest()[:8],
            byteorder="big",
            signed=False,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed)
        indices = torch.randperm(
            total_count,
            generator=generator,
            dtype=torch.int64,
        )[:selected_count]
        indices = indices.sort().values
    if device is not None:
        indices = indices.to(device=device)
    return indices


def relative_force_norm_difference(
    prediction_force: Tensor,
    target_force: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Return one relative force norm difference for every sample."""
    if prediction_force.shape != target_force.shape:
        raise ValueError("prediction_force and target_force shapes must match")
    if prediction_force.ndim != 2 or prediction_force.shape[1] != 3:
        raise ValueError("force tensors must have shape count by three")
    if prediction_force.shape[0] == 0:
        raise ValueError("force tensors must not be empty")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    metric_dtype = (
        torch.float64
        if torch.float64 in (prediction_force.dtype, target_force.dtype)
        else torch.float32
    )
    prediction = prediction_force.detach().to(dtype=metric_dtype)
    target = target_force.detach().to(
        device=prediction.device,
        dtype=metric_dtype,
    )
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("prediction_force and target_force must be finite")
    prediction_norm = torch.linalg.vector_norm(prediction, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    denominator = target_norm.clamp_min(float(epsilon))
    return (prediction_norm - target_norm).abs() / denominator


def summarize_relative_force_norm_difference(
    prediction_force: Tensor,
    target_force: Tensor,
    *,
    partition: str,
    maximum_samples: int | None = DEFAULT_FORCE_NORM_SAMPLE_COUNT,
    seed: int = 0,
    epsilon: float | None = None,
    epsilon_fraction: float = DEFAULT_EPSILON_FRACTION,
) -> dict[str, Any]:
    """Sample and summarize relative differences in force magnitude.

    Prediction and target may both use physical units or the same normalized
    scale. Automatic epsilon follows the target RMS, so a common scale change
    leaves the ratios unchanged.
    """
    if prediction_force.shape != target_force.shape:
        raise ValueError("prediction_force and target_force shapes must match")
    if prediction_force.ndim != 2 or prediction_force.shape[1] != 3:
        raise ValueError("force tensors must have shape count by three")
    if prediction_force.shape[0] == 0:
        raise ValueError("force tensors must not be empty")
    if not math.isfinite(epsilon_fraction) or epsilon_fraction <= 0.0:
        raise ValueError("epsilon_fraction must be finite and positive")

    indices = deterministic_partition_indices(
        int(prediction_force.shape[0]),
        partition=partition,
        maximum_samples=maximum_samples,
        seed=seed,
        device=prediction_force.device,
    )
    metric_dtype = (
        torch.float64
        if torch.float64 in (prediction_force.dtype, target_force.dtype)
        else torch.float32
    )
    prediction = prediction_force.index_select(0, indices).detach().to(
        dtype=metric_dtype
    )
    target = target_force.detach().to(
        device=prediction_force.device,
        dtype=metric_dtype,
    ).index_select(0, indices)
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("sampled prediction_force and target_force must be finite")

    target_norm = torch.linalg.vector_norm(target.detach(), dim=1)
    target_norm_rms_tensor = torch.sqrt(target_norm.square().mean())
    target_norm_rms = float(target_norm_rms_tensor)
    if epsilon is None:
        if target_norm_rms <= 0.0:
            raise ValueError(
                "epsilon is required when all sampled target force norms are zero"
            )
        resolved_epsilon = target_norm_rms * float(epsilon_fraction)
        epsilon_source = "sample_target_norm_rms"
    else:
        resolved_epsilon = float(epsilon)
        epsilon_source = "explicit"
    if not math.isfinite(resolved_epsilon) or resolved_epsilon <= 0.0:
        raise ValueError("resolved epsilon must be finite and positive")

    differences = relative_force_norm_difference(
        prediction,
        target,
        epsilon=resolved_epsilon,
    )
    quantile_levels = differences.new_tensor((0.5, 0.9, 0.95, 0.99))
    quantiles = torch.quantile(differences, quantile_levels)
    near_zero_count = int((target_norm <= resolved_epsilon).sum().item())
    sampled_count = int(differences.numel())

    result: dict[str, Any] = {
        "metric": "relative_difference_in_force_norm",
        "partition": partition,
        "total_count": int(prediction_force.shape[0]),
        "count": sampled_count,
        "sampling_seed": int(seed),
        "epsilon": float(resolved_epsilon),
        "epsilon_source": epsilon_source,
        "epsilon_fraction": float(epsilon_fraction),
        "target_norm_rms": target_norm_rms,
        "near_zero_target_count": near_zero_count,
        "near_zero_target_fraction": float(near_zero_count / sampled_count),
        "min": float(differences.min().item()),
        "max": float(differences.max().item()),
        "median": float(quantiles[0].item()),
        "mean": float(differences.mean().item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
    }
    numeric_values = (
        value
        for value in result.values()
        if isinstance(value, float)
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError("force norm difference summary must be finite")
    return result
