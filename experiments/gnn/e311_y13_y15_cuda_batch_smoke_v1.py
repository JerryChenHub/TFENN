"""Exact batch CUDA memory smoke for the Y13 to Y15 formal controls."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Sequence

import torch
from torch import Tensor, nn

from experiments.benzene_pair.e_series import runner as e_runner
from experiments.gnn.e311_y13_y15_pair_control_core_v1 import (
    HISTORICAL_E311_MODEL_ID,
    HISTORICAL_E311_PARAMETER_COUNT,
    HISTORICAL_MODEL_SEED,
    build_e311_odd_graph_core_v1,
    build_y14_two_node_control_v1,
    complete_pair_index_v1,
)


def _inputs(batch_size: int, node_count: int, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260901)
    centers = torch.randn(
        (batch_size, node_count, 3),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    frames = torch.eye(3, dtype=torch.float32, device=device).expand(
        batch_size,
        node_count,
        3,
        3,
    ).clone()
    return centers, frames


def _parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def _finite_gradients(model: nn.Module) -> bool:
    return all(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in model.parameters()
        if value.requires_grad
    )


def run_smoke(experiment_id: str, hold_seconds: float) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    torch.manual_seed(HISTORICAL_MODEL_SEED)
    torch.cuda.manual_seed_all(HISTORICAL_MODEL_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if experiment_id == "Y13":
        spec = e_runner.get_model_spec(HISTORICAL_E311_MODEL_ID)
        model = e_runner._build_model(spec, str(device))
        centers, frames = _inputs(10_000, 2, device)
        target = torch.randn((10_000, 3), dtype=torch.float32, device=device)
        prediction = model(centers, frames)
    elif experiment_id == "Y14":
        model = build_y14_two_node_control_v1(
            dtype=torch.float32,
            device=device,
            seed=HISTORICAL_MODEL_SEED,
        )
        centers, frames = _inputs(10_000, 2, device)
        target = torch.randn((10_000, 3), dtype=torch.float32, device=device)
        prediction = model(centers, frames)
    elif experiment_id == "Y15":
        model = build_e311_odd_graph_core_v1(
            dtype=torch.float32,
            device=device,
            seed=HISTORICAL_MODEL_SEED,
        )
        centers, frames = _inputs(1_000, 5, device)
        pair_index = complete_pair_index_v1(5).to(device)
        target = torch.randn((1_000, 10, 3), dtype=torch.float32, device=device)
        prediction = model(centers, frames, pair_index)
    else:
        raise ValueError(experiment_id)

    if _parameter_count(model) != HISTORICAL_E311_PARAMETER_COUNT:
        raise RuntimeError("E311 parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1.0e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("smoke loss is nonfinite")
    loss.backward()
    if not _finite_gradients(model):
        raise RuntimeError("smoke gradients are missing or nonfinite")
    optimizer.step()
    torch.cuda.synchronize(device)
    result = {
        "status": "passed",
        "experiment_id": experiment_id,
        "parameter_count": _parameter_count(model),
        "loss": float(loss.detach().cpu()),
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "maximum_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "maximum_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    if not math.isfinite(float(result["loss"])):
        raise RuntimeError("smoke result is nonfinite")
    print(json.dumps(result, sort_keys=True), flush=True)
    if hold_seconds > 0.0:
        time.sleep(hold_seconds)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", choices=("Y13", "Y14", "Y15"))
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    arguments = parser.parse_args(argv)
    run_smoke(arguments.experiment_id, arguments.hold_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
