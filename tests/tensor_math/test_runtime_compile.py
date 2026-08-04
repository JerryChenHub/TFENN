"""Validate the differentiable runtime core under complete graph capture."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from TFENN.tensor_math import (
    PoseEncoder,
    compile_anchors,
    scalar_contraction,
    stf_tensor_coupling,
    vector_covariant,
)

from ._groups import ATOL, DTYPE, RTOL, benzene_generators, rotation


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_pose_and_covariant_runtime_capture_one_complete_graph() -> None:
    """Check ordinary runtime avoids data validation graph breaks."""
    encoder = PoseEncoder(
        compile_anchors(benzene_generators(), output_ranks=(2, 6))
    )

    def runtime(
        pose: Tensor,
        position: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        blocks = encoder.encode_blocks(pose)
        return (
            encoder(pose),
            scalar_contraction(position, blocks[2], 2),
            vector_covariant(position, blocks[2], 2),
            stf_tensor_coupling(blocks[2][..., 0], position, 2, 1, 1),
        )

    compiled = torch.compile(runtime, backend="eager", fullgraph=True)
    pose = rotation((0.13, -0.17, 0.29)).requires_grad_()
    position = torch.tensor(
        (0.41, -0.37, 0.23), dtype=DTYPE, requires_grad=True
    )
    expected = runtime(pose, position)
    actual = compiled(pose, position)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=ATOL, rtol=RTOL)
    sum(value.square().sum() for value in actual).backward()
    assert pose.grad is not None and bool(torch.isfinite(pose.grad).all())
    assert position.grad is not None and bool(torch.isfinite(position.grad).all())
