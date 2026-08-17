from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

import TFENN.models.invariant_gate_pipeline_v2 as pipeline_module
from TFENN.models import (
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    build_invariant_gate_pipeline_v2,
)


DTYPE = torch.float64


def generators() -> torch.Tensor:
    angle = 2.0 * math.pi / 3.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        (
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            (
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
            ),
        ),
        dtype=DTYPE,
    )


def group() -> torch.Tensor:
    threefold, twofold = generators()
    powers = tuple(torch.linalg.matrix_power(threefold, power) for power in range(3))
    return torch.stack(powers + tuple(value @ twofold for value in powers))


def pairs(count: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.zeros((count, 2, 3), dtype=DTYPE)
    centers[:, 1] = torch.tensor((4.2, -0.7, 3.8), dtype=DTYPE)
    frames = torch.eye(3, dtype=DTYPE).expand(count, 2, 3, 3).clone()
    return centers, frames


def config(
    final: InvariantGateStageV2Config,
) -> InvariantGatePipelineV2Config:
    return InvariantGatePipelineV2Config(
        stages=(
            InvariantGateStageV2Config(
                "hidden",
                "A",
                ("x", "r"),
                1,
                trunk_width=2,
                skip_policy="none",
                covariant_include_symmetric_unary=False,
                covariant_include_raw_mixed_pairs=False,
                covariant_include_stf_shortcuts=False,
                invariant_include_symmetric_unary=False,
                invariant_include_raw_mixed_pairs=False,
                invariant_include_stf_shortcuts=False,
            ),
            final,
        ),
        output_stage="out",
        anchor_ranks=(2,),
        max_constraint_entries=2_000_000,
        max_gate_coefficients=100_000,
        max_invariant_channels=10_000,
    )


@pytest.mark.parametrize("skip_policy", ("none", "id", "local_proj", "dense_proj"))
def test_explicit_skip_policies_are_independent(skip_policy: str) -> None:
    final = InvariantGateStageV2Config(
        "out",
        "A",
        ("hidden",),
        1,
        invariant_source_names=("hidden",),
        skip_source_names=("x", "hidden"),
        trunk_width=2,
        skip_policy=skip_policy,
        covariant_include_symmetric_unary=False,
        covariant_include_raw_mixed_pairs=False,
        covariant_include_stf_shortcuts=False,
        invariant_include_symmetric_unary=False,
        invariant_include_raw_mixed_pairs=False,
        invariant_include_stf_shortcuts=False,
    )
    model = build_invariant_gate_pipeline_v2(generators(), config(final))
    output = model(*pairs())
    assert output.shape == (2, 3)
    if skip_policy == "none":
        assert not model.skip_projections
    elif skip_policy == "id":
        assert not model.skip_projections
    else:
        assert model.skip_projections


def test_covariant_and_invariant_path_switches_are_separate() -> None:
    stage = InvariantGateStageV2Config(
        "out",
        "A",
        ("x", "r"),
        1,
        trunk_width=2,
        covariant_include_symmetric_unary=False,
        invariant_include_symmetric_unary=True,
        covariant_include_raw_mixed_pairs=False,
        invariant_include_raw_mixed_pairs=True,
        covariant_include_stf_shortcuts=False,
        invariant_include_stf_shortcuts=True,
    )
    model = build_invariant_gate_pipeline_v2(
        generators(),
        InvariantGatePipelineV2Config(
            stages=(stage,),
            output_stage="out",
            anchor_ranks=(2, 3),
            max_constraint_entries=2_000_000,
            max_gate_coefficients=100_000,
        ),
    )
    roles = tuple(item["role"] for item in model.candidate_manifest)
    assert any(".scalar.symmetric2." in role for role in roles)
    assert any(".scalar.pair." in role for role in roles)
    assert any(".scalar.stf." in role for role in roles)
    assert not any(".a.symmetric2." in role for role in roles)
    assert not any(".a.pair." in role for role in roles)
    assert not any(".a.stf." in role for role in roles)


@pytest.mark.parametrize("policy", ("sym3", "a2b", "ab2", "union", "all"))
def test_degree_three_policies_compile_and_audit(policy: str) -> None:
    stage = InvariantGateStageV2Config(
        "out",
        "A",
        ("x", "r"),
        1,
        trunk_width=2,
        degree3_policy=policy,
        covariant_include_symmetric_unary=False,
        covariant_include_raw_mixed_pairs=False,
        covariant_include_stf_shortcuts=False,
        invariant_include_symmetric_unary=False,
        invariant_include_raw_mixed_pairs=False,
        invariant_include_stf_shortcuts=False,
    )
    model = build_invariant_gate_pipeline_v2(
        generators(),
        InvariantGatePipelineV2Config(
            stages=(stage,),
            output_stage="out",
            anchor_ranks=(2, 3),
            max_constraint_entries=2_000_000,
            max_gate_coefficients=100_000,
        ),
    )
    degree = tuple(
        item for item in model.candidate_manifest if ".degree3." in item["role"]
    )
    assert degree
    assert any(item["status"] == "compiled" for item in degree)
    assert not any(item["status"] == "unsupported" for item in degree)
    if policy in {"ab2", "union", "all"}:
        assert any("independent" in item["role"] for item in degree)


def test_degree_three_reynolds_compiler_ignores_dense_constraint_guard() -> None:
    stage = InvariantGateStageV2Config(
        "out",
        "A",
        ("x", "r"),
        1,
        trunk_width=2,
        degree3_policy="sym3",
        covariant_include_symmetric_unary=False,
        covariant_include_raw_mixed_pairs=False,
        covariant_include_stf_shortcuts=False,
        invariant_include_symmetric_unary=False,
        invariant_include_raw_mixed_pairs=False,
        invariant_include_stf_shortcuts=False,
    )
    model = build_invariant_gate_pipeline_v2(
        generators(),
        InvariantGatePipelineV2Config(
            stages=(stage,),
            output_stage="out",
            anchor_ranks=(2, 3),
            max_constraint_entries=10_000,
            max_gate_coefficients=4,
            degree3_overflow_policy="audit_skip",
        ),
    )
    degree = tuple(
        item for item in model.candidate_manifest if ".degree3." in item["role"]
    )
    assert any(item["status"] == "over_budget" for item in degree)
    assert model(*pairs()).shape == (2, 3)


def test_invariant_metric_handles_a_nonorthogonal_equivalent_basis() -> None:
    transform = torch.tensor(
        ((2.0, 0.3, 0.0), (0.0, 0.7, 0.2), (0.0, 0.0, 1.4)),
        dtype=DTYPE,
    )
    inverse = torch.linalg.inv(transform)
    actions = transform @ group() @ inverse
    metric = pipeline_module._invariant_metric_from_actions(actions)
    assert not torch.allclose(metric, torch.eye(3, dtype=DTYPE))
    for action in actions:
        torch.testing.assert_close(
            action.mT @ metric @ action,
            metric,
            rtol=2.0e-10,
            atol=2.0e-10,
        )


@pytest.mark.parametrize(
    ("head", "rank"),
    (("factorized", 1), ("factorized", 4), ("orthogonal", 2)),
)
def test_head_trunk_descriptor_forward_backward_and_checkpoint(
    head: str, rank: int
) -> None:
    stage = InvariantGateStageV2Config(
        "out",
        "A",
        ("x", "r"),
        1,
        trunk_width=4,
        skip_policy="dense_proj",
        skip_source_names=("x",),
        coefficient_head=head,
        coefficient_rank=rank,
        coefficient_activation="tanh",
        descriptor_mask="mixed",
        trunk_depth=2,
    )
    model = build_invariant_gate_pipeline_v2(
        generators(),
        InvariantGatePipelineV2Config(
            stages=(stage,),
            output_stage="out",
            anchor_ranks=(2,),
            max_constraint_entries=2_000_000,
            max_gate_coefficients=100_000,
        ),
    )
    centers, frames = pairs()
    output = model(centers, frames)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    model.eval()
    reference = model(centers, frames)
    moved = frames.clone()
    moved[:, 0] = frames[:, 0] @ group()[1]
    torch.testing.assert_close(
        model(centers, moved), reference, rtol=1.0e-9, atol=1.0e-9
    )
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = build_invariant_gate_pipeline_v2(generators(), model.config)
    restored.load_state_dict(torch.load(payload, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(centers, frames), reference)


def test_descriptor_projection_collection_and_restore() -> None:
    stage = InvariantGateStageV2Config("out", "A", ("x", "r"), 1, trunk_width=2)
    model = build_invariant_gate_pipeline_v2(
        generators(),
        InvariantGatePipelineV2Config(
            stages=(stage,), output_stage="out", anchor_ranks=(2,)
        ),
    )
    model.eval()
    centers, frames = pairs(3)
    descriptors = model.collect_normalized_descriptors(centers, frames)
    dimension = descriptors["out"].shape[-1]
    components = torch.eye(dimension, dtype=DTYPE)[: max(1, dimension // 2)]
    model.set_descriptor_projection("out", components)
    state = model.descriptor_projection_state_dict()
    model.clear_descriptor_projections()
    model.load_descriptor_projection_state_dict(state)
    torch.testing.assert_close(
        model.descriptor_transforms["out"].components, components
    )


@pytest.mark.skipif(
    os.environ.get("TFENN_RUN_D50_PROCESS_TEST") != "1",
    reason="explicit D50 cross process regression",
)
def test_d50_reynolds_artifacts_are_byte_stable_across_fresh_processes() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import hashlib
        import json

        import TFENN.models.invariant_gate_pipeline_v2 as pipeline_module
        from experiments.benzene_pair.d_series.catalog import build_d_series_model
        from experiments.benzene_pair.train import _proper_d6_generators

        model = build_d_series_model("D50", _proper_d6_generators())
        manifest = json.dumps(
            model.candidate_manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf8")
        buffers = hashlib.sha256()
        artifact_count = 0
        for module_name, module in model.covariants.items():
            if not isinstance(module, pipeline_module._RegisteredReynoldsCovariant):
                continue
            artifact_count += 1
            for buffer_name, value in module.named_buffers():
                frozen = value.detach().cpu().contiguous()
                buffers.update(module_name.encode("utf8"))
                buffers.update(buffer_name.encode("utf8"))
                buffers.update(str(frozen.dtype).encode("ascii"))
                buffers.update(str(tuple(frozen.shape)).encode("ascii"))
                buffers.update(frozen.numpy().tobytes())
        result = {
            "parameter_count": model.trainable_parameter_count,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "reynolds_buffer_sha256": buffers.hexdigest(),
            "reynolds_artifact_count": artifact_count,
            "status": model.offline_compilation_summary["candidate_status_counts"],
        }
        print("D50_FINGERPRINT=" + json.dumps(result, sort_keys=True))
        """
    )

    def build_once() -> dict[str, object]:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        line = next(
            value
            for value in completed.stdout.splitlines()
            if value.startswith("D50_FINGERPRINT=")
        )
        return json.loads(line.removeprefix("D50_FINGERPRINT="))

    first = build_once()
    second = build_once()
    assert first == second
    assert first["parameter_count"] == 145_106
    assert first["reynolds_artifact_count"] == 118
    assert first["status"] == {
        "compiled": 487,
        "empty_hom": 7,
        "failed": 0,
        "over_budget": 0,
        "unsupported": 0,
    }
