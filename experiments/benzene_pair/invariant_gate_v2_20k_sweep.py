"""Define the thirty model invariant gate version two sweep."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from torch import Tensor

from TFENN.models import (
    InvariantGatePipelineV2,
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    build_invariant_gate_pipeline_v2,
)


__all__ = [
    "ANCHOR_RANKS",
    "MODEL_SPECS",
    "ModelSpec",
    "build_model_specs",
    "build_sweep_model",
    "get_model_spec",
]


ANCHOR_RANKS = (2, 6)
ComparisonRole = Literal["candidate", "primary", "lower_control", "upper_control"]
PathFlags = Mapping[str, bool | str]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Store one named architecture and its expected compiled size."""

    model_id: str
    description: str
    purpose: str
    expected_parameter_count: int
    pipeline: InvariantGatePipelineV2Config
    comparison_role: ComparisonRole = "candidate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "description": self.description,
            "purpose": self.purpose,
            "expected_parameter_count": self.expected_parameter_count,
            "comparison_role": self.comparison_role,
            "pipeline": self.pipeline.as_dict(),
        }


def _stage(
    name: str,
    stream: Literal["A", "B"],
    sources: Sequence[str],
    channels: int,
    width: int,
    *,
    invariant_sources: Sequence[str] | None = None,
    activation: Literal["silu", "tanh", "relu", "gelu"] = "silu",
    symmetric: bool = True,
    raw_mixed: bool = True,
    stf: bool = True,
) -> InvariantGateStageV2Config:
    return InvariantGateStageV2Config(
        name=name,
        output_stream=stream,
        source_names=tuple(sources),
        channels=channels,
        invariant_source_names=None
        if invariant_sources is None
        else tuple(invariant_sources),
        trunk_width=width,
        activation=activation,
        include_symmetric_unary=symmetric,
        include_raw_mixed_pairs=raw_mixed,
        include_stf_shortcuts=stf,
    )


def _stage_name(stream: Literal["A", "B"], counts: dict[str, int]) -> str:
    counts[stream] += 1
    return f"{stream.lower()}{counts[stream]}"


def _sequential_stages(
    route: Sequence[tuple[Literal["A", "B"], int]],
    widths: Sequence[int],
    *,
    flags: Mapping[int, PathFlags] | None = None,
    invariant_context: Literal["all", "raw", "raw_latest"] = "all",
) -> tuple[InvariantGateStageV2Config, ...]:
    if len(widths) != len(route) + 1:
        raise ValueError("widths must contain one value per hidden stage and output")
    if invariant_context not in {"all", "raw", "raw_latest"}:
        raise ValueError("unsupported invariant context")
    stages = []
    previous: list[str] = []
    counts = {"A": 0, "B": 0}
    stage_flags = {} if flags is None else flags
    for index, ((stream, channels), width) in enumerate(zip(route, widths)):
        name = _stage_name(stream, counts)
        sources = ("x", "r", *previous)
        invariant_sources = _invariant_sources(invariant_context, previous)
        stages.append(
            _stage(
                name,
                stream,
                sources,
                channels,
                width,
                invariant_sources=invariant_sources,
                **stage_flags.get(index, {}),
            )
        )
        previous.append(name)
    stages.append(
        _stage(
            "out",
            "A",
            ("x", "r", *previous),
            1,
            widths[-1],
            invariant_sources=_invariant_sources(invariant_context, previous),
            **stage_flags.get(len(route), {}),
        )
    )
    return tuple(stages)


def _invariant_sources(
    context: Literal["all", "raw", "raw_latest"], previous: Sequence[str]
) -> tuple[str, ...] | None:
    if context == "all":
        return None
    if context == "raw":
        return ("x", "r")
    return ("x", "r", *previous[-1:])


def _parallel_stages(
    branches: Sequence[tuple[Literal["A", "B"], int]],
    width: int,
) -> tuple[InvariantGateStageV2Config, ...]:
    stages = []
    names = []
    counts = {"A": 0, "B": 0}
    for stream, channels in branches:
        name = _stage_name(stream, counts)
        stages.append(_stage(name, stream, ("x", "r"), channels, width))
        names.append(name)
    stages.append(_stage("out", "A", ("x", "r", *names), 1, width))
    return tuple(stages)


def _pipeline(
    model_id: str,
    stages: Sequence[InvariantGateStageV2Config],
) -> InvariantGatePipelineV2Config:
    return InvariantGatePipelineV2Config(
        stages=tuple(stages),
        output_stage="out",
        architecture_id=f"benzene_pair_sweep_{model_id.lower()}",
        anchor_ranks=ANCHOR_RANKS,
    )


def _spec(
    model_id: str,
    description: str,
    purpose: str,
    expected_parameter_count: int,
    stages: Sequence[InvariantGateStageV2Config],
    comparison_role: ComparisonRole = "candidate",
) -> ModelSpec:
    return ModelSpec(
        model_id,
        description,
        purpose,
        expected_parameter_count,
        _pipeline(model_id, stages),
        comparison_role,
    )


def build_model_specs() -> tuple[ModelSpec, ...]:
    """Return the ordered sweep specification from C01 through C30."""
    full_w8 = (8, 8, 8, 8)
    stf_only: PathFlags = {
        "symmetric": False,
        "raw_mixed": False,
        "stf": True,
    }
    specs = (
        _spec(
            "C01",
            "direct output with width 192",
            "wide direct readout baseline",
            20_234,
            (_stage("out", "A", ("x", "r"), 1, 192),),
        ),
        _spec(
            "C02",
            "A5 with width 32",
            "single A balanced baseline",
            20_240,
            _sequential_stages((("A", 5),), (32, 32)),
        ),
        _spec(
            "C03",
            "A7 with width 24",
            "more A channels versus trunk width",
            20_228,
            _sequential_stages((("A", 7),), (24, 24)),
        ),
        _spec(
            "C04",
            "A8 width 24 then output width 16",
            "channel heavy single A model",
            19_927,
            _sequential_stages((("A", 8),), (24, 16)),
        ),
        _spec(
            "C05",
            "B2 width 12 then output width 20",
            "single B balanced baseline",
            20_234,
            _sequential_stages((("B", 2),), (12, 20)),
        ),
        _spec(
            "C06",
            "B3 width 4 then output width 20",
            "B channel heavy model with narrow trunk",
            19_207,
            _sequential_stages((("B", 3),), (4, 20)),
        ),
        _spec(
            "C07",
            "parallel A2 and B1 then output with width 20",
            "parallel A and B without a serial bottleneck",
            19_722,
            _parallel_stages((("A", 2), ("B", 1)), 20),
        ),
        _spec(
            "C08",
            "parallel A3 and B3 then output with width 8",
            "wider parallel A and B",
            19_557,
            _parallel_stages((("A", 3), ("B", 3)), 8),
        ),
        _spec(
            "C09",
            "A2 then B1 then output with width 16",
            "short serial A and B reference",
            20_020,
            _sequential_stages((("A", 2), ("B", 1)), (16, 16, 16)),
        ),
        _spec(
            "C10",
            "A3 then B2 then output with width 8",
            "wider A and B with narrow trunks",
            20_676,
            _sequential_stages((("A", 3), ("B", 2)), (8, 8, 8)),
        ),
        _spec(
            "C11",
            "B1 then A2 then output with width 16",
            "reverse ordering of C09",
            20_932,
            _sequential_stages((("B", 1), ("A", 2)), (16, 16, 16)),
        ),
        _spec(
            "C12",
            "A2 then A3 then output with width 24",
            "test whether a hidden B stage is needed",
            20_016,
            _sequential_stages((("A", 2), ("A", 3)), (24, 24, 24)),
        ),
        _spec(
            "C13",
            "A4 then A2 then output with width 20",
            "contracting A channels",
            20_044,
            _sequential_stages((("A", 4), ("A", 2)), (20, 20, 20)),
        ),
        _spec(
            "C14",
            "B1 then B1 then output with width 8",
            "B only depth test",
            19_195,
            _sequential_stages((("B", 1), ("B", 1)), (8, 8, 8)),
        ),
        _spec(
            "C15",
            "A2 then A2 then B1 then output with widths 10 6 6 24",
            "primary narrow B model with wide readout",
            20_005,
            _sequential_stages(
                (("A", 2), ("A", 2), ("B", 1)),
                (10, 6, 6, 24),
            ),
            "primary",
        ),
        _spec(
            "C16",
            "A2 then A4 then B1 then output with width 8",
            "expand A before a narrow B stage",
            20_459,
            _sequential_stages(
                (("A", 2), ("A", 4), ("B", 1)),
                full_w8,
            ),
        ),
        _spec(
            "C17",
            "A1 then B2 then B1 then output with width 4",
            "two stage B contraction",
            19_939,
            _sequential_stages(
                (("A", 1), ("B", 2), ("B", 1)),
                (4, 4, 4, 4),
            ),
        ),
        _spec(
            "C18",
            "A2 then B1 then A1 then output with width 12",
            "A B A interleaving",
            19_984,
            _sequential_stages(
                (("A", 2), ("B", 1), ("A", 1)),
                (12, 12, 12, 12),
            ),
        ),
        _spec(
            "C19",
            "B1 then A1 then B2 then output with width 4",
            "B A B interleaving",
            20_012,
            _sequential_stages(
                (("B", 1), ("A", 1), ("B", 2)),
                (4, 4, 4, 4),
            ),
        ),
        _spec(
            "C20",
            "A3 then B1 then A2 then B1 then output with width 4",
            "deeper alternating model",
            20_352,
            _sequential_stages(
                (("A", 3), ("B", 1), ("A", 2), ("B", 1)),
                (4, 4, 4, 4, 4),
            ),
        ),
        _spec(
            "C21",
            "A4 then A4 then B1 with width 8 and no symmetric paths",
            "symmetric unary ablation",
            20_288,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 1)),
                full_w8,
                flags={index: {"symmetric": False} for index in range(4)},
            ),
        ),
        _spec(
            "C22",
            "A4 then A4 then B1 with width 21 and no raw mixed paths",
            "joint information negative control",
            20_013,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 1)),
                (21, 21, 21, 21),
                flags={index: {"raw_mixed": False} for index in range(4)},
            ),
        ),
        _spec(
            "C23",
            "A4 then A4 then B1 with width 6 and no STF shortcuts",
            "generic C paths versus STF conditioning",
            20_093,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 1)),
                (6, 6, 6, 6),
                flags={index: {"stf": False} for index in range(4)},
            ),
        ),
        _spec(
            "C24",
            "A4 then A4 then B1 with width 6 and raw gate context",
            "minimal invariant context",
            19_061,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 1)),
                (6, 6, 6, 6),
                invariant_context="raw",
            ),
        ),
        _spec(
            "C25",
            "A4 then A4 then B1 with width 6 and local gate context",
            "raw plus latest invariant context",
            19_757,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 1)),
                (6, 6, 6, 6),
                invariant_context="raw_latest",
            ),
        ),
        _spec(
            "C26",
            "A4 then A4 then B1 with width 6 and GELU",
            "activation control",
            20_333,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 1)),
                (6, 6, 6, 6),
                flags={index: {"activation": "gelu"} for index in range(4)},
            ),
        ),
        _spec(
            "C27",
            "A4 then A4 then B1 with width 8 and selective paths",
            "allocate optional paths by stage",
            19_667,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 1)),
                full_w8,
                flags={
                    0: {"symmetric": False},
                    1: stf_only,
                },
            ),
        ),
        _spec(
            "C28",
            "A6 then A6 then B1 with width 20 and light hidden paths",
            "spend capacity on channels and the full readout",
            19_917,
            _sequential_stages(
                (("A", 6), ("A", 6), ("B", 1)),
                (20, 20, 20, 20),
                flags={index: stf_only for index in range(3)},
            ),
        ),
        _spec(
            "C29",
            "direct output with width 24",
            "small capacity lower bound",
            2_594,
            (_stage("out", "A", ("x", "r"), 1, 24),),
            "lower_control",
        ),
        _spec(
            "C30",
            "default A4 then A4 then B4 with width 32",
            "full capacity upper bound",
            240_746,
            _sequential_stages(
                (("A", 4), ("A", 4), ("B", 4)),
                (32, 32, 32, 32),
            ),
            "upper_control",
        ),
    )
    expected_ids = tuple(f"C{index:02d}" for index in range(1, 31))
    if tuple(item.model_id for item in specs) != expected_ids:
        raise RuntimeError("model catalog identifiers must run from C01 through C30")
    if len({item.pipeline.architecture_id for item in specs}) != len(specs):
        raise RuntimeError("model catalog architecture identifiers must be unique")
    return specs


MODEL_SPECS = build_model_specs()
_MODEL_LOOKUP = {item.model_id: item for item in MODEL_SPECS}


def get_model_spec(model_id: str) -> ModelSpec:
    """Return one model specification by its case insensitive identifier."""
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    key = model_id.upper()
    try:
        return _MODEL_LOOKUP[key]
    except KeyError as error:
        raise KeyError(f"unknown sweep model {model_id}") from error


def build_sweep_model(
    model: str | ModelSpec,
    generators: Tensor,
    *,
    generator_names: Sequence[str] | None = None,
) -> InvariantGatePipelineV2:
    """Compile one model and enforce the recorded parameter count."""
    spec = get_model_spec(model) if isinstance(model, str) else model
    if not isinstance(spec, ModelSpec):
        raise TypeError("model must be a model identifier or ModelSpec")
    result = build_invariant_gate_pipeline_v2(
        generators,
        spec.pipeline,
        generator_names=generator_names,
    )
    if result.trainable_parameter_count != spec.expected_parameter_count:
        raise RuntimeError(
            f"{spec.model_id} compiled with {result.trainable_parameter_count} "
            f"parameters instead of {spec.expected_parameter_count}"
        )
    return result
