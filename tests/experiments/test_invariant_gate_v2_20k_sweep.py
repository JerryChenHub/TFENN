from __future__ import annotations

import math

import pytest
import torch

from experiments.benzene_pair.invariant_gate_v2_20k_sweep import (
    ANCHOR_RANKS,
    MODEL_SPECS,
    ModelSpec,
    build_model_specs,
    build_sweep_model,
    get_model_spec,
)
from TFENN.models import InvariantGatePipelineV2Config


def _proper_d6_generators() -> torch.Tensor:
    cosine = math.cos(math.pi / 3.0)
    sine = math.sin(math.pi / 3.0)
    sixfold = torch.tensor(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    twofold = torch.diag(torch.tensor((1.0, -1.0, -1.0), dtype=torch.float64))
    return torch.stack((sixfold, twofold))


def test_catalog_is_complete_ordered_and_serializable() -> None:
    expected_ids = tuple(f"C{index:02d}" for index in range(1, 31))
    assert tuple(item.model_id for item in MODEL_SPECS) == expected_ids
    assert build_model_specs() == MODEL_SPECS
    assert len({item.pipeline.architecture_id for item in MODEL_SPECS}) == 30
    assert all(item.pipeline.anchor_ranks == ANCHOR_RANKS for item in MODEL_SPECS)
    assert all(item.description and item.purpose for item in MODEL_SPECS)
    assert get_model_spec("c15") is MODEL_SPECS[14]
    assert get_model_spec("C15").comparison_role == "primary"
    assert get_model_spec("C29").comparison_role == "lower_control"
    assert get_model_spec("C30").comparison_role == "upper_control"
    for item in MODEL_SPECS:
        restored = InvariantGatePipelineV2Config.from_dict(item.as_dict()["pipeline"])
        assert restored == item.pipeline

    with pytest.raises(KeyError, match="unknown sweep model"):
        get_model_spec("C31")
    with pytest.raises(TypeError, match="model_id must be a string"):
        get_model_spec(15)


def test_routes_contexts_and_path_controls_match_the_instruction() -> None:
    parallel = get_model_spec("C07").pipeline.stages
    assert tuple(stage.source_names for stage in parallel) == (
        ("x", "r"),
        ("x", "r"),
        ("x", "r", "a1", "b1"),
    )

    primary = get_model_spec("C15").pipeline.stages
    assert tuple(
        (stage.output_stream, stage.channels, stage.trunk_width) for stage in primary
    ) == (
        ("A", 2, 10),
        ("A", 2, 6),
        ("B", 1, 6),
        ("A", 1, 24),
    )
    assert primary[-1].source_names == ("x", "r", "a1", "a2", "b1")

    assert all(
        not stage.include_symmetric_unary
        for stage in get_model_spec("C21").pipeline.stages
    )
    assert all(
        not stage.include_raw_mixed_pairs
        for stage in get_model_spec("C22").pipeline.stages
    )
    assert all(
        not stage.include_stf_shortcuts
        for stage in get_model_spec("C23").pipeline.stages
    )
    assert all(
        stage.invariant_source_names == ("x", "r")
        for stage in get_model_spec("C24").pipeline.stages
    )
    assert tuple(
        stage.invariant_source_names for stage in get_model_spec("C25").pipeline.stages
    ) == (
        ("x", "r"),
        ("x", "r", "a1"),
        ("x", "r", "a2"),
        ("x", "r", "b1"),
    )
    assert all(
        stage.activation == "gelu" for stage in get_model_spec("C26").pipeline.stages
    )

    selective = get_model_spec("C27").pipeline.stages
    assert not selective[0].include_symmetric_unary
    assert selective[0].include_raw_mixed_pairs
    assert selective[0].include_stf_shortcuts
    assert not selective[1].include_symmetric_unary
    assert not selective[1].include_raw_mixed_pairs
    assert selective[1].include_stf_shortcuts
    assert all(
        stage.include_symmetric_unary
        and stage.include_raw_mixed_pairs
        and stage.include_stf_shortcuts
        for stage in selective[2:]
    )

    light_hidden = get_model_spec("C28").pipeline.stages
    assert all(
        not stage.include_symmetric_unary
        and not stage.include_raw_mixed_pairs
        and stage.include_stf_shortcuts
        for stage in light_hidden[:-1]
    )
    assert (
        light_hidden[-1].include_symmetric_unary
        and light_hidden[-1].include_raw_mixed_pairs
        and light_hidden[-1].include_stf_shortcuts
    )


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda item: item.model_id)
def test_each_model_compiles_to_the_instruction_parameter_count(
    spec: ModelSpec,
) -> None:
    model = build_sweep_model(
        spec,
        _proper_d6_generators(),
        generator_names=("sixfold", "twofold"),
    )
    assert model.trainable_parameter_count == spec.expected_parameter_count
    assert tuple(item.stf_rank for item in model.manifest) == ANCHOR_RANKS
