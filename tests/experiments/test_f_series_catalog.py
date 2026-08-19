"""Validate the complete paired F series catalog."""

from __future__ import annotations

import json

import torch

from experiments.benzene_pair import sweep30 as common
from experiments.benzene_pair.f_series.catalog import (
    F0_SPECS,
    F1_SPECS,
    F2_SPECS,
    F3_SPECS,
    F_SERIES_SPECS,
    get_experiment_specs,
    get_model_spec,
)
from experiments.benzene_pair.f_series.model_factory import (
    build_f_series_model,
    strict_config_from_spec,
)


def test_catalog_has_one_control_and_one_hundred_paired_strict_models() -> None:
    assert tuple(map(len, (F0_SPECS, F1_SPECS, F2_SPECS, F3_SPECS))) == (
        1,
        34,
        32,
        34,
    )
    assert len(F_SERIES_SPECS) == 101
    assert tuple(spec.model_id for spec in F0_SPECS) == ("F100",)
    assert tuple(spec.model_id for spec in F1_SPECS) == (
        *(f"F{index:03d}" for index in range(101, 118)),
        *(f"F{index:03d}" for index in range(201, 218)),
    )
    assert tuple(spec.model_id for spec in F2_SPECS) == (
        *(f"F{index:03d}" for index in range(118, 134)),
        *(f"F{index:03d}" for index in range(218, 234)),
    )
    assert tuple(spec.model_id for spec in F3_SPECS) == (
        *(f"F{index:03d}" for index in range(134, 151)),
        *(f"F{index:03d}" for index in range(234, 251)),
    )
    assert get_model_spec("f250") is F3_SPECS[-1]
    assert get_experiment_specs("experiment_2") is F2_SPECS
    for spec in F_SERIES_SPECS:
        json.dumps(spec.as_dict(), allow_nan=False)


def test_every_f1_model_has_one_raw_only_pair_with_identical_topology() -> None:
    strict = tuple(spec for spec in F_SERIES_SPECS if spec.family == "strict_flow")
    assert len(strict) == 100
    for first in strict:
        pair = get_model_spec(first.pair_model_id or "")
        assert pair.pair_model_id == first.model_id
        assert pair.experiment_id == first.experiment_id
        assert pair.options["topology"] == first.options["topology"]
        assert pair.options["channels"] == first.options["channels"]
        assert pair.options["stages"] == first.options["stages"]
        assert pair.planned_parameter_count == first.planned_parameter_count
        assert {pair.descriptor_mask, first.descriptor_mask} == {"full", "raw_only"}


def test_three_strict_topologies_encode_only_declared_edges() -> None:
    expected_names = {
        "T1": ("a1", "b1", "a2", "b2", "a3", "out"),
        "T2": ("a1", "b1", "a2", "a3", "out"),
        "T3": ("a1", "b1", "a2", "b2", "a3", "out"),
    }
    expected_sources = {
        "T1": (
            ("x",),
            ("r",),
            ("a1", "b1"),
            ("b1", "a1"),
            ("a2", "b2"),
            ("a3",),
        ),
        "T2": (("x",), ("r",), ("a1", "b1"), ("a2",), ("a3",)),
        "T3": (
            ("x",),
            ("r",),
            ("a1",),
            ("b1",),
            ("a2", "b2"),
            ("a3",),
        ),
    }
    for model_id in ("F101", "F118", "F134"):
        spec = get_model_spec(model_id)
        topology = str(spec.options["topology"])
        stages = spec.options["stages"]
        assert tuple(stage["name"] for stage in stages) == expected_names[topology]
        assert (
            tuple(stage["source_names"] for stage in stages)
            == expected_sources[topology]
        )
        assert all(
            stage["source_names"] == stage["covariant_required_source_names"]
            for stage in stages
        )
        config = strict_config_from_spec(spec)
        assert config.descriptor_mask == "full"
        assert config.gate_width == 8
        assert config.output_stage == "out"


def test_representative_strict_pairs_compile_to_exact_equal_counts() -> None:
    generators = common._proper_d6_generators()
    for first_id, second_id, expected in (
        ("F101", "F201", 12_635),
        ("F118", "F218", 6_972),
        ("F134", "F234", 11_137),
    ):
        values = []
        for model_id in (first_id, second_id):
            torch.manual_seed(20260822)
            model = build_f_series_model(
                model_id,
                generators,
                generator_names=("sixfold", "twofold"),
            )
            values.append(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            )
            manifest = model.strict_flow_manifest
            assert manifest["trainable_parameter_count"] == expected
            assert not any(
                item["missing_edges"]
                or item["undeclared_sources"]
                or item["live_live_path_roles"]
                for item in manifest["edge_audit"]
            )
        assert values == [expected, expected]
