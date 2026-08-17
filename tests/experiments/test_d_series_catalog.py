"""Validate the complete D series experiment catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benzene_pair.d_series.catalog import (
    D_SERIES_SPECS,
    EXPERIMENT_1_SPECS,
    EXPERIMENT_2_SPECS,
    EXPERIMENT_3_SPECS,
    get_experiment_specs,
    get_model_spec,
)
from experiments.benzene_pair.d_series.model_factory import pipeline_config_from_spec


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "experiments" / "benzene_pair" / "d_series"


def test_catalog_has_three_ordered_independent_groups() -> None:
    assert tuple(item.model_id for item in D_SERIES_SPECS) == tuple(
        f"D{index:02d}" for index in range(1, 76)
    )
    assert tuple(
        map(len, (EXPERIMENT_1_SPECS, EXPERIMENT_2_SPECS, EXPERIMENT_3_SPECS))
    ) == (
        25,
        25,
        25,
    )
    assert get_experiment_specs(1) is EXPERIMENT_1_SPECS
    assert get_experiment_specs("d26_d50") is EXPERIMENT_2_SPECS
    assert get_experiment_specs("experiment_3") is EXPERIMENT_3_SPECS
    assert get_model_spec("d75") is EXPERIMENT_3_SPECS[-1]
    assert all("group" not in item.description.lower() for item in D_SERIES_SPECS)


def test_every_spec_is_complete_and_json_serializable() -> None:
    for spec in D_SERIES_SPECS:
        value = spec.as_dict()
        assert value["purpose"]
        assert value["source_policy"]
        assert value["invariant_source_policy"]
        assert value["skip_policy"]
        assert value["path_policy"]
        assert value["gate_policy"]
        assert value["stages"][-1]["name"] == "out"
        assert value["stages"][-1]["output_stream"] == "A"
        json.dumps(value, allow_nan=False)


def test_experiment_1_joint_sources_and_custom_deletions_are_exact() -> None:
    for model_id in (
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
        "D06",
        "D07",
        "D08",
        "D09",
        "D10",
    ):
        spec = get_model_spec(model_id)
        assert tuple(stage.source_names for stage in spec.stages) == tuple(
            stage.invariant_source_names for stage in spec.stages
        )
        assert spec.skip_policy == "none"
    d17 = get_model_spec("D17")
    assert d17.stages[0].source_names == ("x", "r")
    assert all("x" not in stage.source_names for stage in d17.stages[1:])
    d19 = get_model_spec("D19")
    assert "r" in d19.stages[1].source_names
    assert all("r" not in stage.source_names for stage in d19.stages[2:])
    assert get_model_spec("D22").stages[-1].source_names == (
        "a1",
        "b1",
        "a2",
        "b2",
    )
    assert get_model_spec("D23").stages[-1].source_names == ("x", "r", "b2")
    assert get_model_spec("D24").expected_parameter_count == 19_939
    assert get_model_spec("D25").expected_parameter_count == 20_352


def test_experiment_2_counts_paths_and_cubic_policies_match_plan() -> None:
    counts = (
        9_896,
        9_688,
        14_954,
        14_640,
        10_064,
        10_232,
        9_803,
        9_918,
        10_033,
        10_579,
        11_235,
        15_904,
        15_808,
        10_838,
        10_878,
        15_003,
        14_858,
        9_154,
        9_911,
        9_876,
    )
    assert (
        tuple(item.expected_parameter_count for item in EXPERIMENT_2_SPECS[:20])
        == counts
    )
    assert tuple(
        get_model_spec(f"D{index}").expected_parameter_count for index in range(46, 50)
    ) == (21_019, 12_348, 30_018, 32_470)
    assert get_model_spec("D43").path_policy == "NO_SYM2"
    assert get_model_spec("D44").path_policy == "NO_RAW_MIXED"
    assert get_model_spec("D45").path_policy == "NO_STF"
    assert tuple(
        get_model_spec(f"D{index}").options["degree3_policy"] for index in range(46, 51)
    ) == (
        "sym3",
        "a2b",
        "ab2",
        "union",
        "all",
    )
    assert tuple(
        get_model_spec(f"D{index}").options["degree3_overflow_policy"]
        for index in range(46, 51)
    ) == ("raise", "raise", "raise", "raise", "audit_skip")
    assert all(
        get_model_spec(f"D{index}").options["max_constraint_entries"] == 10_000_000
        for index in range(46, 51)
    )
    assert tuple(
        pipeline_config_from_spec(get_model_spec(f"D{index}")).degree3_overflow_policy
        for index in range(46, 51)
    ) == ("raise", "raise", "raise", "raise", "audit_skip")
    assert all(
        pipeline_config_from_spec(get_model_spec(f"D{index}")).max_constraint_entries
        == 10_000_000
        for index in range(46, 51)
    )


def test_experiment_3_gate_controls_match_plan() -> None:
    assert get_model_spec("D52").options["coefficient_activation"] == "sigmoid"
    assert get_model_spec("D53").options["coefficient_activation"] == "tanh"
    assert get_model_spec("D55").options["coefficient_head"] == "static_mixing"
    assert get_model_spec("D55").options["metric_gate"] == "norm"
    assert get_model_spec("D57").options["metric_gate"] == "skip_identity"
    assert tuple(
        get_model_spec(f"D{index}").options["coefficient_rank"]
        for index in range(58, 62)
    ) == (
        1,
        2,
        3,
        4,
    )
    assert get_model_spec("D63").options["depends_on"] == "D51"
    assert get_model_spec("D68").options["retained_variance"] == pytest.approx(0.99)
    assert get_model_spec("D69").options["retained_rank_from"] == "D68"
    assert get_model_spec("D70").options["trunk_linearized"] is True
    assert get_model_spec("D74").options["trunk_depth"] == 3
    assert get_model_spec("D75").options["trunk_residual"] is True


def test_three_configs_share_protocol_but_isolate_outputs_and_comet() -> None:
    configs = []
    for experiment_id in (1, 2, 3):
        path = CONFIG_ROOT / f"experiment_{experiment_id}.json"
        value = json.loads(path.read_text(encoding="utf_8"))
        assert value["experiment_id"] == experiment_id
        assert value["epochs"] == 500
        assert value["effective_batch_size"] == 10_000
        assert value["micro_batch_size"] == 10_000
        assert value["expected_sample_count"] == 400_000
        assert value["model_ids"] == [
            item.model_id for item in get_experiment_specs(experiment_id)
        ]
        configs.append(value)
    shared = (
        "shard_paths",
        "learning_rate",
        "weight_decay",
        "scheduler_step_size",
        "scheduler_gamma",
        "split_seed",
        "model_seed",
        "shuffle_seed",
        "split_fractions",
    )
    for key in shared:
        assert configs[0][key] == configs[1][key] == configs[2][key]
    assert len({item["study_directory_name"] for item in configs}) == 3
    assert len({item["comet"]["project_name"] for item in configs}) == 3
