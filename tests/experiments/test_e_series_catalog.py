"""Validate the complete E series experiment catalog."""

from __future__ import annotations

import json

from experiments.benzene_pair.e_series.catalog import (
    E0_SPECS,
    E1_SPECS,
    E2_SPECS,
    E3_SPECS,
    E4_SPECS,
    E_SERIES_SPECS,
    get_experiment_specs,
    get_model_spec,
)
from TFENN.models.e_series import _compact_config


def test_catalog_has_all_ordered_identifiers_and_five_groups() -> None:
    expected = (
        *(f"E{index:03d}" for index in range(1, 9)),
        *(
            f"E{experiment}{index:02d}"
            for experiment in range(1, 5)
            for index in range(1, 26)
        ),
    )
    assert tuple(spec.model_id for spec in E_SERIES_SPECS) == expected
    assert tuple(map(len, (E0_SPECS, E1_SPECS, E2_SPECS, E3_SPECS, E4_SPECS))) == (
        8,
        25,
        25,
        25,
        25,
    )
    assert len(E_SERIES_SPECS) == 108
    for experiment, expected_specs in enumerate(
        (E0_SPECS, E1_SPECS, E2_SPECS, E3_SPECS, E4_SPECS)
    ):
        assert get_experiment_specs(experiment) is expected_specs
        assert get_experiment_specs(f"E{experiment}") is expected_specs
        assert get_experiment_specs(f"experiment_{experiment}") is expected_specs
    assert get_model_spec("e425") is E4_SPECS[-1]


def test_planned_counts_never_become_compiled_count_assertions() -> None:
    assert all(spec.expected_parameter_count is None for spec in E_SERIES_SPECS)
    assert all(spec.description and spec.purpose for spec in E_SERIES_SPECS)
    for spec in E_SERIES_SPECS:
        json.dumps(spec.as_dict(), allow_nan=False)
    assert tuple(spec.planned_parameter_count for spec in E0_SPECS) == (
        20_160,
        50_204,
        20_160,
        50_204,
        19_939,
        20_352,
        9_911,
        240_746,
    )
    assert all(spec.target_parameter_range == (7_800, 8_200) for spec in E4_SPECS)


def test_e1_raw_schedules_bypasses_widths_and_estimates_match_plan() -> None:
    expected_counts = (
        19_939,
        19_926,
        20_101,
        20_088,
        19_918,
        19_906,
        19_922,
        19_915,
        19_981,
        19_974,
        19_934,
        19_927,
        19_932,
        19_925,
        19_759,
        19_753,
        19_971,
        19_965,
        20_106,
        20_099,
        19_920,
        19_959,
        19_949,
        19_964,
        19_958,
    )
    assert tuple(spec.planned_parameter_count for spec in E1_SPECS) == expected_counts
    assert all(spec.options["path_policy"] == "FULL" for spec in E1_SPECS)
    assert all(len(spec.options["stages"]) == 4 for spec in E1_SPECS)
    assert all(
        tuple(stage["name"] for stage in spec.options["stages"])
        == ("a1", "b1", "b2", "out")
        for spec in E1_SPECS
    )
    for left, right in zip(E1_SPECS[:18:2], E1_SPECS[1:18:2]):
        assert left.options["bypass"] == "L"
        assert right.options["bypass"] == "N"
        assert left.options["raw_schedules"] == right.options["raw_schedules"]
        assert tuple(
            tuple(
                stage[key] for key in ("name", "source_names", "invariant_source_names")
            )
            for stage in left.options["stages"]
        ) == tuple(
            tuple(
                stage[key] for key in ("name", "source_names", "invariant_source_names")
            )
            for stage in right.options["stages"]
        )
        assert all(stage["skip_policy"] == "legacy" for stage in left.options["stages"])
        assert all(stage["skip_policy"] == "none" for stage in right.options["stages"])
    assert get_model_spec("E117").options["raw_schedules"] == {
        "Cx": "F",
        "Cr": "F",
        "Ix": "A",
        "Ir": "A",
    }
    assert get_model_spec("E125").options["raw_schedules"] == {
        "Cx": "A",
        "Cr": "A",
        "Ix": "F",
        "Ir": "F",
    }


def test_e2_profiles_exchange_policies_and_synchronous_levels_are_exact() -> None:
    expected_schedules = {
        "P1": ((1, 1), (1, 1), (1, 1)),
        "P2": ((1, 2), (1, 2), (1, 2)),
        "P3": ((2, 1), (2, 1), (2, 1)),
        "P4": ((1, 2), (1, 1), (1, 1)),
        "P5": ((3, 1), (2, 1), (1, 1)),
    }
    for profile_index, profile in enumerate(expected_schedules, start=0):
        specs = E2_SPECS[profile_index * 5 : (profile_index + 1) * 5]
        assert tuple(spec.options["exchange_policy"] for spec in specs) == (
            "X0",
            "X1",
            "X2",
            "X3",
            "X4",
        )
        assert all(spec.options["profile"] == profile for spec in specs)
        assert all(
            spec.options["channel_schedule"] == expected_schedules[profile]
            for spec in specs
        )
        assert len({spec.options["trunk_width"] for spec in specs}) == 1
        assert len({spec.options["path_head_quota"] for spec in specs}) == 1
        for spec in specs:
            stages = spec.options["stages"]
            assert tuple(stage["execution_level"] for stage in stages) == (
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
            )
            assert tuple(stage["name"] for stage in stages) == (
                "a0",
                "b0",
                "a1",
                "b1",
                "a2",
                "b2",
                "a3",
                "b3",
                "out",
            )
            assert all(stage["covariant_live_mixed_only"] for stage in stages[2:])
            assert all(
                stage["path_head_quota"] == spec.options["path_head_quota"]
                for stage in stages[2:]
            )
            for left, right in ((2, 3), (4, 5), (6, 7)):
                assert stages[right]["name"] not in stages[left]["source_names"]
                assert stages[left]["name"] not in stages[right]["source_names"]


def test_e3_parameter_prechecks_and_path_cells_match_plan() -> None:
    expected = (
        12_425,
        22_541,
        5_383,
        9_911,
        11_869,
        19_939,
        36_079,
        68_359,
        4_756,
        8_146,
        14_926,
        28_486,
        9_896,
        14_954,
        14_439,
        20_012,
        20_099,
        11_536,
        15_904,
        16_621,
        20_011,
        20_359,
        24_207,
        28_775,
        22_555,
    )
    assert tuple(spec.planned_parameter_count for spec in E3_SPECS) == expected
    assert tuple(
        get_model_spec(model).options["path_policy"]
        for model in (
            "E301",
            "E302",
            "E303",
            "E304",
        )
    ) == ("FULL", "FULL", "NO_RAW_MIXED", "NO_RAW_MIXED")
    assert tuple(
        tuple(stage["trunk_width"] for stage in get_model_spec(model).options["stages"])
        for model in ("E322", "E323", "E324", "E325")
    ) == (
        (8, 4, 4, 4),
        (4, 8, 4, 4),
        (4, 4, 8, 4),
        (4, 4, 4, 8),
    )


def test_e4_has_twenty_five_distinct_mechanisms_and_complete_context_contract() -> None:
    mechanisms = tuple(spec.options["compact_mechanism"] for spec in E4_SPECS)
    assert len(set(mechanisms)) == 25
    assert len({spec.architecture_name for spec in E4_SPECS}) == 25
    assert tuple(spec.options["tier"] for spec in E4_SPECS).count("C") == 3
    for spec in E4_SPECS:
        assert spec.options["full_invariant_context"] is True
        assert spec.options["minimum_bridge_count"] >= 1
        assert spec.options["initial_factorization_rank"] == 1
        assert spec.options["budget_compiler"] == "coverage_preserving_nearest_8k"


def test_e4_executable_blueprints_are_distinct_and_keep_full_context() -> None:
    signatures = []
    for spec in E4_SPECS:
        config = _compact_config(spec.model_id, spec.architecture_name, 1)
        value = config.as_dict()
        value.pop("architecture_id")
        value.pop("implemented_mechanism")
        signatures.append(json.dumps(value, sort_keys=True))
        committed: list[str] = []
        active_level: int | None = None
        pending: list[str] = []
        previous_level = -1
        resolved_levels = []
        for stage in config.stages:
            level = (
                previous_level + 1
                if stage.execution_level is None
                else stage.execution_level
            )
            resolved_levels.append(level)
            previous_level = level
        for stage, level in zip(config.stages, resolved_levels):
            if active_level is None:
                active_level = level
            elif level != active_level:
                committed.extend(pending)
                pending = []
                active_level = level
            assert stage.invariant_source_names == ("x", "r", *committed)
            assert stage.descriptor_mask == "full"
            assert stage.coefficient_activation == "identity"
            pending.append(stage.name)
        assert config.output_stage == "out"
        assert config.stages[-1].output_stream == "A"
        assert config.stages[-1].channels == 1
    assert len(set(signatures)) == 25


def test_e4_specialized_blueprints_activate_the_named_mechanisms() -> None:
    def stages(model_id: str) -> tuple[object, ...]:
        spec = get_model_spec(model_id)
        return _compact_config(model_id, spec.architecture_name, 2).stages

    assert any(stage.trunk_residual for stage in stages("E401"))
    assert any(stage.type_channel_overrides for stage in stages("E404"))
    assert any(stage.reversible_coupling for stage in stages("E406"))
    assert any(stage.parameter_share_group for stage in stages("E407"))
    assert any(stage.parameter_share_group for stage in stages("E408"))
    assert any(stage.parameter_share_group for stage in stages("E420"))
    assert {stage.channel_projection for stage in stages("E417")} == {"tucker"}
    assert {stage.channel_projection for stage in stages("E418")} == {"tensor_train"}
    assert {stage.coefficient_head for stage in stages("E419")} == {"context_lora"}
    assert {stage.channel_projection for stage in stages("E421")} == {"toeplitz"}
    assert {stage.coefficient_head for stage in stages("E422")} == {"axis_cp"}
    assert {stage.path_aggregation for stage in stages("E423")} == {"attention"}
    assert {stage.path_aggregation for stage in stages("E424")} == {"soft_moe"}
    assert {stage.channel_projection for stage in stages("E425")} == {"cayley"}
    assert any(stage.trunk_residual for stage in stages("E425"))
