"""Validate the complete fixed-shape G series catalog."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benzene_pair.g_series.catalog import (
    CARRIER_SPECS,
    FACTORIAL_SPECS,
    G_SERIES_SPECS,
    LEGACY_SPECS,
    SEED_BLOCK_SPECS,
    VARIANT_SPECS,
    get_group_specs,
    get_model_spec,
    get_seed_specs,
    get_variant_specs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT / "experiments" / "benzene_pair" / "g_series" / "config.json"
)


def test_catalog_has_fourteen_variants_across_five_paired_seeds() -> None:
    expected = tuple(
        f"G{seed_index}{variant_id:02d}"
        for seed_index in range(1, 6)
        for variant_id in range(1, 15)
    )
    assert tuple(spec.model_id for spec in G_SERIES_SPECS) == expected
    assert len(G_SERIES_SPECS) == len(set(expected)) == 70
    assert tuple(SEED_BLOCK_SPECS) == (1, 2, 3, 4, 5)
    assert tuple(VARIANT_SPECS) == tuple(range(1, 15))
    assert all(len(SEED_BLOCK_SPECS[index]) == 14 for index in range(1, 6))
    assert all(len(VARIANT_SPECS[index]) == 5 for index in range(1, 15))
    assert get_model_spec("g514") is G_SERIES_SPECS[-1]
    assert get_seed_specs("seed_1") is SEED_BLOCK_SPECS[1]
    assert get_variant_specs("variant_14") is VARIANT_SPECS[14]
    for spec in G_SERIES_SPECS:
        json.dumps(spec.as_dict(), allow_nan=False)
        assert spec.expected_parameter_count is None
        assert not spec.d6_covariance_exempt


def test_paired_seed_blocks_use_the_registered_seed_offsets() -> None:
    for seed_index in range(1, 6):
        block = get_seed_specs(seed_index)
        assert {spec.seed_index for spec in block} == {seed_index}
        assert {spec.execution_shard_id for spec in block} == {seed_index - 1}
        assert {spec.model_seed for spec in block} == {
            20_260_822 + 10 * (seed_index - 1)
        }
        assert {spec.shuffle_seed for spec in block} == {
            20_260_823 + 10 * (seed_index - 1)
        }


def test_factorial_contains_every_generic_pair_and_stf_combination() -> None:
    assert len(FACTORIAL_SPECS) == 40
    assert get_group_specs("factorial") is FACTORIAL_SPECS
    assert get_group_specs("g1") is FACTORIAL_SPECS
    for seed_index in range(1, 6):
        combinations = {
            (
                bool(spec.options["generic_pair_enabled"]),
                bool(spec.options["stf_a1_enabled"]),
                bool(spec.options["stf_out_enabled"]),
            )
            for spec in FACTORIAL_SPECS
            if spec.seed_index == seed_index
        }
        assert combinations == {
            (generic, stf_a1, stf_out)
            for generic in (False, True)
            for stf_a1 in (False, True)
            for stf_out in (False, True)
        }


def test_every_variant_keeps_the_e311_graph_and_full_scalar_context() -> None:
    expected_sources = (
        ("x", "r"),
        ("x", "r", "a1"),
        ("x", "r", "a1", "b1"),
        ("x", "r", "a1", "b1", "b2"),
    )
    for spec in G_SERIES_SPECS:
        assert spec.options["backbone"] == "C17"
        assert spec.options["source_graph"] == "C17_DENSE_HISTORY"
        assert spec.options["gate_width"] == 8
        assert spec.options["scalar_invariants"] == "full"
        assert spec.options["fixed_shape_supernet"] is True
        assert tuple(
            tuple(stage["sources"]) for stage in spec.options["stage_sources"]
        ) == expected_sources


def test_legacy_and_carrier_groups_have_registered_masks_and_modes() -> None:
    assert len(LEGACY_SPECS) == 20
    assert len(CARRIER_SPECS) == 20
    assert get_group_specs("legacy") is LEGACY_SPECS
    assert get_group_specs("carrier") is CARRIER_SPECS
    assert {spec.variant_id for spec in LEGACY_SPECS} == {1, 9, 10, 11}
    assert {spec.variant_id for spec in CARRIER_SPECS} == {1, 12, 13, 14}

    no_carrier = get_variant_specs(9)[0]
    assert no_carrier.options["carrier_mode"] == "none"
    assert not any(no_carrier.options["carrier_group_mask"].values())
    assert get_variant_specs(10)[0].options["gated_identity_initialization"] == (
        "residual_zero"
    )
    assert get_variant_specs(11)[0].options["gated_identity_initialization"] == (
        "default"
    )
    no_raw = get_variant_specs(12)[0].options["carrier_group_mask"]
    no_hidden = get_variant_specs(13)[0].options["carrier_group_mask"]
    no_both = get_variant_specs(14)[0].options["carrier_group_mask"]
    assert no_raw == {
        "stem": True,
        "adjacent": True,
        "raw_deep": False,
        "hidden_deep": True,
    }
    assert no_hidden == {
        "stem": True,
        "adjacent": True,
        "raw_deep": True,
        "hidden_deep": False,
    }
    assert no_both == {
        "stem": True,
        "adjacent": True,
        "raw_deep": False,
        "hidden_deep": False,
    }


def test_config_is_the_cpu_batch_one_thousand_comet_protocol() -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf_8"))
    assert value["schema_name"] == "tfenn_benzene_pair_g_series"
    assert value["schema_version"] == 1
    assert value["model_count"] == len(G_SERIES_SPECS) == 70
    assert tuple(value["model_ids"]) == tuple(
        spec.model_id for spec in G_SERIES_SPECS
    )
    assert value["device"] == "cpu"
    assert value["batch_size"] == 1000
    assert value["effective_batch_size"] == 1000
    assert value["micro_batch_size"] == 1000
    assert value["epochs"] == 500
    assert value["learning_rate"] == 0.003
    assert value["weight_decay"] == 0.0001
    assert value["scheduler_step_size"] == 125
    assert value["scheduler_gamma"] == 0.5
    assert value["threads"] == 4
    assert value["enable_tf32"] is False
    assert value["comet"] == {
        "enabled": True,
        "required_online": True,
        "project_name": "tfenn_g_series_e311_mechanisms_cpu",
        "workspace": None,
        "upload_checkpoints": True,
        "tags": [
            "tfenn",
            "benzene_pair",
            "opls_2_0_0",
            "g_series",
            "e311_mechanisms",
            "cpu",
        ],
    }
