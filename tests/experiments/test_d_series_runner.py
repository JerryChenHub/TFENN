from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.benzene_pair import sweep30 as common
from experiments.benzene_pair.d_series import runner


def test_three_experiments_have_independent_projects_and_common_protocol(
    tmp_path: Path,
) -> None:
    configs = tuple(
        runner.make_config(index, study_root=tmp_path) for index in range(1, 4)
    )
    assert len({item.study_directory for item in configs}) == 3
    assert len({item.comet.project_name for item in configs}) == 3
    assert {item.effective_batch_size for item in configs} == {10_000}
    assert {item.micro_batch_size for item in configs} == {10_000}
    assert {item.epochs for item in configs} == {500}
    assert {item.split_seed for item in configs} == {20260821}
    assert {item.model_seed for item in configs} == {20260822}
    assert {item.shuffle_seed for item in configs} == {20260823}
    assert len({item.shard_paths for item in configs}) == 1
    assert all(item.comet.enabled and item.comet.required_online for item in configs)
    assert {item.as_dict(device="cuda")["schema_name"] for item in configs} == {
        "tfenn_benzene_pair_d_series"
    }


def test_each_experiment_selects_exactly_twenty_five_models() -> None:
    groups = tuple(runner._select_specs(index, ()) for index in range(1, 4))
    assert tuple(len(group) for group in groups) == (25, 25, 25)
    assert tuple(item.model_id for item in groups[0]) == tuple(
        f"D{index:02d}" for index in range(1, 26)
    )
    assert tuple(item.model_id for item in groups[1]) == tuple(
        f"D{index:02d}" for index in range(26, 51)
    )
    assert tuple(item.model_id for item in groups[2]) == tuple(
        f"D{index:02d}" for index in range(51, 76)
    )
    assert len({item.model_id for group in groups for item in group}) == 75


def test_model_filter_rejects_cross_experiment_and_duplicates() -> None:
    assert tuple(item.model_id for item in runner._select_specs(2, ("d26", "D50"))) == (
        "D26",
        "D50",
    )
    with pytest.raises(ValueError, match="outside experiment"):
        runner._select_specs(1, ("D26",))
    with pytest.raises(ValueError, match="duplicates"):
        runner._select_specs(3, ("D51", "d51"))


def test_config_files_match_the_declared_experiments() -> None:
    for experiment_id, path in runner.DEFAULT_CONFIG_PATHS.items():
        value = json.loads(path.read_text(encoding="utf_8"))
        definition = runner.EXPERIMENTS[experiment_id]
        assert value["experiment_id"] == experiment_id
        assert value["study_directory_name"] == definition.directory_name
        assert value["comet"]["project_name"] == definition.comet_project
        assert value["effective_batch_size"] == 10_000
        assert value["micro_batch_size"] == 10_000
        assert value["concurrent_run"] is True
        assert value["shared_gpu_process_count"] == 3
        assert tuple(value["model_ids"]) == tuple(
            item.model_id for item in runner._select_specs(experiment_id, ())
        )


class _CalibratedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("projection", torch.eye(2))

    def descriptor_projection_state_dict(self) -> dict[str, torch.Tensor]:
        return {"projection": self.projection.detach().clone()}

    def load_descriptor_projection_state_dict(
        self, state: dict[str, torch.Tensor]
    ) -> None:
        self.projection.copy_(state["projection"])


class _Spec:
    model_id = "D67"


def test_checkpoint_round_trip_includes_calibrated_fixed_state() -> None:
    model = _CalibratedModel()
    model.projection.copy_(torch.tensor(((1.0, 0.0), (0.0, 0.0))))
    payload = common._checkpoint_payload(
        model,
        spec=_Spec(),
        epoch=4,
        target_scale=torch.tensor(1.0),
        metrics={"normalized_mse": 0.25},
        trial_hash="trial",
        calibration_report={"policy": "pca_99"},
    )
    assert payload["calibration_report"] == {"policy": "pca_99"}
    assert set(payload["calibration_state_dict"]) == {"projection"}
    restored = _CalibratedModel()
    common._restore_model_state(
        restored,
        payload["parameter_state_dict"],
        payload["normalization_state_dict"],
        payload["calibration_state_dict"],
    )
    torch.testing.assert_close(restored.weight, model.weight)
    torch.testing.assert_close(restored.projection, model.projection)


def test_parser_supports_run_trial_and_selective_smoke() -> None:
    parser = runner.build_parser()
    prepare = parser.parse_args(("prepare",))
    assert prepare.study_root == runner.DEFAULT_STUDY_ROOT
    run = parser.parse_args(("run", "--experiment", "1", "--model", "D01"))
    assert run.experiment == 1
    assert run.model == ["D01"]
    trial = parser.parse_args(
        (
            "trial",
            "--experiment",
            "2",
            "--model",
            "D26",
            "--sample_limit",
            "100",
            "--disable_comet",
        )
    )
    assert trial.disable_comet
    smoke = parser.parse_args(
        ("smoke", "--experiment", "3", "--model", "D51", "--epochs", "1")
    )
    assert smoke.model == ["D51"]
    assert smoke.epochs == 1
