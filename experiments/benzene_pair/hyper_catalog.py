"""Build the fixed catalog of one hundred benzene pair experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from TFENN.models import (
    MLPConfig,
    PairPipelineConfig,
    RadialConfig,
    StageConfig,
    build_invariant_gate_pipeline,
)


GATE_MLP_WIDTH = 64
MAX_PARAMETER_COUNT = 2_000_000
MAX_GATE_COUNT = 96
MAX_INVARIANT_CHANNELS = 128
MAX_GATE_COEFFICIENTS = 500_000


@dataclass(frozen=True, slots=True)
class TopologySpec:
    code: str
    name: str
    stages: tuple[tuple[str, str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    code: str
    anchor_ranks: tuple[int, ...]
    channel_policy: str
    lift_orders: tuple[int, ...]
    mix_inputs: bool
    mix_components: bool
    mix_channels: bool
    cross_grams: bool
    radial_name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    scheduler_step_size: int
    scheduler_gamma: float


@dataclass(frozen=True, slots=True)
class TrialSpec:
    topology_code: str
    topology_name: str
    profile_code: str
    pipeline: PairPipelineConfig
    learning_rate: float
    weight_decay: float
    batch_size: int
    scheduler_step_size: int
    scheduler_gamma: float

    @property
    def candidate_id(self) -> str:
        return self.pipeline.architecture_id

    def functional_dict(self) -> dict[str, Any]:
        pipeline = self.pipeline.as_dict()
        pipeline.pop("architecture_id")
        return {
            "pipeline": pipeline,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "topology_code": self.topology_code,
            "topology_name": self.topology_name,
            "profile_code": self.profile_code,
            "pipeline": self.pipeline.as_dict(),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
        }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf_8")
    return hashlib.sha256(payload).hexdigest()


def _topologies() -> tuple[TopologySpec, ...]:
    return (
        TopologySpec(
            "t01",
            "direct_fusion",
            (
                ("b1", "B", ("x", "r")),
                ("out", "A", ("b1",)),
            ),
        ),
        TopologySpec(
            "t02",
            "single_weave",
            (
                ("a1", "A", ("x",)),
                ("b1", "B", ("a1", "r")),
                ("out", "A", ("b1",)),
            ),
        ),
        TopologySpec(
            "t03",
            "a_depth",
            (
                ("a1", "A", ("x",)),
                ("a2", "A", ("a1",)),
                ("b1", "B", ("a2", "r")),
                ("out", "A", ("b1",)),
            ),
        ),
        TopologySpec(
            "t04",
            "early_pose",
            (
                ("b1", "B", ("r",)),
                ("a1", "A", ("x", "b1")),
                ("out", "A", ("a1",)),
            ),
        ),
        TopologySpec(
            "t05",
            "parallel_fusion",
            (
                ("a1", "A", ("x",)),
                ("b1", "B", ("r",)),
                ("a2", "A", ("a1", "b1")),
                ("out", "A", ("a2",)),
            ),
        ),
        TopologySpec(
            "t06",
            "residual_single_weave",
            (
                ("a1", "A", ("x",)),
                ("b1", "B", ("a1", "r")),
                ("a2", "A", ("b1", "a1")),
                ("out", "A", ("a2",)),
            ),
        ),
        TopologySpec(
            "t07",
            "double_weave",
            (
                ("a1", "A", ("x",)),
                ("b1", "B", ("a1", "r")),
                ("a2", "A", ("b1",)),
                ("b2", "B", ("a2", "r")),
                ("out", "A", ("b2",)),
            ),
        ),
        TopologySpec(
            "t08",
            "residual_double_weave",
            (
                ("a1", "A", ("x",)),
                ("b1", "B", ("a1", "r")),
                ("a2", "A", ("b1", "a1")),
                ("b2", "B", ("a2", "r")),
                ("out", "A", ("b2", "b1")),
            ),
        ),
        TopologySpec(
            "t09",
            "b_refinement",
            (
                ("a1", "A", ("x",)),
                ("a2", "A", ("a1",)),
                ("b1", "B", ("a2", "r")),
                ("b2", "B", ("b1", "r")),
                ("out", "A", ("b2", "a2")),
            ),
        ),
        TopologySpec(
            "t10",
            "parallel_deep_weave",
            (
                ("a1", "A", ("x",)),
                ("b1", "B", ("r",)),
                ("a2", "A", ("a1", "b1")),
                ("b2", "B", ("a2", "b1")),
                ("a3", "A", ("b2", "a2")),
                ("out", "A", ("a3",)),
            ),
        ),
    )


def _profiles() -> tuple[ProfileSpec, ...]:
    rank2 = (1, 2)
    full = tuple(range(1, 7))
    return (
        ProfileSpec(
            "p01",
            rank2,
            "flat2",
            (1,),
            False,
            False,
            False,
            False,
            "small",
            0.003,
            0.0,
            256,
            500,
            0.3,
        ),
        ProfileSpec(
            "p02",
            rank2,
            "flat4",
            (1,),
            True,
            False,
            False,
            False,
            "small",
            0.003,
            0.00001,
            256,
            500,
            0.3,
        ),
        ProfileSpec(
            "p03",
            rank2,
            "grow24",
            (1, 2),
            False,
            False,
            False,
            False,
            "medium",
            0.002,
            0.00001,
            128,
            400,
            0.5,
        ),
        ProfileSpec(
            "p04",
            rank2,
            "shrink42",
            (1, 2),
            True,
            False,
            False,
            True,
            "medium",
            0.002,
            0.0001,
            128,
            400,
            0.5,
        ),
        ProfileSpec(
            "p05",
            rank2,
            "flat6",
            (1, 2, 3),
            True,
            False,
            False,
            True,
            "large",
            0.001,
            0.0002,
            128,
            300,
            0.5,
        ),
        ProfileSpec(
            "p06",
            full,
            "flat2",
            (1,),
            False,
            False,
            False,
            False,
            "small",
            0.002,
            0.00001,
            256,
            500,
            0.3,
        ),
        ProfileSpec(
            "p07",
            full,
            "flat3",
            (1,),
            True,
            False,
            False,
            True,
            "medium",
            0.0015,
            0.00005,
            128,
            400,
            0.5,
        ),
        ProfileSpec(
            "p08",
            full,
            "flat4",
            (1, 2),
            True,
            False,
            False,
            True,
            "medium",
            0.001,
            0.0001,
            128,
            300,
            0.5,
        ),
        ProfileSpec(
            "p09",
            full,
            "flat4",
            (1, 2),
            True,
            True,
            False,
            True,
            "large",
            0.00075,
            0.0002,
            128,
            300,
            0.5,
        ),
        ProfileSpec(
            "p10",
            full,
            "flat3",
            (1, 2),
            True,
            True,
            True,
            True,
            "large",
            0.0005,
            0.0005,
            64,
            300,
            0.5,
        ),
    )


def _radial(name: str) -> RadialConfig:
    values = {
        "small": RadialConfig(
            distance_scale=10.0,
            rbf_centers=(0.4, 0.6, 0.8, 1.0, 1.2),
            rbf_width=0.25,
            inverse_powers=(1, 2),
        ),
        "medium": RadialConfig(
            distance_scale=10.0,
            rbf_centers=(0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35),
            rbf_width=0.18,
            inverse_powers=(1, 2, 3),
        ),
        "large": RadialConfig(
            distance_scale=10.0,
            rbf_centers=(
                0.25,
                0.375,
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
                1.125,
                1.25,
                1.375,
                1.5,
            ),
            rbf_width=0.14,
            inverse_powers=(1, 2, 3, 4),
        ),
    }
    return values[name]


def _channel_schedule(policy: str, count: int) -> tuple[int, ...]:
    constant = {"flat2": 2, "flat3": 3, "flat4": 4, "flat6": 6}
    if policy in constant:
        return (constant[policy],) * count
    middle = math.ceil(count / 2)
    if policy == "grow24":
        return (2,) * middle + (4,) * (count - middle)
    if policy == "shrink42":
        return (4,) * middle + (2,) * (count - middle)
    raise ValueError(f"unknown channel policy {policy}")


def build_trial_specs() -> tuple[TrialSpec, ...]:
    """Return exactly one hundred unique candidate specifications."""
    mlp = MLPConfig(hidden_widths=(GATE_MLP_WIDTH,))
    trials = []
    functional_hashes = set()
    for topology in _topologies():
        for profile in _profiles():
            nonoutput_count = len(topology.stages) - 1
            channels = _channel_schedule(profile.channel_policy, nonoutput_count)
            channel_index = 0
            stages = []
            for name, stream, inputs in topology.stages:
                output = name == "out"
                stage_channels = 1 if output else channels[channel_index]
                channel_index += 0 if output else 1
                stages.append(
                    StageConfig(
                        name=name,
                        output_stream=stream,
                        inputs=inputs,
                        channels=stage_channels,
                        lift_orders=profile.lift_orders,
                        mix_inputs=profile.mix_inputs,
                        mix_components=profile.mix_components,
                        mix_channels=profile.mix_channels,
                        cross_grams=profile.cross_grams,
                        mlp=mlp,
                    )
                )
            architecture_id = f"pair_hp_{topology.code}_{profile.code}"
            pipeline = PairPipelineConfig(
                stages=tuple(stages),
                output_stage="out",
                architecture_id=architecture_id,
                anchor_ranks=profile.anchor_ranks,
                max_constraint_entries=10_000_000,
                max_gate_coefficients=MAX_GATE_COEFFICIENTS,
                max_invariant_channels=MAX_INVARIANT_CHANNELS,
                radial=_radial(profile.radial_name),
            )
            trial = TrialSpec(
                topology_code=topology.code,
                topology_name=topology.name,
                profile_code=profile.code,
                pipeline=pipeline,
                learning_rate=profile.learning_rate,
                weight_decay=profile.weight_decay,
                batch_size=profile.batch_size,
                scheduler_step_size=profile.scheduler_step_size,
                scheduler_gamma=profile.scheduler_gamma,
            )
            functional_hash = _canonical_sha256(trial.functional_dict())
            if functional_hash in functional_hashes:
                raise RuntimeError("catalog contains a duplicate functional design")
            functional_hashes.add(functional_hash)
            trials.append(trial)
    if len(trials) != 100 or len(functional_hashes) != 100:
        raise RuntimeError("catalog must contain exactly one hundred unique designs")
    return tuple(trials)


def _benzene_generators() -> torch.Tensor:
    cosine = math.cos(math.pi / 3.0)
    sine = math.sin(math.pi / 3.0)
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
        dtype=torch.float64,
    )


def _preflight_trial(
    trial: TrialSpec,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if progress is not None:
        progress(trial.candidate_id)
    model = build_invariant_gate_pipeline(
        _benzene_generators(),
        trial.pipeline,
        generator_names=("sixfold", "twofold"),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    gates = tuple(gate for _name, gate in model.named_gates())
    record = {
        "parameter_count": parameter_count,
        "gate_count": len(gates),
        "basis_dimension_sum": sum(gate.basis_dimension for gate in gates),
        "maximum_gate_basis_dimension": max(gate.basis_dimension for gate in gates),
        "maximum_gate_coefficient_count": max(gate.coefficient_count for gate in gates),
        "sum_gate_coefficient_count": sum(gate.coefficient_count for gate in gates),
        "maximum_invariant_channels": max(gate.invariant_channels for gate in gates),
        "fixed_buffer_bytes_float64": buffer_bytes,
        "estimated_two_model_files_bytes_float32": parameter_count * 8,
        "primitive_b_ranks": list(model.b_ranks),
        "stage_count": len(trial.pipeline.stages),
        "maximum_lift_order": max(
            max(stage.lift_orders) for stage in trial.pipeline.stages
        ),
        "active_mix_flag_count": sum(
            stage.mix_inputs + stage.mix_components + stage.mix_channels
            for stage in trial.pipeline.stages
        ),
    }
    failures = []
    if parameter_count > MAX_PARAMETER_COUNT:
        failures.append("parameter_count")
    if len(gates) > MAX_GATE_COUNT:
        failures.append("gate_count")
    if record["maximum_invariant_channels"] > MAX_INVARIANT_CHANNELS:
        failures.append("maximum_invariant_channels")
    if record["maximum_gate_coefficient_count"] > MAX_GATE_COEFFICIENTS:
        failures.append("maximum_gate_coefficient_count")
    if any(stage.channels > 6 for stage in trial.pipeline.stages):
        failures.append("stage_channels")
    if any(len(stage.inputs) > 2 for stage in trial.pipeline.stages):
        failures.append("stage_inputs")
    if any(
        stage.mlp.hidden_widths != (GATE_MLP_WIDTH,) for stage in trial.pipeline.stages
    ):
        failures.append("gate_mlp_width")
    if failures:
        raise RuntimeError(
            f"preflight budget failed for {trial.candidate_id}: {tuple(failures)}"
        )
    return record


def prepare_catalog(
    output_path: str | Path,
    *,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Compile every candidate and write the ordered validated manifest."""
    candidates = build_trial_specs()
    records = []
    for trial in candidates:
        design = trial.as_dict()
        config_hash = _canonical_sha256(design)
        functional_hash = _canonical_sha256(trial.functional_dict())
        preflight = _preflight_trial(trial, progress)
        records.append(
            {
                **design,
                "config_hash": config_hash,
                "functional_hash": functional_hash,
                "preflight": preflight,
            }
        )

    def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        check = record["preflight"]
        return (
            check["parameter_count"],
            check["gate_count"],
            check["basis_dimension_sum"],
            check["stage_count"],
            len(check["primitive_b_ranks"]),
            check["maximum_lift_order"],
            check["active_mix_flag_count"],
            record["topology_code"],
            record["profile_code"],
            record["candidate_id"],
        )

    records.sort(key=sort_key)
    for index, record in enumerate(records, start=1):
        record["trial_id"] = f"trial_{index:03d}"
    manifest = {
        "schema_name": "tfenn_benzene_pair_hyper_catalog",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_count": len(records),
        "gate_mlp_hidden_width": GATE_MLP_WIDTH,
        "ordering": [
            "parameter_count",
            "gate_count",
            "basis_dimension_sum",
            "stage_count",
            "primitive_b_rank_count",
            "maximum_lift_order",
            "active_mix_flag_count",
            "topology_code",
            "profile_code",
        ],
        "budgets": {
            "maximum_parameter_count": MAX_PARAMETER_COUNT,
            "maximum_gate_count": MAX_GATE_COUNT,
            "maximum_invariant_channels": MAX_INVARIANT_CHANNELS,
            "maximum_gate_coefficients": MAX_GATE_COEFFICIENTS,
            "maximum_stage_channels": 6,
            "maximum_stage_inputs": 2,
        },
        "designs": records,
    }
    manifest["catalog_sha256"] = _canonical_sha256(manifest["designs"])
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    partial.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    partial.replace(target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    print(prepare_catalog(arguments.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
