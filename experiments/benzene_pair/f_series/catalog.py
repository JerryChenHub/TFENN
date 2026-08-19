"""Define the F1/F2 science catalogs and independent execution shards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


__all__ = [
    "FModelSpec",
    "F_SERIES_SPECS",
    "F0_SPECS",
    "F1_SPECS",
    "F2_SPECS",
    "EXECUTION_SHARD_SPECS",
    "get_experiment_specs",
    "get_science_experiment_specs",
    "get_execution_shard_specs",
    "get_model_spec",
]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FModelSpec:
    """Store one executable F series model definition."""

    model_id: str
    experiment_id: int
    execution_shard_id: int
    family: str
    architecture_name: str
    description: str
    purpose: str
    planned_parameter_count: int
    comparison_role: str = "candidate"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.experiment_id not in range(3):
            raise ValueError("science experiment id must be zero through two")
        if self.execution_shard_id not in range(5):
            raise ValueError("execution shard id must be zero through four")
        if not self.model_id.startswith("F") or len(self.model_id) != 4:
            raise ValueError("model id must use the F three digit form")
        number = int(self.model_id[1:])
        expected_experiment = 0 if number == 100 else 1 if number < 200 else 2
        if self.experiment_id != expected_experiment:
            raise ValueError("model id and science experiment id disagree")
        if self.execution_shard_id != _execution_shard_id(number):
            raise ValueError("model id and execution shard id disagree")
        if self.planned_parameter_count < 1:
            raise ValueError("planned parameter count must be positive")
        object.__setattr__(self, "options", _freeze(self.options))

    @property
    def expected_parameter_count(self) -> int | None:
        """Keep only the exact historical control as a strict count."""
        return self.planned_parameter_count if self.family == "reference" else None

    @property
    def descriptor_mask(self) -> str:
        return str(self.options.get("descriptor_mask", "full"))

    @property
    def pair_model_id(self) -> str | None:
        value = self.options.get("pair_model_id")
        return None if value is None else str(value)

    @property
    def d6_covariance_exempt(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        value = {
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "execution_shard_id": self.execution_shard_id,
            "family": self.family,
            "architecture_name": self.architecture_name,
            "description": self.description,
            "purpose": self.purpose,
            "planned_parameter_count": self.planned_parameter_count,
            "comparison_role": self.comparison_role,
            "d6_covariance_exempt": False,
            "options": _thaw(self.options),
        }
        json.dumps(value, allow_nan=False)
        return value


def _stage(
    name: str,
    stream: str,
    sources: Sequence[str],
    channels: int,
    execution_level: int,
) -> dict[str, Any]:
    hidden = tuple(source for source in sources if source not in {"x", "r"})
    return {
        "name": name,
        "output_stream": stream,
        "source_names": tuple(sources),
        "invariant_source_names": ("x", "r", *hidden),
        "channels": int(channels),
        "execution_level": int(execution_level),
    }


def _topology_stages(
    topology: str,
    channels: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    if topology == "T1":
        a1, a2, a3, b1, b2 = channels
        return (
            _stage("a1", "A", ("x",), a1, 0),
            _stage("b1", "B", ("r",), b1, 0),
            _stage("a2", "A", ("a1", "b1"), a2, 1),
            _stage("b2", "B", ("b1", "a1"), b2, 1),
            _stage("a3", "A", ("a2", "b2"), a3, 2),
            _stage("out", "A", ("a3",), 1, 3),
        )
    if topology == "T2":
        a1, a2, a3, b1 = channels
        return (
            _stage("a1", "A", ("x",), a1, 0),
            _stage("b1", "B", ("r",), b1, 0),
            _stage("a2", "A", ("a1", "b1"), a2, 1),
            _stage("a3", "A", ("a2",), a3, 2),
            _stage("out", "A", ("a3",), 1, 3),
        )
    if topology == "T3":
        a1, a2, a3, b1, b2 = channels
        return (
            _stage("a1", "A", ("x",), a1, 0),
            _stage("b1", "B", ("r",), b1, 0),
            _stage("a2", "A", ("a1",), a2, 1),
            _stage("b2", "B", ("b1",), b2, 1),
            _stage("a3", "A", ("a2", "b2"), a3, 2),
            _stage("out", "A", ("a3",), 1, 3),
        )
    raise ValueError(f"unknown topology {topology}")


_T1_ROWS = (
    (101, (1, 1, 1, 2, 1), "baseline", 12_635),
    (102, (2, 1, 1, 2, 1), "A1=2", 12_969),
    (103, (4, 1, 1, 2, 1), "A1=4", 13_649),
    (104, (8, 1, 1, 2, 1), "A1=8", 15_057),
    (105, (1, 2, 1, 2, 1), "A2=2", 13_223),
    (106, (1, 4, 1, 2, 1), "A2=4", 14_435),
    (107, (1, 8, 1, 2, 1), "A2=8", 17_003),
    (108, (1, 16, 1, 2, 1), "A2=16", 22_715),
    (109, (1, 1, 2, 2, 1), "A3=2", 12_998),
    (110, (1, 1, 4, 2, 1), "A3=4", 13_760),
    (111, (1, 1, 8, 2, 1), "A3=8", 15_428),
    (112, (1, 1, 1, 1, 1), "B1=1", 7_932),
    (113, (1, 1, 1, 4, 1), "B1=4", 22_089),
    (114, (1, 1, 1, 8, 1), "B1=8", 41_189),
    (115, (1, 1, 1, 2, 2), "B2=2", 17_078),
    (116, (1, 1, 1, 2, 4), "B2=4", 26_036),
    (117, (1, 1, 1, 2, 8), "B2=8", 44_240),
)

_T2_ROWS = (
    (118, (1, 1, 1, 2), "baseline", 6_972),
    (119, (2, 1, 1, 2), "A1=2", 7_098),
    (120, (4, 1, 1, 2), "A1=4", 7_362),
    (121, (8, 1, 1, 2), "A1=8", 7_938),
    (122, (16, 1, 1, 2), "A1=16", 9_282),
    (123, (1, 2, 1, 2), "A2=2", 7_560),
    (124, (1, 4, 1, 2), "A2=4", 8_772),
    (125, (1, 8, 1, 2), "A2=8", 11_340),
    (126, (1, 16, 1, 2), "A2=16", 17_052),
    (127, (1, 1, 2, 2), "A3=2", 7_098),
    (128, (1, 1, 4, 2), "A3=4", 7_362),
    (129, (1, 1, 8, 2), "A3=8", 7_938),
    (130, (1, 1, 16, 2), "A3=16", 9_282),
    (131, (1, 1, 1, 1), "B1=1", 4_496),
    (132, (1, 1, 1, 4), "B1=4", 11_972),
    (133, (1, 1, 1, 8), "B1=8", 22_164),
)

_T3_ROWS = (
    (134, (1, 1, 1, 2, 1), "baseline", 11_137),
    (135, (2, 1, 1, 2, 1), "A1=2", 11_263),
    (136, (4, 1, 1, 2, 1), "A1=4", 11_527),
    (137, (8, 1, 1, 2, 1), "A1=8", 12_103),
    (138, (1, 2, 1, 2, 1), "A2=2", 11_263),
    (139, (1, 4, 1, 2, 1), "A2=4", 11_527),
    (140, (1, 8, 1, 2, 1), "A2=8", 12_103),
    (141, (1, 16, 1, 2, 1), "A2=16", 13_447),
    (142, (1, 1, 2, 2, 1), "A3=2", 11_500),
    (143, (1, 1, 4, 2, 1), "A3=4", 12_262),
    (144, (1, 1, 8, 2, 1), "A3=8", 13_930),
    (145, (1, 1, 1, 1, 1), "B1=1", 7_075),
    (146, (1, 1, 1, 4, 1), "B1=4", 19_309),
    (147, (1, 1, 1, 8, 1), "B1=8", 35_845),
    (148, (1, 1, 1, 2, 2), "B2=2", 15_424),
    (149, (1, 1, 1, 2, 4), "B2=4", 24_046),
    (150, (1, 1, 1, 2, 8), "B2=8", 41_482),
)


def _execution_shard_id(model_number: int) -> int:
    if model_number == 100:
        return 0
    if 101 <= model_number <= 125:
        return 1
    if 126 <= model_number <= 150:
        return 2
    if 201 <= model_number <= 225:
        return 3
    if 226 <= model_number <= 250:
        return 4
    raise ValueError(f"model number {model_number} is outside the F catalog")


def _strict_specs(
    topology: str,
    rows: Sequence[tuple[int, tuple[int, ...], str, int]],
) -> tuple[tuple[FModelSpec, ...], tuple[FModelSpec, ...]]:
    f1: list[FModelSpec] = []
    f2: list[FModelSpec] = []
    for number, channels, changed_node, count in rows:
        f1_id = f"F{number:03d}"
        f2_id = f"F{number + 100:03d}"
        stages = _topology_stages(topology, channels)
        for collection, model_id, experiment_id, policy, mask, pair_id in (
            (f1, f1_id, 1, "RAW_LOCAL_MIX", "full", f2_id),
            (f2, f2_id, 2, "RAW_ONLY_MASK", "raw_only", f1_id),
        ):
            collection.append(
                FModelSpec(
                    model_id=model_id,
                    experiment_id=experiment_id,
                    execution_shard_id=_execution_shard_id(int(model_id[1:])),
                    family="strict_flow",
                    architecture_name=f"{topology}_{policy}",
                    description=(
                        f"{topology} strict typed flow channels={tuple(channels)} "
                        f"Gate W8 {policy}"
                    ),
                    purpose=f"{topology} one node channel sweep {changed_node}",
                    planned_parameter_count=count,
                    comparison_role="baseline"
                    if changed_node == "baseline"
                    else "candidate",
                    options={
                        "topology": topology,
                        "channels": tuple(channels),
                        "changed_node": changed_node,
                        "invariant_policy": policy,
                        "descriptor_mask": mask,
                        "pair_model_id": pair_id,
                        "path_policy": "NO_RAW_MIXED",
                        "gate_width": 8,
                        "stages": stages,
                    },
                )
            )
    return tuple(f1), tuple(f2)


F0_SPECS = (
    FModelSpec(
        model_id="F100",
        experiment_id=0,
        execution_shard_id=0,
        family="reference",
        architecture_name="E311",
        description="Exact E311 historical control",
        purpose="Non strict historical performance and parameter control",
        planned_parameter_count=14_926,
        comparison_role="control",
        options={
            "reference_model_id": "E311",
            "topology": "historical_dense_access",
            "invariant_policy": "FULL",
            "descriptor_mask": "full",
            "path_policy": "NO_RAW_MIXED",
            "gate_width": 8,
        },
    ),
)
_T1_F1, _T1_F2 = _strict_specs("T1", _T1_ROWS)
_T2_F1, _T2_F2 = _strict_specs("T2", _T2_ROWS)
_T3_F1, _T3_F2 = _strict_specs("T3", _T3_ROWS)
F1_SPECS = tuple(
    sorted((*_T1_F1, *_T2_F1, *_T3_F1), key=lambda spec: spec.model_id)
)
F2_SPECS = tuple(
    sorted((*_T1_F2, *_T2_F2, *_T3_F2), key=lambda spec: spec.model_id)
)
F_SERIES_SPECS = tuple(
    sorted(
        (*F0_SPECS, *F1_SPECS, *F2_SPECS),
        key=lambda spec: spec.model_id,
    )
)

_BY_ID = MappingProxyType({spec.model_id: spec for spec in F_SERIES_SPECS})
_BY_EXPERIMENT = MappingProxyType({0: F0_SPECS, 1: F1_SPECS, 2: F2_SPECS})
EXECUTION_SHARD_SPECS = MappingProxyType(
    {
        shard: tuple(
            spec
            for spec in F_SERIES_SPECS
            if spec.execution_shard_id == shard
        )
        for shard in range(5)
    }
)


def get_model_spec(model_id: str) -> FModelSpec:
    try:
        return _BY_ID[str(model_id).upper()]
    except KeyError as error:
        raise KeyError(f"unknown F model {model_id}") from error


def get_science_experiment_specs(experiment: int | str) -> tuple[FModelSpec, ...]:
    value: int
    if isinstance(experiment, str):
        normalized = experiment.lower()
        for prefix in ("experiment_", "f"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        value = int(normalized)
    else:
        value = int(experiment)
    try:
        return _BY_EXPERIMENT[value]
    except KeyError as error:
        raise KeyError(f"unknown F experiment {experiment}") from error


def get_experiment_specs(experiment: int | str) -> tuple[FModelSpec, ...]:
    """Backward-compatible alias for the scientific F0/F1/F2 grouping."""
    return get_science_experiment_specs(experiment)


def get_execution_shard_specs(shard: int | str) -> tuple[FModelSpec, ...]:
    value: int
    if isinstance(shard, str):
        normalized = shard.lower()
        aliases = {
            "control": 0,
            "f1a": 1,
            "f1b": 2,
            "f2a": 3,
            "f2b": 4,
        }
        if normalized in aliases:
            value = aliases[normalized]
        else:
            for prefix in ("execution_shard_", "shard_"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    break
            value = int(normalized)
    else:
        value = int(shard)
    try:
        return EXECUTION_SHARD_SPECS[value]
    except KeyError as error:
        raise KeyError(f"unknown F execution shard {shard}") from error
