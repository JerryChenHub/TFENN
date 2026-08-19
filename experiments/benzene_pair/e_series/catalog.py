"""Define the five E series experiment catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


__all__ = [
    "EModelSpec",
    "E_SERIES_SPECS",
    "E0_SPECS",
    "E1_SPECS",
    "E2_SPECS",
    "E3_SPECS",
    "E4_SPECS",
    "get_experiment_specs",
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
class EModelSpec:
    """Store one executable E series model definition."""

    model_id: str
    experiment_id: int
    family: str
    architecture_name: str
    description: str
    purpose: str
    planned_parameter_count: int | None = None
    target_parameter_range: tuple[int, int] | None = None
    comparison_role: str = "candidate"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.experiment_id not in range(5):
            raise ValueError("experiment id must be zero through four")
        if not self.model_id.startswith(f"E{self.experiment_id}"):
            raise ValueError("model id does not match its experiment")
        if (
            self.planned_parameter_count is not None
            and self.planned_parameter_count < 1
        ):
            raise ValueError("planned parameter count must be positive")
        if self.target_parameter_range is not None:
            lower, upper = self.target_parameter_range
            if lower < 1 or upper < lower:
                raise ValueError("target parameter range is invalid")
        object.__setattr__(self, "options", _freeze(self.options))

    @property
    def expected_parameter_count(self) -> None:
        """Let the compiled count override every planning estimate."""
        return None

    @property
    def d6_covariance_exempt(self) -> bool:
        """Identify ordinary MLP controls that have no D6 requirement."""
        return bool(self.options.get("d6_covariance_exempt", False))

    def as_dict(self) -> dict[str, Any]:
        value = {
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "family": self.family,
            "architecture_name": self.architecture_name,
            "description": self.description,
            "purpose": self.purpose,
            "planned_parameter_count": self.planned_parameter_count,
            "target_parameter_range": None
            if self.target_parameter_range is None
            else list(self.target_parameter_range),
            "comparison_role": self.comparison_role,
            "d6_covariance_exempt": self.d6_covariance_exempt,
            "options": _thaw(self.options),
        }
        json.dumps(value, allow_nan=False)
        return value


def _stage(
    name: str,
    stream: str,
    channels: int,
    width: int,
    sources: Sequence[str],
    invariant_sources: Sequence[str],
    *,
    skip_policy: str,
    execution_level: int | None = None,
    covariant_live_mixed_only: bool = False,
    path_head_quota: int | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "output_stream": stream,
        "channels": int(channels),
        "trunk_width": int(width),
        "source_names": tuple(sources),
        "invariant_source_names": tuple(invariant_sources),
        "skip_policy": skip_policy,
        "execution_level": execution_level,
        "covariant_live_mixed_only": bool(covariant_live_mixed_only),
        "path_head_quota": path_head_quota,
    }


def _e0() -> tuple[EModelSpec, ...]:
    rows = (
        (
            "E001",
            "raw_mlp",
            "MLP 20k",
            (96, 96, 96),
            20_160,
            "Ordinary noncovariant MLP control",
        ),
        (
            "E002",
            "raw_mlp",
            "MLP 50k",
            (154, 154, 154),
            50_204,
            "Ordinary noncovariant MLP capacity control",
        ),
        (
            "E003",
            "group_mlp",
            "GroupMLP 20k",
            (96, 96, 96),
            20_160,
            "Model level D6 by D6 Reynolds averaged control",
        ),
        (
            "E004",
            "group_mlp",
            "GroupMLP 50k",
            (154, 154, 154),
            50_204,
            "GroupMLP capacity control",
        ),
    )
    result = [
        EModelSpec(
            model_id=model_id,
            experiment_id=0,
            family=family,
            architecture_name=name,
            description=f"hidden widths {widths}",
            purpose=purpose,
            planned_parameter_count=count,
            comparison_role="control",
            options={
                "hidden_widths": widths,
                "seed": 20260822,
                "d6_covariance_exempt": family == "raw_mlp",
            },
        )
        for model_id, family, name, widths, count, purpose in rows
    ]
    for model_id, reference, count, purpose in (
        ("E005", "C17", 19_939, "Main legacy reference"),
        ("E006", "C20", 20_352, "Deeper alternating reference"),
        ("E007", "D44", 9_911, "Sparse path compact reference"),
        ("E008", "C30", 240_746, "Full capacity upper reference"),
    ):
        result.append(
            EModelSpec(
                model_id=model_id,
                experiment_id=0,
                family="reference",
                architecture_name=reference,
                description=f"exact {reference} reference",
                purpose=purpose,
                planned_parameter_count=count,
                comparison_role="upper_control" if model_id == "E008" else "control",
                options={"reference_model_id": reference},
            )
        )
    return tuple(result)


_SCHEDULES = {
    "A": frozenset({"a1", "b1", "b2", "out"}),
    "E": frozenset({"a1", "out"}),
    "F": frozenset({"a1"}),
}


def _e1_stages(
    cx: str,
    cr: str,
    ix: str,
    ir: str,
    bypass: str,
    widths: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    names = ("a1", "b1", "b2", "out")
    streams = ("A", "B", "B", "A")
    channels = (1, 2, 1, 1)
    previous: list[str] = []
    result = []
    for name, stream, channel, width in zip(names, streams, channels, widths):
        covariant_raw = tuple(
            raw
            for raw, schedule in (("x", cx), ("r", cr))
            if name in _SCHEDULES[schedule]
        )
        invariant_raw = tuple(
            raw
            for raw, schedule in (("x", ix), ("r", ir))
            if name in _SCHEDULES[schedule]
        )
        result.append(
            _stage(
                name,
                stream,
                channel,
                width,
                (*covariant_raw, *previous),
                (*invariant_raw, *previous),
                skip_policy="legacy" if bypass == "L" else "none",
            )
        )
        previous.append(name)
    return tuple(result)


def _e1() -> tuple[EModelSpec, ...]:
    rows = (
        (
            "E101",
            "A",
            "A",
            "A",
            "A",
            "L",
            (4, 4, 4, 4),
            19_939,
            "Exact C17 and D24 control",
        ),
        (
            "E102",
            "A",
            "A",
            "A",
            "A",
            "N",
            (4, 4, 4, 4),
            19_926,
            "Remove only direct legacy bypass",
        ),
        (
            "E103",
            "E",
            "A",
            "A",
            "A",
            "L",
            (5, 4, 5, 5),
            20_101,
            "Remove middle raw x covariant access",
        ),
        (
            "E104",
            "E",
            "A",
            "A",
            "A",
            "N",
            (5, 4, 5, 5),
            20_088,
            "E103 without direct bypass",
        ),
        (
            "E105",
            "F",
            "A",
            "A",
            "A",
            "L",
            (5, 5, 5, 4),
            19_918,
            "Also remove raw x from output",
        ),
        (
            "E106",
            "F",
            "A",
            "A",
            "A",
            "N",
            (5, 5, 5, 4),
            19_906,
            "E105 without direct bypass",
        ),
        (
            "E107",
            "A",
            "E",
            "A",
            "A",
            "L",
            (13, 11, 11, 11),
            19_922,
            "Remove middle raw pose covariant access",
        ),
        (
            "E108",
            "A",
            "E",
            "A",
            "A",
            "N",
            (13, 11, 11, 11),
            19_915,
            "E107 without direct bypass",
        ),
        (
            "E109",
            "A",
            "F",
            "A",
            "A",
            "L",
            (14, 14, 14, 14),
            19_981,
            "Also remove raw pose from output",
        ),
        (
            "E110",
            "A",
            "F",
            "A",
            "A",
            "N",
            (14, 14, 14, 14),
            19_974,
            "E109 without direct bypass",
        ),
        (
            "E111",
            "E",
            "E",
            "A",
            "A",
            "L",
            (13, 19, 13, 13),
            19_934,
            "Raw covariant paths only at first and output",
        ),
        (
            "E112",
            "E",
            "E",
            "A",
            "A",
            "N",
            (13, 19, 13, 13),
            19_927,
            "E111 without direct bypass",
        ),
        (
            "E113",
            "E",
            "F",
            "A",
            "A",
            "L",
            (22, 17, 17, 17),
            19_932,
            "Keep output x but not output pose",
        ),
        (
            "E114",
            "E",
            "F",
            "A",
            "A",
            "N",
            (22, 17, 17, 17),
            19_925,
            "E113 without direct bypass",
        ),
        (
            "E115",
            "F",
            "E",
            "A",
            "A",
            "L",
            (14, 14, 14, 14),
            19_759,
            "Keep output pose but not output x",
        ),
        (
            "E116",
            "F",
            "E",
            "A",
            "A",
            "N",
            (14, 14, 14, 14),
            19_753,
            "E115 without direct bypass",
        ),
        (
            "E117",
            "F",
            "F",
            "A",
            "A",
            "L",
            (18, 24, 18, 18),
            19_971,
            "No repeated raw covariant access with full Gate context",
        ),
        (
            "E118",
            "F",
            "F",
            "A",
            "A",
            "N",
            (18, 24, 18, 18),
            19_965,
            "No repeated raw access and no direct bypass",
        ),
        (
            "E119",
            "E",
            "E",
            "E",
            "E",
            "L",
            (15, 15, 15, 15),
            20_106,
            "Reduce Gate raw context to endpoints",
        ),
        (
            "E120",
            "E",
            "E",
            "E",
            "E",
            "N",
            (15, 15, 15, 15),
            20_099,
            "E119 without direct bypass",
        ),
        (
            "E121",
            "F",
            "F",
            "F",
            "F",
            "L",
            (25, 30, 25, 25),
            19_920,
            "Gate raw context only at first stage",
        ),
        (
            "E122",
            "F",
            "F",
            "F",
            "F",
            "N",
            (25, 31, 25, 25),
            19_959,
            "E121 without direct bypass",
        ),
        (
            "E123",
            "A",
            "A",
            "F",
            "A",
            "L",
            (6, 4, 4, 4),
            19_949,
            "Remove repeated raw x Gate context only",
        ),
        (
            "E124",
            "A",
            "A",
            "A",
            "F",
            "L",
            (4, 5, 4, 4),
            19_964,
            "Remove repeated raw pose Gate context only",
        ),
        (
            "E125",
            "A",
            "A",
            "F",
            "F",
            "L",
            (15, 4, 4, 4),
            19_958,
            "Remove both repeated raw Gate contexts",
        ),
    )
    return tuple(
        EModelSpec(
            model_id=model_id,
            experiment_id=1,
            family="sequential_gate",
            architecture_name="C17 raw visibility and bypass cell",
            description=(
                f"Cx={cx} Cr={cr} Ix={ix} Ir={ir} bypass={bypass} W={tuple(widths)}"
            ),
            purpose=purpose,
            planned_parameter_count=count,
            target_parameter_range=(19_740, 20_138),
            options={
                "raw_schedules": {"Cx": cx, "Cr": cr, "Ix": ix, "Ir": ir},
                "bypass": bypass,
                "path_policy": "FULL",
                "stages": _e1_stages(cx, cr, ix, ir, bypass, widths),
            },
        )
        for model_id, cx, cr, ix, ir, bypass, widths, count, purpose in rows
    )


_E2_PROFILES = {
    "P1": ((1, 1), (1, 1), (1, 1)),
    "P2": ((1, 2), (1, 2), (1, 2)),
    "P3": ((2, 1), (2, 1), (2, 1)),
    "P4": ((1, 2), (1, 1), (1, 1)),
    "P5": ((3, 1), (2, 1), (1, 1)),
}


_E2_POLICY_PURPOSES = {
    "X0": "Scalar context coupling without an explicit cross covariant message",
    "X1": "A to B message in every block",
    "X2": "B to A message in every block",
    "X3": "Bidirectional exchange only in the middle block",
    "X4": "Bidirectional exchange in every block",
}


_E2_CAPACITY = {
    "P1": (4, 8),
    "P2": (5, 2),
    "P3": (6, 2),
    "P4": (5, 2),
    "P5": (6, 2),
}


def _exchange(policy: str, block: int) -> tuple[bool, bool]:
    if policy == "X1":
        return False, True
    if policy == "X2":
        return True, False
    if policy == "X3":
        return (True, True) if block == 2 else (False, False)
    if policy == "X4":
        return True, True
    return False, False


def _e2_stages(
    schedule: Sequence[tuple[int, int]],
    policy: str,
    *,
    width: int,
    path_head_quota: int,
) -> tuple[dict[str, Any], ...]:
    stages = [
        _stage(
            "a0",
            "A",
            1,
            width,
            ("x",),
            ("x", "r"),
            skip_policy="legacy",
            execution_level=0,
        ),
        _stage(
            "b0",
            "B",
            1,
            width,
            ("r",),
            ("x", "r"),
            skip_policy="legacy",
            execution_level=0,
        ),
    ]
    history = ["a0", "b0"]
    prior_a, prior_b = "a0", "b0"
    for block, (a_channels, b_channels) in enumerate(schedule, start=1):
        b_to_a, a_to_b = _exchange(policy, block)
        invariant_sources = ("x", "r", *history)
        a_live = (prior_a, prior_b) if b_to_a else (prior_a,)
        b_live = (prior_b, prior_a) if a_to_b else (prior_b,)
        stages.extend(
            (
                _stage(
                    f"a{block}",
                    "A",
                    a_channels,
                    width,
                    ("x", "r", *a_live),
                    invariant_sources,
                    skip_policy="legacy",
                    execution_level=block,
                    covariant_live_mixed_only=True,
                    path_head_quota=path_head_quota,
                ),
                _stage(
                    f"b{block}",
                    "B",
                    b_channels,
                    width,
                    ("x", "r", *b_live),
                    invariant_sources,
                    skip_policy="legacy",
                    execution_level=block,
                    covariant_live_mixed_only=True,
                    path_head_quota=path_head_quota,
                ),
            )
        )
        prior_a, prior_b = f"a{block}", f"b{block}"
        history.extend((prior_a, prior_b))
    stages.append(
        _stage(
            "out",
            "A",
            1,
            width,
            ("x", "r", prior_a, prior_b),
            ("x", "r", *history),
            skip_policy="legacy",
            execution_level=4,
            covariant_live_mixed_only=True,
            path_head_quota=path_head_quota,
        )
    )
    return tuple(stages)


def _e2() -> tuple[EModelSpec, ...]:
    profile_purposes = {
        "P1": "Balanced dual stream baseline",
        "P2": "Persistent extra B capacity",
        "P3": "Persistent extra A capacity",
        "P4": "C17 inspired B contraction",
        "P5": "C20 inspired A contraction",
    }
    result = []
    model_number = 201
    for profile, schedule in _E2_PROFILES.items():
        width, path_head_quota = _E2_CAPACITY[profile]
        for policy in _E2_POLICY_PURPOSES:
            model_id = f"E{model_number}"
            result.append(
                EModelSpec(
                    model_id=model_id,
                    experiment_id=2,
                    family="synchronous_dual_stream",
                    architecture_name=f"{profile} with {policy}",
                    description=f"schedule={schedule} policy={policy}",
                    purpose=f"{profile_purposes[profile]}; {_E2_POLICY_PURPOSES[policy]}",
                    target_parameter_range=(10_000, 20_000),
                    options={
                        "profile": profile,
                        "exchange_policy": policy,
                        "channel_schedule": schedule,
                        "trunk_width": width,
                        "path_head_quota": path_head_quota,
                        "target_parameter_center": 15_000,
                        "stages": _e2_stages(
                            schedule,
                            policy,
                            width=width,
                            path_head_quota=path_head_quota,
                        ),
                    },
                )
            )
            model_number += 1
    return tuple(result)


def _dense_stages(
    channels: Sequence[int],
    widths: Sequence[int],
    *,
    skip_policy: str,
) -> tuple[dict[str, Any], ...]:
    names = ("a1", "b1", "b2", "out")
    streams = ("A", "B", "B", "A")
    previous: list[str] = []
    result = []
    for name, stream, channel, width in zip(names, streams, channels, widths):
        sources = ("x", "r", *previous)
        result.append(
            _stage(
                name,
                stream,
                channel,
                width,
                sources,
                sources,
                skip_policy=skip_policy,
            )
        )
        previous.append(name)
    return tuple(result)


def _e3() -> tuple[EModelSpec, ...]:
    rows = (
        (
            "E301",
            "D44",
            "FULL",
            (4, 4, 4, 4),
            12_425,
            "FULL and sparse by W4 and W8 cell",
        ),
        (
            "E302",
            "D44",
            "FULL",
            (8, 8, 8, 8),
            22_541,
            "FULL and sparse by W4 and W8 cell",
        ),
        (
            "E303",
            "D44",
            "NO_RAW_MIXED",
            (4, 4, 4, 4),
            5_383,
            "FULL and sparse by W4 and W8 cell",
        ),
        ("E304", "D44", "NO_RAW_MIXED", (8, 8, 8, 8), 9_911, "Exact D44 control"),
        ("E305", "C17", "FULL", (2, 2, 2, 2), 11_869, "C17 Gate width curve"),
        ("E306", "C17", "FULL", (4, 4, 4, 4), 19_939, "Exact C17 control"),
        ("E307", "C17", "FULL", (8, 8, 8, 8), 36_079, "C17 Gate width curve"),
        (
            "E308",
            "C17",
            "FULL",
            (16, 16, 16, 16),
            68_359,
            "C17 Gate width upper control",
        ),
        ("E309", "C17", "NO_RAW_MIXED", (2, 2, 2, 2), 4_756, "Sparse bank width curve"),
        ("E310", "C17", "NO_RAW_MIXED", (4, 4, 4, 4), 8_146, "Sparse bank width curve"),
        (
            "E311",
            "C17",
            "NO_RAW_MIXED",
            (8, 8, 8, 8),
            14_926,
            "Sparse bank width curve",
        ),
        (
            "E312",
            "C17",
            "NO_RAW_MIXED",
            (16, 16, 16, 16),
            28_486,
            "Sparse bank width curve",
        ),
        (
            "E313",
            "D44",
            "FULL",
            (3, 3, 3, 3),
            9_896,
            "Approximately 10k FULL versus E304",
        ),
        ("E314", "D44", "FULL", (5, 5, 5, 5), 14_954, "Approximately 15k FULL"),
        (
            "E315",
            "D44",
            "NO_RAW_MIXED",
            (12, 12, 12, 12),
            14_439,
            "Approximately 15k sparse bank pair",
        ),
        ("E316", "D44", "FULL", (7, 7, 7, 7), 20_012, "Approximately 20k FULL"),
        (
            "E317",
            "D44",
            "NO_RAW_MIXED",
            (17, 17, 17, 17),
            20_099,
            "Approximately 20k sparse bank pair",
        ),
        (
            "E318",
            "C17",
            "NO_RAW_MIXED",
            (6, 6, 6, 6),
            11_536,
            "Approximately 12k versus E305",
        ),
        ("E319", "C17", "FULL", (3, 3, 3, 3), 15_904, "Approximately 16k FULL"),
        (
            "E320",
            "C17",
            "NO_RAW_MIXED",
            (9, 9, 9, 9),
            16_621,
            "Approximately 16k sparse bank pair",
        ),
        (
            "E321",
            "C17",
            "NO_RAW_MIXED",
            (11, 11, 11, 11),
            20_011,
            "Approximately 20k versus E306",
        ),
        (
            "E322",
            "C17",
            "FULL",
            (8, 4, 4, 4),
            20_359,
            "Add Gate capacity only at first A",
        ),
        (
            "E323",
            "C17",
            "FULL",
            (4, 8, 4, 4),
            24_207,
            "Add Gate capacity only at first B",
        ),
        (
            "E324",
            "C17",
            "FULL",
            (4, 4, 8, 4),
            28_775,
            "Add Gate capacity only at later B",
        ),
        (
            "E325",
            "C17",
            "FULL",
            (4, 4, 4, 8),
            22_555,
            "Add Gate capacity only at output",
        ),
    )
    return tuple(
        EModelSpec(
            model_id=model_id,
            experiment_id=3,
            family="sequential_gate",
            architecture_name=backbone,
            description=f"{backbone} paths={paths} W={widths}",
            purpose=purpose,
            planned_parameter_count=count,
            comparison_role="control" if model_id in {"E304", "E306"} else "candidate",
            options={
                "backbone": backbone,
                "path_policy": paths,
                "stages": _dense_stages(
                    (1, 1, 1, 1) if backbone == "D44" else (1, 2, 1, 1),
                    widths,
                    skip_policy="dense_proj" if backbone == "D44" else "legacy",
                ),
            },
        )
        for model_id, backbone, paths, widths, count, purpose in rows
    )


def _e4() -> tuple[EModelSpec, ...]:
    rows = (
        (
            "E401",
            "A",
            "U4 ResCP",
            "Four uniform residual blocks with CP rank one or two",
            "Auditable 8k baseline",
        ),
        (
            "E402",
            "A",
            "Funnel 5",
            "Width factors 1.30 1 .78 .60 .45",
            "Spend capacity before compression",
        ),
        (
            "E403",
            "A",
            "Diamond 5",
            "Width factors .55 1 1.35 1 .55",
            "Concentrated fusion",
        ),
        (
            "E404",
            "B",
            "Typed U 6",
            "Typewise encoder and decoder with a narrow waist",
            "Recover fine covariants with a narrow waist",
        ),
        (
            "E405",
            "B",
            "DenseGrow 4",
            "Append small channel groups and apply one final compressor",
            "Feature reuse",
        ),
        (
            "E406",
            "B",
            "RevCouple 8",
            "Alternating additive channel couplings",
            "Deep information preserving trunk",
        ),
        (
            "E407",
            "A",
            "TiedCell 16",
            "One compiler legal residual cell iterated with shared weights",
            "Maximum depth per parameter",
        ),
        (
            "E408",
            "A",
            "AB 2Cycle 12",
            "Tied A from B and B from A cells repeated six times",
            "Cheap bidirectional exchange",
        ),
        (
            "E409",
            "A",
            "Directional Ladder 6",
            "Alternate A to B intratype and B to A path bands",
            "Sparse staged interaction",
        ),
        (
            "E410",
            "A",
            "A Hub Star 5",
            "A hub with compiler certified B spokes",
            "Scale to many B types",
        ),
        (
            "E411",
            "A",
            "TwinTower 3 plus 2",
            "Separate A and B trunks with two late fusion stages",
            "Delay cross coupling",
        ),
        (
            "E412",
            "B",
            "Masked B Experts",
            "Small expert per manifest B type with shared A backbone",
            "Type specialization",
        ),
        (
            "E413",
            "B",
            "TypeGraph 3",
            "Types are nodes and legal intertwiners are edges",
            "Irregular manifest efficiency",
        ),
        (
            "E414",
            "B",
            "TreeFuse",
            "Coverage preserving manifest path tree",
            "Sparse logarithmic fusion",
        ),
        (
            "E415",
            "A",
            "LiftPyramid",
            "Frozen selected polynomial lifts with thin learned mixers",
            "Spend parameters on coefficients not lifts",
        ),
        (
            "E416",
            "A",
            "CP WidePaths 3",
            "Three broad fusion blocks with CP ranks",
            "Strong practical low rank candidate",
        ),
        (
            "E417",
            "B",
            "TuckerCore 3",
            "Tucker factorized high centrality channel paths",
            "Correlated channel interactions",
        ),
        (
            "E418",
            "B",
            "TT Path 4",
            "Tensor train channel factors with asymmetric widths",
            "Unequal multiplicity efficiency",
        ),
        (
            "E419",
            "B",
            "Context LoRA 5",
            "Rank two to four invariant conditioned adaptation",
            "Cheap invariant conditioned adaptation",
        ),
        (
            "E420",
            "B",
            "Global Dictionary 8",
            "Shared path experts with one context head and layer factors",
            "Share paths and coefficients across depth",
        ),
        (
            "E421",
            "B",
            "ToeplitzWide 6",
            "Toeplitz multiplicity mixers and rank one tensor paths",
            "Trade arbitrary mixers for width",
        ),
        (
            "E422",
            "B",
            "AxisCP Head 6",
            "Factor coefficients across layer type channel and path axes",
            "Compress coefficient heads",
        ),
        (
            "E423",
            "C",
            "Invariant Attention 4",
            "Invariant logits with compiler legal values",
            "Global channel interaction",
        ),
        (
            "E424",
            "C",
            "SoftPath MoE 6",
            "Invariant router over sparse covering path experts",
            "Input adaptive path use",
        ),
        (
            "E425",
            "C",
            "CayleyFlow 12",
            "Tied residual flow with Cayley multiplicity mixer",
            "Stable deep dynamics",
        ),
    )
    return tuple(
        EModelSpec(
            model_id=model_id,
            experiment_id=4,
            family="budget_compiled_gate",
            architecture_name=architecture,
            description=mechanism,
            purpose=purpose,
            target_parameter_range=(7_800, 8_200),
            options={
                "tier": tier,
                "compact_mechanism": mechanism,
                "full_invariant_context": True,
                "minimum_bridge_count": 1,
                "initial_factorization_rank": 1,
                "budget_compiler": "coverage_preserving_nearest_8k",
                "implemented_mechanism": architecture,
            },
        )
        for model_id, tier, architecture, mechanism, purpose in rows
    )


E0_SPECS = _e0()
E1_SPECS = _e1()
E2_SPECS = _e2()
E3_SPECS = _e3()
E4_SPECS = _e4()
E_SERIES_SPECS = (*E0_SPECS, *E1_SPECS, *E2_SPECS, *E3_SPECS, *E4_SPECS)
_EXPERIMENT_LOOKUP = {
    0: E0_SPECS,
    1: E1_SPECS,
    2: E2_SPECS,
    3: E3_SPECS,
    4: E4_SPECS,
}
_MODEL_LOOKUP = {spec.model_id: spec for spec in E_SERIES_SPECS}


def _validate_catalog() -> None:
    expected = (
        *(f"E{index:03d}" for index in range(1, 9)),
        *(f"E{group}{index:02d}" for group in range(1, 5) for index in range(1, 26)),
    )
    if tuple(spec.model_id for spec in E_SERIES_SPECS) != expected:
        raise RuntimeError("E series identifiers are incomplete or out of order")
    if len(E_SERIES_SPECS) != 108:
        raise RuntimeError("E series must contain one hundred eight models")
    if len(E0_SPECS) != 8 or any(
        len(_EXPERIMENT_LOOKUP[index]) != 25 for index in range(1, 5)
    ):
        raise RuntimeError("E series experiment sizes do not match the plan")


_validate_catalog()


def get_model_spec(model_id: str) -> EModelSpec:
    """Return one E model by its case insensitive identifier."""
    if not isinstance(model_id, str):
        raise TypeError("model id must be a string")
    try:
        return _MODEL_LOOKUP[model_id.upper()]
    except KeyError as error:
        raise KeyError(f"unknown E series model {model_id}") from error


def get_experiment_specs(experiment: int | str) -> tuple[EModelSpec, ...]:
    """Return one stable E subexperiment catalog."""
    key = str(experiment).strip().lower()
    aliases = {
        **{str(index): values for index, values in _EXPERIMENT_LOOKUP.items()},
        **{f"e{index}": values for index, values in _EXPERIMENT_LOOKUP.items()},
        **{
            f"experiment_{index}": values
            for index, values in _EXPERIMENT_LOOKUP.items()
        },
    }
    try:
        return aliases[key]
    except KeyError as error:
        raise KeyError(f"unknown E series experiment {experiment}") from error
