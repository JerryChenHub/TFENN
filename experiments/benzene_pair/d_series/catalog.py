"""Define the three D series model catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


__all__ = [
    "DModelSpec",
    "DStageSpec",
    "D_SERIES_SPECS",
    "EXPERIMENT_1_SPECS",
    "EXPERIMENT_2_SPECS",
    "EXPERIMENT_3_SPECS",
    "build_d_series_model",
    "get_experiment_specs",
    "get_model_spec",
]


_SOURCE_POLICIES = {"DENSE", "LOCAL", "CHAIN", "HISTORY", "RAW_PARALLEL", "CUSTOM"}
_SKIP_POLICIES = {"legacy", "none", "id", "local_proj", "dense_proj"}


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
class DStageSpec:
    """Store one fully resolved stage definition."""

    name: str
    output_stream: str
    channels: int
    trunk_width: int
    source_names: tuple[str, ...]
    invariant_source_names: tuple[str, ...]
    skip_source_names: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if self.output_stream not in {"A", "B"}:
            raise ValueError("stage stream must be A or B")
        if self.channels < 1 or self.trunk_width < 1:
            raise ValueError("stage channels and width must be positive")
        for names in (
            self.source_names,
            self.invariant_source_names,
            () if self.skip_source_names is None else self.skip_source_names,
        ):
            if len(set(names)) != len(names):
                raise ValueError("stage source names must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_stream": self.output_stream,
            "channels": self.channels,
            "trunk_width": self.trunk_width,
            "source_names": list(self.source_names),
            "invariant_source_names": list(self.invariant_source_names),
            "skip_source_names": None
            if self.skip_source_names is None
            else list(self.skip_source_names),
        }


@dataclass(frozen=True, slots=True)
class DModelSpec:
    """Store one D series architecture and its causal comparison purpose."""

    model_id: str
    experiment_id: int
    description: str
    purpose: str
    stages: tuple[DStageSpec, ...]
    source_policy: str
    invariant_source_policy: str
    skip_policy: str
    path_policy: str
    gate_policy: str
    expected_parameter_count: int | None = None
    comparison_role: str = "candidate"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_policy not in _SOURCE_POLICIES:
            raise ValueError("unknown covariant source policy")
        if self.invariant_source_policy not in _SOURCE_POLICIES:
            raise ValueError("unknown invariant source policy")
        if self.skip_policy not in _SKIP_POLICIES:
            raise ValueError("unknown skip policy")
        if not self.stages or self.stages[-1].name != "out":
            raise ValueError("every model must end in Aout")
        if self.stages[-1].output_stream != "A" or self.stages[-1].channels != 1:
            raise ValueError("Aout must have one A channel")
        if (
            self.expected_parameter_count is not None
            and self.expected_parameter_count < 1
        ):
            raise ValueError("expected parameter count must be positive")
        object.__setattr__(self, "options", _freeze(self.options))

    @property
    def hidden_streams(self) -> tuple[str, ...]:
        return tuple(stage.output_stream for stage in self.stages[:-1])

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(stage.channels for stage in self.stages[:-1])

    @property
    def trunk_widths(self) -> tuple[int, ...]:
        return tuple(stage.trunk_width for stage in self.stages)

    def as_dict(self) -> dict[str, Any]:
        value = {
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "purpose": self.purpose,
            "hidden_streams": list(self.hidden_streams),
            "channels": list(self.channels),
            "trunk_widths": list(self.trunk_widths),
            "source_policy": self.source_policy,
            "invariant_source_policy": self.invariant_source_policy,
            "skip_policy": self.skip_policy,
            "path_policy": self.path_policy,
            "gate_policy": self.gate_policy,
            "expected_parameter_count": self.expected_parameter_count,
            "comparison_role": self.comparison_role,
            "stages": [stage.as_dict() for stage in self.stages],
            "options": _thaw(self.options),
        }
        json.dumps(value, allow_nan=False)
        return value


def _stage_names(route: Sequence[tuple[str, int]]) -> tuple[str, ...]:
    counts = {"A": 0, "B": 0}
    names = []
    for stream, _ in route:
        counts[stream] += 1
        names.append(f"{stream.lower()}{counts[stream]}")
    return (*names, "out")


def _policy_sources(
    policy: str, index: int, hidden_names: Sequence[str]
) -> tuple[str, ...]:
    previous = tuple(hidden_names[:index])
    if policy == "DENSE":
        return ("x", "r", *previous)
    if policy == "LOCAL":
        return ("x", "r", *previous[-1:])
    if policy == "CHAIN":
        return ("x", "r") if index == 0 else previous[-1:]
    if policy == "HISTORY":
        return ("x", "r") if index == 0 else previous
    if policy == "RAW_PARALLEL":
        return ("x", "r", *previous) if index == len(hidden_names) else ("x", "r")
    raise ValueError("custom source policies require explicit stage sources")


def _matching_skip_sources(
    stream: str,
    source_names: Sequence[str],
    stream_by_name: Mapping[str, str],
    skip_policy: str,
) -> tuple[str, ...] | None:
    matching = tuple(
        name for name in source_names if stream_by_name.get(name) == stream
    )
    if skip_policy == "legacy":
        return None
    if skip_policy == "none":
        return ()
    if skip_policy in {"id", "local_proj"}:
        return matching[-1:]
    return matching


def _common_options(
    *,
    skip_policy: str,
    path_policy: str,
    degree3_policy: str = "none",
    **extra: Any,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "skip_policy": skip_policy,
        "covariant_include_symmetric_unary": path_policy != "NO_SYM2",
        "covariant_include_raw_mixed_pairs": path_policy != "NO_RAW_MIXED",
        "covariant_include_stf_shortcuts": path_policy != "NO_STF",
        "invariant_include_symmetric_unary": True,
        "invariant_include_raw_mixed_pairs": True,
        "invariant_include_stf_shortcuts": True,
        "degree3_policy": degree3_policy,
        "coefficient_activation": "identity",
        "coefficient_head": "dense",
        "coefficient_rank": None,
        "descriptor_mask": "full",
        "trunk_depth": 1,
        "trunk_linearized": False,
        "trunk_residual": False,
        "metric_gate": "none",
        "degree3_overflow_policy": "raise",
        "max_constraint_entries": 10_000_000,
    }
    values.update(extra)
    return values


def _make_spec(
    model_id: str,
    experiment_id: int,
    route: Sequence[tuple[str, int]],
    widths: Sequence[int],
    purpose: str,
    *,
    source_policy: str = "DENSE",
    invariant_source_policy: str | None = None,
    skip_policy: str = "dense_proj",
    path_policy: str = "FULL_P2",
    gate_policy: str = "CURRENT_DENSE_LINEAR",
    expected_parameter_count: int | None = None,
    comparison_role: str = "candidate",
    explicit_sources: Mapping[str, Sequence[str]] | None = None,
    explicit_invariant_sources: Mapping[str, Sequence[str]] | None = None,
    degree3_policy: str = "none",
    options: Mapping[str, Any] | None = None,
) -> DModelSpec:
    route = tuple((str(stream), int(channels)) for stream, channels in route)
    widths = tuple(int(width) for width in widths)
    if len(widths) != len(route) + 1:
        raise ValueError("widths must include every hidden stage and Aout")
    invariant_policy = (
        source_policy if invariant_source_policy is None else invariant_source_policy
    )
    names = _stage_names(route)
    hidden_names = names[:-1]
    streams = (*tuple(stream for stream, _ in route), "A")
    channel_values = (*tuple(channels for _, channels in route), 1)
    stream_by_name = {"x": "A", "r": "B"}
    stream_by_name.update(zip(hidden_names, streams[:-1]))
    resolved = []
    for index, (name, stream, channels, width) in enumerate(
        zip(names, streams, channel_values, widths)
    ):
        if explicit_sources is None:
            sources = _policy_sources(source_policy, index, hidden_names)
        else:
            sources = tuple(explicit_sources[name])
        if explicit_invariant_sources is None:
            invariant_sources = _policy_sources(invariant_policy, index, hidden_names)
        else:
            invariant_sources = tuple(explicit_invariant_sources[name])
        resolved.append(
            DStageSpec(
                name=name,
                output_stream=stream,
                channels=channels,
                trunk_width=width,
                source_names=tuple(sources),
                invariant_source_names=tuple(invariant_sources),
                skip_source_names=_matching_skip_sources(
                    stream, sources, stream_by_name, skip_policy
                ),
            )
        )
    route_text = " → ".join(
        [*(f"{stream}[{channels}]" for stream, channels in route), "Aout"]
    )
    width_text = ",".join(str(width) for width in widths)
    merged_options = _common_options(
        skip_policy=skip_policy,
        path_policy=path_policy,
        degree3_policy=degree3_policy,
    )
    if model_id == "D50":
        merged_options["degree3_overflow_policy"] = "audit_skip"
    merged_options.update(dict(options or {}))
    return DModelSpec(
        model_id=model_id,
        experiment_id=experiment_id,
        description=f"{route_text}; W=({width_text})",
        purpose=purpose,
        stages=tuple(resolved),
        source_policy=source_policy,
        invariant_source_policy=invariant_policy,
        skip_policy=skip_policy,
        path_policy=path_policy,
        gate_policy=gate_policy,
        expected_parameter_count=expected_parameter_count,
        comparison_role=comparison_role,
        options=merged_options,
    )


def _custom_t20_sources(model_id: str) -> dict[str, tuple[str, ...]]:
    dense = {
        "a1": ("x", "r"),
        "b1": ("x", "r", "a1"),
        "a2": ("x", "r", "a1", "b1"),
        "b2": ("x", "r", "a1", "b1", "a2"),
        "out": ("x", "r", "a1", "b1", "a2", "b2"),
    }
    if model_id == "D17":
        return {
            name: tuple(item for item in values if item != "x")
            if name != "a1"
            else values
            for name, values in dense.items()
        }
    if model_id == "D18":
        dense["out"] = tuple(item for item in dense["out"] if item != "x")
    elif model_id == "D19":
        for name in ("a2", "b2", "out"):
            dense[name] = tuple(item for item in dense[name] if item != "r")
    elif model_id == "D20":
        dense["out"] = tuple(item for item in dense["out"] if item != "r")
    elif model_id == "D21":
        for name in ("a2", "b2", "out"):
            dense[name] = tuple(item for item in dense[name] if item not in {"x", "r"})
    elif model_id == "D22":
        dense["out"] = ("a1", "b1", "a2", "b2")
    elif model_id == "D23":
        dense["out"] = ("x", "r", "b2")
    else:
        raise ValueError("unknown custom T20 source experiment")
    return dense


def _experiment_1() -> tuple[DModelSpec, ...]:
    t17 = (("A", 1), ("B", 1), ("B", 1))
    t20 = (("A", 1), ("B", 1), ("A", 1), ("B", 1))
    rows = []
    purposes_17 = (
        ("D01", "DENSE", "Dense covariant baseline without explicit residual"),
        ("D02", "LOCAL", "Test whether all history covariant visibility is needed"),
        ("D03", "CHAIN", "Test whether repeated raw injection is needed"),
        ("D04", "HISTORY", "Separate history retention from raw retention"),
        (
            "D05",
            "RAW_PARALLEL",
            "Compare iterative propagation with parallel raw branches",
        ),
    )
    purposes_20 = (
        ("D06", "DENSE", "C20 like dense covariant baseline"),
        ("D07", "LOCAL", "Test local history in an alternating topology"),
        ("D08", "CHAIN", "Strict ABAB chain"),
        ("D09", "HISTORY", "History only ABAB propagation"),
        ("D10", "RAW_PARALLEL", "Parallel raw branches with final fusion"),
    )
    for model_id, policy, purpose in purposes_17:
        rows.append(
            _make_spec(
                model_id,
                1,
                t17,
                (5, 5, 5, 5),
                purpose,
                source_policy=policy,
                skip_policy="none",
            )
        )
    for model_id, policy, purpose in purposes_20:
        rows.append(
            _make_spec(
                model_id,
                1,
                t20,
                (4, 4, 4, 4, 4),
                purpose,
                source_policy=policy,
                skip_policy="none",
            )
        )
    for model_id, route, widths, skip, purpose in (
        ("D11", t17, (5, 5, 5, 5), "id", "Test parameter free typed residual"),
        (
            "D12",
            t17,
            (5, 5, 5, 5),
            "local_proj",
            "Compare identity and learned local residual",
        ),
        ("D13", t17, (5, 5, 5, 5), "dense_proj", "Test learned dense typed residual"),
        ("D14", t20, (4, 4, 4, 4, 4), "id", "Repeat identity residual test on T20s"),
        ("D15", t20, (4, 4, 4, 4, 4), "local_proj", "Local projected residual on T20s"),
        (
            "D16",
            t20,
            (4, 4, 4, 4, 4),
            "dense_proj",
            "Current style dense projected residual",
        ),
    ):
        rows.append(_make_spec(model_id, 1, route, widths, purpose, skip_policy=skip))
    custom_purposes = {
        "D17": "Test continuous displacement retention",
        "D18": "Test final raw x access",
        "D19": "Test whether hidden B preserves pose",
        "D20": "Test final raw pose access",
        "D21": "Test whether hidden history preserves the full input",
        "D22": "Isolate final raw bypass",
        "D23": "Isolate final all history fusion",
    }
    for model_id, purpose in custom_purposes.items():
        sources = _custom_t20_sources(model_id)
        rows.append(
            _make_spec(
                model_id,
                1,
                t20,
                (4, 4, 4, 4, 4),
                purpose,
                source_policy="CUSTOM",
                invariant_source_policy="CUSTOM",
                skip_policy="none",
                explicit_sources=sources,
                explicit_invariant_sources=sources,
            )
        )
    rows.extend(
        (
            _make_spec(
                "D24",
                1,
                (("A", 1), ("B", 2), ("B", 1)),
                (4, 4, 4, 4),
                "C17 control; 19939 parameters",
                skip_policy="legacy",
                expected_parameter_count=19_939,
                comparison_role="control",
            ),
            _make_spec(
                "D25",
                1,
                (("A", 3), ("B", 1), ("A", 2), ("B", 1)),
                (4, 4, 4, 4, 4),
                "C20 control; 20352 parameters",
                skip_policy="legacy",
                expected_parameter_count=20_352,
                comparison_role="control",
            ),
        )
    )
    return tuple(rows)


def _route(code: str, channels: Sequence[int]) -> tuple[tuple[str, int], ...]:
    if len(code) != len(channels):
        raise ValueError("hidden code and channels differ")
    return tuple(zip(code, map(int, channels)))


def _experiment_2() -> tuple[DModelSpec, ...]:
    rows = (
        (
            "D26",
            "ABB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2",
            "none",
            9_896,
            "10k C17 like anchor",
        ),
        (
            "D27",
            "ABAB",
            (1, 1, 1, 1),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            9_688,
            "10k C20 like anchor",
        ),
        (
            "D28",
            "ABB",
            (1, 1, 1),
            (5, 5, 5, 5),
            "FULL_P2",
            "none",
            14_954,
            "15k C17 like capacity point",
        ),
        (
            "D29",
            "ABAB",
            (1, 1, 1, 1),
            (4, 4, 4, 4, 4),
            "FULL_P2",
            "none",
            14_640,
            "15k C20 like capacity point",
        ),
        (
            "D30",
            "BAB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2",
            "none",
            10_064,
            "Put A between two B stages",
        ),
        (
            "D31",
            "BBA",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2",
            "none",
            10_232,
            "Put both B stages before A",
        ),
        (
            "D32",
            "ABBA",
            (1, 1, 1, 1),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            9_803,
            "Cluster B stages internally",
        ),
        (
            "D33",
            "BABA",
            (1, 1, 1, 1),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            9_918,
            "B first alternating order",
        ),
        (
            "D34",
            "BBAA",
            (1, 1, 1, 1),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            10_033,
            "B first clustered order",
        ),
        (
            "D35",
            "BB",
            (1, 1),
            (4, 4, 4),
            "FULL_P2",
            "none",
            10_579,
            "Scaled C14; test whether hidden A is needed",
        ),
        (
            "D36",
            "ABB",
            (2, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2",
            "none",
            11_235,
            "Allocate capacity to early A",
        ),
        (
            "D37",
            "ABB",
            (1, 2, 1),
            (3, 3, 3, 3),
            "FULL_P2",
            "none",
            15_904,
            "Test B contraction 2 to 1",
        ),
        (
            "D38",
            "ABB",
            (1, 1, 2),
            (3, 3, 3, 3),
            "FULL_P2",
            "none",
            15_808,
            "Test B expansion 1 to 2",
        ),
        (
            "D39",
            "ABAB",
            (2, 1, 1, 1),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            10_838,
            "Extra early A capacity",
        ),
        (
            "D40",
            "ABAB",
            (1, 1, 2, 1),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            10_878,
            "Extra late A capacity",
        ),
        (
            "D41",
            "ABAB",
            (1, 2, 1, 1),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            15_003,
            "Extra early B capacity",
        ),
        (
            "D42",
            "ABAB",
            (1, 1, 1, 2),
            (2, 2, 2, 2, 4),
            "FULL_P2",
            "none",
            14_858,
            "Extra output near B capacity",
        ),
        (
            "D43",
            "ABB",
            (1, 1, 1),
            (4, 4, 4, 4),
            "NO_SYM2",
            "none",
            9_154,
            "Controlled symmetric path ablation",
        ),
        (
            "D44",
            "ABB",
            (1, 1, 1),
            (8, 8, 8, 8),
            "NO_RAW_MIXED",
            "none",
            9_911,
            "Controlled joint coupling ablation",
        ),
        (
            "D45",
            "ABB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "NO_STF",
            "none",
            9_876,
            "Controlled STF path ablation",
        ),
        (
            "D46",
            "ABB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2_PLUS_SYM3",
            "sym3",
            21_019,
            "Test same variable cubic paths",
        ),
        (
            "D47",
            "ABB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2_PLUS_A2B",
            "a2b",
            12_348,
            "Test displacement dominant cubic coupling",
        ),
        (
            "D48",
            "ABB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2_PLUS_AB2",
            "ab2",
            30_018,
            "Test pose dominant cubic coupling",
        ),
        (
            "D49",
            "ABB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P2_PLUS_A2B_AB2",
            "union",
            32_470,
            "Test both sparse mixed cubic families",
        ),
        (
            "D50",
            "ABB",
            (1, 1, 1),
            (3, 3, 3, 3),
            "FULL_P3",
            "all",
            None,
            "Full cubic upper control",
        ),
    )
    return tuple(
        _make_spec(
            model_id,
            2,
            _route(code, channels),
            widths,
            purpose,
            path_policy=path_policy,
            degree3_policy=degree3,
            expected_parameter_count=expected,
            comparison_role="upper_control" if model_id == "D50" else "candidate",
        )
        for model_id, code, channels, widths, path_policy, degree3, expected, purpose in rows
    )


def _gate_options(model_id: str) -> tuple[str, dict[str, Any]]:
    gate_policy = "CURRENT_DENSE_LINEAR"
    values: dict[str, Any] = {}
    if model_id in {"D52", "D53", "D54"}:
        activation = {"D52": "sigmoid", "D53": "tanh", "D54": "silu"}[model_id]
        gate_policy = f"DENSE_{activation.upper()}"
        values["coefficient_activation"] = activation
    elif model_id == "D55":
        gate_policy = "METRIC_NORM_STATIC_MIXING"
        values.update(metric_gate="norm", coefficient_head="static_mixing")
    elif model_id == "D56":
        gate_policy = "DENSE_LINEAR_TIMES_METRIC_NORM"
        values["metric_gate"] = "multiply"
    elif model_id == "D57":
        gate_policy = "IDENTITY_AWARE_SKIP_TANH"
        values.update(metric_gate="skip_identity", coefficient_activation="tanh")
    elif model_id in {"D58", "D59", "D60", "D61"}:
        rank = int(model_id[1:]) - 57
        gate_policy = f"FACTORIZED_RANK_{rank}"
        values.update(coefficient_head="factorized", coefficient_rank=rank)
    elif model_id in {"D62", "D63"}:
        gate_policy = "ORTHOGONAL_RANK_2"
        values.update(coefficient_head="orthogonal", coefficient_rank=2)
        if model_id == "D63":
            values.update(initialization="trained_D51_truncated_svd", depends_on="D51")
    elif model_id in {"D64", "D65", "D66"}:
        mask = {"D64": "raw_only", "D65": "unary", "D66": "mixed"}[model_id]
        gate_policy = f"DESCRIPTOR_{mask.upper()}"
        values["descriptor_mask"] = mask
    elif model_id == "D67":
        gate_policy = "TRAIN_RANK_REVEALING"
        values.update(
            descriptor_transform="rank_revealing_qr_svd", calibration_partition="train"
        )
    elif model_id == "D68":
        gate_policy = "TRAIN_PCA_99"
        values.update(
            descriptor_transform="pca",
            retained_variance=0.99,
            calibration_partition="train",
        )
    elif model_id == "D69":
        gate_policy = "RANDOM_ORTHOGONAL_MATCH_D68"
        values.update(
            descriptor_transform="random_orthogonal",
            retained_rank_from="D68",
            depends_on="D68",
        )
    elif model_id == "D70":
        gate_policy = "LINEARIZED_WIDTH_4"
        values["trunk_linearized"] = True
    elif model_id in {"D71", "D72"}:
        gate_policy = f"SILU_WIDTH_{int(model_id[1:]) - 70}"
    elif model_id in {"D73", "D74"}:
        depth = int(model_id[1:]) - 71
        gate_policy = f"SILU_DEPTH_{depth}"
        values["trunk_depth"] = depth
    elif model_id == "D75":
        gate_policy = "RESIDUAL_WIDTH_4_DEPTH_3"
        values.update(
            trunk_depth=3,
            trunk_residual=True,
            trunk_formula="h1=silu(L1(s));h2=h1+L3(silu(L2(h1)))",
        )
    return gate_policy, values


def _experiment_3() -> tuple[DModelSpec, ...]:
    purposes = {
        "D51": "Current Invariant Gate reference",
        "D52": "Positive coefficient control",
        "D53": "Signed bounded coefficients",
        "D54": "Asymmetric positive unbounded coefficients",
        "D55": "Group independent norm only alternative",
        "D56": "Test norm only versus global joint conditioning",
        "D57": "Identity aware information preserving gate",
        "D58": "Strong routing bottleneck",
        "D59": "Rank 1 versus rank 2 comparison",
        "D60": "Intermediate rank comparison",
        "D61": "Same affine rank as D51 with different parameterization",
        "D62": "Test orthogonal coefficient parameterization",
        "D63": "Test SVD initialization after D51",
        "D64": "Minimal context control",
        "D65": "Test single source scale information",
        "D66": "Test joint conditioning",
        "D67": "Test true descriptor redundancy",
        "D68": "Test data aware compression",
        "D69": "Distinguish PCA structure from generic dimension reduction",
        "D70": "Test whether scalar trunk nonlinearity is necessary",
        "D71": "Small width point",
        "D72": "Intermediate width point",
        "D73": "Depth 2 comparison",
        "D74": "Depth 3 comparison",
        "D75": "Internal Gate residual comparison",
    }
    result = []
    for index in range(51, 76):
        model_id = f"D{index:02d}"
        width = 1 if model_id == "D71" else 2 if model_id == "D72" else 4
        gate_policy, options = _gate_options(model_id)
        result.append(
            _make_spec(
                model_id,
                3,
                (("A", 1), ("B", 2), ("B", 1)),
                (width, width, width, width),
                purposes[model_id],
                gate_policy=gate_policy,
                expected_parameter_count=19_939 if model_id == "D51" else None,
                comparison_role="reference" if model_id == "D51" else "candidate",
                options=options,
            )
        )
    return tuple(result)


EXPERIMENT_1_SPECS = _experiment_1()
EXPERIMENT_2_SPECS = _experiment_2()
EXPERIMENT_3_SPECS = _experiment_3()
D_SERIES_SPECS = (*EXPERIMENT_1_SPECS, *EXPERIMENT_2_SPECS, *EXPERIMENT_3_SPECS)
_MODEL_LOOKUP = {spec.model_id: spec for spec in D_SERIES_SPECS}


def _validate_catalog() -> None:
    expected = tuple(f"D{index:02d}" for index in range(1, 76))
    actual = tuple(spec.model_id for spec in D_SERIES_SPECS)
    if actual != expected:
        raise RuntimeError("D series identifiers must run from D01 through D75")
    if any(
        len(group) != 25
        for group in (EXPERIMENT_1_SPECS, EXPERIMENT_2_SPECS, EXPERIMENT_3_SPECS)
    ):
        raise RuntimeError("each D series experiment must contain twenty five models")


_validate_catalog()


def get_model_spec(model_id: str) -> DModelSpec:
    """Return one model specification by case insensitive identifier."""
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    try:
        return _MODEL_LOOKUP[model_id.upper()]
    except KeyError as error:
        raise KeyError(f"unknown D series model {model_id}") from error


def get_experiment_specs(experiment: int | str) -> tuple[DModelSpec, ...]:
    """Return one stable twenty five model experiment catalog."""
    key = str(experiment).strip().lower()
    aliases = {
        "1": EXPERIMENT_1_SPECS,
        "experiment_1": EXPERIMENT_1_SPECS,
        "d01_d25": EXPERIMENT_1_SPECS,
        "2": EXPERIMENT_2_SPECS,
        "experiment_2": EXPERIMENT_2_SPECS,
        "d26_d50": EXPERIMENT_2_SPECS,
        "3": EXPERIMENT_3_SPECS,
        "experiment_3": EXPERIMENT_3_SPECS,
        "d51_d75": EXPERIMENT_3_SPECS,
    }
    try:
        return aliases[key]
    except KeyError as error:
        raise KeyError(f"unknown D series experiment {experiment}") from error


def build_d_series_model(
    model: str | DModelSpec, generators: Any, **options: Any
) -> Any:
    """Build one model through the D series factory."""
    from experiments.benzene_pair.d_series.model_factory import (
        build_d_series_model as build,
    )

    return build(model, generators, **options)
