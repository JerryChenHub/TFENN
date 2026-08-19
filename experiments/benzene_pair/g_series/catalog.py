"""Define the fixed-shape G series mechanism catalog around E311."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


__all__ = [
    "CARRIER_SPECS",
    "FACTORIAL_SPECS",
    "GModelSpec",
    "G_SERIES_SPECS",
    "LEGACY_SPECS",
    "SEED_BLOCK_SPECS",
    "VARIANT_SPECS",
    "get_group_specs",
    "get_model_spec",
    "get_seed_specs",
    "get_variant_specs",
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


_STAGE_SOURCES = (
    {"name": "a1", "output_stream": "A", "channels": 1, "sources": ("x", "r")},
    {
        "name": "b1",
        "output_stream": "B",
        "channels": 2,
        "sources": ("x", "r", "a1"),
    },
    {
        "name": "b2",
        "output_stream": "B",
        "channels": 1,
        "sources": ("x", "r", "a1", "b1"),
    },
    {
        "name": "out",
        "output_stream": "A",
        "channels": 1,
        "sources": ("x", "r", "a1", "b1", "b2"),
    },
)

_ALL_CARRIER_GROUPS = {
    "stem": True,
    "adjacent": True,
    "raw_deep": True,
    "hidden_deep": True,
}


@dataclass(frozen=True, slots=True)
class GModelSpec:
    """Store one paired-seed G series training definition."""

    model_id: str
    variant_id: int
    seed_index: int
    model_seed: int
    shuffle_seed: int
    family: str
    architecture_name: str
    description: str
    purpose: str
    comparison_role: str = "candidate"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variant_id not in range(1, 15):
            raise ValueError("G variant id must be one through fourteen")
        if self.seed_index not in range(1, 6):
            raise ValueError("G seed index must be one through five")
        expected_id = f"G{self.seed_index}{self.variant_id:02d}"
        if self.model_id != expected_id:
            raise ValueError(f"G model id must equal {expected_id}")
        if self.model_seed < 0 or self.shuffle_seed < 0:
            raise ValueError("G seeds must be nonnegative")
        if self.family != "e311_fixed_shape_audit":
            raise ValueError("G models must use the E311 fixed-shape audit family")
        if self.comparison_role not in {"control", "candidate"}:
            raise ValueError("G comparison role must be control or candidate")
        object.__setattr__(self, "options", _freeze(self.options))

    @property
    def expected_parameter_count(self) -> None:
        """Use preflight compilation as the nominal parameter-count authority."""
        return None

    @property
    def execution_shard_id(self) -> int:
        """Map the one-based paired seed to its zero-based execution shard."""
        return self.seed_index - 1

    @property
    def descriptor_mask(self) -> str:
        return "full"

    @property
    def d6_covariance_exempt(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        value = {
            "model_id": self.model_id,
            "variant_id": self.variant_id,
            "seed_index": self.seed_index,
            "execution_shard_id": self.execution_shard_id,
            "model_seed": self.model_seed,
            "shuffle_seed": self.shuffle_seed,
            "family": self.family,
            "architecture_name": self.architecture_name,
            "description": self.description,
            "purpose": self.purpose,
            "comparison_role": self.comparison_role,
            "expected_parameter_count": None,
            "d6_covariance_exempt": False,
            "options": _thaw(self.options),
        }
        json.dumps(value, allow_nan=False)
        return value


@dataclass(frozen=True, slots=True)
class _Variant:
    variant_id: int
    description: str
    purpose: str
    generic_pair_enabled: bool = False
    stf_a1_enabled: bool = True
    stf_out_enabled: bool = True
    carrier_mode: str = "direct"
    gated_identity_initialization: str | None = None
    carrier_group_mask: Mapping[str, bool] = field(
        default_factory=lambda: dict(_ALL_CARRIER_GROUPS)
    )

    def __post_init__(self) -> None:
        if self.carrier_mode not in {"direct", "none", "gated_identity"}:
            raise ValueError("unknown G carrier mode")
        if self.gated_identity_initialization not in {
            None,
            "residual_zero",
            "default",
        }:
            raise ValueError("unknown gated identity initialization")
        if self.carrier_mode == "gated_identity":
            if self.gated_identity_initialization is None:
                raise ValueError("gated identity requires an initialization")
        elif self.gated_identity_initialization is not None:
            raise ValueError("only gated identity uses a gated initialization")
        if set(self.carrier_group_mask) != set(_ALL_CARRIER_GROUPS):
            raise ValueError("carrier group mask must define all four groups")
        object.__setattr__(self, "carrier_group_mask", _freeze(self.carrier_group_mask))


def _carrier_mask(**changes: bool) -> dict[str, bool]:
    result = dict(_ALL_CARRIER_GROUPS)
    unknown = set(changes) - set(result)
    if unknown:
        raise ValueError(f"unknown carrier groups {tuple(sorted(unknown))}")
    result.update(changes)
    return result


_VARIANTS = (
    _Variant(
        1,
        "E311 control with generic pairs off and both STF directions active",
        "Fresh paired-seed E311 control under the G CPU protocol",
    ),
    _Variant(
        2,
        "Generic pairs off with only the a1 STF direction active",
        "Measure the out-stage STF contribution",
        stf_out_enabled=False,
    ),
    _Variant(
        3,
        "Generic pairs off with only the out STF direction active",
        "Measure the a1-stage STF contribution",
        stf_a1_enabled=False,
    ),
    _Variant(
        4,
        "Generic pairs off with both covariant STF directions inactive",
        "Measure the combined covariant STF contribution",
        stf_a1_enabled=False,
        stf_out_enabled=False,
    ),
    _Variant(
        5,
        "Generic raw-containing pair covariants and both STF directions active",
        "Add the generic pair bank to the E311 mechanism",
        generic_pair_enabled=True,
    ),
    _Variant(
        6,
        "Generic pair covariants with only the a1 STF direction active",
        "Test generic-pair substitution for the out STF direction",
        generic_pair_enabled=True,
        stf_out_enabled=False,
    ),
    _Variant(
        7,
        "Generic pair covariants with only the out STF direction active",
        "Test generic-pair substitution for the a1 STF direction",
        generic_pair_enabled=True,
        stf_a1_enabled=False,
    ),
    _Variant(
        8,
        "Generic pair covariants active with both STF directions inactive",
        "Test the generic pair bank without covariant STF directions",
        generic_pair_enabled=True,
        stf_a1_enabled=False,
        stf_out_enabled=False,
    ),
    _Variant(
        9,
        "All direct same-TypeKey carriers inactive",
        "Test whether the complete direct carrier bundle is necessary",
        carrier_mode="none",
        carrier_group_mask=_carrier_mask(
            stem=False,
            adjacent=False,
            raw_deep=False,
            hidden_deep=False,
        ),
    ),
    _Variant(
        10,
        "Gated identity carriers with residual-zero initialization",
        "Test a learnable carrier initialized as the direct identity route",
        carrier_mode="gated_identity",
        gated_identity_initialization="residual_zero",
    ),
    _Variant(
        11,
        "Gated identity carriers with default initialization",
        "Separate carrier availability from its initialization and optimization",
        carrier_mode="gated_identity",
        gated_identity_initialization="default",
    ),
    _Variant(
        12,
        "Direct carriers active except deep raw shortcuts",
        "Measure r to b2 and x to out carrier shortcuts",
        carrier_group_mask=_carrier_mask(raw_deep=False),
    ),
    _Variant(
        13,
        "Direct carriers active except the a1 to out hidden shortcut",
        "Measure the learned hidden-history carrier shortcut",
        carrier_group_mask=_carrier_mask(hidden_deep=False),
    ),
    _Variant(
        14,
        "Direct carriers active without deep raw or hidden shortcuts",
        "Measure the interaction of raw-deep and hidden-deep carriers",
        carrier_group_mask=_carrier_mask(raw_deep=False, hidden_deep=False),
    ),
)

_PAIRED_SEEDS = tuple(
    (index, 20_260_822 + 10 * (index - 1), 20_260_823 + 10 * (index - 1))
    for index in range(1, 6)
)


def _options(variant: _Variant) -> dict[str, Any]:
    return {
        "backbone": "C17",
        "source_graph": "C17_DENSE_HISTORY",
        "stage_sources": _STAGE_SOURCES,
        "gate_width": 8,
        "scalar_invariants": "full",
        "fixed_shape_supernet": True,
        "compiled_covariant_union": "generic_pairs_plus_stf",
        "generic_pair_enabled": variant.generic_pair_enabled,
        "stf_a1_enabled": variant.stf_a1_enabled,
        "stf_out_enabled": variant.stf_out_enabled,
        "carrier_mode": variant.carrier_mode,
        "gated_identity_initialization": variant.gated_identity_initialization,
        "carrier_group_mask": variant.carrier_group_mask,
    }


def _build_specs() -> tuple[GModelSpec, ...]:
    result = []
    for seed_index, model_seed, shuffle_seed in _PAIRED_SEEDS:
        for variant in _VARIANTS:
            result.append(
                GModelSpec(
                    model_id=f"G{seed_index}{variant.variant_id:02d}",
                    variant_id=variant.variant_id,
                    seed_index=seed_index,
                    model_seed=model_seed,
                    shuffle_seed=shuffle_seed,
                    family="e311_fixed_shape_audit",
                    architecture_name="E311 fixed-shape mechanism audit",
                    description=variant.description,
                    purpose=variant.purpose,
                    comparison_role="control"
                    if variant.variant_id == 1
                    else "candidate",
                    options=_options(variant),
                )
            )
    return tuple(result)


G_SERIES_SPECS = _build_specs()
_BY_ID = MappingProxyType({spec.model_id: spec for spec in G_SERIES_SPECS})
SEED_BLOCK_SPECS = MappingProxyType(
    {
        seed_index: tuple(
            spec for spec in G_SERIES_SPECS if spec.seed_index == seed_index
        )
        for seed_index in range(1, 6)
    }
)
VARIANT_SPECS = MappingProxyType(
    {
        variant_id: tuple(
            spec for spec in G_SERIES_SPECS if spec.variant_id == variant_id
        )
        for variant_id in range(1, 15)
    }
)

_GROUP_VARIANTS = MappingProxyType(
    {
        "factorial": tuple(range(1, 9)),
        "legacy": (1, 9, 10, 11),
        "carrier": (1, 12, 13, 14),
    }
)
FACTORIAL_SPECS = tuple(
    spec for spec in G_SERIES_SPECS if spec.variant_id in _GROUP_VARIANTS["factorial"]
)
LEGACY_SPECS = tuple(
    spec for spec in G_SERIES_SPECS if spec.variant_id in _GROUP_VARIANTS["legacy"]
)
CARRIER_SPECS = tuple(
    spec for spec in G_SERIES_SPECS if spec.variant_id in _GROUP_VARIANTS["carrier"]
)


def get_model_spec(model_id: str) -> GModelSpec:
    try:
        return _BY_ID[str(model_id).upper()]
    except KeyError as error:
        raise KeyError(f"unknown G model {model_id}") from error


def _number(value: int | str, prefixes: Sequence[str]) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
        return int(normalized)
    return int(value)


def get_variant_specs(variant: int | str) -> tuple[GModelSpec, ...]:
    value = _number(variant, ("variant_", "variant", "g"))
    try:
        return VARIANT_SPECS[value]
    except KeyError as error:
        raise KeyError(f"unknown G variant {variant}") from error


def get_seed_specs(seed: int | str) -> tuple[GModelSpec, ...]:
    value = _number(seed, ("seed_", "seed", "block_", "block", "g"))
    try:
        return SEED_BLOCK_SPECS[value]
    except KeyError as error:
        raise KeyError(f"unknown G seed block {seed}") from error


def get_group_specs(group: str) -> tuple[GModelSpec, ...]:
    normalized = str(group).strip().lower()
    aliases = {
        "g1": "factorial",
        "g2": "factorial",
        "dual_stf": "factorial",
        "g3": "legacy",
        "g4": "carrier",
    }
    normalized = aliases.get(normalized, normalized)
    values = {
        "factorial": FACTORIAL_SPECS,
        "legacy": LEGACY_SPECS,
        "carrier": CARRIER_SPECS,
    }
    try:
        return values[normalized]
    except KeyError as error:
        raise KeyError(f"unknown G group {group}") from error
