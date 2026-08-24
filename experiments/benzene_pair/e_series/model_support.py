"""Small model and compilation utilities used by the E experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from TFENN.models.invariant_gate_pipeline_v2 import (
    InvariantGatePipelineV2,
    InvariantGatePipelineV2Config,
    InvariantGateStageV2Config,
    build_invariant_gate_pipeline_v2,
)


__all__ = [
    "E4_COMPACT_BUDGET_VERSION",
    "E4_FROZEN_BUDGETS",
    "InvariantGateBudgetError",
    "OrdinaryRawMLP",
    "OrdinaryRawMLPConfig",
    "build_ordinary_raw_mlp",
    "build_budget_compiled_invariant_gate",
    "compact_blueprint_manifest",
    "compile_invariant_gate_budget",
]


E4_COMPACT_BUDGET_VERSION = "e4_budget_v2_20260817"
E4_FROZEN_BUDGETS: dict[str, tuple[int, int, int]] = {
    "E401": (11, 15, 7_998),
    "E402": (7, 5, 7_994),
    "E403": (6, 4, 8_008),
    "E404": (1, 23, 8_001),
    "E405": (10, 5, 7_978),
    "E406": (1, 8, 8_004),
    "E407": (21, 16, 7_997),
    "E408": (3, 10, 8_037),
    "E409": (8, 2, 8_067),
    "E410": (1, 35, 7_995),
    "E411": (6, 3, 8_001),
    "E412": (2, 30, 8_006),
    "E413": (9, 6, 7_987),
    "E414": (1, 20, 8_021),
    "E415": (6, 26, 7_995),
    "E416": (8, 3, 7_994),
    "E417": (6, 16, 8_006),
    "E418": (1, 22, 7_957),
    "E419": (10, 2, 7_990),
    "E420": (12, 14, 7_993),
    "E421": (5, 3, 8_000),
    "E422": (5, 15, 7_996),
    "E423": (3, 4, 8_043),
    "E424": (3, 21, 7_996),
    "E425": (2, 7, 8_014),
}


ActivationName = Literal["silu", "gelu", "tanh"]


def _positive_widths(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_widths must be a sequence")
    result = tuple(value)
    if not result or any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        for width in result
    ):
        raise ValueError("hidden_widths must contain positive integers")
    return result


@dataclass(frozen=True, slots=True)
class OrdinaryRawMLPConfig:
    """Configure the direct twelve scalar input MLP control."""

    hidden_widths: tuple[int, ...] = (96, 96, 96)
    activation: ActivationName = "silu"
    distance_scale: float = 6.0
    seed: int = 20260817

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_widths", _positive_widths(self.hidden_widths))
        if self.activation not in ("silu", "gelu", "tanh"):
            raise ValueError("activation must be silu, gelu, or tanh")
        if (
            isinstance(self.distance_scale, bool)
            or not isinstance(self.distance_scale, (int, float))
            or not math.isfinite(float(self.distance_scale))
            or float(self.distance_scale) <= 0.0
        ):
            raise ValueError("distance_scale must be finite and positive")
        object.__setattr__(self, "distance_scale", float(self.distance_scale))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_widths": list(self.hidden_widths),
            "activation": self.activation,
            "distance_scale": self.distance_scale,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OrdinaryRawMLPConfig":
        if not isinstance(value, Mapping):
            raise TypeError("configuration must be a mapping")
        return cls(
            hidden_widths=tuple(value.get("hidden_widths", (96, 96, 96))),
            activation=value.get("activation", "silu"),
            distance_scale=value.get("distance_scale", 6.0),
            seed=value.get("seed", 20260817),
        )


def _activation(value: Tensor, name: ActivationName) -> Tensor:
    if name == "silu":
        return torch.nn.functional.silu(value)
    if name == "gelu":
        return torch.nn.functional.gelu(value)
    return torch.tanh(value)


class OrdinaryRawMLP(nn.Module):
    """Predict local force directly from raw displacement and relative frame."""

    input_width = 12
    output_width = 3

    def __init__(self, config: OrdinaryRawMLPConfig) -> None:
        super().__init__()
        self.config = config
        widths = (self.input_width, *config.hidden_widths, self.output_width)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        layers: list[nn.Linear] = []
        for index, (input_width, output_width) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layer = nn.Linear(
                input_width,
                output_width,
                bias=index < len(widths) - 2,
                dtype=torch.float64,
            )
            scale = (
                math.sqrt(2.0 / input_width)
                if index < len(widths) - 2
                else 1.0 / math.sqrt(input_width)
            )
            with torch.no_grad():
                layer.weight.copy_(
                    torch.randn(
                        layer.weight.shape,
                        dtype=layer.weight.dtype,
                        generator=generator,
                    )
                    * scale
                )
                if layer.bias is not None:
                    layer.bias.zero_()
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "model_family": "ordinary_raw_mlp",
            "input_width": self.input_width,
            "output_width": self.output_width,
            "trainable_parameter_count": self.trainable_parameter_count,
            "config": self.config.as_dict(),
        }

    def _base_mlp(self, value: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            value = _activation(layer(value), self.config.activation)
        return self.layers[-1](value)

    def _root_geometry(self, centers: Tensor, frames: Tensor) -> tuple[Tensor, Tensor]:
        if centers.shape[-2:] != (2, 3):
            raise ValueError("centers must describe one ordered pair")
        if frames.shape != centers.shape[:-2] + (2, 3, 3):
            raise ValueError("frames must match the ordered pair shape")
        if centers.dtype != frames.dtype or centers.device != frames.device:
            raise ValueError("centers and frames must share dtype and device")
        reference = self.layers[0].weight
        if centers.dtype != reference.dtype or centers.device != reference.device:
            raise ValueError("inputs must match network dtype and device")
        root = frames[..., 0, :, :]
        displacement = centers[..., 1, :] - centers[..., 0, :]
        local = torch.einsum("...ji,...j->...i", root, displacement)
        return local, root.mT @ frames[..., 1, :, :]

    def force_from_local(self, displacement: Tensor, relative_frame: Tensor) -> Tensor:
        if displacement.shape[-1:] != (3,):
            raise ValueError("displacement must end with dimension three")
        if relative_frame.shape != displacement.shape[:-1] + (3, 3):
            raise ValueError("relative_frame must match displacement")
        features = torch.cat(
            (
                displacement / self.config.distance_scale,
                relative_frame.flatten(start_dim=-2),
            ),
            dim=-1,
        )
        return self._base_mlp(features)

    def forward_local(self, centers: Tensor, frames: Tensor) -> Tensor:
        displacement, relative_frame = self._root_geometry(centers, frames)
        return self.force_from_local(displacement, relative_frame)

    def forward(self, centers: Tensor, frames: Tensor) -> Tensor:
        local = self.forward_local(centers, frames)
        return torch.einsum("...ij,...j->...i", frames[..., 0, :, :], local)


def build_ordinary_raw_mlp(
    config: OrdinaryRawMLPConfig | Mapping[str, Any] | None = None,
) -> OrdinaryRawMLP:
    resolved = (
        OrdinaryRawMLPConfig()
        if config is None
        else OrdinaryRawMLPConfig.from_dict(config)
        if isinstance(config, Mapping)
        else config
    )
    if not isinstance(resolved, OrdinaryRawMLPConfig):
        raise TypeError("config must be an OrdinaryRawMLPConfig or mapping")
    return OrdinaryRawMLP(resolved)


class InvariantGateBudgetError(RuntimeError):
    """Report candidate counts when no compiled model reaches a budget."""

    def __init__(self, message: str, candidate_counts: Sequence[int]) -> None:
        super().__init__(message)
        self.candidate_counts = tuple(candidate_counts)


def compile_invariant_gate_budget(
    generators: Tensor,
    candidates: Sequence[InvariantGatePipelineV2Config | Mapping[str, Any]],
    *,
    target_range: tuple[int, int] = (7_800, 8_200),
    generator_names: Sequence[str] | None = None,
) -> InvariantGatePipelineV2:
    """Compile real candidates and select the closest model inside a budget."""
    if not candidates:
        raise ValueError("candidates cannot be empty")
    minimum, maximum = target_range
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum <= 0
        or maximum < minimum
    ):
        raise ValueError("target_range must contain ordered positive integers")
    target = (minimum + maximum) / 2.0
    compiled: list[InvariantGatePipelineV2] = []
    for candidate in candidates:
        compiled.append(
            build_invariant_gate_pipeline_v2(
                generators,
                candidate,
                generator_names=generator_names,
            )
        )
    eligible = tuple(
        model
        for model in compiled
        if minimum <= model.trainable_parameter_count <= maximum
    )
    counts = tuple(model.trainable_parameter_count for model in compiled)
    if not eligible:
        raise InvariantGateBudgetError(
            f"no compiled candidate is inside parameter range {target_range}",
            counts,
        )
    return min(
        eligible,
        key=lambda model: (
            abs(model.trainable_parameter_count - target),
            model.trainable_parameter_count,
        ),
    )


def _compact_stage_names(route: str) -> tuple[str, ...]:
    counts = {"A": 0, "B": 0}
    result = []
    for stream in route:
        counts[stream] += 1
        result.append(f"{stream.lower()}{counts[stream]}")
    return tuple(result)


def _compact_strategy(model_id: str) -> dict[str, Any]:
    values: dict[str, dict[str, Any]] = {
        "E401": dict(
            route="ABBA",
            channels=(1, 1, 1, 1),
            head="factorized",
            head_rank=1,
            skip="id",
            residual=True,
            mechanism="residual_cp",
        ),
        "E402": dict(
            route="ABABB",
            channels=(3, 2, 2, 1, 1),
            factors=(1.30, 1.0, 0.78, 0.60, 0.45, 0.45),
            sources="local",
            mechanism="funnel",
        ),
        "E403": dict(
            route="ABABA",
            channels=(1, 2, 3, 2, 1),
            factors=(0.55, 1.0, 1.35, 1.0, 0.55, 0.55),
            sources="waist",
            mechanism="diamond",
        ),
        "E404": dict(
            route="ABBBAA",
            channels=(2, 2, 1, 1, 2, 2),
            sources="local",
            typed=True,
            mechanism="typed_u",
        ),
        "E405": dict(
            route="AABB", channels=(1, 2, 2, 3), sources="dense", mechanism="dense_grow"
        ),
        "E406": dict(
            route="ABABABAB",
            channels=(2,) * 8,
            sources="local",
            skip="local_proj",
            reversible=True,
            mechanism="reversible_coupling",
        ),
        "E407": dict(
            route="A" * 16,
            channels=(1,) * 16,
            sources="local",
            share="tied_cell",
            skip="id",
            residual=True,
            mechanism="tied_cell",
        ),
        "E409": dict(
            route="ABBABA",
            channels=(1,) * 6,
            sources="directional",
            source_schedule=(
                ("x",),
                ("r", "a1"),
                ("r", "b1"),
                ("x", "b2"),
                ("r", "a2"),
                ("x", "b3"),
            ),
            mechanism="directional_ladder",
        ),
        "E410": dict(
            route="ABABA",
            channels=(2, 1, 2, 1, 1),
            sources="hub",
            mechanism="a_hub_star",
        ),
        "E412": dict(
            route="ABB",
            channels=(2, 2, 2),
            sources="dense",
            quota=5,
            typed=True,
            mechanism="masked_b_experts",
        ),
        "E414": dict(
            route="BBAABB",
            channels=(1,) * 6,
            sources="coverage_tree",
            source_schedule=(
                ("r",),
                ("r", "b1"),
                ("x", "b1", "b2"),
                ("x", "a1"),
                ("r", "a2", "b1"),
                ("r", "a2", "b2"),
            ),
            quota=6,
            mechanism="tree_fuse",
        ),
        "E415": dict(
            route="ABA",
            channels=(1, 2, 1),
            sources="dense",
            quota=6,
            mechanism="lift_pyramid",
        ),
        "E416": dict(
            route="ABB",
            channels=(2, 2, 2),
            sources="dense",
            quota=8,
            head="factorized",
            head_rank=2,
            mechanism="cp_wide_paths",
        ),
        "E417": dict(
            route="ABA",
            channels=(2, 3, 1),
            sources="dense",
            projection="tucker",
            projection_rank=2,
            mechanism="tucker_core",
        ),
        "E418": dict(
            route="ABAB",
            channels=(1, 3, 2, 1),
            sources="local",
            projection="tensor_train",
            projection_rank=2,
            mechanism="tensor_train_path",
        ),
        "E419": dict(
            route="ABABA",
            channels=(1,) * 5,
            sources="local",
            head="context_lora",
            head_rank=2,
            mechanism="context_lora",
        ),
        "E420": dict(
            route="ABABABAB",
            channels=(1,) * 8,
            sources="local",
            share="global_dictionary",
            mechanism="global_dictionary",
        ),
        "E421": dict(
            route="ABABAB",
            channels=(2,) * 6,
            sources="local",
            projection="toeplitz",
            head="factorized",
            head_rank=1,
            mechanism="toeplitz_wide",
        ),
        "E422": dict(
            route="ABABAB",
            channels=(1,) * 6,
            sources="local",
            head="axis_cp",
            head_rank=2,
            mechanism="axis_cp_head",
        ),
        "E424": dict(
            route="ABABAB",
            channels=(1,) * 6,
            sources="local",
            quota=5,
            aggregation="soft_moe",
            temperature=0.5,
            mechanism="soft_path_moe",
        ),
        "E425": dict(
            route="AB" * 6,
            channels=(2,) * 12,
            sources="local",
            share="cayley_flow",
            skip="local_proj",
            projection="cayley",
            residual=True,
            mechanism="cayley_flow",
        ),
    }
    if model_id not in values:
        raise KeyError(model_id)
    return values[model_id]


def compact_blueprint_manifest(model_id: str) -> dict[str, Any]:
    """Return the single executable blueprint for one compact model."""
    synchronous = {
        "E408": dict(
            topology="dual_stream",
            rounds=6,
            share="two_cycle",
            mechanism="tied_ab_two_cycle",
        ),
        "E411": dict(
            topology="twin_tower",
            tower_levels=3,
            fusion_levels=2,
            mechanism="twin_tower_late_fusion",
        ),
        "E413": dict(
            topology="dual_stream", rounds=3, share="type_graph", mechanism="type_graph"
        ),
        "E423": dict(
            topology="dual_stream",
            rounds=4,
            share=None,
            aggregation="attention",
            mechanism="invariant_attention",
        ),
    }
    if model_id in synchronous:
        return dict(synchronous[model_id])
    return dict(_compact_strategy(model_id))


def _stage_widths(
    strategy: Mapping[str, Any],
    base_width: int,
    readout_width: int | None = None,
) -> tuple[int, ...]:
    count = len(str(strategy["route"])) + 1
    factors = tuple(strategy.get("factors", (1.0,) * count))
    if len(factors) != count:
        raise ValueError("compact width factors do not match the stage count")
    result = tuple(max(1, int(round(base_width * float(value)))) for value in factors)
    if readout_width is not None:
        result = (*result[:-1], readout_width)
    return result


def _sequential_compact_config(
    model_id: str,
    architecture_name: str,
    strategy: Mapping[str, Any],
    base_width: int,
    readout_width: int | None = None,
) -> InvariantGatePipelineV2Config:
    route = str(strategy["route"])
    channels = tuple(int(value) for value in strategy["channels"])
    names = _compact_stage_names(route)
    widths = _stage_widths(strategy, base_width, readout_width)
    stages: list[Any] = []
    streams: dict[str, str] = {"x": "A", "r": "B"}
    previous: list[str] = []
    source_policy = str(strategy.get("sources", "dense"))
    for index, (name, stream, channel, width) in enumerate(
        zip(names, route, channels, widths[:-1])
    ):
        if source_policy in ("directional", "coverage_tree"):
            source_schedule = tuple(strategy["source_schedule"])
            if len(source_schedule) != len(route):
                raise ValueError("directional source schedule does not match route")
            sources = tuple(str(value) for value in source_schedule[index])
        elif source_policy == "dense":
            sources = ("x", "r", *previous)
        elif source_policy == "waist" and index == len(route) // 2:
            sources = ("x", "r", *previous)
        elif source_policy == "hub":
            a_history = tuple(item for item in previous if streams[item] == "A")
            sources = ("x", "r", *a_history, *previous[-1:])
        elif source_policy == "tree":
            sources = ("x", "r", *previous[-2:])
        else:
            sources = ("x", "r", *previous[-2:])
        invariant_sources = ("x", "r", *previous)
        share_group = strategy.get("share")
        required_sources = tuple(dict.fromkeys(sources))
        path_quota = max(
            int(strategy.get("quota", 4)),
            sum(streams[source] != stream for source in required_sources),
        )
        stage = InvariantGateStageV2Config(
            name=name,
            output_stream=stream,
            source_names=tuple(dict.fromkeys(sources)),
            channels=channel,
            invariant_source_names=invariant_sources,
            trunk_width=width,
            skip_policy=str(strategy.get("skip", "legacy")),
            trunk_depth=3 if bool(strategy.get("residual", False)) else 1,
            trunk_residual=bool(strategy.get("residual", False)),
            coefficient_head=str(strategy.get("head", "dense")),
            coefficient_rank=strategy.get("head_rank"),
            channel_projection=str(strategy.get("projection", "dense")),
            channel_projection_rank=strategy.get("projection_rank"),
            parameter_share_group=None if share_group is None else str(share_group),
            covariant_path_quota=path_quota,
            covariant_required_source_names=required_sources,
            path_aggregation=str(strategy.get("aggregation", "linear")),
            path_temperature=float(strategy.get("temperature", 1.0)),
            type_channel_overrides=(
                ((0, channel), (1, max(1, channel - 1)))
                if bool(strategy.get("typed", False)) and stream == "B"
                else ()
            ),
            reversible_coupling=bool(strategy.get("reversible", False)),
        )
        stages.append(stage)
        streams[name] = stream
        previous.append(name)
    output_sources = ("x", "r", *previous)
    output_quota = max(
        int(strategy.get("quota", 4)),
        sum(streams[source] != "A" for source in output_sources),
    )
    stages.append(
        InvariantGateStageV2Config(
            name="out",
            output_stream="A",
            source_names=output_sources,
            channels=1,
            invariant_source_names=("x", "r", *previous),
            trunk_width=widths[-1],
            skip_policy="legacy",
            coefficient_head=str(strategy.get("head", "dense")),
            coefficient_rank=strategy.get("head_rank"),
            channel_projection=str(strategy.get("projection", "dense")),
            channel_projection_rank=strategy.get("projection_rank"),
            covariant_path_quota=output_quota,
            covariant_required_source_names=output_sources,
            path_aggregation=str(strategy.get("aggregation", "linear")),
            path_temperature=float(strategy.get("temperature", 1.0)),
        )
    )
    return InvariantGatePipelineV2Config(
        stages=tuple(stages),
        output_stage="out",
        architecture_id=f"benzene_pair_e_series_{model_id.lower()}",
        anchor_ranks=(2, 6),
        max_constraint_entries=10_000_000,
        implemented_mechanism=(
            f"{model_id}:{architecture_name}:{strategy['mechanism']}"
        ),
    )


def _synchronous_compact_config(
    model_id: str,
    architecture_name: str,
    rounds: int,
    base_width: int,
    *,
    readout_width: int | None = None,
    aggregation: str = "linear",
    share: str | None = None,
) -> InvariantGatePipelineV2Config:
    stages: list[Any] = []
    history: list[str] = []
    stages.extend(
        (
            InvariantGateStageV2Config(
                "a0",
                "A",
                ("x",),
                1,
                invariant_source_names=("x", "r"),
                trunk_width=base_width,
                execution_level=0,
                covariant_path_quota=4,
                covariant_required_source_names=("x",),
                path_aggregation=aggregation,
                parameter_share_group=None if share is None else f"{share}_a",
            ),
            InvariantGateStageV2Config(
                "b0",
                "B",
                ("r",),
                1,
                invariant_source_names=("x", "r"),
                trunk_width=base_width,
                execution_level=0,
                covariant_path_quota=4,
                covariant_required_source_names=("r",),
                path_aggregation=aggregation,
                parameter_share_group=None if share is None else f"{share}_b",
            ),
        )
    )
    history.extend(("a0", "b0"))
    previous_a, previous_b = "a0", "b0"
    for level in range(1, rounds + 1):
        a_name, b_name = f"a{level}", f"b{level}"
        context = ("x", "r", *history)
        common = dict(
            source_names=("x", "r", previous_a, previous_b),
            channels=1,
            invariant_source_names=context,
            trunk_width=base_width,
            execution_level=level,
            covariant_live_mixed_only=True,
            covariant_path_quota=4,
            covariant_required_source_names=("x", "r", previous_a, previous_b),
            path_aggregation=aggregation,
        )
        stages.append(
            InvariantGateStageV2Config(
                a_name,
                "A",
                parameter_share_group=None if share is None else f"{share}_a",
                **common,
            )
        )
        stages.append(
            InvariantGateStageV2Config(
                b_name,
                "B",
                parameter_share_group=None if share is None else f"{share}_b",
                **common,
            )
        )
        history.extend((a_name, b_name))
        previous_a, previous_b = a_name, b_name
    stages.append(
        InvariantGateStageV2Config(
            "out",
            "A",
            (previous_a, previous_b),
            1,
            invariant_source_names=("x", "r", *history),
            trunk_width=base_width if readout_width is None else readout_width,
            execution_level=rounds + 1,
            covariant_path_quota=4,
            covariant_required_source_names=(previous_a, previous_b),
            path_aggregation=aggregation,
        )
    )
    mechanism = {
        "E408": "tied_ab_two_cycle",
        "E411": "twin_tower_late_fusion",
        "E413": "type_graph",
        "E423": "invariant_attention",
    }[model_id]
    return InvariantGatePipelineV2Config(
        stages=tuple(stages),
        output_stage="out",
        architecture_id=f"benzene_pair_e_series_{model_id.lower()}",
        anchor_ranks=(2, 6),
        max_constraint_entries=10_000_000,
        implemented_mechanism=f"{model_id}:{architecture_name}:{mechanism}",
    )


def _twin_tower_compact_config(
    architecture_name: str,
    base_width: int,
    readout_width: int | None = None,
) -> InvariantGatePipelineV2Config:
    stages: list[Any] = []
    history: list[str] = []
    previous_a = "x"
    previous_b = "r"
    for level in range(1, 4):
        a_name = f"a{level}"
        b_name = f"b{level}"
        context = ("x", "r", *history)
        stages.extend(
            (
                InvariantGateStageV2Config(
                    a_name,
                    "A",
                    tuple(dict.fromkeys(("x", previous_a))),
                    1,
                    invariant_source_names=context,
                    trunk_width=base_width,
                    execution_level=level,
                    covariant_path_quota=4,
                    covariant_required_source_names=tuple(
                        dict.fromkeys(("x", previous_a))
                    ),
                ),
                InvariantGateStageV2Config(
                    b_name,
                    "B",
                    tuple(dict.fromkeys(("r", previous_b))),
                    1,
                    invariant_source_names=context,
                    trunk_width=base_width,
                    execution_level=level,
                    covariant_path_quota=4,
                    covariant_required_source_names=tuple(
                        dict.fromkeys(("r", previous_b))
                    ),
                ),
            )
        )
        history.extend((a_name, b_name))
        previous_a = a_name
        previous_b = b_name
    stages.append(
        InvariantGateStageV2Config(
            "fusion1",
            "A",
            ("x", "r", previous_a, previous_b),
            1,
            invariant_source_names=("x", "r", *history),
            trunk_width=base_width,
            execution_level=4,
            covariant_live_mixed_only=True,
            covariant_path_quota=4,
            covariant_required_source_names=("x", "r", previous_a, previous_b),
        )
    )
    history.append("fusion1")
    stages.append(
        InvariantGateStageV2Config(
            "out",
            "A",
            ("x", "r", previous_a, previous_b, "fusion1"),
            1,
            invariant_source_names=("x", "r", *history),
            trunk_width=(base_width if readout_width is None else readout_width),
            execution_level=5,
            covariant_live_mixed_only=True,
            covariant_path_quota=4,
            covariant_required_source_names=(
                "x",
                "r",
                previous_a,
                previous_b,
                "fusion1",
            ),
        )
    )
    return InvariantGatePipelineV2Config(
        stages=tuple(stages),
        output_stage="out",
        architecture_id="benzene_pair_e_series_e411",
        anchor_ranks=(2, 6),
        max_constraint_entries=10_000_000,
        implemented_mechanism=(f"E411:{architecture_name}:twin_tower_late_fusion"),
    )


def _compact_config(
    model_id: str,
    architecture_name: str,
    base_width: int,
    readout_width: int | None = None,
) -> InvariantGatePipelineV2Config:
    if model_id == "E408":
        return _synchronous_compact_config(
            model_id,
            architecture_name,
            6,
            base_width,
            readout_width=readout_width,
            share="two_cycle",
        )
    if model_id == "E411":
        return _twin_tower_compact_config(
            architecture_name,
            base_width,
            readout_width,
        )
    if model_id == "E413":
        return _synchronous_compact_config(
            model_id,
            architecture_name,
            3,
            base_width,
            readout_width=readout_width,
            share="type_graph",
        )
    if model_id == "E423":
        return _synchronous_compact_config(
            model_id,
            architecture_name,
            4,
            base_width,
            readout_width=readout_width,
            aggregation="attention",
        )
    return _sequential_compact_config(
        model_id,
        architecture_name,
        _compact_strategy(model_id),
        base_width,
        readout_width,
    )


def _estimate_compact_parameter_count(
    blueprint: InvariantGatePipelineV2,
    config: InvariantGatePipelineV2Config,
) -> int:
    """Count a width variant using one already compiled path blueprint."""
    from TFENN.models.invariant_gate_pipeline_v2 import (
        _ReversibleChannelCoupling,
        _build_channel_projection,
        _build_coefficient_head,
        _build_trunk,
        _stage_channel_count,
        _type_label,
    )

    invariant_counts = blueprint._invariant_counts
    shared_context_widths: dict[str, int] = {}
    for stage in config.stages:
        if stage.parameter_share_group is not None:
            shared_context_widths[stage.parameter_share_group] = max(
                shared_context_widths.get(stage.parameter_share_group, 0),
                invariant_counts[stage.name],
            )
    trunk_inputs = {
        stage.name: invariant_counts[stage.name]
        if stage.parameter_share_group is None
        else shared_context_widths[stage.parameter_share_group]
        for stage in config.stages
    }
    shared: dict[tuple[Any, ...], nn.Module] = {}
    modules: list[nn.Module] = []

    def add(
        stage: InvariantGateStageV2Config,
        key: tuple[Any, ...],
        factory: Any,
    ) -> nn.Module:
        if stage.parameter_share_group is None:
            module = factory()
        else:
            shared_key = (stage.parameter_share_group, *key)
            module = shared.get(shared_key)
            if module is None:
                module = factory()
                shared[shared_key] = module
        modules.append(module)
        return module

    channels = blueprint._channels
    for stage in config.stages:
        trunk_input = trunk_inputs[stage.name]
        add(
            stage,
            (
                "trunk",
                trunk_input,
                stage.trunk_width,
                stage.activation,
                stage.trunk_depth,
                stage.trunk_linearized,
                stage.trunk_residual,
                stage.metric_gate,
                stage.coefficient_head,
            ),
            lambda stage=stage, trunk_input=trunk_input: _build_trunk(
                stage, trunk_input, dtype=torch.float64
            ),
        )
        for target, paths in blueprint._stage_paths[stage.name].items():
            output_channels = _stage_channel_count(stage, target)
            for path_index, path in enumerate(paths):
                add(
                    stage,
                    (
                        "head",
                        _type_label(target),
                        path_index,
                        path.coefficient_channels,
                        stage.trunk_width,
                        stage.coefficient_head,
                        stage.coefficient_rank,
                    ),
                    lambda stage=stage, path=path: _build_coefficient_head(
                        stage, path.coefficient_channels, dtype=torch.float64
                    ),
                )
            skip_names = (
                stage.source_names
                if stage.skip_source_names is None
                else stage.skip_source_names
            )
            available = tuple(name for name in skip_names if (name, target) in channels)
            path_channels = output_channels * len(paths)
            if stage.skip_policy == "legacy":
                selected = available
                concat_channels = path_channels + sum(
                    channels[(name, target)] for name in selected
                )
            else:
                concat_channels = path_channels
                if stage.skip_policy == "id":
                    selected = tuple(
                        name
                        for name in available
                        if channels[(name, target)] == output_channels
                    )[-1:]
                elif stage.skip_policy == "local_proj":
                    selected = available[-1:]
                elif stage.skip_policy == "dense_proj":
                    selected = available
                else:
                    selected = ()
                if stage.skip_policy in ("local_proj", "dense_proj"):
                    skip_input = sum(channels[(name, target)] for name in selected)
                    add(
                        stage,
                        (
                            "skip_projection",
                            _type_label(target),
                            skip_input,
                            output_channels,
                            stage.channel_projection,
                            stage.channel_projection_rank,
                        ),
                        lambda stage=stage, skip_input=skip_input, output_channels=output_channels: (
                            _build_channel_projection(
                                stage,
                                skip_input,
                                output_channels,
                                dtype=torch.float64,
                            )
                        ),
                    )
                if stage.metric_gate == "skip_identity":
                    modules.append(
                        nn.Linear(
                            stage.trunk_width,
                            output_channels,
                            dtype=torch.float64,
                        )
                    )
            add(
                stage,
                (
                    "projection",
                    _type_label(target),
                    concat_channels,
                    output_channels,
                    stage.channel_projection,
                    stage.channel_projection_rank,
                ),
                lambda stage=stage, concat_channels=concat_channels, output_channels=output_channels: (
                    _build_channel_projection(
                        stage,
                        concat_channels,
                        output_channels,
                        dtype=torch.float64,
                    )
                ),
            )
            if stage.reversible_coupling:
                add(
                    stage,
                    (
                        "reversible",
                        _type_label(target),
                        output_channels,
                        stage.trunk_width,
                    ),
                    lambda output_channels=output_channels, stage=stage: (
                        _ReversibleChannelCoupling(
                            output_channels,
                            stage.trunk_width,
                            dtype=torch.float64,
                        )
                    ),
                )
    parameters = {
        id(parameter): parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    return sum(parameter.numel() for parameter in parameters.values())


def _compact_coverage_audit(
    model: InvariantGatePipelineV2,
) -> dict[str, Any]:
    from TFENN.models.invariant_gate_pipeline_v2 import A, _type_label

    output_streams = {stage.output_stream for stage in model.config.stages}
    required_targets = (
        *((A,) if "A" in output_streams else ()),
        *(model._b_keys if "B" in output_streams else ()),
    )
    live_lane_counts = {target: 0 for target in required_targets}
    represented_inputs: set[str] = set()
    bridge_roles: list[str] = []
    reachable_sources = {"x", "r"}
    stage_source_coverage: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for stage in model.config.stages:
        stage_reachable = False
        target_coverage: dict[str, Any] = {}
        for target, paths in model._stage_paths[stage.name].items():
            if paths:
                live_lane_counts[target] += model._channels[(stage.name, target)]
            for path in paths:
                endpoint_sources = {
                    endpoint.source for endpoint in path.candidate.endpoints
                }
                represented_inputs.update(endpoint_sources.intersection(("x", "r")))
                if endpoint_sources.issubset(reachable_sources):
                    stage_reachable = True
                target_is_a = target == A
                if any(
                    (endpoint.key == A) != target_is_a
                    for endpoint in path.candidate.endpoints
                ):
                    bridge_roles.append(path.candidate.role)
            skip_sources = model._skip_sources[(stage.name, target)]
            represented_inputs.update(set(skip_sources).intersection(("x", "r")))
            if set(skip_sources).intersection(reachable_sources):
                stage_reachable = True
            path_sources = {
                endpoint.source
                for path in paths
                for endpoint in path.candidate.endpoints
            }
            required_sources = (
                ()
                if stage.covariant_required_source_names is None
                else stage.covariant_required_source_names
            )
            covered_sources = path_sources.union(skip_sources)
            missing_sources = tuple(
                source for source in required_sources if source not in covered_sources
            )
            target_coverage[_type_label(target)] = {
                "required_sources": required_sources,
                "path_sources": tuple(sorted(path_sources)),
                "skip_sources": skip_sources,
                "missing_sources": missing_sources,
            }
            if missing_sources:
                failures.append(
                    f"stage {stage.name} target {_type_label(target)} "
                    f"misses sources {missing_sources}"
                )
        stage_source_coverage[stage.name] = target_coverage
        if stage_reachable:
            reachable_sources.add(stage.name)

    requested_inputs = tuple(
        source
        for source in ("x", "r")
        if any(source in stage.source_names for stage in model.config.stages)
    )
    missing_types = tuple(
        _type_label(target)
        for target, lane_count in live_lane_counts.items()
        if lane_count < 1
    )
    missing_inputs = tuple(
        source for source in requested_inputs if source not in represented_inputs
    )
    output_stage = model.config.output_stage
    output_config = next(
        stage for stage in model.config.stages if stage.name == output_stage
    )
    final_a_reachable = (
        output_config.output_stream == "A" and output_stage in reachable_sources
    )
    if missing_types:
        failures.append(f"missing live lanes for {missing_types}")
    if missing_inputs:
        failures.append(f"missing requested inputs {missing_inputs}")
    if not bridge_roles:
        failures.append("no selected A B bridge")
    if not final_a_reachable:
        failures.append("final A is not reachable")
    if failures:
        raise RuntimeError(
            f"{model.config.architecture_id} coverage audit failed: "
            + "; ".join(failures)
        )
    return {
        "status": "passed",
        "required_type_live_lanes": {
            _type_label(target): lane_count
            for target, lane_count in live_lane_counts.items()
        },
        "requested_inputs": requested_inputs,
        "represented_inputs": tuple(sorted(represented_inputs)),
        "final_a_reachable": final_a_reachable,
        "a_b_bridge_count": len(bridge_roles),
        "a_b_bridge_roles": tuple(bridge_roles),
        "stage_source_coverage": stage_source_coverage,
    }


def build_budget_compiled_invariant_gate(
    generators: Tensor,
    *,
    model_id: str,
    architecture_name: str,
    mechanism: Mapping[str, Any],
    target_range: tuple[int, int] | None,
    generator_names: Sequence[str] | None = None,
    search: bool = False,
) -> InvariantGatePipelineV2:
    """Compile one real compact topology and choose a width inside its budget."""
    if not isinstance(mechanism, Mapping):
        raise TypeError("mechanism must be a mapping")
    requested_mechanism = dict(mechanism)
    if requested_mechanism.get("full_invariant_context") is False:
        raise ValueError("compact models require the complete invariant context")
    blueprint_manifest = compact_blueprint_manifest(model_id)
    if target_range is None:
        raise ValueError("target_range is required")
    minimum, maximum = target_range
    target = (minimum + maximum) / 2.0
    if not search:
        try:
            selected_width, selected_readout_width, expected_count = E4_FROZEN_BUDGETS[
                model_id
            ]
        except KeyError as error:
            raise KeyError(f"no frozen compact budget for {model_id}") from error
        selected_config = _compact_config(
            model_id,
            architecture_name,
            selected_width,
            selected_readout_width,
        )
        result = build_invariant_gate_pipeline_v2(
            generators,
            selected_config,
            generator_names=generator_names,
        )
        coverage_audit = _compact_coverage_audit(result)
        if result.trainable_parameter_count != expected_count:
            raise RuntimeError(
                f"{model_id} frozen count {expected_count} differs from "
                f"compiled count {result.trainable_parameter_count}"
            )
        if not minimum <= expected_count <= maximum:
            raise RuntimeError(
                f"{model_id} frozen count is outside parameter range {target_range}"
            )
        result.budget_compilation_manifest = {
            "budget_version": E4_COMPACT_BUDGET_VERSION,
            "model_id": model_id,
            "architecture_name": architecture_name,
            "target_range": target_range,
            "selected_parameter_count": expected_count,
            "selected_base_width": selected_width,
            "selected_readout_width": selected_readout_width,
            "blueprint": blueprint_manifest,
            "requested_mechanism": requested_mechanism,
            "compiled_stage_config": [
                stage.as_dict() for stage in result.config.stages
            ],
            "selected_covariant_roles": result.selected_covariant_roles,
            "coverage_audit": coverage_audit,
            "factorization_ranks": tuple(
                {
                    "stage": stage.name,
                    "coefficient_rank": stage.coefficient_rank,
                    "channel_projection_rank": stage.channel_projection_rank,
                }
                for stage in result.config.stages
                if stage.coefficient_rank is not None
                or stage.channel_projection_rank is not None
            ),
        }
        return result
    blueprint_config = _compact_config(model_id, architecture_name, 1, 1)
    blueprint = build_invariant_gate_pipeline_v2(
        generators,
        blueprint_config,
        generator_names=generator_names,
    )
    minimum_count = _estimate_compact_parameter_count(blueprint, blueprint_config)
    if minimum_count != blueprint.trainable_parameter_count:
        raise RuntimeError(
            f"{model_id} minimum budget estimate {minimum_count} differs from "
            f"compiled count {blueprint.trainable_parameter_count}"
        )
    if minimum_count > maximum:
        raise InvariantGateBudgetError(
            f"{model_id} minimum topology exceeds parameter range {target_range}",
            (minimum_count,),
        )
    estimates: dict[tuple[int, int], int] = {}
    for base_width in range(1, 65):
        for readout_width in range(1, 129):
            config = _compact_config(
                model_id, architecture_name, base_width, readout_width
            )
            count = _estimate_compact_parameter_count(blueprint, config)
            estimates[(base_width, readout_width)] = count
            if count > maximum:
                break
        if estimates[(base_width, 1)] > maximum:
            break
    eligible = tuple(
        (key, count) for key, count in estimates.items() if minimum <= count <= maximum
    )
    if not eligible:
        raise InvariantGateBudgetError(
            f"{model_id} has no compiled width pair inside parameter range {target_range}",
            tuple(sorted(set(estimates.values()))),
        )
    (selected_width, selected_readout_width), selected_estimate = min(
        eligible,
        key=lambda item: (
            abs(item[1] - target),
            item[1],
            item[0],
        ),
    )
    selected_config = _compact_config(
        model_id,
        architecture_name,
        selected_width,
        selected_readout_width,
    )
    result = build_invariant_gate_pipeline_v2(
        generators,
        selected_config,
        generator_names=generator_names,
    )
    coverage_audit = _compact_coverage_audit(result)
    if result.trainable_parameter_count != selected_estimate:
        raise RuntimeError(
            f"{model_id} budget estimate {selected_estimate} differs from "
            f"compiled count {result.trainable_parameter_count}"
        )
    result.budget_compilation_manifest = {
        "budget_version": E4_COMPACT_BUDGET_VERSION,
        "model_id": model_id,
        "architecture_name": architecture_name,
        "target_range": target_range,
        "selected_parameter_count": result.trainable_parameter_count,
        "selected_base_width": selected_width,
        "selected_readout_width": selected_readout_width,
        "estimated_candidate_count": len(estimates),
        "blueprint": blueprint_manifest,
        "requested_mechanism": requested_mechanism,
        "compiled_stage_config": [stage.as_dict() for stage in result.config.stages],
        "selected_covariant_roles": result.selected_covariant_roles,
        "coverage_audit": coverage_audit,
        "factorization_ranks": tuple(
            {
                "stage": stage.name,
                "coefficient_rank": stage.coefficient_rank,
                "channel_projection_rank": stage.channel_projection_rank,
            }
            for stage in result.config.stages
            if stage.coefficient_rank is not None
            or stage.channel_projection_rank is not None
        ),
    }
    return result
