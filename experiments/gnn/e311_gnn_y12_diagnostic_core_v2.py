"""Core model variants for the Y01 to Y12 causal diagnostic study.

The pair kernel, registered paths, invariant Gate, channel counts, and raw mix
setting remain unchanged. This module changes only graph stack transitions,
the scale applied after receiver local SUM, and the RunningRMS update rule.
Dataset routing and optimization belong in the external runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from experiments.gnn.e311_gnn_12_experiment_core_v1 import (
    BAggregationV1,
    E311GraphCoreConfigV1,
    E311MessageStackCoreV1,
    complete_pair_index_v1,
)
from TFENN.models import GraphMessageBlockOutputV1
from TFENN.tensor_math import TypeKey


CURRENT_X01_STEPS = 8_000
LEGACY_5K_V3_STEPS = 31_500
EMA_DECAY = 0.99


class EdgeTransitionV2(str, Enum):
    """Describe how a Message Block candidate becomes the accepted edge state."""

    REPLACE = "replace"
    IDENTITY_INTERPOLATION = "identity_interpolation"


class RMSPolicyV2(str, Enum):
    CUMULATIVE = "cumulative"
    EMA = "ema"


@dataclass(frozen=True, slots=True)
class YExperimentSpecV2:
    experiment_id: str
    dataset: str
    layer_count: int
    aggregation: BAggregationV1
    aggregation_scale: float
    edge_transition: EdgeTransitionV2
    edge_alpha_init: float
    ema_layers: tuple[int, ...]
    optimizer_protocol: str
    optimizer_steps: int
    purpose: str
    decisive_comparison: str


Y_EXPERIMENT_SPECS_V2: tuple[YExperimentSpecV2, ...] = (
    YExperimentSpecV2(
        "Y01",
        "pair_2k_current",
        1,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (),
        "x01_current",
        CURRENT_X01_STEPS,
        "Reproduce the current X01 result.",
        "X01",
    ),
    YExperimentSpecV2(
        "Y02",
        "pair_2k_current",
        1,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (),
        "x01_continue",
        LEGACY_5K_V3_STEPS,
        "Change only the optimizer update budget.",
        "Y01",
    ),
    YExperimentSpecV2(
        "Y03",
        "pair_2k_current",
        1,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (),
        "legacy_5k_v3_protocol",
        LEGACY_5K_V3_STEPS,
        "Test the old 5k v3 optimizer protocol at fixed data and steps.",
        "Y02",
    ),
    YExperimentSpecV2(
        "Y04",
        "pair_5k_legacy_v3",
        1,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (),
        "legacy_5k_v3_protocol",
        LEGACY_5K_V3_STEPS,
        "Run the current block on the old 5k v3 data and protocol.",
        "Y03 and the E311 test component MAE 0.0018",
    ),
    YExperimentSpecV2(
        "Y05",
        "pair_2k_current",
        1,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (0,),
        "legacy_5k_v3_protocol",
        LEGACY_5K_V3_STEPS,
        "Test cumulative RMS lag in the single block model.",
        "Y03",
    ),
    YExperimentSpecV2(
        "Y06",
        "five_benzene_1k",
        1,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (),
        "selected_pair_protocol",
        LEGACY_5K_V3_STEPS,
        "Establish the corrected five benzene one layer anchor.",
        "Y07",
    ),
    YExperimentSpecV2(
        "Y07",
        "five_benzene_1k",
        2,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (),
        "selected_pair_protocol",
        LEGACY_5K_V3_STEPS,
        "Reproduce depth damage without B communication.",
        "Y06",
    ),
    YExperimentSpecV2(
        "Y08",
        "five_benzene_1k",
        2,
        BAggregationV1.SUM,
        1.0,
        EdgeTransitionV2.REPLACE,
        1.0,
        (),
        "selected_pair_protocol",
        LEGACY_5K_V3_STEPS,
        "Reproduce the current raw SUM two layer stack.",
        "Y07",
    ),
    YExperimentSpecV2(
        "Y09",
        "five_benzene_1k",
        2,
        BAggregationV1.NONE,
        1.0,
        EdgeTransitionV2.IDENTITY_INTERPOLATION,
        0.0,
        (),
        "selected_pair_protocol",
        LEGACY_5K_V3_STEPS,
        "Test exact identity initialization of the second edge transition.",
        "Y07",
    ),
    YExperimentSpecV2(
        "Y10",
        "five_benzene_1k",
        2,
        BAggregationV1.SUM,
        1.0,
        EdgeTransitionV2.IDENTITY_INTERPOLATION,
        0.0,
        (),
        "selected_pair_protocol",
        LEGACY_5K_V3_STEPS,
        "Measure raw SUM after stabilizing the edge transition.",
        "Y09 and Y08",
    ),
    YExperimentSpecV2(
        "Y11",
        "five_benzene_1k",
        2,
        BAggregationV1.SUM,
        0.25,
        EdgeTransitionV2.IDENTITY_INTERPOLATION,
        0.0,
        (),
        "selected_pair_protocol",
        LEGACY_5K_V3_STEPS,
        "Diagnose degree four SUM scale without changing message semantics.",
        "Y10",
    ),
    YExperimentSpecV2(
        "Y12",
        "five_benzene_1k",
        2,
        BAggregationV1.SUM,
        0.25,
        EdgeTransitionV2.IDENTITY_INTERPOLATION,
        0.0,
        (1,),
        "selected_pair_protocol",
        LEGACY_5K_V3_STEPS,
        "Test layer two RMS lag after controlling edge and B scale.",
        "Y11",
    ),
)

_Y_SPEC_BY_ID = MappingProxyType(
    {spec.experiment_id: spec for spec in Y_EXPERIMENT_SPECS_V2}
)
if len(_Y_SPEC_BY_ID) != 12:
    raise RuntimeError("the Y diagnostic registry must contain exactly 12 experiments")


def get_y_experiment_spec_v2(experiment_id: str) -> YExperimentSpecV2:
    try:
        return _Y_SPEC_BY_ID[experiment_id.upper()]
    except KeyError as error:
        raise KeyError(f"unknown Y experiment: {experiment_id}") from error


class EMARunningRMSV2(nn.Module):
    """Normalize each schema with an exponential nonstationary RMS estimate."""

    def __init__(
        self,
        epsilon: float,
        decay: float,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.epsilon = float(epsilon)
        self.decay = float(decay)
        self.register_buffer("mean_square", torch.ones((), dtype=dtype, device=device))
        self.register_buffer(
            "sample_count", torch.zeros((), dtype=torch.int64, device=device)
        )

    def reset(self) -> None:
        with torch.no_grad():
            self.mean_square.fill_(1.0)
            self.sample_count.zero_()

    def forward(self, value: Tensor) -> Tensor:
        if self.training:
            detached = value.detach()
            if detached.numel() == 0:
                raise ValueError("cannot normalize an empty invariant schema")
            batch_mean_square = detached.square().mean()
            with torch.no_grad():
                if int(self.sample_count.item()) == 0:
                    self.mean_square.copy_(batch_mean_square)
                else:
                    self.mean_square.mul_(self.decay).add_(
                        batch_mean_square, alpha=1.0 - self.decay
                    )
                self.sample_count.add_(detached.numel())
        scale = torch.where(
            self.sample_count > 0,
            torch.sqrt(self.mean_square.clamp_min(0.0) + self.epsilon),
            torch.ones_like(self.mean_square),
        )
        return value / scale


def _replace_running_rms_with_ema(
    parent: nn.Module,
    decay: float,
) -> int:
    """Replace cumulative RMS leaves while preserving the public stage API."""

    replaced = 0
    for name, child in tuple(parent.named_children()):
        if child.__class__.__name__ == "_RunningRMS":
            replacement = EMARunningRMSV2(
                child.epsilon,
                decay,
                dtype=child.mean_square.dtype,
                device=child.mean_square.device,
            )
            if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(name)] = replacement
            elif isinstance(parent, nn.ModuleDict):
                parent[name] = replacement
            else:
                setattr(parent, name, replacement)
            replaced += 1
        else:
            replaced += _replace_running_rms_with_ema(child, decay)
    return replaced


@dataclass(frozen=True, slots=True)
class YCoreConfigV2:
    layer_count: int
    aggregation: BAggregationV1
    aggregation_scale: float = 1.0
    edge_transition: EdgeTransitionV2 = EdgeTransitionV2.REPLACE
    edge_alpha_init: float = 1.0
    ema_layers: tuple[int, ...] = ()
    ema_decay: float = EMA_DECAY

    def __post_init__(self) -> None:
        if not 1 <= self.layer_count <= 4:
            raise ValueError("layer_count must be in [1, 4]")
        if not math.isfinite(self.aggregation_scale) or self.aggregation_scale <= 0:
            raise ValueError("aggregation_scale must be finite and positive")
        if not math.isfinite(self.edge_alpha_init):
            raise ValueError("edge_alpha_init must be finite")
        if any(index < 0 or index >= self.layer_count for index in self.ema_layers):
            raise ValueError("EMA layer index is outside the stack")


@dataclass(frozen=True, slots=True)
class LayerAuditV2:
    layer_index: int
    edge_input_rms: Tensor
    edge_candidate_rms: Tensor
    edge_accepted_rms: Tensor
    edge_correction_ratio: Tensor
    aggregated_b_rms: Mapping[TypeKey, Tensor]
    hidden_input_rms: Mapping[TypeKey, Tensor]
    hidden_output_rms: Mapping[TypeKey, Tensor]


@dataclass(frozen=True, slots=True)
class YCoreOutputV2:
    normalized_node_force_world: Tensor
    normalized_pair_force_world: Tensor
    edge_a_world: Tensor
    hidden_b_input_to_last_layer: Mapping[TypeKey, Tensor]
    final_aggregated_b_local: Mapping[TypeKey, Tensor]
    layer_outputs: tuple[GraphMessageBlockOutputV1, ...]
    layer_audits: tuple[LayerAuditV2, ...]


class E311YDiagnosticCoreV2(E311MessageStackCoreV1):
    """Extend the existing E311 stack with explicit causal controls."""

    def __init__(
        self,
        force_scale: float,
        y_config: YCoreConfigV2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            force_scale,
            E311GraphCoreConfigV1(
                layer_count=y_config.layer_count,
                aggregation=y_config.aggregation,
                share_all_layers=False,
            ),
            dtype,
        )
        self.y_config = y_config
        transition_count = max(y_config.layer_count - 1, 0)
        initial = torch.full(
            (transition_count,), y_config.edge_alpha_init, dtype=dtype
        )
        if y_config.edge_transition is EdgeTransitionV2.IDENTITY_INTERPOLATION:
            self.edge_alpha = nn.Parameter(initial)
        else:
            self.register_buffer("edge_alpha", initial)

        for layer_index in y_config.ema_layers:
            count = _replace_running_rms_with_ema(
                self._block_at(layer_index).pair_kernel,
                y_config.ema_decay,
            )
            if count == 0:
                raise RuntimeError(f"layer {layer_index} contains no RunningRMS modules")

    def reset_running_rms(self) -> None:
        for block in self.message_blocks:
            block.pair_kernel.reset_running_rms()
            for module in block.pair_kernel.modules():
                if isinstance(module, EMARunningRMSV2):
                    module.reset()

    def architecture_record(self) -> Mapping[str, Any]:
        record = dict(super().architecture_record())
        ema_layer_set = set(self.y_config.ema_layers)
        record.update(
            {
                "core_version": "e311_gnn_y12_diagnostic_core_v2",
                "aggregation_scale": self.y_config.aggregation_scale,
                "edge_transition": self.y_config.edge_transition.value,
                "edge_alpha_init": self.y_config.edge_alpha_init,
                "edge_alpha_current": tuple(
                    float(value) for value in self.edge_alpha.detach().cpu().tolist()
                ),
                "edge_alpha_trainable": isinstance(self.edge_alpha, nn.Parameter),
                "ema_layers": self.y_config.ema_layers,
                "ema_decay": self.y_config.ema_decay,
                "running_rms_policy_by_layer": tuple(
                    RMSPolicyV2.EMA.value
                    if layer_index in ema_layer_set
                    else RMSPolicyV2.CUMULATIVE.value
                    for layer_index in range(self.y_config.layer_count)
                ),
            }
        )
        return MappingProxyType(record)

    @staticmethod
    def _bank_rms(bank: Mapping[TypeKey, Tensor]) -> Mapping[TypeKey, Tensor]:
        return MappingProxyType(
            {
                key: value.detach().square().mean().sqrt()
                for key, value in bank.items()
            }
        )

    @staticmethod
    def _scale_bank(
        bank: Mapping[TypeKey, Tensor], scale: float
    ) -> Mapping[TypeKey, Tensor]:
        return MappingProxyType({key: value * scale for key, value in bank.items()})

    @staticmethod
    def _scatter_pair_force(
        pair_force_world: Tensor,
        pair_index: Tensor,
        node_count: int,
    ) -> Tensor:
        receiver, sender = pair_index
        batch_shape = pair_force_world.shape[:-2]
        node_axis = len(batch_shape)
        result = pair_force_world.new_zeros(batch_shape + (node_count, 3))
        result = result.index_add(node_axis, receiver, pair_force_world)
        return result.index_add(node_axis, sender, -pair_force_world)

    def _accept_edge(
        self,
        layer_index: int,
        previous: Tensor,
        candidate: Tensor,
    ) -> Tensor:
        if (
            layer_index == 0
            or self.y_config.edge_transition is EdgeTransitionV2.REPLACE
        ):
            return candidate
        alpha = self.edge_alpha[layer_index - 1]
        return previous + alpha * (candidate - previous)

    def core_output(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor | None = None,
        *,
        collect_trace: bool = False,
    ) -> YCoreOutputV2:
        self._validate_geometry(centers_world, frames_body_to_world)
        node_count = centers_world.shape[-2]
        if pair_index is None:
            pair_index = complete_pair_index_v1(node_count, centers_world.device)
        if pair_index.ndim != 2 or pair_index.shape[0] != 2:
            raise ValueError("pair_index must have shape (2, E)")
        if pair_index.device != centers_world.device:
            raise ValueError("pair_index must be on the graph device")

        hidden = self._initial_hidden_b(centers_world)
        edge = self._initial_edge_a(centers_world, pair_index.shape[1])
        outputs: list[GraphMessageBlockOutputV1] = []
        audits: list[LayerAuditV2] = []
        hidden_input_to_last = hidden
        final_aggregated: Mapping[TypeKey, Tensor] | None = None

        for layer_index in range(self.y_config.layer_count):
            edge_input = edge
            hidden_input = hidden
            if layer_index == self.y_config.layer_count - 1:
                hidden_input_to_last = hidden_input

            raw = self._block_at(layer_index)(
                centers_world,
                frames_body_to_world,
                pair_index,
                hidden_input,
                edge_input,
                None,
                collect_trace=collect_trace,
            )
            edge = self._accept_edge(layer_index, edge_input, raw.edge_a_world)
            pair_force = edge[..., 0, :]
            node_force = self._scatter_pair_force(pair_force, pair_index, node_count)
            accepted = GraphMessageBlockOutputV1(
                pair_force_world=pair_force,
                node_force_world=node_force,
                edge_a_world=edge,
                message_j_to_i_local=raw.message_j_to_i_local,
                message_i_to_j_local=raw.message_i_to_j_local,
                node_b_local=raw.node_b_local,
                trace=raw.trace,
            )
            outputs.append(accepted)

            aggregated = self._scale_bank(
                self.b_aggregator(raw, pair_index, node_count),
                self.y_config.aggregation_scale,
            )
            final_aggregated = aggregated
            if layer_index + 1 < self.y_config.layer_count:
                hidden = self._node_update_at(layer_index)(hidden_input, aggregated)
            else:
                hidden = hidden_input

            denominator = edge_input.detach().square().mean().sqrt().clamp_min(1.0e-12)
            correction = (raw.edge_a_world - edge_input).detach()
            audits.append(
                LayerAuditV2(
                    layer_index=layer_index,
                    edge_input_rms=edge_input.detach().square().mean().sqrt(),
                    edge_candidate_rms=raw.edge_a_world.detach().square().mean().sqrt(),
                    edge_accepted_rms=edge.detach().square().mean().sqrt(),
                    edge_correction_ratio=correction.square().mean().sqrt()
                    / denominator,
                    aggregated_b_rms=self._bank_rms(aggregated),
                    hidden_input_rms=self._bank_rms(hidden_input),
                    hidden_output_rms=self._bank_rms(hidden),
                )
            )

        if final_aggregated is None:
            raise RuntimeError("the stack produced no layer")
        final = outputs[-1]
        return YCoreOutputV2(
            normalized_node_force_world=final.node_force_world,
            normalized_pair_force_world=final.pair_force_world,
            edge_a_world=final.edge_a_world,
            hidden_b_input_to_last_layer=hidden_input_to_last,
            final_aggregated_b_local=final_aggregated,
            layer_outputs=tuple(outputs),
            layer_audits=tuple(audits),
        )

    @torch.no_grad()
    def warm_start_first_block(self, source: E311MessageStackCoreV1) -> None:
        if self.path_manifest_sha256 != source.path_manifest_sha256:
            raise ValueError("source and target path manifests differ")
        self._block_at(0).load_state_dict(source._block_at(0).state_dict(), strict=True)

    def layer_rms_snapshot(self, layer_index: int) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for name, module in self._block_at(layer_index).pair_kernel.named_modules():
            if not hasattr(module, "mean_square") or not hasattr(module, "sample_count"):
                continue
            result[name] = {
                "mean_square": float(module.mean_square.detach().cpu()),
                "sample_count": int(module.sample_count.detach().cpu()),
            }
        return result

    def layer_gradient_norms(self) -> tuple[Tensor, ...]:
        norms = []
        for layer_index in range(self.y_config.layer_count):
            squares = [
                parameter.grad.detach().square().sum()
                for parameter in self._block_at(layer_index).parameters()
                if parameter.grad is not None
            ]
            norms.append(
                torch.stack(squares).sum().sqrt()
                if squares
                else self.force_scale.new_zeros(())
            )
        return tuple(norms)


def y_core_config_from_spec_v2(spec: YExperimentSpecV2) -> YCoreConfigV2:
    return YCoreConfigV2(
        layer_count=spec.layer_count,
        aggregation=spec.aggregation,
        aggregation_scale=spec.aggregation_scale,
        edge_transition=spec.edge_transition,
        edge_alpha_init=spec.edge_alpha_init,
        ema_layers=spec.ema_layers,
    )


def build_y_diagnostic_core_v2(
    experiment_id: str,
    force_scale: float,
    dtype: torch.dtype = torch.float32,
) -> E311YDiagnosticCoreV2:
    spec = get_y_experiment_spec_v2(experiment_id)
    return E311YDiagnosticCoreV2(
        force_scale,
        y_core_config_from_spec_v2(spec),
        dtype,
    )


@torch.no_grad()
def assert_one_layer_stack_preflight_v2(
    model: E311YDiagnosticCoreV2,
    centers_world: Tensor,
    frames_body_to_world: Tensor,
    pair_index: Tensor,
    *,
    atol: float = 2.0e-6,
) -> None:
    """Stop before training if a one layer stack changes its block output."""

    if model.y_config.layer_count != 1:
        raise ValueError("preflight requires a one layer model")
    was_training = model.training
    model.eval()
    hidden = model._initial_hidden_b(centers_world)
    edge = model._initial_edge_a(centers_world, pair_index.shape[1])
    raw = model._block_at(0)(
        centers_world, frames_body_to_world, pair_index, hidden, edge, None
    )
    stacked = model.core_output(centers_world, frames_body_to_world, pair_index)
    model.train(was_training)
    for actual, expected, name in (
        (stacked.edge_a_world, raw.edge_a_world, "edge"),
        (stacked.normalized_pair_force_world, raw.pair_force_world, "pair force"),
        (stacked.normalized_node_force_world, raw.node_force_world, "node force"),
    ):
        if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
            error = float((actual - expected).abs().max())
            raise RuntimeError(f"one layer {name} preflight failed: {error:.3e}")


__all__ = [
    "CURRENT_X01_STEPS",
    "LEGACY_5K_V3_STEPS",
    "EMA_DECAY",
    "EdgeTransitionV2",
    "RMSPolicyV2",
    "YExperimentSpecV2",
    "Y_EXPERIMENT_SPECS_V2",
    "YCoreConfigV2",
    "YCoreOutputV2",
    "E311YDiagnosticCoreV2",
    "EMARunningRMSV2",
    "get_y_experiment_spec_v2",
    "y_core_config_from_spec_v2",
    "build_y_diagnostic_core_v2",
    "assert_one_layer_stack_preflight_v2",
]
