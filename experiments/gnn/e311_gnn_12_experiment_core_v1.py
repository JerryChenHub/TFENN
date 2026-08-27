"""Core E311 graph stacks for the twelve controlled GNN experiments.

This module deliberately contains no dataset, optimizer, checkpoint, or CLI
code.  It reuses ``E311MultibodyMessageBlockV1`` without changing its path
registry, invariant descriptors, Gate, or typed A1-B2-B1-A1 schedule.  The
only experimental variables implemented here are graph depth, receiver-local
B aggregation, and whole-layer parameter sharing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from TFENN.models import (
    E311MessageBlockConfigV1,
    E311MultibodyMessageBlockV1,
    GraphMessageBlockOutputV1,
    build_e311_multibody_message_block_v1,
)
from TFENN.tensor_math import (
    GeneratorSystem,
    PoseEncoder,
    TypeKey,
    build_primitive_b_manifest,
    build_type_catalog,
    compile_anchors,
)


LOCKED_MESSAGE_BLOCK_CONFIG_V1 = E311MessageBlockConfigV1(
    edge_a_channels=1,
    a_mid_channels=1,
    b_wide_channels=2,
    b_out_channels=1,
    gate_width=8,
    molecular_scalar_dim=0,
    distance_scale=6.0,
    rbf_centers=(0.0, 0.5, 1.0, 1.5, 2.0),
    rbf_width=0.4,
    inverse_powers=(1, 2, 3),
    distance_epsilon=1.0e-12,
    rms_epsilon=1.0e-8,
    max_constraint_entries=10_000_000,
)

LOCKED_BLOCK_SIGNATURE_V1: Mapping[str, Any] = MappingProxyType(
    {
        "typed_schedule": "A1-B2-B1-A1",
        "edge_a_channels": 1,
        "hidden_b_channels_per_type": 1,
        "b_wide_channels": 2,
        "b_out_channels": 1,
        "gate_width": 8,
        "gate_readout": "signed_unbounded",
        "generic_raw_mixed_covariants": False,
        "force_readout": "last_layer_only",
    }
)


class BAggregationV1(str, Enum):
    """Receiver-local aggregation used between message-block layers."""

    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


class SupervisionV1(str, Enum):
    """Training target expected from the external experiment runner."""

    AUDIT_ONLY = "audit_only"
    NODE_FORCE = "node_force"
    PAIR_FORCE = "pair_force"


@dataclass(frozen=True, slots=True)
class E311GNNExperimentSpecV1:
    """A model-side declaration for one controlled experiment."""

    experiment_id: str
    title: str
    dataset: str
    layer_count: int
    aggregation: BAggregationV1
    share_all_layers: bool
    supervision: SupervisionV1
    train_node_counts: tuple[int, ...]
    test_node_counts: tuple[int, ...]
    topology: str
    initialization: str
    purpose: str
    decisive_comparison: str
    expected_result: str


EXPERIMENT_SPECS_V1: tuple[E311GNNExperimentSpecV1, ...] = (
    E311GNNExperimentSpecV1(
        "X01",
        "two-benzene one-block reproduction",
        "opls_two_benzene_pair",
        1,
        BAggregationV1.NONE,
        False,
        SupervisionV1.PAIR_FORCE,
        (2,),
        (2,),
        "complete",
        "scratch",
        "Lock the already converged one-block result as the reference.",
        "current converged checkpoint and ledger",
        "The reproduced error and convergence curve agree within seed variation.",
    ),
    E311GNNExperimentSpecV1(
        "X02",
        "frozen pair checkpoint explicit multi-pair sum",
        "opls_multi_benzene_pair_decomposition",
        1,
        BAggregationV1.NONE,
        False,
        SupervisionV1.AUDIT_ONLY,
        (),
        (3, 4, 5),
        "complete",
        "X01_checkpoint",
        "Measure zero-shot pair-error accumulation before graph-level training.",
        "explicit pair loop versus one-block graph wrapper",
        "The two implementations match numerically; error growth is attributable "
        "to pair transfer.",
    ),
    E311GNNExperimentSpecV1(
        "X03",
        "five-benzene one-block node-force training",
        "opls_five_benzene",
        1,
        BAggregationV1.NONE,
        False,
        SupervisionV1.NODE_FORCE,
        (5,),
        (5,),
        "complete",
        "scratch",
        "Test graph-level signed scatter and node-force credit assignment.",
        "X04 pair supervision; X02 is a secondary transfer diagnostic",
        "It approaches the pair-supervised ceiling if total-force supervision is sufficient.",
    ),
    E311GNNExperimentSpecV1(
        "X04",
        "five-benzene one-block pair-force supervision",
        "opls_five_benzene_pair_decomposition",
        1,
        BAggregationV1.NONE,
        False,
        SupervisionV1.PAIR_FORCE,
        (5,),
        (5,),
        "complete",
        "scratch",
        "Establish a pair-supervised optimization reference without graph credit assignment.",
        "X03 node-force supervision",
        "A large X03-X04 gap diagnoses supervision ambiguity rather than missing paths.",
    ),
    E311GNNExperimentSpecV1(
        "X05",
        "five-benzene two-block depth control without B communication",
        "opls_five_benzene",
        2,
        BAggregationV1.NONE,
        False,
        SupervisionV1.NODE_FORCE,
        (5,),
        (5,),
        "complete",
        "scratch",
        "Isolate extra edge-local refinement from cross-edge communication.",
        "X06, with equal depth and parameter count",
        "Any gain over X03 is a depth effect and is not evidence of many-body communication.",
    ),
    E311GNNExperimentSpecV1(
        "X06",
        "five-benzene two-block receiver-local B sum",
        "opls_five_benzene",
        2,
        BAggregationV1.SUM,
        False,
        SupervisionV1.NODE_FORCE,
        (5,),
        (5,),
        "complete",
        "scratch",
        "Measure the effect of cross-edge B communication at fixed depth.",
        "X05 no-communication control",
        "Pair-additive OPLS should not require a large gain; instability indicates "
        "graph-state scaling problems.",
    ),
    E311GNNExperimentSpecV1(
        "X07",
        "five-benzene recurrent shared-all two-block stack",
        "opls_five_benzene",
        2,
        BAggregationV1.SUM,
        True,
        SupervisionV1.NODE_FORCE,
        (5,),
        (5,),
        "complete",
        "scratch",
        "Test a recurrent inductive bias and parameter efficiency.",
        "X06 independent layers",
        "Similar validation error with fewer parameters supports sharing; shared "
        "RMS semantics must be reported.",
    ),
    E311GNNExperimentSpecV1(
        "X08",
        "five-benzene three-block B-sum depth probe",
        "opls_five_benzene",
        3,
        BAggregationV1.SUM,
        False,
        SupervisionV1.NODE_FORCE,
        (5,),
        (5,),
        "complete",
        "scratch",
        "Check whether a third communication round helps or only adds optimization burden.",
        "X06 two independent layers",
        "OPLS should plateau early; exploding B norms or worse validation error "
        "exposes depth instability.",
    ),
    E311GNNExperimentSpecV1(
        "X09",
        "variable-size two-block B-sum generalization",
        "opls_variable_benzene_count",
        2,
        BAggregationV1.SUM,
        False,
        SupervisionV1.NODE_FORCE,
        (3, 4),
        (5,),
        "complete_bucketed_by_node_count",
        "scratch",
        "Test extensivity and transfer across graph size with the draft-preferred sum.",
        "X10 mean aggregation on identical splits",
        "Per-node error remains approximately stable under the pair-additive target.",
    ),
    E311GNNExperimentSpecV1(
        "X10",
        "variable-size two-block B-mean normalization control",
        "opls_variable_benzene_count",
        2,
        BAggregationV1.MEAN,
        False,
        SupervisionV1.NODE_FORCE,
        (3, 4),
        (5,),
        "complete_bucketed_by_node_count",
        "scratch",
        "Determine whether mean normalization erases degree-dependent environment strength.",
        "X09 sum aggregation on identical splits",
        "A fixed-N comparison is not decisive; the N=3,4 to N=5 shift makes the "
        "aggregation law identifiable.",
    ),
    E311GNNExperimentSpecV1(
        "X11",
        "conservative three-body chain without B communication",
        "synthetic_conservative_three_body_chain",
        2,
        BAggregationV1.NONE,
        False,
        SupervisionV1.NODE_FORCE,
        (3,),
        (3,),
        "chain_0-1-2",
        "scratch",
        "Establish the irreducible error of an edge-local stack on a true three-body target.",
        "X12 on the same samples, split, depth, and parameter count",
        "The model cannot make edge (0,1) depend on node 2, and its corresponding "
        "Jacobian stays zero.",
    ),
    E311GNNExperimentSpecV1(
        "X12",
        "conservative three-body chain with B communication",
        "synthetic_conservative_three_body_chain",
        2,
        BAggregationV1.SUM,
        False,
        SupervisionV1.NODE_FORCE,
        (3,),
        (3,),
        "chain_0-1-2",
        "scratch",
        "Provide a positive test that aggregation creates learnable many-body dependence.",
        "X11 no-communication control",
        "Error falls and the edge-(0,1) response to node 2 becomes nonzero through "
        "node-1 B state.",
    ),
)

_EXPERIMENT_BY_ID_V1 = MappingProxyType(
    {spec.experiment_id: spec for spec in EXPERIMENT_SPECS_V1}
)
if len(_EXPERIMENT_BY_ID_V1) != 12:
    raise RuntimeError("the v1 experiment registry must contain exactly twelve IDs")


def get_experiment_spec_v1(experiment_id: str) -> E311GNNExperimentSpecV1:
    """Return one immutable experiment declaration."""

    try:
        return _EXPERIMENT_BY_ID_V1[experiment_id.upper()]
    except KeyError as error:
        known = ", ".join(_EXPERIMENT_BY_ID_V1)
        raise KeyError(f"unknown experiment {experiment_id!r}; expected one of {known}") from error


@contextmanager
def _fixed_geometry_compilation_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    if previous != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


def benzene_generators_v1(dtype: torch.dtype = torch.float64) -> Tensor:
    """Return the proper D6 generators used by the existing benzene model."""

    angle = math.pi / 3.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
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
        dtype=dtype,
    )


def complete_pair_index_v1(node_count: int, device: torch.device | None = None) -> Tensor:
    """Return one canonical entry for every unordered edge."""

    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 2:
        raise ValueError("node_count must be an integer of at least two")
    return torch.triu_indices(node_count, node_count, offset=1, device=device)


@dataclass(frozen=True, slots=True)
class E311GraphCoreConfigV1:
    """Graph-level variables; the internal message block remains locked.

    ``share_all_layers`` shares the message block, its RunningRMS buffers, and
    the external typed node update.  It is therefore a recurrent engineering
    configuration, not a pure learnable-weight-sharing intervention.
    """

    layer_count: int = 2
    aggregation: BAggregationV1 = BAggregationV1.SUM
    share_all_layers: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_count, bool)
            or not isinstance(self.layer_count, int)
            or not 1 <= self.layer_count <= 4
        ):
            raise ValueError("layer_count must be an integer in [1, 4]")
        if not isinstance(self.aggregation, BAggregationV1):
            try:
                object.__setattr__(self, "aggregation", BAggregationV1(self.aggregation))
            except (TypeError, ValueError) as error:
                raise ValueError("aggregation must be none, sum, or mean") from error
        if not isinstance(self.share_all_layers, bool):
            raise TypeError("share_all_layers must be bool")


class ReceiverLocalBAggregatorV1(nn.Module):
    """Aggregate directed B messages without changing representation channels."""

    def __init__(self, mode: BAggregationV1, b_keys: tuple[TypeKey, ...]) -> None:
        super().__init__()
        self.mode = BAggregationV1(mode)
        self._b_keys = tuple(b_keys)

    @staticmethod
    def _index_add_nodes(values: Tensor, indices: Tensor, node_count: int) -> Tensor:
        batch_shape = values.shape[:-3]
        result = values.new_zeros(batch_shape + (node_count,) + values.shape[-2:])
        return result.index_add(len(batch_shape), indices, values)

    def forward(
        self,
        output: GraphMessageBlockOutputV1,
        pair_index: Tensor,
        node_count: int,
    ) -> Mapping[TypeKey, Tensor]:
        receiver, sender = pair_index
        expected = set(self._b_keys)
        if (
            set(output.message_j_to_i_local) != expected
            or set(output.message_i_to_j_local) != expected
        ):
            raise ValueError("the two directed message banks must contain every B TypeKey")
        summed = {
            key: self._index_add_nodes(
                output.message_j_to_i_local[key], receiver, node_count
            )
            + self._index_add_nodes(
                output.message_i_to_j_local[key], sender, node_count
            )
            for key in output.message_j_to_i_local
        }
        if self.mode is BAggregationV1.NONE:
            return MappingProxyType(
                {key: torch.zeros_like(value) for key, value in summed.items()}
            )
        if self.mode is BAggregationV1.SUM:
            return MappingProxyType(summed)

        degree = torch.bincount(pair_index.reshape(-1), minlength=node_count).clamp_min(1)
        mean = {}
        for key, value in summed.items():
            view_shape = (1,) * (value.ndim - 3) + (node_count, 1, 1)
            divisor = degree.to(device=value.device, dtype=value.dtype).reshape(view_shape)
            mean[key] = value / divisor
        return MappingProxyType(mean)


class TypedNodeUpdateV1(nn.Module):
    """Apply ``(P_h tensor I) Cat[h, m_bar]`` independently per B TypeKey."""

    def __init__(
        self,
        b_keys: tuple[TypeKey, ...],
        hidden_channels: Mapping[TypeKey, int],
        message_channels: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if message_channels != 1:
            raise ValueError("v1 locks message_channels at one")
        self._b_keys = tuple(b_keys)
        self._hidden_channels = MappingProxyType(dict(hidden_channels))
        projections = []
        for key in self._b_keys:
            channels = self._hidden_channels[key]
            if channels != 1:
                raise ValueError("v1 locks every hidden B TypeKey at one channel")
            weight = nn.Parameter(
                torch.zeros((channels, channels + message_channels), dtype=dtype)
            )
            with torch.no_grad():
                weight[:, channels:].copy_(torch.eye(channels, message_channels, dtype=dtype))
            projections.append(weight)
        self.projections = nn.ParameterList(projections)

    def forward(
        self,
        hidden_b_local: Mapping[TypeKey, Tensor],
        aggregated_b_local: Mapping[TypeKey, Tensor],
    ) -> Mapping[TypeKey, Tensor]:
        expected = set(self._b_keys)
        if set(hidden_b_local) != expected or set(aggregated_b_local) != expected:
            raise ValueError("node update inputs must contain every B TypeKey")
        updated = {}
        for key, weight in zip(self._b_keys, self.projections, strict=True):
            hidden = hidden_b_local[key]
            message = aggregated_b_local[key]
            if hidden.shape[:-2] != message.shape[:-2] or hidden.shape[-1] != message.shape[-1]:
                raise ValueError(f"node update geometry mismatch for {key}")
            if hidden.shape[-2] != 1 or message.shape[-2] != 1:
                raise ValueError(f"node update channel mismatch for {key}")
            if (
                hidden.dtype != weight.dtype
                or message.dtype != hidden.dtype
                or hidden.device != weight.device
                or message.device != hidden.device
            ):
                raise ValueError(f"node update dtype/device mismatch for {key}")
            carriers = torch.cat((hidden, message), dim=-2)
            updated[key] = torch.einsum("oc,...ncd->...nod", weight, carriers)
        return MappingProxyType(updated)


@dataclass(frozen=True, slots=True)
class E311GraphCoreOutputV1:
    """Last-layer prediction plus states retained for scientific audits."""

    normalized_node_force_world: Tensor
    normalized_pair_force_world: Tensor
    edge_a_world: Tensor
    hidden_b_input_to_last_layer: Mapping[TypeKey, Tensor]
    final_aggregated_b_local: Mapping[TypeKey, Tensor]
    layer_outputs: tuple[GraphMessageBlockOutputV1, ...]


class E311MessageStackCoreV1(nn.Module):
    """Stack unchanged E311 message blocks with an external graph-state update."""

    def __init__(
        self,
        force_scale: float,
        config: E311GraphCoreConfigV1 = E311GraphCoreConfigV1(),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not math.isfinite(force_scale) or force_scale <= 0.0:
            raise ValueError("force_scale must be finite and positive")
        if not isinstance(config, E311GraphCoreConfigV1):
            raise TypeError("config must be E311GraphCoreConfigV1")
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("dtype must be torch.float32 or torch.float64")
        self.config = config

        with _fixed_geometry_compilation_threads():
            generators = benzene_generators_v1(torch.float64)
            anchors = compile_anchors(generators, output_ranks=(2, 6))
            manifest = build_primitive_b_manifest(anchors)
            catalog = build_type_catalog(
                GeneratorSystem(("sixfold", "twofold"), generators),
                manifest,
            )
            pose_encoder = PoseEncoder(anchors)
            hidden_channels = {item.key: 1 for item in manifest}

            def build_block() -> E311MultibodyMessageBlockV1:
                return build_e311_multibody_message_block_v1(
                    catalog,
                    manifest,
                    pose_encoder,
                    hidden_channels,
                    LOCKED_MESSAGE_BLOCK_CONFIG_V1,
                    dtype,
                )

            registered_block_count = 1 if config.share_all_layers else config.layer_count
            blocks = [build_block() for _ in range(registered_block_count)]

        self.catalog = catalog
        self.manifest = manifest
        self.message_blocks = nn.ModuleList(blocks)
        self.b_aggregator = ReceiverLocalBAggregatorV1(
            config.aggregation, blocks[0].b_keys
        )

        path_fingerprints = tuple(
            self._path_manifest_sha256(block) for block in blocks
        )
        if len(set(path_fingerprints)) != 1:
            raise RuntimeError("all registered layers must compile the same path manifest")
        self.path_manifest_sha256 = path_fingerprints[0]

        update_count = max(config.layer_count - 1, 0)
        registered_update_count = 1 if config.share_all_layers and update_count else update_count
        self.node_updates = nn.ModuleList(
            [
                TypedNodeUpdateV1(
                    blocks[0].b_keys,
                    blocks[0].pair_kernel.hidden_channels,
                    LOCKED_MESSAGE_BLOCK_CONFIG_V1.b_out_channels,
                    dtype,
                )
                for _ in range(registered_update_count)
            ]
        )
        self.register_buffer("force_scale", torch.tensor(float(force_scale), dtype=dtype))

    @property
    def b_keys(self) -> tuple[TypeKey, ...]:
        return self.message_blocks[0].b_keys

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _block_at(self, layer_index: int) -> E311MultibodyMessageBlockV1:
        return self.message_blocks[0 if self.config.share_all_layers else layer_index]

    @staticmethod
    def _path_manifest_sha256(block: E311MultibodyMessageBlockV1) -> str:
        records = [dict(record) for record in block.pair_kernel.path_manifest]
        payload = json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _node_update_at(self, layer_index: int) -> TypedNodeUpdateV1:
        return self.node_updates[0 if self.config.share_all_layers else layer_index]

    def reset_running_rms(self) -> None:
        """Reset each registered block once; a shared block owns shared RMS buffers."""

        for block in self.message_blocks:
            block.pair_kernel.reset_running_rms()

    def _validate_geometry(self, centers_world: Tensor, frames_body_to_world: Tensor) -> None:
        if centers_world.ndim < 2 or centers_world.shape[-1] != 3:
            raise ValueError("centers_world must have shape (..., N, 3)")
        expected_frames = centers_world.shape[:-1] + (3, 3)
        if frames_body_to_world.shape != expected_frames:
            raise ValueError("frames_body_to_world must have shape (..., N, 3, 3)")
        if (
            centers_world.dtype != self.force_scale.dtype
            or frames_body_to_world.dtype != centers_world.dtype
        ):
            raise TypeError("geometry must match the model dtype")
        if (
            centers_world.device != self.force_scale.device
            or frames_body_to_world.device != centers_world.device
        ):
            raise ValueError("geometry must be on the model device")

    def _initial_hidden_b(self, centers_world: Tensor) -> Mapping[TypeKey, Tensor]:
        batch_shape = centers_world.shape[:-2]
        node_count = centers_world.shape[-2]
        block = self.message_blocks[0]
        return MappingProxyType(
            {
                key: centers_world.new_zeros(
                    batch_shape
                    + (
                        node_count,
                        block.pair_kernel.hidden_channels[key],
                        self.catalog.resolve(key).representation_dim,
                    )
                )
                for key in self.b_keys
            }
        )

    def _initial_edge_a(self, centers_world: Tensor, edge_count: int) -> Tensor:
        return centers_world.new_zeros(
            centers_world.shape[:-2]
            + (edge_count, LOCKED_MESSAGE_BLOCK_CONFIG_V1.edge_a_channels, 3)
        )

    def core_output(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor | None = None,
        *,
        collect_trace: bool = False,
    ) -> E311GraphCoreOutputV1:
        """Evaluate one fixed-topology graph batch and return scientific states."""

        self._validate_geometry(centers_world, frames_body_to_world)
        node_count = centers_world.shape[-2]
        if pair_index is None:
            pair_index = complete_pair_index_v1(node_count, centers_world.device)
        if pair_index.ndim != 2 or pair_index.shape[0] != 2:
            raise ValueError("pair_index must have shape (2, E)")
        if pair_index.device != centers_world.device:
            raise ValueError("pair_index must be on the graph device")

        hidden_b = self._initial_hidden_b(centers_world)
        edge_a = self._initial_edge_a(centers_world, pair_index.shape[1])
        layer_outputs: list[GraphMessageBlockOutputV1] = []
        hidden_input_to_last = hidden_b

        for layer_index in range(self.config.layer_count):
            if layer_index == self.config.layer_count - 1:
                hidden_input_to_last = hidden_b
            output = self._block_at(layer_index)(
                centers_world,
                frames_body_to_world,
                pair_index,
                hidden_b,
                edge_a,
                None,
                collect_trace=collect_trace,
            )
            layer_outputs.append(output)
            edge_a = output.edge_a_world
            if layer_index + 1 < self.config.layer_count:
                aggregated = self.b_aggregator(output, pair_index, node_count)
                hidden_b = self._node_update_at(layer_index)(hidden_b, aggregated)

        final_output = layer_outputs[-1]
        final_aggregated = self.b_aggregator(final_output, pair_index, node_count)
        return E311GraphCoreOutputV1(
            final_output.node_force_world,
            final_output.pair_force_world,
            final_output.edge_a_world,
            hidden_input_to_last,
            final_aggregated,
            tuple(layer_outputs),
        )

    def normalized_forces_world(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor | None = None,
    ) -> Tensor:
        return self.core_output(
            centers_world, frames_body_to_world, pair_index
        ).normalized_node_force_world

    def forward_world(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor | None = None,
    ) -> Tensor:
        return self.normalized_forces_world(
            centers_world, frames_body_to_world, pair_index
        ) * self.force_scale

    def forward(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor | None = None,
    ) -> Tensor:
        return self.forward_world(centers_world, frames_body_to_world, pair_index)

    def architecture_record(self) -> Mapping[str, Any]:
        """Return the model facts that every experiment ledger must save."""

        return MappingProxyType(
            {
                "core_version": "e311_gnn_12_experiment_core_v1",
                "layer_count": self.config.layer_count,
                "aggregation": self.config.aggregation.value,
                "share_all_layers": self.config.share_all_layers,
                "shared_scope": (
                    "message_block+node_update+running_rms"
                    if self.config.share_all_layers
                    else "none"
                ),
                "registered_message_blocks": len(self.message_blocks),
                "registered_node_updates": len(self.node_updates),
                "node_update": "bias_free_same-TypeKey_concat_project",
                "dtype": str(self.force_scale.dtype),
                "force_scale": float(self.force_scale.detach().cpu().item()),
                "running_rms": (
                    "shared_across_depth"
                    if self.config.share_all_layers
                    else "independent_per_layer"
                ),
                "message_block": asdict(LOCKED_MESSAGE_BLOCK_CONFIG_V1),
                "locked_signature": dict(LOCKED_BLOCK_SIGNATURE_V1),
                "compiled_path_count": len(
                    self.message_blocks[0].pair_kernel.path_manifest
                ),
                "compiled_path_manifest_sha256": self.path_manifest_sha256,
                "b_manifest_stf_ranks": tuple(item.stf_rank for item in self.manifest),
                "b_type_keys": tuple(str(key) for key in self.b_keys),
                "trainable_parameter_count": self.trainable_parameter_count,
            }
        )


def build_experiment_core_v1(
    experiment_id: str,
    force_scale: float,
    dtype: torch.dtype = torch.float32,
) -> E311MessageStackCoreV1:
    """Build only the core architecture declared by one experiment ID."""

    spec = get_experiment_spec_v1(experiment_id)
    return E311MessageStackCoreV1(
        force_scale=force_scale,
        config=E311GraphCoreConfigV1(
            layer_count=spec.layer_count,
            aggregation=spec.aggregation,
            share_all_layers=spec.share_all_layers,
        ),
        dtype=dtype,
    )


__all__ = [
    "BAggregationV1",
    "E311GNNExperimentSpecV1",
    "E311GraphCoreConfigV1",
    "E311GraphCoreOutputV1",
    "E311MessageStackCoreV1",
    "EXPERIMENT_SPECS_V1",
    "LOCKED_BLOCK_SIGNATURE_V1",
    "LOCKED_MESSAGE_BLOCK_CONFIG_V1",
    "ReceiverLocalBAggregatorV1",
    "SupervisionV1",
    "TypedNodeUpdateV1",
    "benzene_generators_v1",
    "build_experiment_core_v1",
    "complete_pair_index_v1",
    "get_experiment_spec_v1",
]
