"""Exact-E311 pairwise graph controls for the Y13--Y15 study.

Y13 is the unchanged historical E311 model and is launched by the companion
runner. Y14 and Y15 wrap that exact pair kernel with a parameter-free graph
operation: evaluate both endpoint orders in one vectorized kernel call, apply
a world-frame odd projection, and scatter the pair force with opposite signs.

The newer receiver-local multibody MessageBlock is deliberately not used.
E311's own world-to-root-local-to-world evaluation remains unchanged because
removing it would no longer be an E311 control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from experiments.benzene_pair import sweep30 as e_common
from experiments.benzene_pair.e_series.catalog import get_model_spec
from experiments.benzene_pair.e_series.model_factory import (
    build_e_series_model,
    pipeline_config_from_spec,
)


HISTORICAL_E311_PARAMETER_COUNT = 14_926
HISTORICAL_E311_MODEL_ID = "E311"
HISTORICAL_SPLIT_SEED = 20260821
HISTORICAL_MODEL_SEED = 20260822
HISTORICAL_SHUFFLE_SEED = 20260823


@dataclass(frozen=True, slots=True)
class YPairControlSpecV1:
    """One locked causal-control experiment."""

    experiment_id: str
    architecture_name: str
    description: str
    purpose: str
    comparison_role: str
    sample_count: int
    node_count: int
    unordered_edge_count: int
    epochs: int
    graph_batch_size: int
    scheduler_step_size: int
    uses_graph_wrapper: bool
    expected_parameter_count: int = HISTORICAL_E311_PARAMETER_COUNT

    @property
    def model_id(self) -> str:
        """Compatibility with the historical sweep runner."""

        return self.experiment_id

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Y_PAIR_CONTROL_SPECS_V1 = (
    YPairControlSpecV1(
        experiment_id="Y13",
        architecture_name="E311-Exact-400K",
        description="Unchanged historical non-GNN E311 reproduction",
        purpose="Verify that the historical 0.0018-level pair result is reproducible",
        comparison_role="exact_reference",
        sample_count=400_000,
        node_count=2,
        unordered_edge_count=1,
        epochs=500,
        graph_batch_size=10_000,
        scheduler_step_size=125,
        uses_graph_wrapper=False,
    ),
    YPairControlSpecV1(
        experiment_id="Y14",
        architecture_name="E311-OddGraph-400K",
        description=(
            "Exact shared E311 kernel in both directions, world OddPair, "
            "and signed scatter"
        ),
        purpose=(
            "Isolate endpoint symmetrization and graph wrapping "
            "from kernel capacity"
        ),
        comparison_role="graph_causal_control",
        sample_count=400_000,
        node_count=2,
        unordered_edge_count=1,
        epochs=500,
        graph_batch_size=10_000,
        scheduler_step_size=125,
        uses_graph_wrapper=True,
    ),
    YPairControlSpecV1(
        experiment_id="Y15",
        architecture_name="E311-OddGraph-5B100K",
        description=(
            "One shared exact E311 kernel over ten five-benzene edges, "
            "world OddPair, and SUM scatter"
        ),
        purpose="Test a strictly pairwise graph model on five-benzene OPLS data",
        comparison_role="graph_scaling_control",
        sample_count=100_000,
        node_count=5,
        unordered_edge_count=10,
        epochs=200,
        graph_batch_size=1_000,
        scheduler_step_size=50,
        uses_graph_wrapper=True,
    ),
)
_SPEC_LOOKUP = MappingProxyType(
    {spec.experiment_id: spec for spec in Y_PAIR_CONTROL_SPECS_V1}
)


def get_y_pair_control_spec_v1(experiment_id: str) -> YPairControlSpecV1:
    if not isinstance(experiment_id, str):
        raise TypeError("experiment_id must be a string")
    try:
        return _SPEC_LOOKUP[experiment_id.upper()]
    except KeyError as error:
        raise KeyError(f"unknown Y pair control {experiment_id}") from error


def complete_pair_index_v1(
    node_count: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return each unordered edge exactly once in lexicographic order."""

    if isinstance(node_count, bool) or not isinstance(node_count, int):
        raise TypeError("node_count must be an integer")
    if node_count < 2:
        raise ValueError("node_count must be at least two")
    return torch.triu_indices(
        node_count,
        node_count,
        offset=1,
        device=None if device is None else torch.device(device),
    )


def validate_pair_index_v1(
    pair_index: Tensor,
    node_count: int,
    *,
    device: torch.device,
) -> None:
    """Validate the canonical first-receives-from-second edge convention."""

    if not isinstance(pair_index, Tensor):
        raise TypeError("pair_index must be a tensor")
    if pair_index.dtype != torch.int64 or pair_index.ndim != 2:
        raise TypeError("pair_index must be int64 with shape (2, E)")
    if pair_index.shape[0] != 2 or pair_index.shape[1] < 1:
        raise ValueError("pair_index must have shape (2, E) with E positive")
    if pair_index.device != device:
        raise ValueError("pair_index and geometry must be on the same device")
    first, second = pair_index
    if bool(((first < 0) | (second >= node_count)).any()):
        raise ValueError("pair_index contains an out-of-range node")
    if bool((first >= second).any()):
        raise ValueError("pair_index must contain canonical first < second edges")
    encoded = first * node_count + second
    if int(torch.unique(encoded).numel()) != int(encoded.numel()):
        raise ValueError("pair_index contains duplicate unordered edges")


def signed_scatter_pair_force_v1(
    pair_force_world: Tensor,
    pair_index: Tensor,
    node_count: int,
) -> Tensor:
    """SUM pair forces at receivers and their negatives at senders."""

    if pair_force_world.ndim < 2 or pair_force_world.shape[-1] != 3:
        raise ValueError("pair_force_world must have shape (..., E, 3)")
    validate_pair_index_v1(
        pair_index,
        node_count,
        device=pair_force_world.device,
    )
    if pair_force_world.shape[-2] != pair_index.shape[1]:
        raise ValueError("pair force edge count does not match pair_index")
    node_axis = pair_force_world.ndim - 2
    result = pair_force_world.new_zeros(
        pair_force_world.shape[:-2] + (node_count, 3)
    )
    first, second = pair_index
    result = result.index_add(node_axis, first, pair_force_world)
    return result.index_add(node_axis, second, -pair_force_world)


def assert_historical_e311_definition_v1() -> None:
    """Fail closed if the catalog no longer describes the historical E311."""

    spec = get_model_spec(HISTORICAL_E311_MODEL_ID)
    stages = tuple(spec.options["stages"])
    observed = tuple(
        (
            str(stage["name"]),
            str(stage["output_stream"]),
            int(stage["channels"]),
            int(stage["trunk_width"]),
            str(stage["skip_policy"]),
        )
        for stage in stages
    )
    expected = (
        ("a1", "A", 1, 8, "legacy"),
        ("b1", "B", 2, 8, "legacy"),
        ("b2", "B", 1, 8, "legacy"),
        ("out", "A", 1, 8, "legacy"),
    )
    if (
        spec.family != "sequential_gate"
        or spec.architecture_name != "C17"
        or spec.options.get("path_policy") != "NO_RAW_MIXED"
        or spec.planned_parameter_count != HISTORICAL_E311_PARAMETER_COUNT
        or observed != expected
    ):
        raise RuntimeError("the historical E311 catalog definition changed")
    config = pipeline_config_from_spec(spec)
    if (
        config.output_stage != "out"
        or config.architecture_id != "benzene_pair_e_series_e311"
        or config.anchor_ranks != (2, 6)
        or config.max_constraint_entries != 10_000_000
        or config.max_gate_coefficients != 2_000_000
        or config.max_invariant_channels != 20_000
        or config.degree3_overflow_policy != "raise"
        or config.implemented_mechanism != "C17"
        or config.radial.as_dict()
        != {
            "distance_scale": 6.0,
            "rbf_centers": [0.0, 0.5, 1.0, 1.5, 2.0],
            "rbf_width": 0.4,
            "inverse_powers": [1, 2, 3],
            "distance_epsilon": 1.0e-12,
            "rms_epsilon": 1.0e-8,
        }
    ):
        raise RuntimeError("E311 pipeline or radial definition changed")
    expected_sources = (
        ("x", "r"),
        ("x", "r", "a1"),
        ("x", "r", "a1", "b1"),
        ("x", "r", "a1", "b1", "b2"),
    )
    for stage, sources in zip(config.stages, expected_sources, strict=True):
        if (
            stage.source_names != sources
            or stage.invariant_source_names != sources
            or stage.skip_source_names is not None
            or stage.activation != "silu"
            or not stage.include_symmetric_unary
            or not stage.include_raw_mixed_pairs
            or not stage.include_stf_shortcuts
            or not stage.covariant_include_symmetric_unary
            or stage.coefficient_activation != "identity"
            or stage.coefficient_head != "dense"
            or stage.descriptor_mask != "full"
            or stage.covariant_include_raw_mixed_pairs
            or not stage.invariant_include_raw_mixed_pairs
            or not stage.covariant_include_stf_shortcuts
            or not stage.invariant_include_symmetric_unary
            or not stage.invariant_include_stf_shortcuts
            or stage.degree3_policy != "none"
            or stage.trunk_depth != 1
            or stage.trunk_linearized
            or stage.trunk_residual
            or stage.metric_gate != "none"
            or stage.execution_level is not None
            or stage.covariant_live_mixed_only
            or stage.covariant_path_quota is not None
            or stage.covariant_required_source_names is not None
            or stage.parameter_share_group is not None
            or stage.channel_projection != "dense"
            or stage.channel_projection_rank is not None
            or stage.path_aggregation != "linear"
            or stage.path_temperature != 1.0
            or stage.type_channel_overrides
            or stage.reversible_coupling
        ):
            raise RuntimeError(f"E311 stage semantics changed at {stage.name}")


def _parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def build_historical_e311_kernel_v1(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = HISTORICAL_MODEL_SEED,
) -> nn.Module:
    """Build exact catalog E311 and enforce its frozen identity."""

    assert_historical_e311_definition_v1()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        kernel = build_e_series_model(
            HISTORICAL_E311_MODEL_ID,
            e_common._proper_d6_generators(),
            generator_names=("sixfold", "twofold"),
        )
    kernel = kernel.to(device=torch.device(device), dtype=dtype)
    actual = _parameter_count(kernel)
    if actual != HISTORICAL_E311_PARAMETER_COUNT:
        raise RuntimeError(
            f"E311 compiled with {actual} trainable parameters; "
            f"expected {HISTORICAL_E311_PARAMETER_COUNT}"
        )
    return kernel


@dataclass(frozen=True, slots=True)
class E311OddGraphOutputV1:
    """All normalized outputs needed for training and diagnosis."""

    normalized_pair_force_world: Tensor
    normalized_node_force_world: Tensor
    raw_forward_world: Tensor
    raw_reverse_world: Tensor


class E311OddGraphCoreV1(nn.Module):
    """Parameter-free permutation-equivariant graph wrapper around E311."""

    def __init__(self, pair_kernel: nn.Module) -> None:
        super().__init__()
        if not isinstance(pair_kernel, nn.Module):
            raise TypeError("pair_kernel must be a torch module")
        self.pair_kernel = pair_kernel
        kernel_count = _parameter_count(pair_kernel)
        adapter_count = _parameter_count(self)
        if (
            kernel_count != HISTORICAL_E311_PARAMETER_COUNT
            or adapter_count != kernel_count
        ):
            raise RuntimeError(
                "the graph wrapper must contain exactly one 14,926-parameter E311"
            )

    @property
    def trainable_parameter_count(self) -> int:
        return _parameter_count(self)

    @property
    def candidate_manifest(self) -> tuple[Any, ...]:
        return tuple(getattr(self.pair_kernel, "candidate_manifest", ()))

    @property
    def architecture_metadata(self) -> Mapping[str, Any]:
        kernel_metadata = getattr(self.pair_kernel, "architecture_metadata", {})
        return {
            "model_family": "e311_odd_pair_graph_control_v1",
            "pair_kernel": dict(kernel_metadata)
            if isinstance(kernel_metadata, Mapping)
            else kernel_metadata,
            "trainable_parameter_count": self.trainable_parameter_count,
            "ordered_calls_per_unordered_edge": 2,
            "ordered_evaluations_share_one_runtime_call": True,
            "odd_projection_frame": "world",
            "aggregation": "signed_sum",
            "uses_receiver_local_multibody_message_block": False,
            "has_hidden_node_state": False,
        }

    def reset_normalization_stats(self) -> None:
        self.pair_kernel.reset_normalization_stats()

    def normalization_state_dict(self) -> dict[str, Tensor]:
        return self.pair_kernel.normalization_state_dict()

    def load_normalization_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.pair_kernel.load_normalization_state_dict(state)

    def descriptor_projection_state_dict(self) -> dict[str, Tensor]:
        return self.pair_kernel.descriptor_projection_state_dict()

    def load_descriptor_projection_state_dict(
        self,
        state: Mapping[str, Tensor],
    ) -> None:
        self.pair_kernel.load_descriptor_projection_state_dict(state)

    def core_output(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor | None = None,
    ) -> E311OddGraphOutputV1:
        if (
            centers_world.ndim < 2
            or centers_world.shape[-1] != 3
            or not torch.is_floating_point(centers_world)
        ):
            raise ValueError("centers_world must have shape (..., N, 3)")
        expected_frames = centers_world.shape + (3,)
        if frames_body_to_world.shape != expected_frames:
            raise ValueError(
                "frames_body_to_world must have shape (..., N, 3, 3)"
            )
        if (
            frames_body_to_world.device != centers_world.device
            or frames_body_to_world.dtype != centers_world.dtype
        ):
            raise ValueError("centers and frames must share device and dtype")
        node_count = int(centers_world.shape[-2])
        edges = (
            complete_pair_index_v1(node_count, device=centers_world.device)
            if pair_index is None
            else pair_index
        )
        validate_pair_index_v1(edges, node_count, device=centers_world.device)
        first, second = edges

        first_center = centers_world.index_select(-2, first)
        second_center = centers_world.index_select(-2, second)
        forward_centers = torch.stack((first_center, second_center), dim=-2)
        reverse_centers = torch.stack((second_center, first_center), dim=-2)

        first_frame = frames_body_to_world.index_select(-3, first)
        second_frame = frames_body_to_world.index_select(-3, second)
        forward_frames = torch.stack((first_frame, second_frame), dim=-3)
        reverse_frames = torch.stack((second_frame, first_frame), dim=-3)

        # Both directions enter one call, so every RunningRMS sees one frozen
        # pre-update snapshot rather than a direction-dependent call order.
        ordered_centers = torch.cat((forward_centers, reverse_centers), dim=-3)
        ordered_frames = torch.cat((forward_frames, reverse_frames), dim=-4)
        directed = self.pair_kernel(ordered_centers, ordered_frames)
        edge_count = int(edges.shape[1])
        raw_forward, raw_reverse = directed.split(edge_count, dim=-2)
        pair_force = 0.5 * (raw_forward - raw_reverse)
        node_force = signed_scatter_pair_force_v1(
            pair_force,
            edges,
            node_count,
        )
        return E311OddGraphOutputV1(
            normalized_pair_force_world=pair_force,
            normalized_node_force_world=node_force,
            raw_forward_world=raw_forward,
            raw_reverse_world=raw_reverse,
        )

    def forward(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
        pair_index: Tensor | None = None,
    ) -> Tensor:
        """Return normalized pair forces; core_output also exposes node forces."""

        return self.core_output(
            centers_world,
            frames_body_to_world,
            pair_index,
        ).normalized_pair_force_world


class E311TwoNodeOddControlV1(nn.Module):
    """Two-node adapter matching the historical runner's batch-by-three API."""

    def __init__(self, graph_core: E311OddGraphCoreV1) -> None:
        super().__init__()
        if not isinstance(graph_core, E311OddGraphCoreV1):
            raise TypeError("graph_core must be E311OddGraphCoreV1")
        self.graph_core = graph_core
        self.register_buffer(
            "_pair_index",
            torch.tensor(((0,), (1,)), dtype=torch.int64),
            persistent=False,
        )
        if _parameter_count(self) != HISTORICAL_E311_PARAMETER_COUNT:
            raise RuntimeError("the two-node adapter added trainable parameters")

    @property
    def trainable_parameter_count(self) -> int:
        return _parameter_count(self)

    @property
    def pair_kernel(self) -> nn.Module:
        return self.graph_core.pair_kernel

    @property
    def candidate_manifest(self) -> tuple[Any, ...]:
        return self.graph_core.candidate_manifest

    @property
    def architecture_metadata(self) -> Mapping[str, Any]:
        return self.graph_core.architecture_metadata

    def reset_normalization_stats(self) -> None:
        self.graph_core.reset_normalization_stats()

    def normalization_state_dict(self) -> dict[str, Tensor]:
        return self.graph_core.normalization_state_dict()

    def load_normalization_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.graph_core.load_normalization_state_dict(state)

    def descriptor_projection_state_dict(self) -> dict[str, Tensor]:
        return self.graph_core.descriptor_projection_state_dict()

    def load_descriptor_projection_state_dict(
        self,
        state: Mapping[str, Tensor],
    ) -> None:
        self.graph_core.load_descriptor_projection_state_dict(state)

    def core_output(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
    ) -> E311OddGraphOutputV1:
        if centers_world.shape[-2] != 2:
            raise ValueError("Y14 requires exactly two nodes")
        return self.graph_core.core_output(
            centers_world,
            frames_body_to_world,
            self._pair_index,
        )

    def forward(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
    ) -> Tensor:
        return self.core_output(
            centers_world,
            frames_body_to_world,
        ).normalized_pair_force_world[..., 0, :]

    def forward_local(
        self,
        centers_world: Tensor,
        frames_body_to_world: Tensor,
    ) -> Tensor:
        """Return the OddPair prediction in endpoint zero's body frame."""

        world = self.forward(centers_world, frames_body_to_world)
        return torch.einsum(
            "...ji,...j->...i",
            frames_body_to_world[..., 0, :, :],
            world,
        )


def build_e311_odd_graph_core_v1(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = HISTORICAL_MODEL_SEED,
) -> E311OddGraphCoreV1:
    return E311OddGraphCoreV1(
        build_historical_e311_kernel_v1(
            dtype=dtype,
            device=device,
            seed=seed,
        )
    )


def build_y14_two_node_control_v1(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = HISTORICAL_MODEL_SEED,
) -> E311TwoNodeOddControlV1:
    return E311TwoNodeOddControlV1(
        build_e311_odd_graph_core_v1(
            dtype=dtype,
            device=device,
            seed=seed,
        )
    )


__all__ = [
    "E311OddGraphCoreV1",
    "E311OddGraphOutputV1",
    "E311TwoNodeOddControlV1",
    "HISTORICAL_E311_MODEL_ID",
    "HISTORICAL_E311_PARAMETER_COUNT",
    "HISTORICAL_MODEL_SEED",
    "HISTORICAL_SHUFFLE_SEED",
    "HISTORICAL_SPLIT_SEED",
    "YPairControlSpecV1",
    "Y_PAIR_CONTROL_SPECS_V1",
    "assert_historical_e311_definition_v1",
    "build_e311_odd_graph_core_v1",
    "build_historical_e311_kernel_v1",
    "build_y14_two_node_control_v1",
    "complete_pair_index_v1",
    "get_y_pair_control_spec_v1",
    "signed_scatter_pair_force_v1",
    "validate_pair_index_v1",
]
