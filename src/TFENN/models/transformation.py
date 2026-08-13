# References
# Yarotsky, Universal Approximations of Invariant Maps by Neural Networks,
# 2018, Lemma 2.1 and Proposition 2.4.
# https://arxiv.org/abs/1804.10306
# Satorras, Hoogeboom, Welling, E(n) Equivariant Graph Neural Networks,
# ICML 2021, equations 3 and 4, Appendix C.
# https://arxiv.org/abs/2102.09844
# Weiler et al., 3D Steerable CNNs, NeurIPS 2018, section 4.3.
# https://arxiv.org/abs/1807.02547
"""Invariant controlled transformations between registered typed pathways.

The caller supplies a complete ``CovariantCompilation`` produced before model
execution.  Construction registers that fixed artifact as buffers.  Forward
contains only the invariant MLP, fixed lifts, fixed C contractions, and channel
sums.  It never compiles a basis or performs an SVD or nullspace solve.

Every input uses layout ``prefix + (channels, representation_dimension)``.
The result uses layout ``prefix + (out_channels, output_dimension)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from TFENN.tensor_math import (
    CSignature,
    CovariantCompilation,
    RegisteredCovariant,
)


__all__ = ["InvariantGate"]


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "tanh": nn.Tanh,
}


def _activation_name(value: str, name: str) -> str:
    """Return one supported smooth scalar activation name."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    key = value.lower()
    if key not in _ACTIVATIONS:
        choices = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"{name} must be one of {choices}")
    return key


def _signature_descriptor(signature: CSignature) -> tuple[object, ...]:
    """Return primitive metadata for one exact typed signature."""
    output = signature.output
    output_descriptor = (
        type(output).__name__,
        getattr(output, "stream", None),
        getattr(output, "component", None),
        getattr(output, "name", None),
    )
    inputs = tuple(
        (
            slot.name,
            slot.type_key.stream,
            slot.type_key.component,
            slot.power,
            slot.mode,
        )
        for slot in signature.inputs
    )
    return output_descriptor, inputs


def _exact_metadata_equal(left: object, right: object) -> bool:
    """Compare checkpoint metadata without numeric type aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(_exact_metadata_equal(left[key], right[key]) for key in left)
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _exact_metadata_equal(first, second) for first, second in zip(left, right)
        )
    return bool(left == right)


def _positive_int(value: int, name: str) -> int:
    """Return one strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _hidden_widths(
    value: int | Sequence[int],
    num_linear_layers: int | None,
) -> tuple[int, ...]:
    """Return the exact hidden width of every scalar MLP layer."""
    if isinstance(value, bool):
        raise ValueError("hidden_channels must contain positive integers")
    if isinstance(value, int):
        layer_count = (
            2
            if num_linear_layers is None
            else _positive_int(
                num_linear_layers,
                "num_linear_layers",
            )
        )
        width = _positive_int(value, "hidden_channels")
        return (width,) * (layer_count - 1)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("hidden_channels must be an integer or integer sequence")
    widths = tuple(
        _positive_int(width, f"hidden_channels[{index}]")
        for index, width in enumerate(value)
    )
    if num_linear_layers is not None:
        layer_count = _positive_int(num_linear_layers, "num_linear_layers")
        if layer_count != len(widths) + 1:
            raise ValueError("num_linear_layers must equal hidden width count plus one")
    return widths


class InvariantGate(nn.Module):
    """Transform one typed pathway with invariant controlled C coefficients.

    ``num_linear_layers`` counts every linear layer, including the coefficient
    head.  The default is therefore Linear, SiLU, Linear.  The coefficient
    head is unbounded by default so every signed direction in the complete C
    basis remains available.
    """

    def __init__(
        self,
        signature: CSignature,
        artifact: CovariantCompilation,
        input_channels: Mapping[str, int],
        out_channels: int,
        invariant_channels: int,
        *,
        hidden_channels: int | Sequence[int] = 64,
        num_linear_layers: int | None = None,
        activation: str = "silu",
        output_activation: str = "identity",
        use_bias: bool = True,
        check_signature: bool = True,
    ) -> None:
        """Register one offline C artifact and build its invariant coefficient MLP."""
        super().__init__()
        if not isinstance(signature, CSignature):
            raise TypeError("signature must be a CSignature")
        if not isinstance(artifact, CovariantCompilation):
            raise TypeError("artifact must be a CovariantCompilation")
        if not isinstance(check_signature, bool):
            raise TypeError("check_signature must be bool")
        if check_signature and artifact.signature != signature:
            raise ValueError("artifact C basis does not match signature")
        if not isinstance(input_channels, Mapping):
            raise TypeError("input_channels must map slot names to channel counts")
        if not isinstance(use_bias, bool):
            raise TypeError("use_bias must be bool")
        if artifact.basis_dimension <= 0:
            raise ValueError("artifact C basis is empty, so the pathway does not exist")

        runtime_slots = artifact.signature.inputs
        slot_names = tuple(slot.name for slot in runtime_slots)
        actual_names = tuple(input_channels.keys())
        if any(not isinstance(name, str) for name in actual_names):
            raise TypeError("input_channels keys must be slot name strings")
        if len(actual_names) != len(slot_names) or set(actual_names) != set(slot_names):
            raise ValueError(
                "input_channels keys must exactly match artifact slot names"
            )

        self.signature = artifact.signature
        self.check_signature = check_signature
        self._slot_names = slot_names
        self._input_channels = tuple(
            _positive_int(input_channels[name], f"input channel count for {name}")
            for name in slot_names
        )
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.invariant_channels = _positive_int(
            invariant_channels,
            "invariant_channels",
        )
        self.hidden_widths = _hidden_widths(hidden_channels, num_linear_layers)
        self.hidden_channels = (
            self.hidden_widths[0]
            if self.hidden_widths and len(set(self.hidden_widths)) == 1
            else self.hidden_widths
        )
        self.num_linear_layers = len(self.hidden_widths) + 1
        self.basis_dimension = artifact.basis_dimension
        self.output_dimension = artifact.output_dimension
        self.input_dimensions = artifact.input_dimensions
        self.channel_product = math.prod(self._input_channels)
        self.coefficient_count = (
            self.out_channels * self.channel_product * self.basis_dimension
        )
        self.activation = _activation_name(activation, "activation")
        self.output_activation_name = _activation_name(
            output_activation,
            "output_activation",
        )
        self.use_bias = use_bias

        self.covariant = RegisteredCovariant(artifact)
        basis = artifact.basis
        factory_kwargs = {"dtype": basis.dtype, "device": basis.device}
        layers: list[nn.Module] = []
        current_channels = self.invariant_channels
        for hidden_width in self.hidden_widths:
            linear = nn.Linear(
                current_channels,
                hidden_width,
                bias=use_bias,
                **factory_kwargs,
            )
            self._initialize_linear(linear)
            layers.append(linear)
            layers.append(_ACTIVATIONS[self.activation]().to(**factory_kwargs))
            current_channels = hidden_width
        coefficient_head = nn.Linear(
            current_channels,
            self.coefficient_count,
            bias=use_bias,
            **factory_kwargs,
        )
        self._initialize_linear(
            coefficient_head,
            output_scale=1.0 / math.sqrt(self.channel_product * self.basis_dimension),
        )
        layers.append(coefficient_head)
        self.invariant_mlp = nn.Sequential(*layers)
        self.output_activation = _ACTIVATIONS[self.output_activation_name]().to(
            **factory_kwargs
        )
        self.register_buffer(
            "_runtime_reference",
            torch.empty(0, **factory_kwargs),
            persistent=False,
        )
        self._state_metadata = {
            "schema_version": 2,
            "artifact_fingerprint": artifact.artifact_fingerprint,
            "signature": _signature_descriptor(artifact.signature),
            "input_channels": tuple(zip(self._slot_names, self._input_channels)),
            "out_channels": self.out_channels,
            "invariant_channels": self.invariant_channels,
            "hidden_widths": self.hidden_widths,
            "num_linear_layers": self.num_linear_layers,
            "activation": self.activation,
            "output_activation": self.output_activation_name,
            "use_bias": self.use_bias,
        }

    @staticmethod
    def _initialize_linear(linear: nn.Linear, output_scale: float = 1.0) -> None:
        """Initialize a scalar map with fan input variance and zero bias."""
        nn.init.normal_(
            linear.weight,
            mean=0.0,
            std=output_scale / math.sqrt(linear.in_features),
        )
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    @property
    def input_channels(self) -> dict[str, int]:
        """Return channel counts in exact C slot order."""
        return dict(zip(self._slot_names, self._input_channels))

    def get_extra_state(self) -> dict[str, object]:
        """Bind checkpoints to the exact pathway and coefficient axis layout."""
        return dict(self._state_metadata)

    def set_extra_state(self, state: object) -> None:
        """Reject a checkpoint created for another pathway configuration."""
        if not isinstance(state, Mapping) or not _exact_metadata_equal(
            dict(state),
            self._state_metadata,
        ):
            raise RuntimeError("InvariantGate checkpoint configuration does not match")

    def _require_runtime_tensor(self, value: Tensor, name: str) -> Tensor:
        """Require dtype and device agreement with the registered pathway."""
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != self._runtime_reference.dtype:
            raise TypeError(f"{name} dtype must match the registered pathway")
        if value.device != self._runtime_reference.device:
            raise ValueError(f"{name} device must match the registered pathway")
        return value

    def coefficients(self, invariants: Tensor) -> Tensor:
        """Return gamma with output, input channel, and C basis axes."""
        value = self._require_runtime_tensor(invariants, "invariants")
        if value.ndim < 1 or value.shape[-1] != self.invariant_channels:
            raise ValueError(
                "invariants must end with the declared invariant channel count"
            )
        raw = self.invariant_mlp(value)
        activated = self.output_activation(raw).to(
            dtype=self._runtime_reference.dtype,
        )
        return activated.reshape(
            value.shape[:-1]
            + (self.out_channels, *self._input_channels, self.basis_dimension)
        )

    def _expanded_inputs(
        self,
        inputs: Mapping[str, Tensor],
    ) -> tuple[dict[str, Tensor], tuple[int, ...]]:
        """Insert explicit output and input channel broadcast axes."""
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must map slot names to tensors")
        actual_names = tuple(inputs.keys())
        if any(not isinstance(name, str) for name in actual_names):
            raise ValueError("input keys must be slot name strings")
        if len(actual_names) != len(self._slot_names) or set(actual_names) != set(
            self._slot_names
        ):
            raise ValueError("input keys must exactly match artifact slot names")

        prefix: tuple[int, ...] | None = None
        expanded: dict[str, Tensor] = {}
        slot_count = len(self._slot_names)
        for index, (name, channels, dimension) in enumerate(
            zip(self._slot_names, self._input_channels, self.input_dimensions)
        ):
            value = self._require_runtime_tensor(inputs[name], f"input {name}")
            if value.ndim < 2 or value.shape[-2:] != (channels, dimension):
                raise ValueError(
                    f"input {name} must end with shape {(channels, dimension)}"
                )
            value_prefix = tuple(value.shape[:-2])
            if prefix is None:
                prefix = value_prefix
            elif value_prefix != prefix:
                raise ValueError("all input prefix axes must match exactly")
            channel_grid = [1] * (slot_count + 1)
            channel_grid[index + 1] = channels
            expanded[name] = value.reshape(
                value_prefix + tuple(channel_grid) + (dimension,)
            )
        return expanded, () if prefix is None else prefix

    def forward(
        self,
        inputs: Mapping[str, Tensor],
        invariants: Tensor,
    ) -> Tensor:
        """Evaluate the complete invariant controlled typed transformation."""
        expanded, prefix = self._expanded_inputs(inputs)
        coefficients = self.coefficients(invariants)
        if tuple(invariants.shape[:-1]) != prefix:
            raise ValueError("invariant prefix axes must match input prefix axes")
        contracted = self.covariant.apply_coefficients(expanded, coefficients)
        first_channel_axis = len(prefix) + 1
        channel_axes = tuple(
            range(first_channel_axis, first_channel_axis + len(self._slot_names))
        )
        return contracted.sum(dim=channel_axes).to(
            dtype=self._runtime_reference.dtype,
        )
