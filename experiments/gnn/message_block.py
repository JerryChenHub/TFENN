from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from TFENN.tensor_math import (
    CSignature,
    CovariantCompilation,
    RegisteredCovariant,
    TypeCatalog,
    TypeKey,
)


A = TypeKey("A")


def type_name(key: TypeKey) -> str:
    return "A" if key.stream == "A" else f"B_{key.component}"


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _floating_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")


@dataclass(frozen=True)
class SourceSpec:
    type_key: TypeKey
    channels: int

    def __post_init__(self) -> None:
        if not isinstance(self.type_key, TypeKey):
            raise TypeError("type_key must be a TypeKey")
        _positive_integer(self.channels, "channels")


@dataclass(frozen=True)
class PathSpec:
    signature: CSignature
    source_names: tuple[str, ...]
    artifact: CovariantCompilation

    def __post_init__(self) -> None:
        if not isinstance(self.signature, CSignature):
            raise TypeError("signature must be a CSignature")
        if not isinstance(self.artifact, CovariantCompilation):
            raise TypeError("artifact must be a CovariantCompilation")
        names = tuple(self.source_names)
        if len(names) != len(self.signature.inputs):
            raise ValueError("source_names must match the signature input count")
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("source_names must contain nonempty strings")
        if self.artifact.signature != self.signature:
            raise ValueError("artifact signature does not match path signature")
        object.__setattr__(self, "source_names", names)


@dataclass(frozen=True)
class TargetSpec:
    channels: int
    carrier_names: tuple[str, ...]
    paths: tuple[PathSpec, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.channels, "channels")
        carriers = tuple(self.carrier_names)
        paths = tuple(self.paths)
        if any(not isinstance(name, str) or not name for name in carriers):
            raise TypeError("carrier_names must contain nonempty strings")
        if any(not isinstance(path, PathSpec) for path in paths):
            raise TypeError("paths must contain PathSpec values")
        if not carriers and not paths:
            raise ValueError("a target needs at least one carrier or path")
        object.__setattr__(self, "carrier_names", carriers)
        object.__setattr__(self, "paths", paths)


class ChannelProjection(nn.Module):
    """Mix channels while preserving representation coordinates."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        _positive_integer(c_in, "c_in")
        _positive_integer(c_out, "c_out")
        _floating_dtype(dtype)
        self.weight = nn.Parameter(torch.empty(c_out, c_in, dtype=dtype))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(c_in))

    def forward(self, value: Tensor) -> Tensor:
        return torch.einsum("oi,...id->...od", self.weight, value)


class DenseCovariantPath(nn.Module):
    def __init__(
        self,
        spec: PathSpec,
        sources: Mapping[str, SourceSpec],
        target: TypeKey,
        c_out: int,
        gate_width: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        _positive_integer(c_out, "c_out")
        _positive_integer(gate_width, "gate_width")
        _floating_dtype(dtype)
        if spec.signature.output != target:
            raise ValueError("path output does not match target")
        for slot, name in zip(
            spec.signature.inputs,
            spec.source_names,
            strict=True,
        ):
            if name not in sources:
                raise KeyError(f"source {name} is not declared")
            if sources[name].type_key != slot.type_key:
                raise ValueError(f"source {name} has the wrong TypeKey")
        self.source_names = spec.source_names
        self.slot_names = tuple(slot.name for slot in spec.signature.inputs)
        self.source_channels = tuple(
            sources[name].channels for name in spec.source_names
        )
        self.catalog_fingerprint = spec.artifact.type_catalog_fingerprint
        self.operator = RegisteredCovariant(spec.artifact).to(dtype=dtype)
        channel_product = math.prod(self.source_channels)
        self.primitive_count = channel_product * spec.artifact.basis_dimension
        self.c_out = c_out
        self.head = nn.Linear(
            gate_width,
            c_out * self.primitive_count,
            dtype=dtype,
        )

    def forward(self, state: Mapping[str, Tensor], gate: Tensor) -> Tensor:
        values = tuple(state[name] for name in self.source_names)
        leading = values[0].shape[:-2]
        for name, channels, value in zip(
            self.source_names,
            self.source_channels,
            values,
            strict=True,
        ):
            if value.ndim < 2 or value.shape[:-2] != leading:
                raise ValueError(f"source {name} has incompatible leading axes")
            if value.shape[-2] != channels:
                raise ValueError(f"source {name} has the wrong channel count")
        if gate.shape[:-1] != leading:
            raise ValueError("gate and path sources have incompatible leading axes")
        slot_count = len(values)
        inputs = {}
        for index, (slot_name, value) in enumerate(
            zip(self.slot_names, values, strict=True)
        ):
            channel_axes = (
                (1,) * index
                + (value.shape[-2],)
                + (1,) * (slot_count - index - 1)
            )
            inputs[slot_name] = value.reshape(
                leading + channel_axes + (value.shape[-1],)
            )
        primitive = self.operator.evaluate_basis(inputs)
        primitive = primitive.reshape(
            leading + (self.primitive_count, primitive.shape[-1])
        )
        gamma = self.head(gate).reshape(
            leading + (self.c_out, self.primitive_count)
        )
        return torch.einsum("...op,...pd->...od", gamma, primitive)


class TargetUpdate(nn.Module):
    """Update one target TypeKey without untyped mixing."""

    def __init__(
        self,
        target: TypeKey,
        spec: TargetSpec,
        sources: Mapping[str, SourceSpec],
        gate_width: int,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        for name in spec.carrier_names:
            if name not in sources:
                raise KeyError(f"carrier {name} is not declared")
            if sources[name].type_key != target:
                raise ValueError("carrier must match the target TypeKey")
        self.carrier_names = spec.carrier_names
        self.carrier_channels = tuple(
            sources[name].channels for name in spec.carrier_names
        )
        self.paths = nn.ModuleList(
            DenseCovariantPath(
                path,
                sources,
                target,
                spec.channels,
                gate_width,
                dtype,
            )
            for path in spec.paths
        )
        c_in = (
            sum(self.carrier_channels)
            + len(spec.paths) * spec.channels
        )
        self.project = ChannelProjection(c_in, spec.channels, dtype)

    def forward(
        self,
        state: Mapping[str, Tensor],
        gate: Tensor,
    ) -> Tensor:
        pieces = []
        leading = gate.shape[:-1]
        for name, channels in zip(
            self.carrier_names,
            self.carrier_channels,
            strict=True,
        ):
            value = state[name]
            if value.ndim < 2 or value.shape[:-2] != leading:
                raise ValueError(f"carrier {name} has incompatible leading axes")
            if value.shape[-2] != channels:
                raise ValueError(f"carrier {name} has the wrong channel count")
            pieces.append(value)
        pieces.extend(path(state, gate) for path in self.paths)
        return self.project(torch.cat(pieces, dim=-2))


class TypedStage(nn.Module):
    """Use one shared gate trunk and separate typed updates."""

    def __init__(
        self,
        invariant_dim: int,
        sources: Mapping[str, SourceSpec],
        targets: Mapping[TypeKey, TargetSpec],
        gate_width: int = 8,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        _positive_integer(invariant_dim, "invariant_dim")
        _positive_integer(gate_width, "gate_width")
        _floating_dtype(dtype)
        if not targets:
            raise ValueError("a typed stage needs at least one target")
        self.keys = tuple(targets)
        self.trunk = nn.Sequential(
            nn.Linear(invariant_dim, gate_width, dtype=dtype),
            nn.SiLU(),
        )
        self.updates = nn.ModuleDict(
            {
                type_name(key): TargetUpdate(
                    key,
                    targets[key],
                    sources,
                    gate_width,
                    dtype,
                )
                for key in self.keys
            }
        )

    def forward(
        self,
        state: Mapping[str, Tensor],
        invariants: Tensor,
    ) -> dict[TypeKey, Tensor]:
        gate = self.trunk(invariants)
        return {
            key: self.updates[type_name(key)](state, gate)
            for key in self.keys
        }


class ABMessageUpdate(nn.Module):
    """Compute typed edge messages without constructing a graph."""

    def __init__(
        self,
        catalog: TypeCatalog,
        a_mid: TypedStage,
        b_wide: TypedStage,
        b_out: TypedStage,
        a_out: TypedStage,
    ) -> None:
        super().__init__()
        if not isinstance(catalog, TypeCatalog):
            raise TypeError("catalog must be a TypeCatalog")
        self.b_keys = tuple(
            key for key in catalog.blocks if key.stream == "B"
        )
        if not self.b_keys:
            raise ValueError("catalog contains no B TypeKeys")
        if a_mid.keys != (A,) or a_out.keys != (A,):
            raise ValueError("A stages must contain exactly the A target")
        if set(b_wide.keys) != set(self.b_keys):
            raise ValueError("b_wide must contain every catalog B target")
        if set(b_out.keys) != set(self.b_keys):
            raise ValueError("b_out must contain every catalog B target")
        stages = (a_mid, b_wide, b_out, a_out)
        for stage in stages:
            for update in stage.updates.values():
                for path in update.paths:
                    if path.catalog_fingerprint != catalog.fingerprint:
                        raise ValueError("path artifact belongs to another catalog")
        self.a_mid = a_mid
        self.b_wide = b_wide
        self.b_out = b_out
        self.a_out = a_out

    def forward(
        self,
        d: Tensor,
        e: Tensor,
        B_i: Mapping[TypeKey, Tensor],
        B_j: Mapping[TypeKey, Tensor],
        h_i: Mapping[TypeKey, Tensor],
        h_j: Mapping[TypeKey, Tensor],
        invariants: Callable[[str, Mapping[str, Tensor]], Tensor],
    ) -> tuple[Tensor, dict[TypeKey, Tensor]]:
        for name, bank in (
            ("B_i", B_i),
            ("B_j", B_j),
            ("h_i", h_i),
            ("h_j", h_j),
        ):
            if set(bank) != set(self.b_keys):
                raise ValueError(f"{name} must contain every catalog B TypeKey")
        state = {"d": d, "e": e}
        for key in self.b_keys:
            name = type_name(key)
            state[f"Bi_{name}"] = B_i[key]
            state[f"Bj_{name}"] = B_j[key]
            state[f"hi_{name}"] = h_i[key]
            state[f"hj_{name}"] = h_j[key]
        a1 = self.a_mid(state, invariants("a_mid", state))[A]
        state["a1"] = a1
        b1 = self.b_wide(state, invariants("b_wide", state))
        state.update(
            {f"b1_{type_name(key)}": value for key, value in b1.items()}
        )
        b2 = self.b_out(state, invariants("b_out", state))
        state.update(
            {f"b2_{type_name(key)}": value for key, value in b2.items()}
        )
        aout = self.a_out(state, invariants("a_out", state))[A]
        return aout, b2
