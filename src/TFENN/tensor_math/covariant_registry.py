"""Compile and run generator defined typed covariant map registries.

This file owns the molecule independent typed catalog, complete offline
covariant compiler, immutable artifact registry, basis views, runtime module,
and PoseEncoder layout adapter.  Its major public APIs are
``build_type_catalog``, ``compile_covariant_basis``, ``CRegistry``,
``RegisteredCovariant``, ``CovariantBasisView``, and
``encode_typed_blocks``.

For every ordered supplied generator ``g_q`` and declared signature, the
compiler constructs the source action in slot order and solves

``C S_q = T_q C``.

It calls ``compile_intertwiners`` exactly once and preserves the complete
canonical basis of the resulting Hom space.  No group closure, group average,
irrep table, or Clebsch Gordon database is used.  A scalar target has action
one and output dimension one.  It is a descriptor target rather than a hidden
feature stream.

Every representation coordinate is the final tensor axis.  Runtime inputs
have shape ``(..., d_slot)`` and may broadcast only their leading axes.  The
caller requests channel Cartesian products by inserting explicit singleton
axes.  ``evaluate_basis`` returns ``(..., n_basis, d_out)`` and
``apply_coefficients`` returns ``(..., d_out)``.  ``evaluate_view`` returns
``(..., n_active, d_out)``.

A repeated variable uses the normalized occupation basis
``U[d**p, comb(d+p-1,p)]``.  Occupations are in descending lexicographic
order and tensor words use left major flattening.  Thus
``Sym^p rho = U.T rho_tensor_power U``.  Distinct slots always have power one.
The initial dense compiler supports at most two slots and total polynomial
degree three.  Larger structured compilers belong to a later module.  A guard
checks all predicted dense allocation sizes before constructing Kronecker
actions or constraints.

Compilation detaches inputs and stores CPU float64 tensors.  Runtime modules
register basis, lifts, and enabled views as persistent buffers, contain no
Parameters, preserve buffer dtype and device, and preserve gradients with
respect to live inputs and coefficients.  Compilation never occurs in
forward.  Determinism is limited by the existing SVD backend and its recorded
singular diagnostics.  Canonical JSON serialization validates every layered
SHA256 fingerprint and is byte deterministic for equal registries.

TypeError reports invalid Python types and unsupported tensor dtypes.
ValueError reports malformed shapes, metadata, signatures, fingerprints,
generator provenance, runtime keys, dtype or device mismatches, incomplete
view use without opt in, and allocation guard failures.  RuntimeError reports
obsolete or tampered serialized artifacts and state dictionaries.

The representation construction follows Finzi, Welling, and Wilson,
``A Practical Method for Constructing Equivariant Multilayer Perceptrons for
Arbitrary Matrix Groups``.  Typed homogeneous space semantics follow Cohen,
Geiger, and Weiler, ``A General Theory of Equivariant CNNs on Homogeneous
Spaces``.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from .anchor_compiler import AnchorCompilation
from .intertwiner_compiler import (
    IntertwinerCompilation,
    compile_intertwiners,
    tensor_product_representation,
)
from .pose_encoding import PoseEncoder
from .stf_rep import stf_representation, validate_rotation
from .stf_space import MAX_STF_RANK, TENSOR_CONVENTION_VERSION


__all__ = [
    "BBlockManifest",
    "COVARIANT_CONVENTION_VERSION",
    "CRegistry",
    "CSignature",
    "CSlot",
    "CovariantBasisView",
    "CovariantCompilation",
    "CovariantCost",
    "GeneratorSystem",
    "POSE_TYPED_BLOCK_LAYOUT",
    "PoseTypedBlockAdapter",
    "RegisteredCovariant",
    "SYMMETRIC_POWER_CONVENTION_VERSION",
    "SlotLiftArtifact",
    "TRIVIAL_SCALAR",
    "TrivialScalarKey",
    "TypeBlock",
    "TypeCatalog",
    "TypeKey",
    "build_primitive_b_manifest",
    "build_type_catalog",
    "compile_covariant_basis",
    "encode_typed_blocks",
    "normalized_symmetric_power_basis",
    "normalized_symmetric_power_representation",
]


SYMMETRIC_POWER_CONVENTION_VERSION = (
    "normalized_symmetric_power_descending_lex_left_major_v1"
)
POSE_TYPED_BLOCK_LAYOUT = "leading_anchor_channel_stf_component_v1"
COVARIANT_CONVENTION_VERSION = (
    f"{TENSOR_CONVENTION_VERSION}_covariant_registry_v1_"
    f"{SYMMETRIC_POWER_CONVENTION_VERSION}_signature_order"
)
REGISTRY_SCHEMA_VERSION = 1
REGISTERED_COVARIANT_STATE_SCHEMA_VERSION = 1
MAX_COVARIANT_SLOTS = 2
MAX_COVARIANT_TOTAL_DEGREE = 3
DEFAULT_MAX_CONSTRAINT_ENTRIES = 1_000_000
MAX_SYMMETRIC_BASIS_ENTRIES = 1_000_000


def _freeze_cpu64_tensor(value: Tensor, name: str, ndim: int | None = None) -> Tensor:
    """Return a finite detached canonical CPU float64 tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; actual shape {tuple(value.shape)}")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    if result.numel():
        result[result == 0.0] = 0.0
    return result


def _freeze_metadata_value(value: Any, path: str = "metadata") -> Any:
    """Recursively freeze one canonical metadata tree."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} floating values must be finite")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_metadata_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be nonempty strings")
            frozen[key] = _freeze_metadata_value(item, f"{path}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _plain_metadata(value: Any) -> Any:
    """Convert frozen metadata to canonical JSON values."""
    if isinstance(value, Mapping):
        return {key: _plain_metadata(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_plain_metadata(item) for item in value]
    return value


def _exact_metadata_equal(left: Any, right: Any) -> bool:
    """Compare nested state metadata with exact container and scalar types."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(_exact_metadata_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _exact_metadata_equal(first, second)
            for first, second in zip(left, right)
        )
    return bool(left == right)


def _tensor_payload(value: Tensor) -> dict[str, Any]:
    """Encode one canonical CPU float64 tensor."""
    tensor = _freeze_cpu64_tensor(value, "serialized tensor")
    array = tensor.numpy().astype("<f8", copy=False)
    return {
        "dtype": "float64_le",
        "shape": list(tensor.shape),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _tensor_from_payload(payload: Any, name: str) -> Tensor:
    """Decode and validate one canonical tensor payload."""
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{name} tensor payload must be a mapping")
    if payload.get("dtype") != "float64_le":
        raise RuntimeError(f"{name} tensor dtype is obsolete")
    shape_value = payload.get("shape")
    if not isinstance(shape_value, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape_value
    ):
        raise RuntimeError(f"{name} tensor shape is malformed")
    try:
        raw = base64.b64decode(payload.get("data", ""), validate=True)
    except Exception as error:
        raise RuntimeError(f"{name} tensor data is malformed") from error
    expected_bytes = math.prod(shape_value) * 8
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"{name} tensor byte count {len(raw)} does not match {expected_bytes}"
        )
    array = np.frombuffer(raw, dtype="<f8").copy().reshape(tuple(shape_value))
    return _freeze_cpu64_tensor(torch.from_numpy(array), name)


def _float_text(value: float, name: str, *, allow_infinite: bool = True) -> str:
    """Encode one diagnostic float without JSON ambiguity."""
    number = float(value)
    if math.isnan(number) or (not allow_infinite and not math.isfinite(number)):
        raise ValueError(f"{name} is not a valid diagnostic value")
    return number.hex()


def _float_from_text(value: Any, name: str, *, allow_infinite: bool = True) -> float:
    """Decode one diagnostic float."""
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a hexadecimal float string")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise RuntimeError(f"{name} is malformed") from error
    if math.isnan(result) or (not allow_infinite and not math.isfinite(result)):
        raise RuntimeError(f"{name} is not a valid diagnostic value")
    return result


def _canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic canonical JSON bytes."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf8")


def _reject_json_constant(value: str) -> None:
    """Reject nonfinite constants accepted by the default JSON parser."""
    raise ValueError(f"nonfinite JSON constant {value} is unsupported")


def _fingerprint(tag: str, payload: Any) -> str:
    """Return one versioned SHA256 fingerprint."""
    digest = hashlib.sha256()
    digest.update(tag.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(payload))
    return digest.hexdigest()


def _validate_fingerprint(value: str, name: str) -> None:
    """Require a lowercase SHA256 hexadecimal string."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")


@dataclass(frozen=True, slots=True)
class TypeKey:
    """Identify one A carrier or one explicit B component."""

    stream: Literal["A", "B"]
    component: int | None = None

    def __post_init__(self) -> None:
        if self.stream not in ("A", "B"):
            raise ValueError("TypeKey stream must be A or B")
        if self.stream == "A" and self.component is not None:
            raise ValueError("A TypeKey component must be None")
        if self.stream == "B" and (
            isinstance(self.component, bool)
            or not isinstance(self.component, int)
            or self.component < 0
        ):
            raise ValueError("B TypeKey component must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class TrivialScalarKey:
    """Identify the scalar descriptor target without creating a stream."""

    name: Literal["trivial_scalar"] = "trivial_scalar"

    def __post_init__(self) -> None:
        if self.name != "trivial_scalar":
            raise ValueError("TrivialScalarKey name must be trivial_scalar")


TRIVIAL_SCALAR = TrivialScalarKey()


def _type_key_payload(key: TypeKey) -> dict[str, Any]:
    return {"stream": key.stream, "component": key.component}


def _type_key_from_payload(payload: Any) -> TypeKey:
    if not isinstance(payload, Mapping):
        raise RuntimeError("type key payload must be a mapping")
    return TypeKey(payload.get("stream"), payload.get("component"))


def _type_sort_key(key: TypeKey) -> tuple[int, int]:
    return (0, 0) if key.stream == "A" else (1, int(key.component))


def _generator_system_payload(
    names: tuple[str, ...],
    matrices: Tensor,
    convention_version: str,
) -> dict[str, Any]:
    return {
        "names": list(names),
        "matrices": _tensor_payload(matrices),
        "convention_version": convention_version,
    }


@dataclass(frozen=True, slots=True, eq=False, init=False)
class GeneratorSystem:
    """Store one ordered proper rotation generator system."""

    names: tuple[str, ...]
    _matrices: Tensor = field(repr=False, compare=False)
    fingerprint: str
    convention_version: str

    def __init__(
        self,
        names: Sequence[str],
        matrices: Tensor,
        *,
        convention_version: str = TENSOR_CONVENTION_VERSION,
        expected_fingerprint: str | None = None,
    ) -> None:
        resolved_names = tuple(names)
        if any(not isinstance(name, str) or not name for name in resolved_names):
            raise TypeError("generator names must be nonempty strings")
        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError("generator names must be unique")
        if not isinstance(matrices, Tensor):
            raise TypeError("generator matrices must be a torch.Tensor")
        source = matrices.unsqueeze(0) if matrices.ndim == 2 else matrices
        if source.ndim != 3 or source.shape[1:] != (3, 3):
            raise ValueError(
                "generator matrices must have shape (generator_count, 3, 3)"
            )
        if source.shape[0] != len(resolved_names):
            raise ValueError("generator name count must match matrix count")
        if source.dtype not in (torch.float32, torch.float64):
            raise TypeError("generator matrices must use float32 or float64")
        tolerance = 1e-6 if source.dtype == torch.float32 else 1e-10
        validate_rotation(
            source,
            rotation_atol=tolerance,
            rotation_rtol=tolerance,
        )
        if convention_version != TENSOR_CONVENTION_VERSION:
            raise ValueError("generator system uses an unsupported convention version")
        frozen = _freeze_cpu64_tensor(source, "generator matrices", ndim=3)
        payload = _generator_system_payload(
            resolved_names,
            frozen,
            convention_version,
        )
        fingerprint = _fingerprint("tfenn_generator_system_v1", payload)
        if expected_fingerprint is not None and expected_fingerprint != fingerprint:
            raise RuntimeError("generator system fingerprint validation failed")
        object.__setattr__(self, "names", resolved_names)
        object.__setattr__(self, "_matrices", frozen)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "convention_version", convention_version)

    @property
    def matrices(self) -> Tensor:
        """Return a detached clone of ordered generator matrices."""
        _validate_generator_system_integrity(self)
        return self._matrices.detach().clone()


def _validate_generator_system_integrity(system: GeneratorSystem) -> None:
    """Reject private generator tensor mutation at every trust boundary."""
    expected = _fingerprint(
        "tfenn_generator_system_v1",
        _generator_system_payload(
            system.names,
            system._matrices,
            system.convention_version,
        ),
    )
    if expected != system.fingerprint:
        raise RuntimeError("generator system content does not match fingerprint")


@dataclass(frozen=True, slots=True)
class BBlockManifest:
    """Declare one stable B component and its Pose anchor channel mapping."""

    component: int
    stf_rank: int
    anchor_columns: tuple[int, ...]
    anchor_multiplicity: int
    stable_component_id: str
    source_layout: str = "stf_component_anchor_channel"
    typed_layout: str = POSE_TYPED_BLOCK_LAYOUT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        TypeKey("B", self.component)
        if (
            isinstance(self.stf_rank, bool)
            or not isinstance(self.stf_rank, int)
            or not 1 <= self.stf_rank <= MAX_STF_RANK
        ):
            raise ValueError(f"stf_rank must be between one and {MAX_STF_RANK}")
        columns = tuple(self.anchor_columns)
        if not columns or any(
            isinstance(column, bool) or not isinstance(column, int) or column < 0
            for column in columns
        ):
            raise ValueError("anchor_columns must contain nonnegative integers")
        if len(set(columns)) != len(columns):
            raise ValueError("anchor_columns must be unique")
        if (
            isinstance(self.anchor_multiplicity, bool)
            or not isinstance(self.anchor_multiplicity, int)
            or self.anchor_multiplicity <= 0
        ):
            raise ValueError("anchor_multiplicity must be a positive integer")
        if max(columns) >= self.anchor_multiplicity:
            raise ValueError("anchor_columns exceed declared anchor_multiplicity")
        if not isinstance(self.stable_component_id, str) or not self.stable_component_id:
            raise TypeError("stable_component_id must be a nonempty string")
        if self.source_layout != "stf_component_anchor_channel":
            raise ValueError("source_layout is unsupported")
        if self.typed_layout != POSE_TYPED_BLOCK_LAYOUT:
            raise ValueError("typed_layout is unsupported")
        object.__setattr__(self, "anchor_columns", columns)
        object.__setattr__(self, "metadata", _freeze_metadata_value(self.metadata))

    @property
    def key(self) -> TypeKey:
        """Return the stable typed key declared by this manifest."""
        return TypeKey("B", self.component)


def build_primitive_b_manifest(
    compilation: AnchorCompilation,
) -> tuple[BBlockManifest, ...]:
    """Build typed B blocks from every discovered primitive anchor rank."""
    if not isinstance(compilation, AnchorCompilation):
        raise TypeError("compilation must be an AnchorCompilation")
    manifests: list[BBlockManifest] = []
    for component, rank in enumerate(compilation.output_ranks):
        multiplicity = compilation.dimensions[rank].primitive
        if multiplicity <= 0:
            raise RuntimeError("discovered primitive rank has no anchor channels")
        manifests.append(
            BBlockManifest(
                component=component,
                stf_rank=rank,
                anchor_columns=tuple(range(multiplicity)),
                anchor_multiplicity=multiplicity,
                stable_component_id=f"primitive_rank_{rank}",
                metadata={
                    "source": "AnchorCompilation",
                    "requested_output_ranks": compilation.requested_output_ranks,
                },
            )
        )
    return tuple(manifests)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class TypeBlock:
    """Store one frozen representation block for an ordered generator system."""

    key: TypeKey
    _actions: Tensor = field(repr=False, compare=False)
    representation_dim: int
    generator_fingerprint: str
    coordinate_convention: str
    metadata: Mapping[str, Any]

    def __init__(
        self,
        key: TypeKey,
        actions: Tensor,
        representation_dim: int,
        generator_fingerprint: str,
        coordinate_convention: str,
        metadata: Mapping[str, Any],
    ) -> None:
        if not isinstance(key, TypeKey):
            raise TypeError("TypeBlock key must be a TypeKey")
        if (
            isinstance(representation_dim, bool)
            or not isinstance(representation_dim, int)
            or representation_dim <= 0
        ):
            raise ValueError("representation_dim must be a positive integer")
        _validate_fingerprint(generator_fingerprint, "generator_fingerprint")
        if not isinstance(coordinate_convention, str) or not coordinate_convention:
            raise TypeError("coordinate_convention must be a nonempty string")
        frozen = _freeze_cpu64_tensor(actions, "TypeBlock actions", ndim=3)
        if frozen.shape[-2:] != (representation_dim, representation_dim):
            raise ValueError(
                "TypeBlock action shape does not match representation_dim"
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "_actions", frozen)
        object.__setattr__(self, "representation_dim", representation_dim)
        object.__setattr__(self, "generator_fingerprint", generator_fingerprint)
        object.__setattr__(self, "coordinate_convention", coordinate_convention)
        object.__setattr__(self, "metadata", _freeze_metadata_value(metadata))

    @property
    def actions(self) -> Tensor:
        """Return a detached clone of generator actions."""
        return self._actions.detach().clone()


def _manifest_metadata(manifest: BBlockManifest) -> Mapping[str, Any]:
    return {
        "stable_component_id": manifest.stable_component_id,
        "stf_rank": manifest.stf_rank,
        "anchor_columns": manifest.anchor_columns,
        "anchor_multiplicity": manifest.anchor_multiplicity,
        "source_layout": manifest.source_layout,
        "typed_layout": manifest.typed_layout,
        "user_metadata": manifest.metadata,
    }


def _type_block_payload(block: TypeBlock) -> dict[str, Any]:
    return {
        "key": _type_key_payload(block.key),
        "actions": _tensor_payload(block._actions),
        "representation_dim": block.representation_dim,
        "generator_fingerprint": block.generator_fingerprint,
        "coordinate_convention": block.coordinate_convention,
        "metadata": _plain_metadata(block.metadata),
    }


def _type_catalog_payload(
    generator_system: GeneratorSystem,
    blocks: Mapping[TypeKey, TypeBlock],
) -> dict[str, Any]:
    ordered = sorted(blocks.items(), key=lambda item: _type_sort_key(item[0]))
    return {
        "generator_fingerprint": generator_system.fingerprint,
        "blocks": [_type_block_payload(block) for _, block in ordered],
        "convention_version": COVARIANT_CONVENTION_VERSION,
    }


@dataclass(frozen=True, slots=True, eq=False, init=False)
class TypeCatalog:
    """Store a complete immutable typed carrier catalog."""

    generator_system: GeneratorSystem
    _blocks: Mapping[TypeKey, TypeBlock] = field(repr=False, compare=False)
    fingerprint: str

    def __init__(
        self,
        generator_system: GeneratorSystem,
        blocks: Mapping[TypeKey, TypeBlock],
        *,
        expected_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(generator_system, GeneratorSystem):
            raise TypeError("generator_system must be a GeneratorSystem")
        _validate_generator_system_integrity(generator_system)
        if not isinstance(blocks, Mapping):
            raise TypeError("blocks must be a mapping")
        copied: dict[TypeKey, TypeBlock] = {}
        for key, block in blocks.items():
            if not isinstance(key, TypeKey) or not isinstance(block, TypeBlock):
                raise TypeError("catalog blocks must map TypeKey to TypeBlock")
            if key != block.key:
                raise ValueError("catalog mapping key does not match TypeBlock key")
            if block.generator_fingerprint != generator_system.fingerprint:
                raise ValueError(
                    "TypeBlock generator fingerprint does not match GeneratorSystem"
                )
            if block._actions.shape[0] != len(generator_system.names):
                raise ValueError("TypeBlock generator count does not match system")
            copied[key] = block
        a_key = TypeKey("A")
        if a_key not in copied:
            raise ValueError("TypeCatalog must contain the A block")
        a_block = copied[a_key]
        if a_block.representation_dim != 3 or not torch.equal(
            a_block._actions, generator_system._matrices
        ):
            raise ValueError("A block must use the defining generator actions")
        for key, block in copied.items():
            if key.stream != "B":
                continue
            metadata = block.metadata
            required = {
                "stable_component_id",
                "stf_rank",
                "anchor_columns",
                "anchor_multiplicity",
                "source_layout",
                "typed_layout",
                "user_metadata",
            }
            if set(metadata) != required:
                raise ValueError("B TypeBlock metadata does not match manifest schema")
            manifest = BBlockManifest(
                component=int(key.component),
                stf_rank=metadata["stf_rank"],
                anchor_columns=tuple(metadata["anchor_columns"]),
                anchor_multiplicity=metadata["anchor_multiplicity"],
                stable_component_id=metadata["stable_component_id"],
                source_layout=metadata["source_layout"],
                typed_layout=metadata["typed_layout"],
                metadata=metadata["user_metadata"],
            )
            expected_dim = 2 * manifest.stf_rank + 1
            if block.representation_dim != expected_dim:
                raise ValueError("B TypeBlock dimension does not match manifest rank")
            expected_actions = stf_representation(
                generator_system._matrices,
                manifest.stf_rank,
            )
            if not torch.equal(block._actions, expected_actions):
                raise ValueError("B TypeBlock actions do not match manifest rank")
        ordered = dict(sorted(copied.items(), key=lambda item: _type_sort_key(item[0])))
        payload = _type_catalog_payload(generator_system, ordered)
        fingerprint = _fingerprint("tfenn_type_catalog_v1", payload)
        if expected_fingerprint is not None and expected_fingerprint != fingerprint:
            raise RuntimeError("type catalog fingerprint validation failed")
        object.__setattr__(self, "generator_system", generator_system)
        object.__setattr__(self, "_blocks", MappingProxyType(ordered))
        object.__setattr__(self, "fingerprint", fingerprint)

    @property
    def blocks(self) -> Mapping[TypeKey, TypeBlock]:
        """Return the immutable typed block mapping."""
        _validate_type_catalog_integrity(self)
        return self._blocks

    def resolve(self, key: TypeKey) -> TypeBlock:
        """Resolve one explicit typed block without shape inference."""
        if not isinstance(key, TypeKey):
            raise TypeError("key must be a TypeKey")
        _validate_type_catalog_integrity(self)
        try:
            return self._blocks[key]
        except KeyError as error:
            raise KeyError(f"TypeCatalog does not contain {key}") from error


def _validate_type_catalog_integrity(catalog: TypeCatalog) -> None:
    """Reject private generator or action mutation before typed resolution."""
    _validate_generator_system_integrity(catalog.generator_system)
    expected = _fingerprint(
        "tfenn_type_catalog_v1",
        _type_catalog_payload(catalog.generator_system, catalog._blocks),
    )
    if expected != catalog.fingerprint:
        raise RuntimeError("type catalog content does not match fingerprint")


def build_type_catalog(
    generator_system: GeneratorSystem,
    b_manifest: Sequence[BBlockManifest],
) -> TypeCatalog:
    """Build A and every explicitly declared B representation block."""
    if not isinstance(generator_system, GeneratorSystem):
        raise TypeError("generator_system must be a GeneratorSystem")
    _validate_generator_system_integrity(generator_system)
    manifests = tuple(b_manifest)
    if any(not isinstance(item, BBlockManifest) for item in manifests):
        raise TypeError("b_manifest must contain BBlockManifest values")
    keys = tuple(item.key for item in manifests)
    stable_ids = tuple(item.stable_component_id for item in manifests)
    if len(set(keys)) != len(keys):
        raise ValueError("B manifest component keys must be unique")
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("B manifest stable component ids must be unique")
    a_key = TypeKey("A")
    blocks: dict[TypeKey, TypeBlock] = {
        a_key: TypeBlock(
            a_key,
            generator_system._matrices,
            3,
            generator_system.fingerprint,
            "active_column_cartesian_xyz_v1",
            {"stable_component_id": "A", "layout": "final_representation_axis"},
        )
    }
    for manifest in manifests:
        dimension = 2 * manifest.stf_rank + 1
        blocks[manifest.key] = TypeBlock(
            manifest.key,
            stf_representation(generator_system._matrices, manifest.stf_rank),
            dimension,
            generator_system.fingerprint,
            f"{TENSOR_CONVENTION_VERSION}_stf_rank_{manifest.stf_rank}",
            _manifest_metadata(manifest),
        )
    return TypeCatalog(generator_system, blocks)


@dataclass(frozen=True, slots=True)
class CSlot:
    """Declare one named runtime variable and its representation lift."""

    name: str
    type_key: TypeKey
    power: int = 1
    mode: Literal["distinct", "symmetric_power"] = "distinct"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("CSlot name must be a nonempty string")
        if not isinstance(self.type_key, TypeKey):
            raise TypeError("CSlot type_key must be a TypeKey")
        if isinstance(self.power, bool) or not isinstance(self.power, int):
            raise TypeError("CSlot power must be an integer and cannot be bool")
        if self.mode == "distinct":
            if self.power != 1:
                raise ValueError("distinct CSlot power must equal one")
        elif self.mode == "symmetric_power":
            if self.power < 2:
                raise ValueError("symmetric_power CSlot power must be at least two")
        else:
            raise ValueError("CSlot mode must be distinct or symmetric_power")


@dataclass(frozen=True, slots=True)
class CSignature:
    """Declare one ordered typed source and one typed or scalar target."""

    output: TypeKey | TrivialScalarKey
    inputs: tuple[CSlot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.output, (TypeKey, TrivialScalarKey)):
            raise TypeError("CSignature output must be TypeKey or TrivialScalarKey")
        inputs = tuple(self.inputs)
        if not inputs:
            raise ValueError("CSignature must contain at least one input slot")
        if any(not isinstance(slot, CSlot) for slot in inputs):
            raise TypeError("CSignature inputs must contain CSlot values")
        names = tuple(slot.name for slot in inputs)
        if len(set(names)) != len(names):
            raise ValueError("CSignature slot names must be unique")
        object.__setattr__(self, "inputs", inputs)


def _signature_payload(signature: CSignature) -> dict[str, Any]:
    output = (
        {"kind": "scalar", "name": signature.output.name}
        if isinstance(signature.output, TrivialScalarKey)
        else {"kind": "type", "key": _type_key_payload(signature.output)}
    )
    return {
        "output": output,
        "inputs": [
            {
                "name": slot.name,
                "type_key": _type_key_payload(slot.type_key),
                "power": slot.power,
                "mode": slot.mode,
            }
            for slot in signature.inputs
        ],
    }


def _signature_from_payload(payload: Any) -> CSignature:
    if not isinstance(payload, Mapping):
        raise RuntimeError("signature payload must be a mapping")
    output_payload = payload.get("output")
    if not isinstance(output_payload, Mapping):
        raise RuntimeError("signature output payload is malformed")
    kind = output_payload.get("kind")
    if kind == "scalar":
        output: TypeKey | TrivialScalarKey = TrivialScalarKey(
            output_payload.get("name")
        )
    elif kind == "type":
        output = _type_key_from_payload(output_payload.get("key"))
    else:
        raise RuntimeError("signature output kind is unsupported")
    input_payload = payload.get("inputs")
    if not isinstance(input_payload, list):
        raise RuntimeError("signature inputs payload is malformed")
    slots = []
    for item in input_payload:
        if not isinstance(item, Mapping):
            raise RuntimeError("signature slot payload is malformed")
        slots.append(
            CSlot(
                name=item.get("name"),
                type_key=_type_key_from_payload(item.get("type_key")),
                power=item.get("power"),
                mode=item.get("mode"),
            )
        )
    return CSignature(output=output, inputs=tuple(slots))


@lru_cache(maxsize=None)
def _symmetric_multi_indices(
    dimension: int,
    power: int,
) -> tuple[tuple[int, ...], ...]:
    """Return descending lexicographic occupation tuples."""
    if dimension == 1:
        return ((power,),)
    result = []
    for first in range(power, -1, -1):
        for suffix in _symmetric_multi_indices(dimension - 1, power - first):
            result.append((first, *suffix))
    return tuple(result)


@lru_cache(maxsize=None)
def _normalized_symmetric_power_basis_master(dimension: int, power: int) -> Tensor:
    """Build one canonical normalized occupation lift."""
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if isinstance(power, bool) or not isinstance(power, int) or power < 0:
        raise ValueError("power must be a nonnegative integer")
    tensor_dimension = dimension**power
    effective_dimension = math.comb(dimension + power - 1, power)
    entries = tensor_dimension * effective_dimension
    if entries > MAX_SYMMETRIC_BASIS_ENTRIES:
        raise ValueError(
            f"normalized symmetric basis needs {entries} entries; "
            f"limit is {MAX_SYMMETRIC_BASIS_ENTRIES}"
        )
    occupations = _symmetric_multi_indices(dimension, power)
    positions = {occupation: index for index, occupation in enumerate(occupations)}
    basis = torch.zeros(
        (tensor_dimension, effective_dimension),
        dtype=torch.float64,
    )
    words = itertools.product(range(dimension), repeat=power)
    for row, word in enumerate(words):
        counts = tuple(word.count(axis) for axis in range(dimension))
        column = positions[counts]
        multiplicity = math.factorial(power)
        for count in counts:
            multiplicity //= math.factorial(count)
        basis[row, column] = 1.0 / math.sqrt(multiplicity)
    return basis


def normalized_symmetric_power_basis(dimension: int, power: int) -> Tensor:
    """Return a clone of the canonical normalized symmetric power lift."""
    return _normalized_symmetric_power_basis_master(dimension, power).clone()


def _tensor_power_representation(actions: Tensor, power: int) -> Tensor:
    """Return repeated tensor power actions in left major order."""
    if power == 0:
        return actions.new_ones((actions.shape[0], 1, 1))
    result = actions
    for _ in range(1, power):
        result = tensor_product_representation(result, actions)
    return result


def normalized_symmetric_power_representation(actions: Tensor, power: int) -> Tensor:
    """Return normalized symmetric power actions for arbitrary dimension."""
    frozen = _freeze_cpu64_tensor(actions, "actions", ndim=3)
    if frozen.shape[-1] == 0 or frozen.shape[-2] != frozen.shape[-1]:
        raise ValueError("actions must have positive square trailing dimensions")
    basis = normalized_symmetric_power_basis(frozen.shape[-1], power)
    tensor_power = _tensor_power_representation(frozen, power)
    return torch.einsum("ia,qij,jb->qab", basis, tensor_power, basis)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SlotLiftArtifact:
    """Store one slot lift and its effective source coordinates."""

    slot: CSlot
    input_dimension: int
    effective_dimension: int
    multi_indices: tuple[tuple[int, ...], ...]
    _lift: Tensor = field(repr=False, compare=False)
    convention_version: str
    fingerprint: str

    def __init__(self, slot: CSlot, input_dimension: int, lift: Tensor) -> None:
        if not isinstance(slot, CSlot):
            raise TypeError("slot must be a CSlot")
        frozen = _freeze_cpu64_tensor(lift, f"slot {slot.name} lift", ndim=2)
        expected_tensor_dimension = input_dimension**slot.power
        if frozen.shape[0] != expected_tensor_dimension:
            raise ValueError("slot lift tensor dimension is inconsistent")
        effective_dimension = frozen.shape[1]
        indices = _symmetric_multi_indices(input_dimension, slot.power)
        if effective_dimension != len(indices):
            raise ValueError("slot lift effective dimension is inconsistent")
        canonical = normalized_symmetric_power_basis(input_dimension, slot.power)
        if not torch.equal(frozen, canonical):
            raise ValueError("slot lift does not match the canonical convention")
        gram = frozen.T @ frozen
        identity = torch.eye(effective_dimension, dtype=torch.float64)
        if not torch.allclose(gram, identity, atol=2e-12, rtol=2e-12):
            raise ValueError("slot lift columns must be orthonormal")
        payload = {
            "slot": _signature_payload(
                CSignature(output=TRIVIAL_SCALAR, inputs=(slot,))
            )["inputs"][0],
            "input_dimension": input_dimension,
            "effective_dimension": effective_dimension,
            "multi_indices": [list(index) for index in indices],
            "lift": _tensor_payload(frozen),
            "convention_version": SYMMETRIC_POWER_CONVENTION_VERSION,
        }
        fingerprint = _fingerprint("tfenn_slot_lift_v1", payload)
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "input_dimension", input_dimension)
        object.__setattr__(self, "effective_dimension", effective_dimension)
        object.__setattr__(self, "multi_indices", indices)
        object.__setattr__(self, "_lift", frozen)
        object.__setattr__(
            self,
            "convention_version",
            SYMMETRIC_POWER_CONVENTION_VERSION,
        )
        object.__setattr__(self, "fingerprint", fingerprint)

    @property
    def lift(self) -> Tensor:
        """Return a detached clone of the normalized lift."""
        return self._lift.detach().clone()


@dataclass(frozen=True, slots=True)
class CovariantCost:
    """Store diagnostic dense map costs without changing compilation."""

    effective_input_dimension: int
    output_dimension: int
    full_basis_dimension: int
    active_basis_dimension: int
    coefficient_count: int
    estimated_multiply_add_count: int
    reduction_ratio: float


def _cost_metadata(
    effective_input_dimension: int,
    output_dimension: int,
    full_dimension: int,
    active_dimension: int,
) -> CovariantCost:
    ratio = 1.0 if full_dimension == 0 else active_dimension / full_dimension
    return CovariantCost(
        effective_input_dimension=effective_input_dimension,
        output_dimension=output_dimension,
        full_basis_dimension=full_dimension,
        active_basis_dimension=active_dimension,
        coefficient_count=active_dimension,
        estimated_multiply_add_count=(
            active_dimension * output_dimension * effective_input_dimension
        ),
        reduction_ratio=float(ratio),
    )


def _cost_payload(cost: CovariantCost) -> dict[str, Any]:
    return {
        "effective_input_dimension": cost.effective_input_dimension,
        "output_dimension": cost.output_dimension,
        "full_basis_dimension": cost.full_basis_dimension,
        "active_basis_dimension": cost.active_basis_dimension,
        "coefficient_count": cost.coefficient_count,
        "estimated_multiply_add_count": cost.estimated_multiply_add_count,
        "reduction_ratio": _float_text(cost.reduction_ratio, "reduction_ratio"),
    }


def _slot_lift_payload(lift: SlotLiftArtifact) -> dict[str, Any]:
    """Return the canonical serialized slot lift payload."""
    return {
        "slot": _signature_payload(
            CSignature(output=TRIVIAL_SCALAR, inputs=(lift.slot,))
        )["inputs"][0],
        "input_dimension": lift.input_dimension,
        "effective_dimension": lift.effective_dimension,
        "multi_indices": [list(index) for index in lift.multi_indices],
        "lift": _tensor_payload(lift._lift),
        "convention_version": lift.convention_version,
        "fingerprint": lift.fingerprint,
    }


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CovariantCompilation:
    """Store one complete canonical Hom basis and all offline diagnostics."""

    signature: CSignature
    _basis: Tensor = field(repr=False, compare=False)
    slot_lifts: tuple[SlotLiftArtifact, ...]
    input_dimensions: tuple[int, ...]
    effective_input_dimension: int
    output_dimension: int
    basis_dimension: int
    _singular_values: Tensor = field(repr=False, compare=False)
    threshold: float
    singular_value_gap: float
    threshold_margin: float
    residual: float
    nullspace_atol: float
    nullspace_rtol: float
    generator_fingerprint: str
    type_catalog_fingerprint: str
    convention_version: str
    artifact_fingerprint: str
    cost: CovariantCost

    def __init__(
        self,
        signature: CSignature,
        basis: Tensor,
        slot_lifts: Sequence[SlotLiftArtifact],
        compilation: IntertwinerCompilation,
        output_dimension: int,
        generator_fingerprint: str,
        type_catalog_fingerprint: str,
        *,
        expected_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(signature, CSignature):
            raise TypeError("signature must be a CSignature")
        lifts = tuple(slot_lifts)
        if len(lifts) != len(signature.inputs) or any(
            not isinstance(lift, SlotLiftArtifact) for lift in lifts
        ):
            raise ValueError("slot_lifts must match signature inputs")
        if tuple(lift.slot for lift in lifts) != signature.inputs:
            raise ValueError("slot_lifts must preserve exact signature order")
        if not isinstance(compilation, IntertwinerCompilation):
            raise TypeError("compilation must be an IntertwinerCompilation")
        if compilation.convention_version != TENSOR_CONVENTION_VERSION:
            raise ValueError("intertwiner compilation convention is incompatible")
        _validate_fingerprint(generator_fingerprint, "generator_fingerprint")
        _validate_fingerprint(type_catalog_fingerprint, "type_catalog_fingerprint")
        frozen_basis = _freeze_cpu64_tensor(basis, "covariant basis", ndim=3)
        frozen_singular = _freeze_cpu64_tensor(
            compilation.singular_values,
            "covariant singular values",
            ndim=1,
        )
        effective_input_dimension = math.prod(
            lift.effective_dimension for lift in lifts
        )
        basis_dimension = compilation.dimension
        expected_shape = (
            basis_dimension,
            output_dimension,
            effective_input_dimension,
        )
        if tuple(frozen_basis.shape) != expected_shape:
            raise ValueError(
                f"covariant basis shape {tuple(frozen_basis.shape)} does not match "
                f"expected shape {expected_shape}"
            )
        if not torch.equal(frozen_basis, compilation.basis):
            raise ValueError("covariant basis must preserve backend coordinates exactly")
        cost = _cost_metadata(
            effective_input_dimension,
            output_dimension,
            basis_dimension,
            basis_dimension,
        )
        input_dimensions = tuple(lift.input_dimension for lift in lifts)
        core_payload = {
            "signature": _signature_payload(signature),
            "basis": _tensor_payload(frozen_basis),
            "slot_lifts": [_slot_lift_payload(lift) for lift in lifts],
            "input_dimensions": list(input_dimensions),
            "effective_input_dimension": effective_input_dimension,
            "output_dimension": output_dimension,
            "basis_dimension": basis_dimension,
            "singular_values": _tensor_payload(frozen_singular),
            "threshold": _float_text(compilation.threshold, "threshold"),
            "singular_value_gap": _float_text(
                compilation.singular_value_gap,
                "singular_value_gap",
            ),
            "threshold_margin": _float_text(
                compilation.threshold_margin,
                "threshold_margin",
            ),
            "residual": _float_text(compilation.residual, "residual"),
            "nullspace_atol": _float_text(
                compilation.nullspace_atol,
                "nullspace_atol",
                allow_infinite=False,
            ),
            "nullspace_rtol": _float_text(
                compilation.nullspace_rtol,
                "nullspace_rtol",
                allow_infinite=False,
            ),
            "generator_fingerprint": generator_fingerprint,
            "type_catalog_fingerprint": type_catalog_fingerprint,
            "convention_version": COVARIANT_CONVENTION_VERSION,
            "cost": _cost_payload(cost),
        }
        fingerprint = _fingerprint("tfenn_covariant_artifact_v1", core_payload)
        if expected_fingerprint is not None and expected_fingerprint != fingerprint:
            raise RuntimeError("covariant artifact fingerprint validation failed")
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "_basis", frozen_basis)
        object.__setattr__(self, "slot_lifts", lifts)
        object.__setattr__(self, "input_dimensions", input_dimensions)
        object.__setattr__(
            self,
            "effective_input_dimension",
            effective_input_dimension,
        )
        object.__setattr__(self, "output_dimension", output_dimension)
        object.__setattr__(self, "basis_dimension", basis_dimension)
        object.__setattr__(self, "_singular_values", frozen_singular)
        object.__setattr__(self, "threshold", float(compilation.threshold))
        object.__setattr__(
            self,
            "singular_value_gap",
            float(compilation.singular_value_gap),
        )
        object.__setattr__(
            self,
            "threshold_margin",
            float(compilation.threshold_margin),
        )
        object.__setattr__(self, "residual", float(compilation.residual))
        object.__setattr__(
            self,
            "nullspace_atol",
            float(compilation.nullspace_atol),
        )
        object.__setattr__(
            self,
            "nullspace_rtol",
            float(compilation.nullspace_rtol),
        )
        object.__setattr__(self, "generator_fingerprint", generator_fingerprint)
        object.__setattr__(
            self,
            "type_catalog_fingerprint",
            type_catalog_fingerprint,
        )
        object.__setattr__(
            self,
            "convention_version",
            COVARIANT_CONVENTION_VERSION,
        )
        object.__setattr__(self, "artifact_fingerprint", fingerprint)
        object.__setattr__(self, "cost", cost)

    @property
    def basis(self) -> Tensor:
        """Return a detached clone of the complete canonical basis."""
        return self._basis.detach().clone()

    @property
    def singular_values(self) -> Tensor:
        """Return a detached clone of backend singular values."""
        return self._singular_values.detach().clone()

    @property
    def symmetric_power_lifts(self) -> tuple[SlotLiftArtifact, ...]:
        """Return only genuine repeated variable lift artifacts."""
        return tuple(
            lift for lift in self.slot_lifts if lift.slot.mode == "symmetric_power"
        )


def _validated_signature_dimensions(
    catalog: TypeCatalog,
    signature: CSignature,
) -> tuple[tuple[TypeBlock, ...], int, Tensor]:
    """Resolve all explicit typed inputs and the target action."""
    if not isinstance(catalog, TypeCatalog):
        raise TypeError("catalog must be a TypeCatalog")
    _validate_type_catalog_integrity(catalog)
    if not isinstance(signature, CSignature):
        raise TypeError("signature must be a CSignature")
    if len(signature.inputs) > MAX_COVARIANT_SLOTS:
        raise ValueError(
            f"signature has {len(signature.inputs)} slots; limit is "
            f"{MAX_COVARIANT_SLOTS}"
        )
    total_degree = sum(slot.power for slot in signature.inputs)
    if total_degree > MAX_COVARIANT_TOTAL_DEGREE:
        raise ValueError(
            f"signature total degree {total_degree}; limit is "
            f"{MAX_COVARIANT_TOTAL_DEGREE}"
        )
    blocks = tuple(catalog.resolve(slot.type_key) for slot in signature.inputs)
    generator_count = len(catalog.generator_system.names)
    if isinstance(signature.output, TrivialScalarKey):
        output_dimension = 1
        output_actions = torch.ones(
            (generator_count, 1, 1),
            dtype=torch.float64,
        )
    else:
        output = catalog.resolve(signature.output)
        output_dimension = output.representation_dim
        output_actions = output._actions
    return blocks, output_dimension, output_actions


def _predicted_dense_entries(
    generator_count: int,
    blocks: Sequence[TypeBlock],
    signature: CSignature,
    output_dimension: int,
) -> tuple[int, int, int]:
    """Return maximum lift, action, and constraint entry counts."""
    effective_dimensions = []
    largest_lift = 0
    largest_action = 0
    for slot, block in zip(signature.inputs, blocks):
        dimension = block.representation_dim
        tensor_dimension = dimension**slot.power
        effective_dimension = math.comb(
            dimension + slot.power - 1,
            slot.power,
        )
        effective_dimensions.append(effective_dimension)
        largest_lift = max(largest_lift, tensor_dimension * effective_dimension)
        largest_action = max(
            largest_action,
            generator_count * tensor_dimension * tensor_dimension,
        )
    effective_input = math.prod(effective_dimensions)
    largest_action = max(
        largest_action,
        generator_count * effective_input * effective_input,
    )
    variable_dimension = effective_input * output_dimension
    constraint_entries = generator_count * variable_dimension * variable_dimension
    return largest_lift, largest_action, constraint_entries


def compile_covariant_basis(
    catalog: TypeCatalog,
    signature: CSignature,
    *,
    solver: str = "dense_svd",
    nullspace_atol: float | None = None,
    nullspace_rtol: float | None = None,
    max_constraint_entries: int = DEFAULT_MAX_CONSTRAINT_ENTRIES,
) -> CovariantCompilation:
    """Compile the complete canonical Hom basis for one declared signature."""
    if solver != "dense_svd":
        raise ValueError("solver must be dense_svd")
    if (
        isinstance(max_constraint_entries, bool)
        or not isinstance(max_constraint_entries, int)
        or max_constraint_entries <= 0
    ):
        raise ValueError("max_constraint_entries must be a positive integer")
    blocks, output_dimension, output_actions = _validated_signature_dimensions(
        catalog,
        signature,
    )
    counts = _predicted_dense_entries(
        len(catalog.generator_system.names),
        blocks,
        signature,
        output_dimension,
    )
    largest = max(counts)
    if largest > max_constraint_entries:
        raise ValueError(
            "dense covariant compilation exceeds allocation guard: "
            f"lift entries {counts[0]}, action entries {counts[1]}, "
            f"constraint entries {counts[2]}, limit {max_constraint_entries}"
        )
    lifts = []
    slot_actions = []
    for slot, block in zip(signature.inputs, blocks):
        lift_tensor = normalized_symmetric_power_basis(
            block.representation_dim,
            slot.power,
        )
        lift = SlotLiftArtifact(slot, block.representation_dim, lift_tensor)
        lifts.append(lift)
        if slot.mode == "distinct":
            slot_actions.append(block._actions)
        else:
            slot_actions.append(
                normalized_symmetric_power_representation(
                    block._actions,
                    slot.power,
                )
            )
    source_actions = slot_actions[0]
    for actions in slot_actions[1:]:
        source_actions = tensor_product_representation(source_actions, actions)
    backend = compile_intertwiners(
        source_actions,
        output_actions,
        nullspace_atol=nullspace_atol,
        nullspace_rtol=nullspace_rtol,
    )
    return CovariantCompilation(
        signature,
        backend.basis,
        lifts,
        backend,
        output_dimension,
        catalog.generator_system.fingerprint,
        catalog.fingerprint,
    )


def _view_rank_diagnostics(
    transform: Tensor,
    atol: float,
    rtol: float,
) -> tuple[int, Tensor, float, float, float]:
    """Return deterministic numerical rank diagnostics for a view transform."""
    singular_values = torch.linalg.svdvals(transform)
    sigma_max = float(singular_values[0]) if singular_values.numel() else 0.0
    threshold = atol + rtol * sigma_max
    rank = int((singular_values > threshold).sum())
    if singular_values.numel() == 0:
        gap = math.inf
        margin = math.inf
    elif rank == 0:
        gap = math.inf if sigma_max == 0.0 else float(singular_values[0])
        margin = threshold - float(singular_values[0])
    elif rank == singular_values.numel():
        gap = math.inf
        margin = float(singular_values[-1]) - threshold
    else:
        gap = float(singular_values[rank - 1] - singular_values[rank])
        margin = min(
            float(singular_values[rank - 1]) - threshold,
            threshold - float(singular_values[rank]),
        )
    return rank, singular_values, threshold, gap, margin


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CovariantBasisView:
    """Store a fixed linear view of one immutable complete parent basis."""

    parent_artifact_fingerprint: str
    _transform: Tensor = field(repr=False, compare=False)
    full_dimension: int
    active_dimension: int
    transform_rank: int
    is_complete: bool
    _singular_values: Tensor = field(repr=False, compare=False)
    threshold: float
    singular_value_gap: float
    threshold_margin: float
    rank_atol: float
    rank_rtol: float
    view_fingerprint: str
    cost: CovariantCost

    def __init__(
        self,
        parent: CovariantCompilation,
        transform: Tensor,
        *,
        rank_atol: float | None = None,
        rank_rtol: float | None = None,
        expected_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(parent, CovariantCompilation):
            raise TypeError("parent must be a CovariantCompilation")
        frozen = _freeze_cpu64_tensor(transform, "basis view transform", ndim=2)
        full_dimension = parent.basis_dimension
        if frozen.shape[1] != full_dimension:
            raise ValueError(
                "basis view transform columns must equal parent basis dimension"
            )
        atol = parent.nullspace_atol if rank_atol is None else rank_atol
        rtol = parent.nullspace_rtol if rank_rtol is None else rank_rtol
        if isinstance(atol, bool) or not isinstance(atol, (int, float)):
            raise TypeError("rank_atol must be a real number and cannot be bool")
        if isinstance(rtol, bool) or not isinstance(rtol, (int, float)):
            raise TypeError("rank_rtol must be a real number and cannot be bool")
        atol = float(atol)
        rtol = float(rtol)
        if not math.isfinite(atol) or atol <= 0.0:
            raise ValueError("rank_atol must be finite and positive")
        if not math.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rank_rtol must be finite and positive")
        rank, singular_values, threshold, gap, margin = _view_rank_diagnostics(
            frozen,
            atol,
            rtol,
        )
        active_dimension = frozen.shape[0]
        complete = rank == full_dimension
        cost = _cost_metadata(
            parent.effective_input_dimension,
            parent.output_dimension,
            full_dimension,
            active_dimension,
        )
        core_payload = {
            "parent_artifact_fingerprint": parent.artifact_fingerprint,
            "transform": _tensor_payload(frozen),
            "full_dimension": full_dimension,
            "active_dimension": active_dimension,
            "transform_rank": rank,
            "is_complete": complete,
            "singular_values": _tensor_payload(singular_values),
            "threshold": _float_text(threshold, "view threshold"),
            "singular_value_gap": _float_text(gap, "view singular gap"),
            "threshold_margin": _float_text(margin, "view threshold margin"),
            "rank_atol": _float_text(atol, "view rank_atol"),
            "rank_rtol": _float_text(rtol, "view rank_rtol"),
            "cost": _cost_payload(cost),
            "convention_version": COVARIANT_CONVENTION_VERSION,
        }
        fingerprint = _fingerprint("tfenn_covariant_view_v1", core_payload)
        if expected_fingerprint is not None and expected_fingerprint != fingerprint:
            raise RuntimeError("basis view fingerprint validation failed")
        object.__setattr__(
            self,
            "parent_artifact_fingerprint",
            parent.artifact_fingerprint,
        )
        object.__setattr__(self, "_transform", frozen)
        object.__setattr__(self, "full_dimension", full_dimension)
        object.__setattr__(self, "active_dimension", active_dimension)
        object.__setattr__(self, "transform_rank", rank)
        object.__setattr__(self, "is_complete", complete)
        object.__setattr__(self, "_singular_values", singular_values)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "singular_value_gap", gap)
        object.__setattr__(self, "threshold_margin", margin)
        object.__setattr__(self, "rank_atol", atol)
        object.__setattr__(self, "rank_rtol", rtol)
        object.__setattr__(self, "view_fingerprint", fingerprint)
        object.__setattr__(self, "cost", cost)

    @property
    def transform(self) -> Tensor:
        """Return a detached clone of the basis transform."""
        return self._transform.detach().clone()

    @property
    def singular_values(self) -> Tensor:
        """Return a detached clone of view singular values."""
        return self._singular_values.detach().clone()


def encode_typed_blocks(
    pose_encoder: PoseEncoder,
    rotation: Tensor,
    manifest: Sequence[BBlockManifest],
) -> Mapping[TypeKey, Tensor]:
    """Adapt Pose blocks to explicit typed channel then representation layout."""
    if not isinstance(pose_encoder, PoseEncoder):
        raise TypeError("pose_encoder must be a PoseEncoder")
    manifests = tuple(manifest)
    if any(not isinstance(item, BBlockManifest) for item in manifests):
        raise TypeError("manifest must contain BBlockManifest values")
    if len({item.key for item in manifests}) != len(manifests):
        raise ValueError("typed Pose manifest keys must be unique")
    source = pose_encoder.encode_blocks(rotation)
    multiplicities = pose_encoder.multiplicities
    result: dict[TypeKey, Tensor] = {}
    for item in manifests:
        if item.stf_rank not in source:
            raise ValueError(
                f"PoseEncoder does not contain manifest rank {item.stf_rank}"
            )
        actual_multiplicity = multiplicities[item.stf_rank]
        if actual_multiplicity != item.anchor_multiplicity:
            raise ValueError(
                f"rank {item.stf_rank} anchor multiplicity {actual_multiplicity} "
                f"does not match manifest value {item.anchor_multiplicity}"
            )
        if max(item.anchor_columns) >= actual_multiplicity:
            raise ValueError("manifest anchor column exceeds Pose block multiplicity")
        typed = source[item.stf_rank].movedim(-2, -1)
        result[item.key] = typed[..., item.anchor_columns, :]
    return MappingProxyType(result)


class PoseTypedBlockAdapter(nn.Module):
    """Wrap PoseEncoder without changing its existing block layout API."""

    def __init__(self, pose_encoder: PoseEncoder) -> None:
        super().__init__()
        if not isinstance(pose_encoder, PoseEncoder):
            raise TypeError("pose_encoder must be a PoseEncoder")
        self.pose_encoder = pose_encoder

    def encode_typed_blocks(
        self,
        rotation: Tensor,
        manifest: Sequence[BBlockManifest],
    ) -> Mapping[TypeKey, Tensor]:
        """Return explicit B blocks with channel then representation axes."""
        return encode_typed_blocks(self.pose_encoder, rotation, manifest)

    def forward(
        self,
        rotation: Tensor,
        manifest: Sequence[BBlockManifest],
    ) -> Mapping[TypeKey, Tensor]:
        """Apply the typed Pose layout adapter."""
        return self.encode_typed_blocks(rotation, manifest)


def _require_runtime_tensor(
    value: Any,
    name: str,
    expected_dimension: int,
    reference: Tensor,
) -> Tensor:
    """Validate one runtime feature tensor without data dependent checks."""
    if not isinstance(value, Tensor):
        raise TypeError(f"runtime input {name} must be a torch.Tensor")
    if value.ndim == 0 or value.shape[-1] != expected_dimension:
        raise ValueError(
            f"runtime input {name} final dimension {value.shape[-1:]}; "
            f"expected {expected_dimension}"
        )
    if value.dtype != reference.dtype:
        raise TypeError(f"runtime input {name} dtype must match registered buffers")
    if value.device != reference.device:
        raise ValueError(f"runtime input {name} device must match registered buffers")
    return value


def _require_state_constant(incoming: Any, master: Tensor, name: str) -> None:
    """Require one state tensor to equal its artifact after exact cast."""
    if not isinstance(incoming, Tensor):
        raise RuntimeError(f"RegisteredCovariant state {name} must be a tensor")
    if incoming.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(f"RegisteredCovariant state {name} dtype is unsupported")
    expected = master.to(device=incoming.device, dtype=incoming.dtype)
    if tuple(incoming.shape) != tuple(expected.shape) or not torch.equal(
        incoming,
        expected,
    ):
        raise RuntimeError(
            f"RegisteredCovariant state {name} does not match its artifact"
        )


class RegisteredCovariant(nn.Module):
    """Apply one complete frozen C basis and optional registered basis views."""

    def __init__(
        self,
        artifact: CovariantCompilation,
        basis_views: Sequence[CovariantBasisView] = (),
    ) -> None:
        """Register one artifact and fixed view transforms as buffers."""
        super().__init__()
        if not isinstance(artifact, CovariantCompilation):
            raise TypeError("artifact must be a CovariantCompilation")
        _validate_artifact_integrity(artifact)
        views = tuple(basis_views)
        if any(not isinstance(view, CovariantBasisView) for view in views):
            raise TypeError("basis_views must contain CovariantBasisView values")
        if len({view.view_fingerprint for view in views}) != len(views):
            raise ValueError("basis view fingerprints must be unique")
        for view in views:
            _validate_view_integrity(view)
            if view.parent_artifact_fingerprint != artifact.artifact_fingerprint:
                raise ValueError("basis view parent does not match artifact")
        self._signature = artifact.signature
        self._artifact_fingerprint = artifact.artifact_fingerprint
        self._generator_fingerprint = artifact.generator_fingerprint
        self._catalog_fingerprint = artifact.type_catalog_fingerprint
        self._convention_version = artifact.convention_version
        self._basis_dimension = artifact.basis_dimension
        self._output_dimension = artifact.output_dimension
        self._input_dimensions = artifact.input_dimensions
        self._effective_input_dimension = artifact.effective_input_dimension
        self._slot_descriptors = tuple(
            (
                lift.slot.name,
                lift.slot.mode,
                lift.slot.power,
                lift.input_dimension,
                lift.effective_dimension,
                lift.fingerprint,
            )
            for lift in artifact.slot_lifts
        )
        self._view_descriptors = tuple(
            (
                view.view_fingerprint,
                view.parent_artifact_fingerprint,
                view.active_dimension,
                view.full_dimension,
                view.transform_rank,
                view.is_complete,
            )
            for view in views
        )
        self._view_names = tuple(
            (view.view_fingerprint, f"_view_transform_{index}")
            for index, view in enumerate(views)
        )
        self._master_basis_bytes = _canonical_json_bytes(
            _tensor_payload(artifact._basis)
        )
        self._master_lift_bytes = tuple(
            _canonical_json_bytes(_tensor_payload(lift._lift))
            for lift in artifact.slot_lifts
        )
        self._master_view_bytes = tuple(
            _canonical_json_bytes(_tensor_payload(view._transform)) for view in views
        )
        self.register_buffer("_basis", artifact._basis.detach().clone())
        self._lift_names = tuple(f"_slot_lift_{index}" for index in range(len(artifact.slot_lifts)))
        for name, lift in zip(self._lift_names, artifact.slot_lifts):
            self.register_buffer(name, lift._lift.detach().clone())
        for (_, name), view in zip(self._view_names, views):
            self.register_buffer(name, view._transform.detach().clone())

    @property
    def artifact_fingerprint(self) -> str:
        """Return the bound complete parent artifact fingerprint."""
        return self._artifact_fingerprint

    @property
    def basis(self) -> Tensor:
        """Return a detached clone of the live basis buffer."""
        return self._basis.detach().clone()

    @property
    def view_fingerprints(self) -> tuple[str, ...]:
        """Return registered basis view fingerprints."""
        return tuple(item[0] for item in self._view_names)

    def _runtime_source(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Lift each slot and form the ordered left major source product."""
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping from slot name to tensor")
        expected_names = tuple(slot.name for slot in self._signature.inputs)
        actual_names = tuple(inputs.keys())
        if any(not isinstance(name, str) for name in actual_names):
            raise ValueError("runtime input keys must be slot name strings")
        if set(actual_names) != set(expected_names) or len(actual_names) != len(
            expected_names
        ):
            raise ValueError(
                f"runtime input keys {sorted(actual_names)} must exactly match "
                f"{sorted(expected_names)}"
            )
        lifted = []
        for index, (slot, input_dimension) in enumerate(
            zip(self._signature.inputs, self._input_dimensions)
        ):
            value = _require_runtime_tensor(
                inputs[slot.name],
                slot.name,
                input_dimension,
                self._basis,
            )
            if slot.mode == "distinct":
                lifted.append(value)
                continue
            tensor_power = value
            for degree in range(2, slot.power + 1):
                tensor_power = torch.einsum(
                    "...i,...j->...ij",
                    tensor_power,
                    value,
                ).reshape(value.shape[:-1] + (input_dimension**degree,))
            lift = getattr(self, self._lift_names[index])
            lifted.append(torch.einsum("...i,ij->...j", tensor_power, lift))
        source = lifted[0]
        for value in lifted[1:]:
            try:
                leading = torch.broadcast_shapes(source.shape[:-1], value.shape[:-1])
            except RuntimeError as error:
                raise ValueError("runtime slot leading axes are not broadcast compatible") from error
            source_work = source.expand(leading + (source.shape[-1],))
            value_work = value.expand(leading + (value.shape[-1],))
            source = torch.einsum(
                "...i,...j->...ij",
                source_work,
                value_work,
            ).reshape(leading + (source.shape[-1] * value.shape[-1],))
        return source

    def evaluate_basis(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Evaluate every complete parent basis map."""
        source = self._runtime_source(inputs)
        return torch.einsum("boi,...i->...bo", self._basis, source)

    def apply_coefficients(
        self,
        inputs: Mapping[str, Tensor],
        coefficients: Tensor,
    ) -> Tensor:
        """Apply the fused weighted sum over the complete basis."""
        source = self._runtime_source(inputs)
        coefficient = _require_runtime_tensor(
            coefficients,
            "coefficients",
            self._basis_dimension,
            self._basis,
        )
        try:
            torch.broadcast_shapes(source.shape[:-1], coefficient.shape[:-1])
        except RuntimeError as error:
            raise ValueError(
                "coefficient leading axes are not broadcast compatible with inputs"
            ) from error
        return torch.einsum(
            "...b,boi,...i->...o",
            coefficient,
            self._basis,
            source,
        )

    def evaluate_view(
        self,
        inputs: Mapping[str, Tensor],
        basis_view: CovariantBasisView,
        *,
        allow_incomplete: bool = False,
    ) -> Tensor:
        """Evaluate one constructor registered fixed basis view."""
        if not isinstance(basis_view, CovariantBasisView):
            raise TypeError("basis_view must be a CovariantBasisView")
        if not isinstance(allow_incomplete, bool):
            raise TypeError("allow_incomplete must be bool")
        if basis_view.parent_artifact_fingerprint != self._artifact_fingerprint:
            raise ValueError("basis view parent fingerprint does not match module")
        names = dict(self._view_names)
        if basis_view.view_fingerprint not in names:
            raise ValueError("basis view was not registered with this module")
        if not basis_view.is_complete and not allow_incomplete:
            raise ValueError("incomplete basis view requires explicit opt in")
        transform = getattr(self, names[basis_view.view_fingerprint])
        source = self._runtime_source(inputs)
        view_basis = torch.einsum("af,foi->aoi", transform, self._basis)
        return torch.einsum("aoi,...i->...ao", view_basis, source)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Evaluate the complete basis without learned combination weights."""
        return self.evaluate_basis(inputs)

    def get_extra_state(self) -> dict[str, Any]:
        """Bind checkpoint constants to artifact and view fingerprints."""
        return {
            "state_schema_version": REGISTERED_COVARIANT_STATE_SCHEMA_VERSION,
            "artifact_fingerprint": self._artifact_fingerprint,
            "generator_fingerprint": self._generator_fingerprint,
            "catalog_fingerprint": self._catalog_fingerprint,
            "convention_version": self._convention_version,
            "signature": _signature_payload(self._signature),
            "basis_shape": tuple(self._basis.shape),
            "slot_descriptors": self._slot_descriptors,
            "view_descriptors": self._view_descriptors,
        }

    def set_extra_state(self, state: Any) -> None:
        """Reject state metadata from another artifact or basis view set."""
        if not isinstance(state, Mapping):
            raise RuntimeError("RegisteredCovariant state lacks metadata")
        expected = self.get_extra_state()
        if not _exact_metadata_equal(dict(state), expected):
            raise RuntimeError("RegisteredCovariant state metadata does not match")

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Validate every frozen constant and roll back failed loads."""
        extra_key = f"{prefix}_extra_state"
        if extra_key not in state_dict:
            raise RuntimeError("RegisteredCovariant state is unversioned")
        self.set_extra_state(state_dict[extra_key])
        constants = [("_basis", self._master_basis_bytes)]
        constants.extend(zip(self._lift_names, self._master_lift_bytes))
        constants.extend(
            (name, master)
            for (_, name), master in zip(self._view_names, self._master_view_bytes)
        )
        for name, master_bytes in constants:
            key = f"{prefix}{name}"
            if key not in state_dict:
                raise RuntimeError("RegisteredCovariant state is incomplete")
            master = _tensor_from_payload(
                json.loads(master_bytes),
                f"RegisteredCovariant master {name}",
            )
            _require_state_constant(state_dict[key], master, name)
        previous = {
            name: getattr(self, name).detach().clone() for name, _ in constants
        }
        previous_missing = len(missing_keys)
        previous_unexpected = len(unexpected_keys)
        previous_errors = len(error_msgs)
        try:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            if (
                len(missing_keys) != previous_missing
                or len(unexpected_keys) != previous_unexpected
                or len(error_msgs) != previous_errors
            ):
                raise RuntimeError(
                    "RegisteredCovariant state key validation failed"
                )
            for name, master_bytes in constants:
                incoming = state_dict[f"{prefix}{name}"]
                loaded = getattr(self, name)
                master = _tensor_from_payload(
                    json.loads(master_bytes),
                    f"RegisteredCovariant master {name}",
                )
                expected = master.to(
                    device=loaded.device,
                    dtype=incoming.dtype,
                ).to(dtype=loaded.dtype)
                if not torch.equal(loaded, expected):
                    raise RuntimeError(
                        f"RegisteredCovariant state {name} does not match its artifact"
                    )
        except Exception:
            for name, value in previous.items():
                setattr(self, name, value)
            raise


def _artifact_payload(artifact: CovariantCompilation) -> dict[str, Any]:
    """Return one complete canonical artifact payload."""
    return {
        "signature": _signature_payload(artifact.signature),
        "basis": _tensor_payload(artifact._basis),
        "slot_lifts": [_slot_lift_payload(lift) for lift in artifact.slot_lifts],
        "input_dimensions": list(artifact.input_dimensions),
        "effective_input_dimension": artifact.effective_input_dimension,
        "output_dimension": artifact.output_dimension,
        "basis_dimension": artifact.basis_dimension,
        "singular_values": _tensor_payload(artifact._singular_values),
        "threshold": _float_text(artifact.threshold, "threshold"),
        "singular_value_gap": _float_text(
            artifact.singular_value_gap,
            "singular_value_gap",
        ),
        "threshold_margin": _float_text(
            artifact.threshold_margin,
            "threshold_margin",
        ),
        "residual": _float_text(artifact.residual, "residual"),
        "nullspace_atol": _float_text(
            artifact.nullspace_atol,
            "nullspace_atol",
            allow_infinite=False,
        ),
        "nullspace_rtol": _float_text(
            artifact.nullspace_rtol,
            "nullspace_rtol",
            allow_infinite=False,
        ),
        "generator_fingerprint": artifact.generator_fingerprint,
        "type_catalog_fingerprint": artifact.type_catalog_fingerprint,
        "convention_version": artifact.convention_version,
        "cost": _cost_payload(artifact.cost),
        "artifact_fingerprint": artifact.artifact_fingerprint,
    }


def _artifact_core_payload(artifact: CovariantCompilation) -> dict[str, Any]:
    payload = _artifact_payload(artifact)
    payload.pop("artifact_fingerprint")
    return payload


def _validate_artifact_integrity(artifact: CovariantCompilation) -> None:
    """Recompute every artifact fingerprint before persistence or use."""
    expected = _fingerprint(
        "tfenn_covariant_artifact_v1",
        _artifact_core_payload(artifact),
    )
    if expected != artifact.artifact_fingerprint:
        raise RuntimeError("covariant artifact content does not match fingerprint")
    for lift in artifact.slot_lifts:
        payload = _slot_lift_payload(lift)
        fingerprint = payload.pop("fingerprint")
        expected_lift = _fingerprint("tfenn_slot_lift_v1", payload)
        if fingerprint != expected_lift:
            raise RuntimeError("slot lift content does not match fingerprint")


def _slot_from_payload(payload: Any) -> CSlot:
    if not isinstance(payload, Mapping):
        raise RuntimeError("slot payload must be a mapping")
    return CSlot(
        name=payload.get("name"),
        type_key=_type_key_from_payload(payload.get("type_key")),
        power=payload.get("power"),
        mode=payload.get("mode"),
    )


def _slot_lift_from_payload(payload: Any) -> SlotLiftArtifact:
    if not isinstance(payload, Mapping):
        raise RuntimeError("slot lift payload must be a mapping")
    lift = SlotLiftArtifact(
        _slot_from_payload(payload.get("slot")),
        payload.get("input_dimension"),
        _tensor_from_payload(payload.get("lift"), "slot lift"),
    )
    if _canonical_json_bytes(_slot_lift_payload(lift)) != _canonical_json_bytes(
        dict(payload)
    ):
        raise RuntimeError("slot lift payload validation failed")
    return lift


def _artifact_from_payload(payload: Any) -> CovariantCompilation:
    """Rebuild one artifact without invoking any compiler."""
    if not isinstance(payload, Mapping):
        raise RuntimeError("covariant artifact payload must be a mapping")
    signature = _signature_from_payload(payload.get("signature"))
    lift_payloads = payload.get("slot_lifts")
    if not isinstance(lift_payloads, list):
        raise RuntimeError("covariant artifact slot lifts are malformed")
    lifts = tuple(_slot_lift_from_payload(item) for item in lift_payloads)
    basis = _tensor_from_payload(payload.get("basis"), "covariant basis")
    singular_values = _tensor_from_payload(
        payload.get("singular_values"),
        "covariant singular values",
    )
    backend = IntertwinerCompilation(
        basis=basis,
        dimension=payload.get("basis_dimension"),
        singular_values=singular_values,
        threshold=_float_from_text(payload.get("threshold"), "threshold"),
        singular_value_gap=_float_from_text(
            payload.get("singular_value_gap"),
            "singular_value_gap",
        ),
        threshold_margin=_float_from_text(
            payload.get("threshold_margin"),
            "threshold_margin",
        ),
        residual=_float_from_text(payload.get("residual"), "residual"),
        convention_version=TENSOR_CONVENTION_VERSION,
        nullspace_atol=_float_from_text(
            payload.get("nullspace_atol"),
            "nullspace_atol",
            allow_infinite=False,
        ),
        nullspace_rtol=_float_from_text(
            payload.get("nullspace_rtol"),
            "nullspace_rtol",
            allow_infinite=False,
        ),
    )
    artifact = CovariantCompilation(
        signature,
        basis,
        lifts,
        backend,
        payload.get("output_dimension"),
        payload.get("generator_fingerprint"),
        payload.get("type_catalog_fingerprint"),
        expected_fingerprint=payload.get("artifact_fingerprint"),
    )
    if _canonical_json_bytes(_artifact_payload(artifact)) != _canonical_json_bytes(
        dict(payload)
    ):
        raise RuntimeError("covariant artifact payload validation failed")
    return artifact


def _view_payload(view: CovariantBasisView) -> dict[str, Any]:
    """Return one complete canonical basis view payload."""
    return {
        "parent_artifact_fingerprint": view.parent_artifact_fingerprint,
        "transform": _tensor_payload(view._transform),
        "full_dimension": view.full_dimension,
        "active_dimension": view.active_dimension,
        "transform_rank": view.transform_rank,
        "is_complete": view.is_complete,
        "singular_values": _tensor_payload(view._singular_values),
        "threshold": _float_text(view.threshold, "view threshold"),
        "singular_value_gap": _float_text(
            view.singular_value_gap,
            "view singular gap",
        ),
        "threshold_margin": _float_text(
            view.threshold_margin,
            "view threshold margin",
        ),
        "rank_atol": _float_text(
            view.rank_atol,
            "view rank_atol",
            allow_infinite=False,
        ),
        "rank_rtol": _float_text(
            view.rank_rtol,
            "view rank_rtol",
            allow_infinite=False,
        ),
        "cost": _cost_payload(view.cost),
        "convention_version": COVARIANT_CONVENTION_VERSION,
        "view_fingerprint": view.view_fingerprint,
    }


def _view_core_payload(view: CovariantBasisView) -> dict[str, Any]:
    payload = _view_payload(view)
    payload.pop("view_fingerprint")
    return payload


def _validate_view_integrity(view: CovariantBasisView) -> None:
    expected = _fingerprint("tfenn_covariant_view_v1", _view_core_payload(view))
    if expected != view.view_fingerprint:
        raise RuntimeError("basis view content does not match fingerprint")


def _view_from_payload(
    payload: Any,
    parent: CovariantCompilation,
) -> CovariantBasisView:
    if not isinstance(payload, Mapping):
        raise RuntimeError("basis view payload must be a mapping")
    view = CovariantBasisView(
        parent,
        _tensor_from_payload(payload.get("transform"), "basis view transform"),
        rank_atol=_float_from_text(
            payload.get("rank_atol"),
            "view rank_atol",
            allow_infinite=False,
        ),
        rank_rtol=_float_from_text(
            payload.get("rank_rtol"),
            "view rank_rtol",
            allow_infinite=False,
        ),
        expected_fingerprint=payload.get("view_fingerprint"),
    )
    if _canonical_json_bytes(_view_payload(view)) != _canonical_json_bytes(
        dict(payload)
    ):
        raise RuntimeError("basis view payload validation failed")
    return view


def _registry_primary_key(
    signature: CSignature,
    catalog_fingerprint: str,
    convention_version: str = COVARIANT_CONVENTION_VERSION,
) -> tuple[str, str, str]:
    if not isinstance(signature, CSignature):
        raise TypeError("signature must be a CSignature")
    _validate_fingerprint(catalog_fingerprint, "catalog_fingerprint")
    if convention_version != COVARIANT_CONVENTION_VERSION:
        raise ValueError("registry convention version is unsupported")
    signature_key = _canonical_json_bytes(_signature_payload(signature)).decode("ascii")
    return signature_key, catalog_fingerprint, convention_version


class CRegistry:
    """Store complete artifacts and explicitly selected immutable basis views."""

    def __init__(self) -> None:
        self._artifacts: dict[tuple[str, str, str], CovariantCompilation] = {}
        self._views: dict[
            tuple[tuple[str, str, str], str],
            CovariantBasisView,
        ] = {}

    def _validate_contents(self) -> None:
        """Validate every parent, view, and parent binding in the registry."""
        for artifact in self._artifacts.values():
            _validate_artifact_integrity(artifact)
        parents = {
            artifact.artifact_fingerprint: artifact
            for artifact in self._artifacts.values()
        }
        for view in self._views.values():
            _validate_view_integrity(view)
            try:
                parent = parents[view.parent_artifact_fingerprint]
            except KeyError as error:
                raise RuntimeError("registered basis view parent is missing") from error
            _validate_artifact_integrity(parent)

    @property
    def artifacts(self) -> tuple[CovariantCompilation, ...]:
        """Return complete artifacts in deterministic registry key order."""
        self._validate_contents()
        return tuple(self._artifacts[key] for key in sorted(self._artifacts))

    @property
    def views(self) -> tuple[CovariantBasisView, ...]:
        """Return basis views in deterministic registry key order."""
        self._validate_contents()
        return tuple(self._views[key] for key in sorted(self._views))

    @property
    def fingerprint(self) -> str:
        """Return a fingerprint of all ordered registry entries."""
        self._validate_contents()
        payload = {
            "artifacts": [
                {
                    "key": list(key),
                    "artifact_fingerprint": self._artifacts[key].artifact_fingerprint,
                }
                for key in sorted(self._artifacts)
            ],
            "views": [
                {
                    "key": [*key[0], key[1]],
                    "view_fingerprint": self._views[key].view_fingerprint,
                }
                for key in sorted(self._views)
            ],
            "convention_version": COVARIANT_CONVENTION_VERSION,
        }
        return _fingerprint("tfenn_covariant_registry_v1", payload)

    def register(
        self,
        artifact: CovariantCompilation | CovariantBasisView,
    ) -> None:
        """Register one complete artifact or one view bound to a known parent."""
        if isinstance(artifact, CovariantCompilation):
            _validate_artifact_integrity(artifact)
            key = _registry_primary_key(
                artifact.signature,
                artifact.type_catalog_fingerprint,
                artifact.convention_version,
            )
            existing = self._artifacts.get(key)
            if existing is not None:
                _validate_artifact_integrity(existing)
                if existing.artifact_fingerprint != artifact.artifact_fingerprint:
                    raise ValueError("registry primary key already has another artifact")
                return
            self._artifacts[key] = artifact
            return
        if isinstance(artifact, CovariantBasisView):
            _validate_view_integrity(artifact)
            parent_items = [
                (key, parent)
                for key, parent in self._artifacts.items()
                if parent.artifact_fingerprint
                == artifact.parent_artifact_fingerprint
            ]
            if len(parent_items) != 1:
                raise ValueError("basis view parent artifact is not registered")
            parent_key, parent = parent_items[0]
            _validate_artifact_integrity(parent)
            key = (parent_key, artifact.view_fingerprint)
            existing_view = self._views.get(key)
            if existing_view is not None:
                _validate_view_integrity(existing_view)
                if existing_view.view_fingerprint != artifact.view_fingerprint:
                    raise ValueError("registry view key already has another view")
                return
            self._views[key] = artifact
            return
        raise TypeError("registry accepts CovariantCompilation or CovariantBasisView")

    def resolve(
        self,
        signature: CSignature,
        catalog_fingerprint: str,
        *,
        convention_version: str = COVARIANT_CONVENTION_VERSION,
        view_fingerprint: str | None = None,
        require_complete: bool = True,
    ) -> CovariantCompilation | CovariantBasisView:
        """Resolve a complete parent or one explicitly selected registered view."""
        if not isinstance(require_complete, bool):
            raise TypeError("require_complete must be bool")
        key = _registry_primary_key(
            signature,
            catalog_fingerprint,
            convention_version,
        )
        try:
            parent = self._artifacts[key]
        except KeyError as error:
            raise KeyError("registry does not contain the requested signature") from error
        _validate_artifact_integrity(parent)
        if view_fingerprint is None:
            return parent
        _validate_fingerprint(view_fingerprint, "view_fingerprint")
        try:
            view = self._views[(key, view_fingerprint)]
        except KeyError as error:
            raise KeyError("registry does not contain the requested basis view") from error
        _validate_view_integrity(view)
        if require_complete and not view.is_complete:
            raise ValueError("incomplete basis view requires require_complete False")
        return view

    def _save_payload(self) -> dict[str, Any]:
        artifact_entries = [
            {"key": list(key), "artifact": _artifact_payload(self._artifacts[key])}
            for key in sorted(self._artifacts)
        ]
        view_entries = [
            {"key": [*key[0], key[1]], "view": _view_payload(self._views[key])}
            for key in sorted(self._views)
        ]
        core = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "convention_version": COVARIANT_CONVENTION_VERSION,
            "artifacts": artifact_entries,
            "views": view_entries,
        }
        return {**core, "registry_fingerprint": self.fingerprint}

    def save(self, path: str | Path) -> None:
        """Write deterministic canonical JSON after validating all contents."""
        for artifact in self.artifacts:
            _validate_artifact_integrity(artifact)
        for view in self.views:
            _validate_view_integrity(view)
        destination = Path(path)
        destination.write_bytes(_canonical_json_bytes(self._save_payload()))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_complete: bool = True,
    ) -> CRegistry:
        """Load and validate a registry without compiling any artifact."""
        if not isinstance(require_complete, bool):
            raise TypeError("require_complete must be bool")
        try:
            payload = json.loads(
                Path(path).read_bytes(),
                parse_constant=_reject_json_constant,
            )
        except Exception as error:
            raise RuntimeError("registry file is not valid canonical JSON") from error
        if not isinstance(payload, Mapping):
            raise RuntimeError("registry payload must be a mapping")
        if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise RuntimeError("registry schema version is obsolete")
        if payload.get("convention_version") != COVARIANT_CONVENTION_VERSION:
            raise RuntimeError("registry convention version is obsolete")
        artifact_entries = payload.get("artifacts")
        view_entries = payload.get("views")
        if not isinstance(artifact_entries, list) or not isinstance(view_entries, list):
            raise RuntimeError("registry entry lists are malformed")
        registry = cls()
        parents_by_fingerprint: dict[str, CovariantCompilation] = {}
        for entry in artifact_entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError("registry artifact entry is malformed")
            try:
                artifact = _artifact_from_payload(entry.get("artifact"))
                expected_key = list(
                    _registry_primary_key(
                        artifact.signature,
                        artifact.type_catalog_fingerprint,
                        artifact.convention_version,
                    )
                )
                registry.register(artifact)
            except RuntimeError:
                raise
            except (TypeError, ValueError, KeyError) as error:
                raise RuntimeError(
                    "registry artifact validation failed"
                ) from error
            if entry.get("key") != expected_key:
                raise RuntimeError("registry artifact key validation failed")
            parents_by_fingerprint[artifact.artifact_fingerprint] = artifact
        for entry in view_entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError("registry view entry is malformed")
            view_payload = entry.get("view")
            if not isinstance(view_payload, Mapping):
                raise RuntimeError("registry view payload is malformed")
            parent_fingerprint = view_payload.get("parent_artifact_fingerprint")
            try:
                _validate_fingerprint(
                    parent_fingerprint,
                    "parent_artifact_fingerprint",
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError("registry view parent fingerprint is malformed") from error
            try:
                parent = parents_by_fingerprint[parent_fingerprint]
            except (KeyError, TypeError) as error:
                raise RuntimeError("registry view parent is missing") from error
            try:
                view = _view_from_payload(view_payload, parent)
                registry.register(view)
            except RuntimeError:
                raise
            except (TypeError, ValueError, KeyError) as error:
                raise RuntimeError("registry view validation failed") from error
            if require_complete and not view.is_complete:
                raise RuntimeError(
                    "registry contains an incomplete view and requires explicit opt in"
                )
            parent_key = _registry_primary_key(
                parent.signature,
                parent.type_catalog_fingerprint,
                parent.convention_version,
            )
            expected_key = [*parent_key, view.view_fingerprint]
            if entry.get("key") != expected_key:
                raise RuntimeError("registry view key validation failed")
        expected_top = registry._save_payload()
        if _canonical_json_bytes(dict(payload)) != _canonical_json_bytes(expected_top):
            raise RuntimeError("registry payload or fingerprint validation failed")
        return registry
