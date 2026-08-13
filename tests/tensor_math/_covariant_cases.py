"""Construct explicit typed catalogs used by covariant registry tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from TFENN.tensor_math import (
    BBlockManifest,
    GeneratorSystem,
    TypeCatalog,
    TypeKey,
    build_type_catalog,
)

from ._groups import benzene_generators, c60_generators, methane_generators


@dataclass(frozen=True)
class CovariantCase:
    """Store one molecule generator system and explicit B manifest."""

    name: str
    catalog: TypeCatalog
    manifest: tuple[BBlockManifest, ...]


def benzene_covariant_case() -> CovariantCase:
    """Return the proper D6 catalog with explicit rank two and six blocks."""
    system = GeneratorSystem(("sixfold", "twofold"), benzene_generators())
    manifest = (
        BBlockManifest(0, 2, (0,), 1, "benzene_b_rank_2"),
        BBlockManifest(1, 6, (0,), 1, "benzene_b_rank_6"),
    )
    return CovariantCase("benzene", build_type_catalog(system, manifest), manifest)


def methane_covariant_case() -> CovariantCase:
    """Return the rotational tetrahedral catalog with explicit rank three B."""
    system = GeneratorSystem(("twofold", "threefold"), methane_generators())
    manifest = (BBlockManifest(0, 3, (0,), 1, "methane_b_rank_3"),)
    return CovariantCase("methane", build_type_catalog(system, manifest), manifest)


def c60_covariant_case() -> CovariantCase:
    """Return the rotational icosahedral catalog with primitive rank six B."""
    system = GeneratorSystem(("fivefold", "threefold"), c60_generators())
    manifest = (BBlockManifest(0, 6, (0,), 1, "c60_b_rank_6"),)
    return CovariantCase("c60", build_type_catalog(system, manifest), manifest)


def transformed_inputs(
    inputs: dict[str, torch.Tensor],
    slot_keys: dict[str, TypeKey],
    catalog: TypeCatalog,
    word: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Apply one ordered generator word to every final representation axis."""
    result = {}
    for name, value in inputs.items():
        block = catalog.resolve(slot_keys[name])
        action = torch.eye(block.representation_dim, dtype=torch.float64)
        for generator_index in word:
            action = block.actions[generator_index] @ action
        result[name] = value @ action.T
    return result
