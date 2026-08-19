"""G series fixed-shape E311 mechanism experiments."""

from .catalog import (
    CARRIER_SPECS,
    FACTORIAL_SPECS,
    GModelSpec,
    G_SERIES_SPECS,
    LEGACY_SPECS,
    SEED_BLOCK_SPECS,
    VARIANT_SPECS,
    get_group_specs,
    get_model_spec,
    get_seed_specs,
    get_variant_specs,
)
from .model_factory import build_g_series_model


__all__ = [
    "CARRIER_SPECS",
    "FACTORIAL_SPECS",
    "GModelSpec",
    "G_SERIES_SPECS",
    "LEGACY_SPECS",
    "SEED_BLOCK_SPECS",
    "VARIANT_SPECS",
    "build_g_series_model",
    "get_group_specs",
    "get_model_spec",
    "get_seed_specs",
    "get_variant_specs",
]
