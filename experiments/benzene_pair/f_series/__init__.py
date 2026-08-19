"""F series strict typed flow experiments."""

from .catalog import (
    F_SERIES_SPECS,
    F0_SPECS,
    F1_SPECS,
    F2_SPECS,
    F3_SPECS,
    FModelSpec,
    get_experiment_specs,
    get_model_spec,
)
from .model_factory import build_f_series_model, strict_config_from_spec


__all__ = [
    "F_SERIES_SPECS",
    "F0_SPECS",
    "F1_SPECS",
    "F2_SPECS",
    "F3_SPECS",
    "FModelSpec",
    "build_f_series_model",
    "get_experiment_specs",
    "get_model_spec",
    "strict_config_from_spec",
]
