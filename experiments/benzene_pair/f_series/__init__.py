"""F series strict typed flow experiments."""

from .catalog import (
    F_SERIES_SPECS,
    F0_SPECS,
    F1_SPECS,
    F2_SPECS,
    PROJECT_SPECS,
    EXECUTION_SHARD_SPECS,
    FModelSpec,
    get_experiment_specs,
    get_science_experiment_specs,
    get_execution_shard_specs,
    get_model_spec,
    get_project_specs,
)
from .model_factory import build_f_series_model, strict_config_from_spec


__all__ = [
    "F_SERIES_SPECS",
    "F0_SPECS",
    "F1_SPECS",
    "F2_SPECS",
    "PROJECT_SPECS",
    "EXECUTION_SHARD_SPECS",
    "FModelSpec",
    "build_f_series_model",
    "get_experiment_specs",
    "get_science_experiment_specs",
    "get_execution_shard_specs",
    "get_model_spec",
    "get_project_specs",
    "strict_config_from_spec",
]
