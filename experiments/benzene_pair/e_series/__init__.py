"""E series benzene pair experiments."""

from .catalog import (
    E_SERIES_SPECS,
    E0_SPECS,
    E1_SPECS,
    E2_SPECS,
    E3_SPECS,
    E4_SPECS,
    EModelSpec,
    get_experiment_specs,
    get_model_spec,
)
from .model_factory import build_e_series_model, pipeline_config_from_spec


__all__ = [
    "E_SERIES_SPECS",
    "E0_SPECS",
    "E1_SPECS",
    "E2_SPECS",
    "E3_SPECS",
    "E4_SPECS",
    "EModelSpec",
    "build_e_series_model",
    "get_experiment_specs",
    "get_model_spec",
    "pipeline_config_from_spec",
]
