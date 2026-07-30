"""Tensor basis networks for the benzene pair problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import (
        D6GroupAverageNetV1,
        D6SymmetrizedMLPBaselineV1,
        D6TensorBasisNetV1,
        D6TensorBasisNetV2,
        MLPBaselineV1,
    )

__version__ = "0.1.0"

_MODEL_NAMES = {
    "D6GroupAverageNetV1",
    "D6SymmetrizedMLPBaselineV1",
    "D6TensorBasisNetV1",
    "D6TensorBasisNetV2",
    "MLPBaselineV1",
}


def __getattr__(name: str) -> Any:
    if name not in _MODEL_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import models

    value = getattr(models, name)
    globals()[name] = value
    return value


__all__ = [
    "D6GroupAverageNetV1",
    "D6SymmetrizedMLPBaselineV1",
    "D6TensorBasisNetV1",
    "D6TensorBasisNetV2",
    "MLPBaselineV1",
    "__version__",
]
