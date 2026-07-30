from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class CenteredSigmoidV1(nn.Module):
    """A zero centered sigmoid with configurable output span."""

    def __init__(self, span: float = 3.0) -> None:
        super().__init__()
        self.span = float(span)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.span * (torch.sigmoid(value) - 0.5)


ActivationFactory = Callable[[], nn.Module]


def activation_factory(name: str) -> ActivationFactory:
    """Return a fresh activation constructor for an experiment name."""
    normalized = name.lower()
    factories: dict[str, ActivationFactory] = {
        "sigmoid_centered": CenteredSigmoidV1,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
    }
    try:
        return factories[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(factories))
        raise ValueError(
            f"Unknown activation {name!r}. Expected one of: {choices}."
        ) from error


__all__ = ["ActivationFactory", "CenteredSigmoidV1", "activation_factory"]
