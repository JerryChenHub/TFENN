"""Validate the fixed-shape causal controls used by the G model factory."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from experiments.benzene_pair.e_series.model_factory import build_e_series_model
from experiments.benzene_pair.g_series.model_factory import build_g_series_model
from experiments.benzene_pair.train import _proper_d6_generators


DTYPE = torch.float64
MODEL_SEED = 20_260_822


def _rotation(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))
    return torch.matrix_exp(skew)


def _pairs() -> tuple[Tensor, Tensor]:
    centers = torch.tensor(
        (
            ((0.0, 0.0, 0.0), (4.5, -0.7, 3.9)),
            ((0.1, -0.1, 0.2), (4.7, -0.5, 4.0)),
        ),
        dtype=DTYPE,
    )
    root = _rotation(
        torch.tensor(((0.1, -0.2, 0.05), (0.0, 0.1, -0.1)), dtype=DTYPE)
    )
    sender = _rotation(
        torch.tensor(((-0.1, 0.15, 0.08), (0.2, -0.05, 0.1)), dtype=DTYPE)
    )
    return centers, torch.stack((root, sender), dim=1)


@pytest.fixture(scope="module")
def compiled_controls() -> tuple[nn.Module, nn.Module, nn.Module]:
    generators = _proper_d6_generators()
    names = ("sixfold", "twofold")
    torch.manual_seed(MODEL_SEED)
    native = build_e_series_model(
        "E311", generators, generator_names=names
    ).eval()
    torch.manual_seed(MODEL_SEED)
    control = build_g_series_model(
        "G101", generators, generator_names=names
    ).eval()
    torch.manual_seed(MODEL_SEED)
    pair_on = build_g_series_model(
        "G105", generators, generator_names=names
    ).eval()
    return native, control, pair_on


def test_g_control_reproduces_native_e311_initialization(
    compiled_controls: tuple[nn.Module, nn.Module, nn.Module],
) -> None:
    native, control, _pair_on = compiled_controls
    centers, frames = _pairs()
    with torch.no_grad():
        native_value = native(centers, frames)
        control_value = control(centers, frames)
    torch.testing.assert_close(control_value, native_value, rtol=1e-12, atol=1e-12)
    manifest = control.g_series_manifest
    assert manifest["reference_model_id"] == "E311"
    assert manifest["fixed_shape_supernet"] is True
    assert manifest["inactive_covariant_path_count"] > 0


def test_generic_pair_factor_changes_only_runtime_activity(
    compiled_controls: tuple[nn.Module, nn.Module, nn.Module],
) -> None:
    _native, control, pair_on = compiled_controls
    control_state = control.state_dict()
    pair_state = pair_on.state_dict()
    assert control_state.keys() == pair_state.keys()
    for name in control_state:
        left = control_state[name]
        right = pair_state[name]
        if isinstance(left, Tensor):
            assert isinstance(right, Tensor)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right
    control_roles = {
        str(item["role"]): bool(item["active"])
        for item in control.coefficient_head_role_manifest
    }
    pair_roles = {
        str(item["role"]): bool(item["active"])
        for item in pair_on.coefficient_head_role_manifest
    }
    assert control_roles.keys() == pair_roles.keys()
    changed = {
        role
        for role in control_roles
        if control_roles[role] != pair_roles[role]
    }
    assert changed
    families = {
        str(item["role"]): str(item["path_family"])
        for item in control.coefficient_head_role_manifest
    }
    assert {families[role] for role in changed} == {"pair"}
    assert all(not control_roles[role] and pair_roles[role] for role in changed)
    assert any(
        item["kind"] == "pair" and bool(item["active"])
        for item in control.descriptor_role_manifest
    )
    assert any(
        item["kind"] == "stf" and bool(item["active"])
        for item in control.descriptor_role_manifest
    )


def test_g_fixed_shape_models_keep_identical_nominal_counts(
    compiled_controls: tuple[nn.Module, nn.Module, nn.Module],
) -> None:
    native, control, pair_on = compiled_controls
    control_count = sum(parameter.numel() for parameter in control.parameters())
    pair_count = sum(parameter.numel() for parameter in pair_on.parameters())
    assert control_count == pair_count
    assert control_count > 0
    native_count = sum(parameter.numel() for parameter in native.parameters())
    control_audit = control.g_series_manifest
    pair_audit = pair_on.g_series_manifest
    assert control_audit["nominal_trainable_parameter_count"] == control_count
    assert pair_audit["nominal_trainable_parameter_count"] == pair_count
    assert control_audit["causally_active_parameter_scalar_count"] == native_count
    assert pair_audit["causally_active_parameter_scalar_count"] > native_count
