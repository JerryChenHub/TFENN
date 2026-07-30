from __future__ import annotations

import ast
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    *(f"R{i}{j}" for i in range(1, 4) for j in range(1, 4)),
    "x1",
    "x2",
    "x3",
    "F1",
    "F2",
    "F3",
    "M1",
    "M2",
    "M3",
]


def check_source_syntax() -> int:
    paths = sorted(
        path
        for root in (
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "experiments",
            PROJECT_ROOT / "tests",
        )
        for path in root.rglob("*.py")
    )
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return len(paths)


def check_data() -> tuple[int, int]:
    from TFENN.data import load_benzene_pair_csv

    paths = sorted((PROJECT_ROOT / "data" / "benzene_pair").glob("*.csv"))
    if not paths:
        raise AssertionError("No experiment data files were found")
    for path in paths:
        arrays = load_benzene_pair_csv(path)
        if len(arrays) != 10_000:
            raise AssertionError(f"Unexpected sample count in {path.name}")
    return len(paths), len(EXPECTED_COLUMNS)


def check_model() -> tuple[int, ...]:
    import torch

    from TFENN.models import D6TensorBasisNetV1

    torch.set_default_dtype(torch.float64)
    model = D6TensorBasisNetV1(
        x_hidden_channels=4,
        r_hidden_channels=4,
        num_x_layers=1,
        num_r_layers=1,
        r_to_x_channels=4,
        out_channels=2,
        num_head_layers=1,
        head_hidden_channels=4,
    )
    x = torch.randn(3, 3)
    rotation = torch.linalg.qr(torch.randn(3, 3, 3)).Q
    negative = torch.linalg.det(rotation) < 0
    rotation[negative, :, -1] *= -1
    output = model(x, rotation)
    if output.shape != (3, 2, 3):
        raise AssertionError(f"Unexpected model output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("Model output contains nonfinite values")
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Model gradients are missing or nonfinite")
    return tuple(output.shape)


def check_opls() -> tuple[tuple[int, ...], tuple[int, ...]]:
    import numpy as np

    from TFENN.data import BenzenePairGenerationConfig, sample_benzene_pair
    from opls2020.core.force_field import OPLS2020_Force_Field
    from opls2020.core.molecule import Benzene

    molecule_a = Benzene()
    molecule_b = Benzene()
    parameters_a = molecule_a.opls_params[molecule_a._atom_types[0]]
    parameters_b = molecule_b.opls_params[molecule_b._atom_types[0]]
    force_field = OPLS2020_Force_Field(cutoff=12.0, smoothing="linear")
    force = force_field.Non_bond_Force(
        molecule_a.atom_position[0],
        parameters_a,
        molecule_b.atom_position[0] + np.array([5.0, 0.0, 0.0]),
        parameters_b,
    )
    if force.shape != (3,) or not np.isfinite(force).all():
        raise AssertionError("OPLS compatibility force is invalid")

    rotation, displacement, net_force, net_moment = sample_benzene_pair(
        np.random.default_rng(7),
        BenzenePairGenerationConfig(
            sample_count=1,
            distance_range=(6.5, 6.6),
            min_separation=0.0,
        ),
    )
    arrays = (rotation, displacement, net_force, net_moment)
    if not all(np.isfinite(array).all() for array in arrays):
        raise AssertionError("TFENN data generation returned nonfinite values")
    return tuple(rotation.shape), tuple(net_force.shape)


CHECKS = {
    "source": check_source_syntax,
    "data": check_data,
    "model": check_model,
    "opls": check_opls,
}


def run_check(name: str) -> None:
    if name not in CHECKS:
        raise SystemExit(f"Unknown smoke test: {name}")
    print(json.dumps({name: CHECKS[name]()}))


def main() -> None:
    for name in CHECKS:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), name],
            text=True,
            capture_output=True,
        )
        if completed.stdout:
            print(completed.stdout.strip())
        if completed.returncode:
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
            raise SystemExit(completed.returncode)

    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in (
            "torch",
            "numpy",
            "scipy",
            "opls2020-static",
        )
    }
    print("TFENN smoke test passed", versions)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_check(sys.argv[1])
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("Usage: smoke_test.py [source|data|model|opls]")
