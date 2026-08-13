from __future__ import annotations

import ast
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    from TFENN.data import load_benzene_cluster_csv

    path = PROJECT_ROOT / "data" / "benzene_pair" / "benzene_pair_opls_2_0_0_v1.csv"
    arrays = load_benzene_cluster_csv(path)
    if len(arrays) != 1_000 or arrays.molecule_count != 2:
        raise AssertionError(f"Unexpected dataset shape in {path.name}")
    return len(arrays), arrays.molecule_count


def check_model() -> tuple[int, ...]:
    import torch

    from TFENN.models import (
        PairPipelineConfig,
        StageConfig,
        build_invariant_gate_pipeline,
    )

    torch.set_default_dtype(torch.float64)
    root_three_over_two = 3.0**0.5 / 2.0
    generators = torch.tensor(
        (
            (
                (0.5, -root_three_over_two, 0.0),
                (root_three_over_two, 0.5, 0.0),
                (0.0, 0.0, 1.0),
            ),
            ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        )
    )
    config = PairPipelineConfig(
        stages=(
            StageConfig("a1", "A", ("x",), 2, lift_orders=(1,)),
            StageConfig("b1", "B", ("r",), 2, lift_orders=(1,)),
            StageConfig("a2", "A", ("a1", "b1"), 1, lift_orders=(1,)),
        ),
        output_stage="a2",
        anchor_ranks=(1, 2),
    )
    model = build_invariant_gate_pipeline(
        generators,
        config,
        generator_names=("sixfold", "twofold"),
    )
    centers = torch.randn(3, 2, 3)
    frames = torch.eye(3).expand(3, 2, 3, 3).clone()
    output = model(centers, frames)
    if output.shape != (3, 3):
        raise AssertionError(f"Unexpected model output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("Model output contains nonfinite values")
    output.square().mean().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise AssertionError("Model gradients are missing or nonfinite")
    return tuple(output.shape)


def check_opls() -> tuple[tuple[int, ...], tuple[int, ...]]:
    import numpy as np

    from TFENN.data import BenzenePairGenerationConfig, sample_benzene_pair
    from opls2020 import (
        MoleculeInstance,
        Pose,
        StaticEngine,
        SystemSpec,
        __version__,
        benzene,
    )

    if __version__ != "1.0.0":
        raise AssertionError(f"Unexpected OPLS version: {__version__}")
    species = benzene()
    system = SystemSpec(
        configuration_id="tfenn_docker_smoke",
        species={species.species_id: species},
        molecules=(
            MoleculeInstance("benzene_0001", species.species_id, Pose()),
            MoleculeInstance(
                "benzene_0002",
                species.species_id,
                Pose(center_A=(8.0, 0.0, 0.0)),
            ),
        ),
    )
    result = StaticEngine(use_neighbor_list=False).evaluate(system)
    if (
        result.model.model_semantics_id
        != "opls2020_open_direct_quintic_10_12_codata2022_v1"
    ):
        raise AssertionError("Unexpected OPLS model semantics")
    if result.molecular_forces_kcal_mol_A.shape != (2, 3):
        raise AssertionError("Unexpected OPLS molecular force shape")
    if result.molecular_torques_kcal_mol.shape != (2, 3):
        raise AssertionError("Unexpected OPLS molecular torque shape")
    result_arrays = (
        result.molecular_forces_kcal_mol_A,
        result.molecular_torques_kcal_mol,
        result.virial_kcal_mol,
    )
    if not all(np.isfinite(array).all() for array in result_arrays):
        raise AssertionError("OPLS result contains nonfinite values")
    if not np.allclose(result.total_force, 0.0, atol=1.0e-12, rtol=0.0):
        raise AssertionError("OPLS total force is not conserved")

    rotation, displacement, net_force, net_moment = sample_benzene_pair(
        np.random.default_rng(7),
        BenzenePairGenerationConfig(
            sample_count=1,
            distance_range_A=(6.5, 6.6),
            min_interatomic_distance_A=0.0,
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
