from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .benzene_pair import BENZENE_PAIR_COLUMNS


@dataclass(frozen=True, slots=True)
class BenzenePairGenerationConfig:
    sample_count: int = 10_000
    seed: int | None = None
    distance_range: tuple[float, float] = (4.5, 8.0)
    cutoff: float = 12.0
    min_separation: float = 3.0
    smoothing: str = "linear"
    target_scale: float = 1.0
    max_attempts_per_sample: int = 500

    def __post_init__(self) -> None:
        minimum, maximum = self.distance_range
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive.")
        if not 0.0 < minimum < maximum:
            raise ValueError("distance_range must contain two increasing positive values.")
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive.")
        if self.min_separation < 0.0:
            raise ValueError("min_separation cannot be negative.")
        if not self.smoothing:
            raise ValueError("smoothing cannot be empty.")
        if not np.isfinite(self.target_scale):
            raise ValueError("target_scale must be finite.")
        if self.max_attempts_per_sample < 1:
            raise ValueError("max_attempts_per_sample must be positive.")

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": "benzene_pair",
            "sample_count": self.sample_count,
            "seed": self.seed,
            "distance_range": list(self.distance_range),
            "cutoff": self.cutoff,
            "min_separation": self.min_separation,
            "smoothing": self.smoothing,
            "target_scale": self.target_scale,
            "max_attempts_per_sample": self.max_attempts_per_sample,
            "columns": list(BENZENE_PAIR_COLUMNS),
        }


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _minimum_interatomic_distance(
    first_positions: np.ndarray,
    second_positions: np.ndarray,
) -> float:
    difference = first_positions[:, None, :] - second_positions[None, :, :]
    return float(np.linalg.norm(difference, axis=2).min())


def sample_benzene_pair(
    rng: np.random.Generator,
    config: BenzenePairGenerationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one valid relative pose and compute its OPLS targets."""
    try:
        from opls2020.core.force_field import OPLS2020_Force_Field
        from opls2020.core.molecule import Benzene
    except ImportError as error:
        raise ImportError(
            "Benzene pair generation requires the opls2020 package."
        ) from error

    for _attempt in range(1, config.max_attempts_per_sample + 1):
        first_rotation = _random_rotation_matrix(rng)
        second_rotation = _random_rotation_matrix(rng)

        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        distance = rng.uniform(*config.distance_range)
        displacement = first_rotation.T @ (distance * direction)
        relative_rotation = first_rotation.T @ second_rotation

        first_molecule = Benzene()
        second_molecule = Benzene()
        first_molecule.direction_vectors = (
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        second_molecule.direction_vectors = (
            displacement,
            relative_rotation[:, 0],
            relative_rotation[:, 2],
        )

        first_positions = first_molecule.atom_position
        second_positions = second_molecule.atom_position
        if (
            _minimum_interatomic_distance(first_positions, second_positions)
            >= config.min_separation
        ):
            break
    else:
        raise RuntimeError(
            "Could not sample a configuration satisfying "
            f"min_separation={config.min_separation} within "
            f"{config.max_attempts_per_sample} attempts."
        )

    force_field = OPLS2020_Force_Field(
        cutoff=config.cutoff,
        smoothing=config.smoothing,
    )
    atom_force = np.zeros_like(first_positions)
    first_types = first_molecule._atom_types
    second_types = second_molecule._atom_types
    parameters = first_molecule.opls_params
    for first_index, first_position in enumerate(first_positions):
        first_parameters = parameters[first_types[first_index]]
        for second_index, second_position in enumerate(second_positions):
            second_parameters = parameters[second_types[second_index]]
            atom_force[first_index] += force_field.Non_bond_Force(
                first_position,
                first_parameters,
                second_position,
                second_parameters,
            )

    first_molecule._atom_force = atom_force
    force = np.asarray(first_molecule.net_force, dtype=np.float64)
    moment = np.asarray(first_molecule.net_moment, dtype=np.float64)
    return relative_rotation, displacement, force, moment


def generate_benzene_pair_dataset(
    output_csv: str | Path,
    config: BenzenePairGenerationConfig,
    *,
    progress_every: int | None = 1_000,
) -> tuple[Path, Path]:
    """Write a CSV table and a JSON parameter sidecar with the same stem."""
    csv_path = Path(output_csv)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("output_csv must use the .csv suffix.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = csv_path.with_suffix(".json")
    rng = np.random.default_rng(config.seed)

    with csv_path.open("w", newline="", encoding="utf_8") as stream:
        writer = csv.writer(stream)
        writer.writerow(BENZENE_PAIR_COLUMNS)
        for sample_index in range(config.sample_count):
            rotation, displacement, force, moment = sample_benzene_pair(rng, config)
            target_scale = config.target_scale
            writer.writerow(
                np.concatenate(
                    (
                        rotation.reshape(9),
                        displacement,
                        force * target_scale,
                        moment * target_scale,
                    )
                )
            )
            if progress_every and (sample_index + 1) % progress_every == 0:
                print(f"{sample_index + 1}/{config.sample_count} samples written")

    metadata_path.write_text(
        json.dumps(config.metadata(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf_8",
    )
    return csv_path, metadata_path
