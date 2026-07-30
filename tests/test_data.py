from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from TFENN.data import (
    BENZENE_PAIR_COLUMNS,
    BenzenePairDataset,
    BenzenePairGenerationConfig,
    generate_benzene_pair_dataset,
    load_benzene_pair_csv,
    load_benzene_pair_metadata,
)


def test_real_benzene_pair_data_and_metadata_load(
    benzene_csv_path: Path,
) -> None:
    arrays = load_benzene_pair_csv(benzene_csv_path)
    metadata = load_benzene_pair_metadata(benzene_csv_path)
    dataset = BenzenePairDataset(benzene_csv_path, dtype=torch.float64)

    assert len(arrays) == 10_000
    assert arrays.relative_rotation.shape == (10_000, 3, 3)
    assert arrays.displacement.shape == (10_000, 3)
    assert arrays.target.shape == (10_000, 2, 3)
    assert np.isfinite(arrays.relative_rotation).all()
    assert np.isfinite(arrays.displacement).all()
    assert np.isfinite(arrays.target).all()

    assert metadata["sample_count"] == len(arrays)
    assert tuple(metadata["columns"]) == BENZENE_PAIR_COLUMNS
    assert dataset.metadata == metadata
    (displacement, relative_rotation), target = dataset[0]
    assert displacement.shape == (3,)
    assert relative_rotation.shape == (3, 3)
    assert target.shape == (2, 3)
    assert displacement.dtype == torch.float64
    assert relative_rotation.dtype == torch.float64
    assert target.dtype == torch.float64


def test_generation_writes_compact_parameters_to_json_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from TFENN.data import generation as generation_module

    rotation = np.eye(3, dtype=np.float64)
    displacement = np.array((1.0, 2.0, 3.0), dtype=np.float64)
    force = np.array((4.0, 5.0, 6.0), dtype=np.float64)
    moment = np.array((7.0, 8.0, 9.0), dtype=np.float64)

    def fake_sample(
        rng: np.random.Generator,
        config: BenzenePairGenerationConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert isinstance(rng, np.random.Generator)
        assert config.seed == 23
        return rotation, displacement, force, moment

    monkeypatch.setattr(generation_module, "sample_benzene_pair", fake_sample)
    config = BenzenePairGenerationConfig(
        sample_count=2,
        seed=23,
        distance_range=(5.0, 7.0),
        cutoff=11.0,
        min_separation=3.5,
        smoothing="linear",
        target_scale=0.25,
        max_attempts_per_sample=31,
    )
    csv_path, metadata_path = generate_benzene_pair_dataset(
        tmp_path / "small_benzene_pair.csv",
        config,
        progress_every=None,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf_8"))
    arrays = load_benzene_pair_csv(csv_path)
    assert metadata == config.metadata()
    assert metadata_path == csv_path.with_suffix(".json")
    assert metadata["sample_count"] == 2
    assert metadata["seed"] == 23
    assert metadata["distance_range"] == [5.0, 7.0]
    assert metadata["cutoff"] == 11.0
    assert metadata["min_separation"] == 3.5
    assert metadata["smoothing"] == "linear"
    assert metadata["target_scale"] == 0.25
    assert metadata["max_attempts_per_sample"] == 31
    assert metadata["columns"] == list(BENZENE_PAIR_COLUMNS)
    np.testing.assert_allclose(arrays.force, np.tile(force * 0.25, (2, 1)))
    np.testing.assert_allclose(arrays.moment, np.tile(moment * 0.25, (2, 1)))
