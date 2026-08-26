from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.gnn.e311_one_block_gnn import (
    E311OneBlockGNN,
    E311OneBlockGNNConfig,
)
from experiments.gnn.train_e311_two_benzene_one_block import (
    FORMAL_EPOCHS,
    SHARED_TRAINING_SOURCE,
    SPECIFIED_DESIGN_SHA256,
    SPECIFIED_DESIGN_SOURCE,
    _source_provenance,
    parse_args,
)
from experiments.gnn.two_benzene_training_support import (
    DEFAULT_DATA,
    DEFAULT_METADATA,
    DEFAULT_SEED,
    DEFAULT_VALIDATION,
    TEST_COUNT,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    _load_prepared_data,
    _normalized_single_edge_prediction,
    _single_edge_targets,
    _split_sample_ids,
)


def test_default_contract_and_complete_graph_split() -> None:
    arguments = parse_args([])
    assert arguments.epochs == FORMAL_EPOCHS == 300
    train, validation, test = _split_sample_ids(2000, DEFAULT_SEED)
    assert train.shape == (TRAIN_COUNT,)
    assert validation.shape == (VALIDATION_COUNT,)
    assert test.shape == (TEST_COUNT,)
    assert np.array_equal(
        np.sort(np.concatenate((train, validation, test))),
        np.arange(2000),
    )
    repeated = _split_sample_ids(2000, DEFAULT_SEED)
    for original, replay in zip((train, validation, test), repeated, strict=True):
        assert np.array_equal(original, replay)


def test_prepared_data_single_edge_target_and_e311_smoke() -> None:
    arrays, metadata, validation, data_hash = _load_prepared_data(
        DEFAULT_DATA,
        DEFAULT_METADATA,
        DEFAULT_VALIDATION,
    )
    targets = _single_edge_targets(arrays)
    assert metadata["csv_sha256"] == data_hash
    assert validation["passed"] is True
    assert arrays.centers.shape == (2000, 2, 3)
    assert arrays.rotations.shape == (2000, 2, 3, 3)
    assert targets.shape == (2000, 1, 3)
    np.testing.assert_allclose(
        targets[:, 0],
        arrays.forces[:, 0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        arrays.forces[:, 0] + arrays.forces[:, 1],
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    model = E311OneBlockGNN(
        1.0,
        E311OneBlockGNNConfig(),
        dtype=torch.float32,
    )
    centers = torch.from_numpy(
        np.ascontiguousarray(arrays.centers[:2], dtype=np.float32)
    )
    rotations = torch.from_numpy(
        np.ascontiguousarray(arrays.rotations[:2], dtype=np.float32)
    )
    target = torch.from_numpy(
        np.ascontiguousarray(targets[:2], dtype=np.float32)
    )
    prediction = _normalized_single_edge_prediction(model, centers, rotations)
    loss = (prediction - target).square().mean()
    loss.backward()
    assert model.message_block_count == 1
    assert model.pair_count == 1
    assert prediction.shape == target.shape
    assert bool(torch.isfinite(prediction).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_e311_source_provenance_uses_formal_core_and_support() -> None:
    provenance = _source_provenance()
    design = provenance["specified_design_source"]
    assert design["expected_sha256"] == SPECIFIED_DESIGN_SHA256
    assert design["actual_sha256"] == SPECIFIED_DESIGN_SHA256
    assert design["verification_status"] == "matched"
    assert design["verified"] is True
    assert Path(design["path"]) == SPECIFIED_DESIGN_SOURCE.resolve()
    assert SPECIFIED_DESIGN_SOURCE.parts[-4:] == (
        "src",
        "TFENN",
        "models",
        "e311_multibody_message_block_v1.py",
    )
    support = provenance["shared_training_utility_source"]
    assert Path(support["path"]) == SHARED_TRAINING_SOURCE.resolve()
    assert SHARED_TRAINING_SOURCE.name == "two_benzene_training_support.py"
