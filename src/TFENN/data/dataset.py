from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .benzene_pair import load_benzene_pair_csv, load_benzene_pair_metadata


class BenzenePairDataset(Dataset):
    """Torch dataset for relative pose inputs and force plus moment targets."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        dtype: torch.dtype = torch.float64,
        validate_finite: bool = True,
    ) -> None:
        arrays = load_benzene_pair_csv(
            csv_path,
            validate_finite=validate_finite,
        )
        self.relative_rotation = torch.as_tensor(
            arrays.relative_rotation,
            dtype=dtype,
        )
        self.displacement = torch.as_tensor(arrays.displacement, dtype=dtype)
        self.target = torch.as_tensor(arrays.target, dtype=dtype)

        metadata_path = Path(csv_path).with_suffix(".json")
        self.metadata: dict[str, Any] | None
        if metadata_path.exists():
            self.metadata = load_benzene_pair_metadata(metadata_path)
        else:
            self.metadata = None

    def __len__(self) -> int:
        return self.displacement.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        inputs = self.displacement[index], self.relative_rotation[index]
        return inputs, self.target[index]
