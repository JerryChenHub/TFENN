from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .benzene_pair import (
    BENZENE_PAIR_COLUMNS,
    DISPLACEMENT_COLUMNS,
    FORCE_COLUMNS,
    MOMENT_COLUMNS,
    ROTATION_COLUMNS,
    BenzenePairArrays,
    load_benzene_pair_csv,
    load_benzene_pair_metadata,
)

if TYPE_CHECKING:
    from .dataset import BenzenePairDataset
    from .generation import (
        BenzenePairGenerationConfig,
        generate_benzene_pair_dataset,
        sample_benzene_pair,
    )

__all__ = [
    "BENZENE_PAIR_COLUMNS",
    "DISPLACEMENT_COLUMNS",
    "FORCE_COLUMNS",
    "MOMENT_COLUMNS",
    "ROTATION_COLUMNS",
    "BenzenePairArrays",
    "BenzenePairDataset",
    "BenzenePairGenerationConfig",
    "generate_benzene_pair_dataset",
    "load_benzene_pair_csv",
    "load_benzene_pair_metadata",
    "sample_benzene_pair",
]


def __getattr__(name: str) -> Any:
    if name == "BenzenePairDataset":
        from .dataset import BenzenePairDataset

        value = BenzenePairDataset
    elif name in {
        "BenzenePairGenerationConfig",
        "generate_benzene_pair_dataset",
        "sample_benzene_pair",
    }:
        from .generation import (
            BenzenePairGenerationConfig,
            generate_benzene_pair_dataset,
            sample_benzene_pair,
        )

        value = {
            "BenzenePairGenerationConfig": BenzenePairGenerationConfig,
            "generate_benzene_pair_dataset": generate_benzene_pair_dataset,
            "sample_benzene_pair": sample_benzene_pair,
        }[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
