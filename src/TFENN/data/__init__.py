from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .benzene_cluster import (
    BENZENE_CLUSTER_COLUMNS,
    CENTER_COLUMNS,
    CLUSTER_FORCE_COLUMNS,
    CLUSTER_MOMENT_COLUMNS,
    CLUSTER_ROTATION_COLUMNS,
    BenzeneClusterArrays,
    load_benzene_cluster_csv,
    load_benzene_cluster_metadata,
)
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
        BenzeneClusterGenerationConfig,
        BenzeneClusterSample,
        BenzenePairGenerationConfig,
        BenzeneTripleGenerationConfig,
        compare_legacy_pair_rows,
        generate_benzene_cluster_dataset,
        generate_benzene_pair_dataset,
        generate_benzene_triple_dataset,
        sample_benzene_cluster,
        sample_benzene_pair,
    )

__all__ = [
    "BENZENE_CLUSTER_COLUMNS",
    "BENZENE_PAIR_COLUMNS",
    "CENTER_COLUMNS",
    "CLUSTER_FORCE_COLUMNS",
    "CLUSTER_MOMENT_COLUMNS",
    "CLUSTER_ROTATION_COLUMNS",
    "DISPLACEMENT_COLUMNS",
    "FORCE_COLUMNS",
    "MOMENT_COLUMNS",
    "ROTATION_COLUMNS",
    "BenzeneClusterArrays",
    "BenzeneClusterGenerationConfig",
    "BenzeneClusterSample",
    "BenzenePairArrays",
    "BenzenePairDataset",
    "BenzenePairGenerationConfig",
    "BenzeneTripleGenerationConfig",
    "compare_legacy_pair_rows",
    "generate_benzene_cluster_dataset",
    "generate_benzene_pair_dataset",
    "generate_benzene_triple_dataset",
    "load_benzene_cluster_csv",
    "load_benzene_cluster_metadata",
    "load_benzene_pair_csv",
    "load_benzene_pair_metadata",
    "sample_benzene_cluster",
    "sample_benzene_pair",
]


def __getattr__(name: str) -> Any:
    if name == "BenzenePairDataset":
        from .dataset import BenzenePairDataset

        value = BenzenePairDataset
    elif name in {
        "BenzeneClusterGenerationConfig",
        "BenzeneClusterSample",
        "BenzenePairGenerationConfig",
        "BenzeneTripleGenerationConfig",
        "compare_legacy_pair_rows",
        "generate_benzene_cluster_dataset",
        "generate_benzene_pair_dataset",
        "generate_benzene_triple_dataset",
        "sample_benzene_cluster",
        "sample_benzene_pair",
    }:
        from .generation import (
            BenzeneClusterGenerationConfig,
            BenzeneClusterSample,
            BenzenePairGenerationConfig,
            BenzeneTripleGenerationConfig,
            compare_legacy_pair_rows,
            generate_benzene_cluster_dataset,
            generate_benzene_pair_dataset,
            generate_benzene_triple_dataset,
            sample_benzene_cluster,
            sample_benzene_pair,
        )

        value = {
            "BenzeneClusterGenerationConfig": BenzeneClusterGenerationConfig,
            "BenzeneClusterSample": BenzeneClusterSample,
            "BenzenePairGenerationConfig": BenzenePairGenerationConfig,
            "BenzeneTripleGenerationConfig": BenzeneTripleGenerationConfig,
            "compare_legacy_pair_rows": compare_legacy_pair_rows,
            "generate_benzene_cluster_dataset": generate_benzene_cluster_dataset,
            "generate_benzene_pair_dataset": generate_benzene_pair_dataset,
            "generate_benzene_triple_dataset": generate_benzene_triple_dataset,
            "sample_benzene_cluster": sample_benzene_cluster,
            "sample_benzene_pair": sample_benzene_pair,
        }[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
