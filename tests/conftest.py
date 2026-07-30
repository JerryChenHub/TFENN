from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def benzene_csv_path(repository_root: Path) -> Path:
    return (
        repository_root
        / "data"
        / "benzene_pair"
        / "Benzene_10000_6.0_10.0_4.0_gamma1.csv"
    )
