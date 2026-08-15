# conftest.py — put `src/` on the import path and provide the synthetic fixtures.
# Author: Colile Sibanda

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CLASSES = (
    "invoice",
    "budget",
    "form",
    "letter",
    "memo",
    "handwritten",
    "resume",
    "advertisement",
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Purpose: the repository root, for tests that read committed config files."""
    return ROOT


@pytest.fixture
def manifest() -> pd.DataFrame:
    """Purpose: a synthetic manifest with several pages per source document.

    Output: a DataFrame in manifest shape — eight classes, six groups each,
    between one and three pages per group. No image files are involved.
    """
    rows = []
    for class_id, class_name in enumerate(CLASSES):
        for group_index in range(6):
            group = f"{class_name}/doc{group_index:02d}"
            for page in range(1 + group_index % 3):
                rows.append(
                    {
                        "path": f"/data/{class_name}/doc{group_index:02d}_p{page}.tif",
                        "class_name": class_name,
                        "class_id": class_id,
                        "doc_group_id": group,
                        "width": 754,
                        "height": 1000,
                        "bytes": 20000,
                        "readable": True,
                        "phash": f"{class_id:x}{group_index:x}{page:x}0deadbee",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Purpose: a tiny class-per-directory dataset on disk, for manifest tests.

    Output: a directory with all eight class folders, four documents per class,
    two pages each, plus one deliberately corrupt file in `invoice`. Four
    documents is the smallest number that can populate three splits.
    """
    from PIL import Image

    root = tmp_path / "dataset"
    for class_name in CLASSES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for document in range(4):
            for page in range(2):
                image = Image.new("L", (128, 160), color=32 * document + 8 * page)
                image.save(class_dir / f"doc{document}_p{page}.png")
    (root / "invoice" / "doc9_p0.png").write_bytes(b"not an image at all")
    return root
