# __init__.py — public surface of the data module.
# Author: Colile Sibanda

from docclf.data.acquire import fetch_dataset, verify_checksum
from docclf.data.clean import CleaningReport, clean_manifest
from docclf.data.dataset import DocumentPageDataset
from docclf.data.loaders import build_dataloaders
from docclf.data.manifest import MANIFEST_COLUMNS, build_manifest, derive_group_id
from docclf.data.subsample import cap_per_class
from docclf.data.transforms import build_transforms

__all__ = [
    "CleaningReport",
    "DocumentPageDataset",
    "MANIFEST_COLUMNS",
    "build_dataloaders",
    "build_manifest",
    "build_transforms",
    "cap_per_class",
    "clean_manifest",
    "derive_group_id",
    "fetch_dataset",
    "verify_checksum",
]
