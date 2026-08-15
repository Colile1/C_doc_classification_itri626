# env.py — decide which machine the code is running on, and which device to use.
# Author: Colile Sibanda

from __future__ import annotations

import os
import sys
from typing import Any

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


def detect_environment() -> str:
    """Purpose: name the machine the process is running on.

    Input: none; the decision is made from the interpreter and environment.
    Output: one of `colab`, `hpc` or `local`.
    """
    if "google.colab" in sys.modules or os.path.isdir("/content/drive"):
        return "colab"
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("DOCCLF_ENV") == "hpc":
        return "hpc"
    if os.environ.get("DOCCLF_ENV") in ("local", "colab", "hpc"):
        return os.environ["DOCCLF_ENV"]
    return "local"


def select_device(cfg: Any) -> "Any":
    """Purpose: choose the torch device, honouring the config and the hardware.

    Input: a RunConfig whose `train.device` is `auto`, `cpu` or `cuda`.
    Output: a `torch.device`. A configured `cuda` with no CUDA present falls
    back to CPU with a logged note rather than failing the run.
    """
    import torch

    requested = getattr(cfg.train, "device", "auto")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(
            f"Field 'train.device' must be auto, cpu or cuda, got {requested!r}."
        )
    available = torch.cuda.is_available()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not available:
        _LOGGER.warning(
            "Device fallback: cuda requested but unavailable, using cpu instead"
        )
        return torch.device("cpu")
    if available:
        return torch.device("cuda")
    return torch.device("cpu")
