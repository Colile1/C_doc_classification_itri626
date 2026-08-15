# seed.py — drive every source of randomness in the project from one seed.
# Author: Colile Sibanda

from __future__ import annotations

import os
import random


def set_global_seed(n: int) -> None:
    """Purpose: seed Python, NumPy and torch so a run reproduces exactly.

    Input: the integer seed from `config.seed`.
    Output: None. torch is imported lazily, so seeding works in an environment
    where torch is not installed (config loading and split building do not need it).
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"Seed must be an integer, got {type(n).__name__}.")
    os.environ["PYTHONHASHSEED"] = str(n)
    random.seed(n)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        np = None
    if np is not None:
        np.random.seed(n)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
