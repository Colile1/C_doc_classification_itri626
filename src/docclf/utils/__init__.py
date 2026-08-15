# __init__.py — public surface of the shared helpers.
# Author: Colile Sibanda

from docclf.utils.env import detect_environment, select_device
from docclf.utils.hashing import hamming_distance, perceptual_hash
from docclf.utils.logging import get_logger, log_values
from docclf.utils.timing import timed

__all__ = [
    "detect_environment",
    "get_logger",
    "hamming_distance",
    "log_values",
    "perceptual_hash",
    "select_device",
    "timed",
]
