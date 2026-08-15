# efficiency.py — the cost half of the deployment recommendation (objective O8).
# Author: Colile Sibanda

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from docclf.models.summary import describe_model


@dataclass(frozen=True)
class EfficiencySummary:
    """Latency and size, measured on a stated device at a stated batch size."""

    model_name: str
    device: str
    batch_size: int
    repeats: int
    median_ms_per_image: float
    p90_ms_per_image: float
    total_parameters: int
    size_mb: float

    def as_row(self) -> dict[str, object]:
        """Purpose: one row for the comparison table. Output: a plain dictionary."""
        return asdict(self)


@torch.no_grad()
def measure_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str = "model",
    repeats: int = 50,
    warmup: int = 3,
) -> EfficiencySummary:
    """Purpose: time per-image inference and record the model's size.

    Inputs: the model, a loader supplying real pages, the device, the model name,
    how many batches to time, and how many to discard as warm-up.
    Output: an EfficiencySummary. The median is reported rather than the mean
    because the first batches on any device are not representative; the device
    is recorded because a latency without one means nothing.
    """
    model = model.to(device).eval()
    durations: list[float] = []
    batch_size = int(getattr(loader, "batch_size", 1) or 1)
    iterator = iter(loader)
    for index in range(warmup + repeats):
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                images, _ = next(iterator)
            except StopIteration:
                break
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= warmup:
            durations.append(elapsed / max(images.shape[0], 1) * 1000.0)
    if not durations:
        raise ValueError(
            "The loader yielded no batches, so inference could not be timed. "
            "Check that the split is not empty."
        )
    ordered = sorted(durations)
    summary = describe_model(model, model_name)
    return EfficiencySummary(
        model_name=model_name,
        device=str(device),
        batch_size=batch_size,
        repeats=len(durations),
        median_ms_per_image=round(statistics.median(ordered), 4),
        p90_ms_per_image=round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 4),
        total_parameters=summary.total_parameters,
        size_mb=summary.size_mb,
    )
