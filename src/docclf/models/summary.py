# summary.py — parameter count and on-disk size, for the deployment recommendation.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class ModelSummary:
    """The size half of the accuracy-versus-cost argument (objective O8)."""

    name: str
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    size_mb: float

    def as_row(self) -> dict[str, object]:
        """Purpose: one row for the comparison table. Output: a plain dictionary."""
        return asdict(self)


def describe_model(model: nn.Module, name: str | None = None) -> ModelSummary:
    """Purpose: measure a model's parameter count and memory footprint.

    Inputs: the model and an optional name (its class name is used otherwise).
    Output: a ModelSummary. `size_mb` is the size of the parameters and buffers
    in memory, which is what the on-disk state dict weighs.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return ModelSummary(
        name=name or type(model).__name__,
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        size_mb=round((parameter_bytes + buffer_bytes) / _BYTES_PER_MB, 3),
    )
