# history.py — the per-epoch record, and the source of the learning-curve figures.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of training, as written to `outputs/metrics/`."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate: float
    seconds: float
    train_seconds: float = 0.0
    val_seconds: float = 0.0
    train_images_per_second: float = 0.0
    val_images_per_second: float = 0.0


@dataclass
class TrainingHistory:
    """The full record of one training run."""

    model_name: str
    config_name: str
    records: list[EpochRecord] = field(default_factory=list)

    def append(self, record: EpochRecord) -> None:
        """Purpose: add one epoch, keeping the epoch numbers monotonic.

        Input: the epoch record. Output: None. Raises ValueError if out of order.
        """
        if self.records and record.epoch <= self.records[-1].epoch:
            raise ValueError(
                f"Epoch {record.epoch} follows epoch {self.records[-1].epoch}; "
                f"the history must be recorded in order."
            )
        self.records.append(record)

    def to_frame(self) -> pd.DataFrame:
        """Purpose: the tabular history. Output: one row per epoch."""
        columns = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "learning_rate",
            "seconds",
            "train_seconds",
            "val_seconds",
            "train_images_per_second",
            "val_images_per_second",
        ]
        frame = pd.DataFrame([asdict(item) for item in self.records], columns=columns)
        frame.insert(0, "model", self.model_name)
        return frame

    def best(self, monitor: str = "val_macro_f1") -> EpochRecord:
        """Purpose: the epoch the checkpoint and the report refer to.

        Input: the monitored metric name (a loss is minimised, anything else
        maximised). Output: the best EpochRecord.
        """
        if not self.records:
            raise ValueError("Training history is empty; no best epoch exists.")
        if not hasattr(self.records[0], monitor):
            raise ValueError(
                f"Unknown monitor {monitor!r}. Valid values are: "
                f"{', '.join(vars(self.records[0]))}."
            )
        reverse = not monitor.endswith("loss")
        return sorted(
            self.records, key=lambda item: getattr(item, monitor), reverse=reverse
        )[0]

    def save(self, directory: str | Path) -> Path:
        """Purpose: write the history CSV a curve figure is drawn from.

        Input: the destination directory. Output: the file written.
        """
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"history_{self.model_name}.csv"
        self.to_frame().to_csv(path, index=False, encoding="utf-8")
        return path
