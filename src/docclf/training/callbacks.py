# callbacks.py — early stopping, and a checkpoint every epoch so a disconnect is cheap.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from docclf.utils.logging import get_logger

_LOGGER = get_logger(__name__)


class EarlyStopping:
    """Stop when the monitored validation metric has not improved for `patience` epochs."""

    def __init__(self, patience: int, monitor: str = "val_macro_f1", min_delta: float = 1e-4) -> None:
        """Purpose: configure the stopping rule.

        Inputs: the patience in epochs, the monitored metric name, and the
        smallest change that counts as an improvement.
        Output: None.
        """
        if patience < 1:
            raise ValueError(f"Early stopping patience must be at least 1, got {patience}.")
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.minimise = monitor.endswith("loss")
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        """Purpose: record this epoch's metric and say whether training should stop.

        Input: the monitored metric for the epoch just finished.
        Output: True when training should stop now.
        """
        improved = (
            self.best is None
            or (self.minimise and value < self.best - self.min_delta)
            or (not self.minimise and value > self.best + self.min_delta)
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            _LOGGER.info(
                "Early stopping triggered: %s did not improve for %d epoch(s)",
                self.monitor,
                self.bad_epochs,
            )
            return True
        return False

    def is_best(self) -> bool:
        """Purpose: report whether the last update was an improvement. Output: bool."""
        return self.bad_epochs == 0

    def state_dict(self) -> dict[str, Any]:
        """Purpose: capture the patience counter so a resumed run does not forget
        how many epochs have already failed to improve. Output: the state."""
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Purpose: restore the patience counter on resume. Input: the state."""
        self.best = state["best"]
        self.bad_epochs = int(state["bad_epochs"])


class Checkpoint:
    """Writes the model state every epoch, and keeps a copy of the best epoch."""

    def __init__(self, directory: str | Path, model_name: str) -> None:
        """Purpose: prepare the checkpoint destination.

        Inputs: the checkpoint directory and the model name used in file names.
        Output: None.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    @property
    def last_path(self) -> Path:
        """Purpose: the every-epoch checkpoint path, the one a restart resumes from."""
        return self.directory / f"{self.model_name}_last.pt"

    @property
    def best_path(self) -> Path:
        """Purpose: the best-epoch checkpoint path, the one evaluation loads."""
        return self.directory / f"{self.model_name}_best.pt"

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict[str, float],
        best: bool,
        training_state: dict[str, Any] | None = None,
    ) -> Path:
        """Purpose: write this epoch's checkpoint, and the best copy when it is one.

        Inputs: the model, the epoch number, the epoch's metrics, whether this
        epoch is the best so far, and the optimiser/scheduler/early-stopping
        state that lets another machine resume exactly where this one stopped.
        Output: the path written for this epoch.
        """
        payload = {
            "model_name": self.model_name,
            "epoch": epoch,
            "metrics": dict(metrics),
            "state_dict": model.state_dict(),
        }
        if training_state is not None:
            payload["training_state"] = training_state
        torch.save(payload, self.last_path)
        if best:
            torch.save(payload, self.best_path)
            _LOGGER.info("Best checkpoint written: %s", self.best_path)
        return self.last_path

    def load_best(self, model: nn.Module, map_location: Any = "cpu") -> dict[str, Any]:
        """Purpose: restore the best epoch's weights before evaluation.

        Inputs: the model to load into and the device to map onto.
        Output: the checkpoint payload without its state dict.
        """
        if not self.best_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint at {self.best_path}. Train the model first with "
                f"`python run_train.py --config <config> --model {self.model_name}`."
            )
        payload = torch.load(self.best_path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return {key: value for key, value in payload.items() if key != "state_dict"}
