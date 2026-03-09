"""Simple 1D CNN model for gait event detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from config import OVERLAP, WINDOW_SIZE
from src.models.base import GaitModel


class SimpleCNN(nn.Module):
    """CNN with stacked Conv-BN-ReLU-MaxPool blocks and GAP."""

    def __init__(self, in_channels: int, num_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
                        nn.Conv1d(in_channels, 64, kernel_size=7, padding="same"),
                        nn.BatchNorm1d(64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.3),
                        nn.Conv1d(64, 128, kernel_size=5, padding="same"),
                        nn.BatchNorm1d(128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.3),
                        nn.Conv1d(128, 256, kernel_size=5, padding="same"),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.3),
                        nn.Conv1d(256, 128, kernel_size=3, padding="same"),
                        nn.BatchNorm1d(128),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.3),
        )

        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class CNNModel(GaitModel):
    """Wrapper for CNN inference on full sequences."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        window_size: int = WINDOW_SIZE,
        overlap: float = OVERLAP,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window_size = window_size
        self.step_size = max(1, int(round(window_size * (1.0 - overlap))))

        self.model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(self.device)
        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    def _windowize(self, signals: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        windows = []
        starts = []
        max_start = signals.shape[0] - self.window_size
        for start in range(0, max_start + 1, self.step_size):
            end = start + self.window_size
            windows.append(signals[start:end])
            starts.append(start)
        return np.stack(windows, axis=0), starts

    def _aggregate_logits(self, logits: np.ndarray, starts: List[int], length: int) -> np.ndarray:
        num_classes = logits.shape[1]
        accum = np.zeros((length, num_classes), dtype=np.float32)
        counts = np.zeros((length, 1), dtype=np.float32)

        for logit, start in zip(logits, starts):
            end = start + self.window_size
            accum[start:end] += logit.transpose(1, 0)
            counts[start:end] += 1.0

        counts[counts == 0] = 1.0
        return accum / counts

    def detect(
        self,
        acc_data: np.ndarray,
        gyro_data: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        if gyro_data is not None:
            signals = np.concatenate([acc_data, gyro_data], axis=1)
        else:
            signals = acc_data

        if signals.shape[0] < self.window_size:
            raise ValueError("Signal shorter than window_size")

        windows, starts = self._windowize(signals)
        x = torch.from_numpy(windows.astype(np.float32)).permute(0, 2, 1).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            logits = logits.cpu().numpy()

        logits_seq = self._aggregate_logits(logits, starts, signals.shape[0])
        probs_seq = 1.0 / (1.0 + np.exp(-logits_seq))

        return {
            "logits": logits_seq,
            "probs": probs_seq,
        }
