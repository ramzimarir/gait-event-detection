"""Unidirectional LSTM model for gait event detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from config import OVERLAP, WINDOW_SIZE
from src.models.base import GaitModel


class LSTMModel(nn.Module, GaitModel):
    """LSTM model with full-resolution outputs and windowed inference."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        window_size: int = WINDOW_SIZE,
        overlap: float = OVERLAP,
    ) -> None:
        super().__init__()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window_size = window_size
        self.step_size = max(1, int(round(window_size * (1.0 - overlap))))

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=False,
        )
        self.fc = nn.Linear(128, num_classes)

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.load_state_dict(state)

        self.to(self.device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        outputs, _ = self.lstm(x)
        outputs = self.fc(outputs)
        return outputs.permute(0, 2, 1)

    def _windowize(self, signals: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        windows: List[np.ndarray] = []
        starts: List[int] = []
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
            logits = self(x).cpu().numpy()

        logits_seq = self._aggregate_logits(logits, starts, signals.shape[0])
        probs_seq = 1.0 / (1.0 + np.exp(-logits_seq))

        return {
            "logits": logits_seq,
            "probs": probs_seq,
        }
