"""Temporal Convolutional Network model for gait event detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm

from config import OVERLAP, WINDOW_SIZE
from src.models.base import GaitModel


class Chomp1d(nn.Module):
    """Remove extra padding to ensure causal convolutions."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """Residual block with two causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            weight_norm(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module, GaitModel):
    """Causal TCN with full-resolution outputs and windowed inference."""

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

        num_channels = [128, 128, 128, 128, 128]
        kernel_size = 5
        dropout = 0.35

        layers: List[nn.Module] = []
        in_ch = in_channels
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(in_ch, num_classes, kernel_size=1)

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.load_state_dict(state)

        self.to(self.device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return self.classifier(x)

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
