"""PyTorch dataset for gait event segmentation using unified input files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import INPUT_DATA_DIR, OVERLAP, WINDOW_SIZE
from src.data_loader import DataLoader


@dataclass
class WindowIndex:
    series_idx: int
    start: int
    end: int
    center: int
    subject: str
    side: str


class GaitDataset(Dataset):
    """Sliding-window dataset for CNN models."""

    def __init__(
        self,
        input_dir: Path = INPUT_DATA_DIR,
        window_size: int = WINDOW_SIZE,
        overlap: float = OVERLAP,
        label_mode: str = "sequence",
        downsample_factor: int = 8,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0.0, 1.0)")
        if label_mode not in {"center", "sequence"}:
            raise ValueError("label_mode must be 'center' or 'sequence'")
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")

        self.input_dir = Path(input_dir)
        self.window_size = window_size
        self.step_size = max(1, int(round(window_size * (1.0 - overlap))))
        self.label_mode = label_mode
        self.downsample_factor = downsample_factor

        self._loader = DataLoader()
        self._series: List[Dict[str, object]] = []
        self._windows: List[WindowIndex] = []
        self._subject_index: Dict[str, List[int]] = {}

        self._load_all()
        self._build_windows()

    def _labels_for_side(self, df, suffix: str) -> np.ndarray:
        if suffix == "left":
            label_cols = ["TO_left", "IC_left"]
        else:
            label_cols = ["TO_right", "IC_right"]
        missing = [col for col in label_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")
        return df[label_cols].to_numpy(dtype=np.float32)

    def _load_all(self) -> None:
        files = sorted(self.input_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")

        for file_path in files:
            df = self._loader.load_data(file_path)
            features = self._loader.get_features(df)
            labels = self._loader.get_labels(df)

            subject_id = file_path.stem
            for side in ["left", "right"]:
                side_cols = [col for col in features.columns if col.endswith(f"_{side}")]
                if not side_cols:
                    continue

                signals = features[side_cols].to_numpy(dtype=np.float32)
                label_array = self._labels_for_side(labels, side)

                self._series.append(
                    {
                        "subject": subject_id,
                        "side": side,
                        "signals": signals,
                        "labels": label_array,
                    }
                )

    def _build_windows(self) -> None:
        for series_idx, series in enumerate(self._series):
            signals = series["signals"]
            subject = str(series["subject"])
            side = str(series["side"])

            max_start = signals.shape[0] - self.window_size
            for start in range(0, max_start + 1, self.step_size):
                end = start + self.window_size
                center = start + self.window_size // 2
                self._windows.append(
                    WindowIndex(
                        series_idx=series_idx,
                        start=start,
                        end=end,
                        center=center,
                        subject=subject,
                        side=side,
                    )
                )

                self._subject_index.setdefault(subject, []).append(len(self._windows) - 1)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self._windows[idx]
        series = self._series[window.series_idx]
        signals = series["signals"]
        labels = series["labels"]

        x = signals[window.start : window.end]
        x_tensor = torch.from_numpy(x).transpose(0, 1)

        if self.label_mode == "center":
            y = labels[window.center]
            y_tensor = torch.from_numpy(y)
        else:
            y = labels[window.start : window.end]
            y = y[:: self.downsample_factor]
            y_tensor = torch.from_numpy(y)

        return x_tensor, y_tensor

    @property
    def in_channels(self) -> int:
        if not self._series:
            return 0
        return int(self._series[0]["signals"].shape[1])

    @property
    def num_classes(self) -> int:
        return 2

    def loso_split(self, test_subject_id: str) -> Tuple[List[int], List[int]]:
        test_idx = self._subject_index.get(test_subject_id, [])
        train_idx = [i for i in range(len(self._windows)) if i not in set(test_idx)]
        return train_idx, test_idx

    def subject_ids(self) -> List[str]:
        return sorted(self._subject_index.keys())
