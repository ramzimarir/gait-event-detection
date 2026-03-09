"""Model interface for gait event detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class GaitModel(ABC):
    """Abstract interface for gait models."""

    @abstractmethod
    def detect(
        self,
        acc_data: np.ndarray,
        gyro_data: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Detect gait events from input signals."""
        raise NotImplementedError
