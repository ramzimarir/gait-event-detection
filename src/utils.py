"""Utility functions for gait event detection."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def probs_to_events(
    probs: np.ndarray,
    height: float = 0.5,
    distance: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert continuous probabilities to discrete event detections.

    Args:
        probs: (N, 2) array of [TO, IC] probabilities
        height: Minimum peak height threshold
        distance: Minimum distance between peaks (samples)

    Returns:
        to_events: Binary array for TO events
        ic_events: Binary array for IC events
    """
    n_samples = probs.shape[0]
    to_events = np.zeros(n_samples, dtype=int)
    ic_events = np.zeros(n_samples, dtype=int)

    # Find TO peaks (class 0)
    to_peaks, _ = find_peaks(probs[:, 0], height=height, distance=distance)
    to_events[to_peaks] = 1

    # Find IC peaks (class 1)
    ic_peaks, _ = find_peaks(probs[:, 1], height=height, distance=distance)
    ic_events[ic_peaks] = 1

    return to_events, ic_events
