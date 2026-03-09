"""Data loading utilities for plotting scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config import OUTPUT_DIR


def load_subject_results(subject_id: str, model: str) -> Optional[pd.DataFrame]:
    """Load detailed results for a subject."""
    subject_path = Path(OUTPUT_DIR) / model / f"{subject_id}.csv"
    if not subject_path.exists():
        print(f"[!] Subject results not found: {subject_path}")
        return None
    return pd.read_csv(subject_path)
