"""Evaluation metrics for gait event detection."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GaitEventEvaluator:
    """Compute evaluation metrics for gait event detection (IC and TO)."""

    def __init__(self, fs: int = 100):
        self.fs = fs
        self.dt = 1000 / fs

    def compute_mae(self, pred_events: np.ndarray, true_events: np.ndarray) -> float:
        pred_indices = np.where(pred_events == 1)[0]
        true_indices = np.where(true_events == 1)[0]

        if len(pred_indices) == 0 or len(true_indices) == 0:
            logger.warning("No events detected or ground truth is empty")
            return np.nan

        errors = []
        for true_idx in true_indices:
            distances = np.abs(pred_indices - true_idx)
            errors.append(np.min(distances) * self.dt)

        return float(np.mean(errors))

    def compute_accuracy(
        self,
        pred_events: np.ndarray,
        true_events: np.ndarray,
        tolerance_ms: int = 20,
    ) -> Tuple[float, int, int]:
        tolerance_samples = int(tolerance_ms / self.dt)

        pred_indices = np.where(pred_events == 1)[0]
        true_indices = np.where(true_events == 1)[0]

        if len(true_indices) == 0:
            return np.nan, 0, 0

        n_correct = 0
        for true_idx in true_indices:
            window = np.abs(pred_indices - true_idx) <= tolerance_samples
            if np.any(window):
                n_correct += 1

        accuracy = 100 * n_correct / len(true_indices)
        return accuracy, n_correct, len(true_indices)

    def compute_rmse(self, pred_events: np.ndarray, true_events: np.ndarray) -> float:
        pred_indices = np.where(pred_events == 1)[0]
        true_indices = np.where(true_events == 1)[0]

        if len(pred_indices) == 0 or len(true_indices) == 0:
            return 0.0

        squared_errors = []
        for true_idx in true_indices:
            distances = np.abs(pred_indices - true_idx)
            squared_errors.append((np.min(distances) * self.dt) ** 2)

        return float(np.sqrt(np.mean(squared_errors)))

    def compute_precision(
        self,
        pred_events: np.ndarray,
        true_events: np.ndarray,
        tolerance_ms: int = 20,
    ) -> float:
        tolerance_samples = int(tolerance_ms / self.dt)

        pred_indices = np.where(pred_events == 1)[0]
        true_indices = np.where(true_events == 1)[0]

        if len(pred_indices) == 0 or len(true_indices) == 0:
            return 0.0

        matched_preds = 0
        for pred_idx in pred_indices:
            window = np.abs(true_indices - pred_idx) <= tolerance_samples
            if np.any(window):
                matched_preds += 1

        return matched_preds / len(pred_indices)

    def compute_recall(
        self,
        pred_events: np.ndarray,
        true_events: np.ndarray,
        tolerance_ms: int = 20,
    ) -> float:
        tolerance_samples = int(tolerance_ms / self.dt)

        pred_indices = np.where(pred_events == 1)[0]
        true_indices = np.where(true_events == 1)[0]

        if len(pred_indices) == 0 or len(true_indices) == 0:
            return 0.0

        matched_true = 0
        for true_idx in true_indices:
            window = np.abs(pred_indices - true_idx) <= tolerance_samples
            if np.any(window):
                matched_true += 1

        return matched_true / len(true_indices)

    def compute_f1(
        self,
        pred_events: np.ndarray,
        true_events: np.ndarray,
        tolerance_ms: int = 20,
    ) -> float:
        precision = self.compute_precision(pred_events, true_events, tolerance_ms)
        recall = self.compute_recall(pred_events, true_events, tolerance_ms)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def evaluate_file(
        self,
        pred_to: np.ndarray,
        pred_ic: np.ndarray,
        true_to: np.ndarray,
        true_ic: np.ndarray,
    ) -> Dict[str, float]:
        results = {}

        results["MAE_TO"] = self.compute_mae(pred_to, true_to)
        results["MAE_IC"] = self.compute_mae(pred_ic, true_ic)

        acc_to_20, _, _ = self.compute_accuracy(pred_to, true_to, 20)
        acc_ic_20, _, _ = self.compute_accuracy(pred_ic, true_ic, 20)
        results["Acc_TO_20ms"] = acc_to_20
        results["Acc_IC_20ms"] = acc_ic_20

        acc_to_50, _, _ = self.compute_accuracy(pred_to, true_to, 50)
        acc_ic_50, _, _ = self.compute_accuracy(pred_ic, true_ic, 50)
        results["Acc_TO_50ms"] = acc_to_50
        results["Acc_IC_50ms"] = acc_ic_50

        results["RMSE_TO"] = self.compute_rmse(pred_to, true_to)
        results["RMSE_IC"] = self.compute_rmse(pred_ic, true_ic)
        results["Precision_TO"] = self.compute_precision(pred_to, true_to)
        results["Precision_IC"] = self.compute_precision(pred_ic, true_ic)
        results["Recall_TO"] = self.compute_recall(pred_to, true_to)
        results["Recall_IC"] = self.compute_recall(pred_ic, true_ic)
        results["F1_TO"] = self.compute_f1(pred_to, true_to)
        results["F1_IC"] = self.compute_f1(pred_ic, true_ic)

        return results

    @staticmethod
    def aggregate_results(results_list: list) -> pd.DataFrame:
        df = pd.DataFrame(results_list)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]

        summary = pd.DataFrame(
            {
                "Mean": df_numeric.mean(),
                "Std": df_numeric.std(),
                "Min": df_numeric.min(),
                "Max": df_numeric.max(),
            }
        )

        return summary
