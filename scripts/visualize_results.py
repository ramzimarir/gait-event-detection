"""CLI entry point for generating gait visualization plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path so plots modules can find config
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from plots import (
    plot_all_subjects_signal_segmentation,
    plot_boxplots_per_subject,
    plot_confusion_matrix_equivalent,
    plot_f1_precision_recall,
    plot_metrics_with_std,
    plot_model_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization plots for gait event detection results.")
    parser.add_argument("models", nargs="*", choices=["cnn", "lstm", "tcn"])
    models = parser.parse_args().models or ["cnn", "lstm", "tcn"]
    print(f"[*] Selected models: {', '.join(models)}")
    print("[*] Generating model comparison plot..."); plot_model_comparison(models)
    print("[*] Generating signal segmentation plots...")
    plot_all_subjects_signal_segmentation(side="left")
    plot_all_subjects_signal_segmentation(side="right")
    print("[*] Generating metrics with std plot..."); plot_metrics_with_std(models)
    print("[*] Generating boxplots per subject..."); plot_boxplots_per_subject(models)
    print("[*] Generating F1 / Precision / Recall plot..."); plot_f1_precision_recall(models)
    print("[*] Generating confusion matrix equivalent..."); plot_confusion_matrix_equivalent(models)
    print("[OK] All visualizations complete!")


if __name__ == "__main__":
    main()
