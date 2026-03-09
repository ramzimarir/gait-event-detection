"""Signal visualization plots for gait event detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FS, OUTPUT_DIR
from .data_loader import load_subject_results


sns.set_theme(style="whitegrid")


def plot_signal_segmentation(
    subject_id: str = "Subject_A",
    side: str = "right",
    time_start_ms: int = 2000,
    time_end_ms: int = 6000,
    fs: int = FS,
) -> None:
    """Plot signal with event predictions over a time window."""
    plot_dir = Path(OUTPUT_DIR) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    time_start_idx = int(time_start_ms * fs / 1000)
    time_end_idx = int(time_end_ms * fs / 1000)

    models_to_plot = ["cnn", "lstm", "tcn"]
    dataframes: Dict[str, pd.DataFrame] = {}

    for model in models_to_plot:
        df = load_subject_results(subject_id, model)
        if df is not None:
            dataframes[model] = df
        else:
            print(f"[!] Skipping {model} for {subject_id}")

    if not dataframes:
        print("[!] No subject results found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    first_df = list(dataframes.values())[0]
    time_indices = np.arange(time_start_idx, min(time_end_idx, len(first_df)))
    time_seconds = time_indices / fs

    acc_col = f"Acc_Norm_{side}"
    label_col = f"IC_{side}"

    if acc_col in first_df.columns:
        acc_signal = first_df[acc_col].iloc[time_start_idx:time_end_idx].values
        ax1.plot(time_seconds, acc_signal, "k-", linewidth=2, label=f"Acc Norm {side.upper()}")
        ax1.set_ylabel("Acceleration (m/s²)", fontsize=11, fontweight="bold")
        ax1.set_title(f"Signal Segmentation: {subject_id} - {side.upper()}", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

    if label_col in first_df.columns:
        label_events = first_df[label_col].iloc[time_start_idx:time_end_idx].values
        label_indices = np.where(label_events == 1)[0]
        for idx in label_indices:
            ax1.axvline(x=time_seconds[idx], color="black", linestyle="-", linewidth=2, alpha=0.7)
        ax1.text(0.02, 0.95, "Ground Truth (IC)", transform=ax1.transAxes, fontsize=10,
                 verticalalignment="top", bbox=dict(boxstyle="round", facecolor="black", alpha=0.3))

    colors = {"cnn": "blue", "lstm": "green", "tcn": "red"}
    linestyles = {"cnn": "-", "lstm": "--", "tcn": ":"}

    for model in dataframes.keys():
        df = dataframes[model]
        pred_col = f"Pred_{model.upper()}_IC_{side}"
        prob_col = f"Prob_{model.upper()}_IC_{side}"

        if pred_col in df.columns:
            pred_events = df[pred_col].iloc[time_start_idx:time_end_idx].values
            pred_indices = np.where(pred_events == 1)[0]
            for idx in pred_indices:
                ax1.axvline(
                    x=time_seconds[idx],
                    color=colors.get(model, "gray"),
                    linestyle=linestyles.get(model, "-"),
                    linewidth=1.5,
                    alpha=0.6,
                )

        if prob_col in df.columns:
            prob_signal = df[prob_col].iloc[time_start_idx:time_end_idx].values
            ax2.plot(
                time_seconds,
                prob_signal,
                label=f"{model.upper()} Prob(IC)",
                color=colors.get(model, "gray"),
                linestyle=linestyles.get(model, "-"),
                linewidth=2,
            )

    if label_col in first_df.columns:
        label_events = first_df[label_col].iloc[time_start_idx:time_end_idx].values
        label_indices = np.where(label_events == 1)[0]
        for idx in label_indices:
            ax2.axvline(x=time_seconds[idx], color="black", linestyle="-", linewidth=2, alpha=0.7)

    ax2.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability IC", fontsize=11, fontweight="bold")
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = plot_dir / f"signal_zoom_{subject_id}_{side}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {output_path}")
    plt.close()


def plot_all_subjects_signal_segmentation(side: str = "right") -> None:
    """Generate signal plots for all available subjects."""
    output_dir_path = Path(OUTPUT_DIR)

    subjects_set = set()
    for model_dir in output_dir_path.iterdir():
        if model_dir.is_dir() and model_dir.name in {"cnn", "lstm", "tcn"}:
            for csv_file in model_dir.glob("Subject_*.csv"):
                subjects_set.add(csv_file.stem)

    subjects = sorted(subjects_set)

    if not subjects:
        print("[!] No subject results found.")
        return

    for subject in subjects:
        plot_signal_segmentation(subject_id=subject, side=side)
