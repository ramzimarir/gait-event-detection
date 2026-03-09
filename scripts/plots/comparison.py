"""Model comparison plots for gait event detection."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR


sns.set_theme(style="whitegrid")


def plot_model_comparison(models: List[str]) -> None:
    """Compare metrics across all models."""
    combined_df = None

    for model in models:
        summary_path = OUTPUT_DIR / model / "evaluation_summary.csv"
        if not summary_path.exists():
            print(f"  [!] {model}: {summary_path} not found, skipping")
            continue

        df = pd.read_csv(summary_path)
        # Normalize the metric column name.
        df = df.rename(columns={"Unnamed: 0": "Metric"})
        df["Model"] = model

        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df is None:
        print("  [!] No evaluation_summary.csv file found")
        return

    # Reshape for grouped plotting.
    combined_df_melted = combined_df.melt(
        id_vars=["Metric", "Model"],
        value_vars=["Mean", "Std", "Min", "Max"],
        var_name="Stat",
        value_name="Value"
    )

    # Keep key metrics for the comparison view.
    key_metrics = ["MAE_TO", "MAE_IC", "Acc_TO_50ms", "Acc_IC_50ms"]
    combined_df_melted = combined_df_melted[combined_df_melted["Metric"].isin(key_metrics)]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Compare model means only.
    plot_data = combined_df_melted[combined_df_melted["Stat"] == "Mean"]

    sns.barplot(
        data=plot_data,
        x="Metric",
        y="Value",
        hue="Model",
        ax=ax,
        palette="Set2"
    )

    ax.set_title("Model Comparison: Key Metrics (Mean ± Std)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Value (ms or %)", fontsize=12)
    ax.legend(title="Model", loc="best")
    ax.grid(True, alpha=0.3)

    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_path = plots_dir / "model_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {output_path}")
    plt.close(fig)


def plot_metrics_with_std(models: List[str]) -> None:
    """Grouped barplot for MAE/Accuracy with Mean ± Std."""
    summary_frames: List[pd.DataFrame] = []

    for model in models:
        summary_path = OUTPUT_DIR / model / "evaluation_summary.csv"
        if not summary_path.exists():
            print(f"  [!] {model}: {summary_path} not found, skipping")
            continue
        df = pd.read_csv(summary_path)
        df = df.rename(columns={"Unnamed: 0": "Metric"})
        df["Model"] = model
        summary_frames.append(df)

    if not summary_frames:
        print("  [!] No evaluation_summary.csv files found")
        return

    combined_df = pd.concat(summary_frames, ignore_index=True)
    metrics = ["MAE_TO", "MAE_IC", "Acc_TO_20ms", "Acc_IC_20ms", "Acc_TO_50ms", "Acc_IC_50ms"]
    combined_df = combined_df[combined_df["Metric"].isin(metrics)]

    mae_metrics = ["MAE_TO", "MAE_IC"]
    acc_metrics = ["Acc_TO_20ms", "Acc_IC_20ms", "Acc_TO_50ms", "Acc_IC_50ms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    mae_df = combined_df[combined_df["Metric"].isin(mae_metrics)]
    acc_df = combined_df[combined_df["Metric"].isin(acc_metrics)]

    sns.barplot(
        data=mae_df,
        x="Metric",
        y="Mean",
        hue="Model",
        ax=ax1,
        palette="Set2",
        errorbar=("sd", 1),
    )
    ax1.set_title("MAE Metrics (ms)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Metric", fontsize=11)
    ax1.set_ylabel("Value (ms)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    sns.barplot(
        data=acc_df,
        x="Metric",
        y="Mean",
        hue="Model",
        ax=ax2,
        palette="Set2",
        errorbar=("sd", 1),
    )
    ax2.set_title("Accuracy Metrics (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Metric", fontsize=11)
    ax2.set_ylabel("Value (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        for container in ax.containers:
            if hasattr(container, "patches"):
                ax.bar_label(container, fmt="%.1f", padding=3)

    fig.suptitle("Model Comparison: Mean ± Std", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "metrics_with_std.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {output_path}")
    plt.close(fig)


def plot_f1_precision_recall(models: List[str]) -> None:
    """Grouped barplots for Precision, Recall, and F1 (TO/IC)."""
    summary_rows = []

    for model in models:
        details_path = OUTPUT_DIR / model / "evaluation_details.csv"
        if not details_path.exists():
            print(f"  [!] {model}: {details_path} not found, skipping")
            continue
        df = pd.read_csv(details_path)

        # Check if new metrics columns exist
        required_cols = ["Precision_TO", "Recall_TO", "F1_TO", "Precision_IC", "Recall_IC", "F1_IC"]
        if not all(col in df.columns for col in required_cols):
            print(f"  [!] {model}: Missing new metric columns. Please re-run evaluation to generate Precision/Recall/F1 metrics.")
            return

        summary_rows.append(
            {
                "Model": model,
                "Precision_TO": df["Precision_TO"].mean(),
                "Recall_TO": df["Recall_TO"].mean(),
                "F1_TO": df["F1_TO"].mean(),
                "Precision_IC": df["Precision_IC"].mean(),
                "Recall_IC": df["Recall_IC"].mean(),
                "F1_IC": df["F1_IC"].mean(),
            }
        )

    if not summary_rows:
        print("  [!] No evaluation_details.csv files found")
        return

    summary_df = pd.DataFrame(summary_rows)

    to_df = summary_df.melt(
        id_vars="Model",
        value_vars=["Precision_TO", "Recall_TO", "F1_TO"],
        var_name="Metric",
        value_name="Value",
    )
    to_df["Metric"] = to_df["Metric"].str.replace("_TO", "", regex=False)

    ic_df = summary_df.melt(
        id_vars="Model",
        value_vars=["Precision_IC", "Recall_IC", "F1_IC"],
        var_name="Metric",
        value_name="Value",
    )
    ic_df["Metric"] = ic_df["Metric"].str.replace("_IC", "", regex=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.barplot(
        data=to_df,
        x="Metric",
        y="Value",
        hue="Model",
        ax=ax1,
        palette="Set2",
        errorbar=None,
    )
    ax1.set_title("TO Detection: Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Metric", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    sns.barplot(
        data=ic_df,
        x="Metric",
        y="Value",
        hue="Model",
        ax=ax2,
        palette="Set2",
        errorbar=None,
    )
    ax2.set_title("IC Detection: Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Metric", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout()
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "f1_precision_recall.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {output_path}")
    plt.close(fig)


def plot_confusion_matrix_equivalent(models: List[str]) -> None:
    """Heatmap of recall and FP rate (1 - precision) per model and event type."""
    rows = []

    for model in models:
        details_path = OUTPUT_DIR / model / "evaluation_details.csv"
        if not details_path.exists():
            print(f"  [!] {model}: {details_path} not found, skipping")
            continue
        df = pd.read_csv(details_path)

        # Check if new metrics columns exist
        required_cols = ["Precision_TO", "Recall_TO", "Precision_IC", "Recall_IC"]
        if not all(col in df.columns for col in required_cols):
            print(f"  [!] {model}: Missing new metric columns. Please re-run evaluation to generate Precision/Recall metrics.")
            return

        recall_to = df["Recall_TO"].mean()
        recall_ic = df["Recall_IC"].mean()
        precision_to = df["Precision_TO"].mean()
        precision_ic = df["Precision_IC"].mean()
        rows.append(
            {
                "Model": model,
                "Recall_TO": recall_to,
                "Recall_IC": recall_ic,
                "FP_rate_TO": 1 - precision_to,
                "FP_rate_IC": 1 - precision_ic,
            }
        )

    if not rows:
        print("  [!] No evaluation_details.csv files found")
        return

    heatmap_df = pd.DataFrame(rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(
        "Detection Quality Matrix (Recall = TP rate, FP rate = 1 - Precision)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    plt.tight_layout()
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "confusion_matrix_equiv.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {output_path}")
    plt.close(fig)
