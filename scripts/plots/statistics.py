"""Per-subject statistical plots for gait event detection."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR


sns.set_theme(style="whitegrid")


def plot_boxplots_per_subject(models: List[str]) -> None:
    """Boxplots of MAE per subject for each model."""
    detail_frames: List[pd.DataFrame] = []

    for model in models:
        details_path = OUTPUT_DIR / model / "evaluation_details.csv"
        if not details_path.exists():
            print(f"  [!] {model}: {details_path} not found, skipping")
            continue
        df = pd.read_csv(details_path)
        df["Model"] = model
        df["Subject"] = df["filename"].str.replace(".csv", "", regex=False)
        detail_frames.append(df)

    if not detail_frames:
        print("  [!] No evaluation_details.csv files found")
        return

    combined_df = pd.concat(detail_frames, ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    sns.boxplot(
        data=combined_df,
        x="Subject",
        y="MAE_TO",
        hue="Model",
        ax=ax1,
        palette="Set2",
    )
    ax1.axhline(20, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title("MAE TO per Subject and Model", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Subject", fontsize=11)
    ax1.set_ylabel("MAE (ms)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    sns.boxplot(
        data=combined_df,
        x="Subject",
        y="MAE_IC",
        hue="Model",
        ax=ax2,
        palette="Set2",
    )
    ax2.axhline(20, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title("MAE IC per Subject and Model", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Subject", fontsize=11)
    ax2.set_ylabel("MAE (ms)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax1.legend(title="Model", loc="best")
    ax2.legend(title="Model", loc="best")

    plt.tight_layout()
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "boxplots_per_subject.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  [OK] Saved: {output_path}")
    plt.close(fig)
