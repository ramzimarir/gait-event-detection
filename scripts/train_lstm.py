"""Train a simple LSTM model with LOSO evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is available on PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset

from config import OUTPUT_DIR, WINDOW_SIZE
from src.dataset import GaitDataset
from src.models.lstm import LSTMModel
from train_common import compute_pos_weight, evaluate, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM with LOSO protocol")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    patience = 10
    output_dir = Path(OUTPUT_DIR) / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = GaitDataset(
        window_size=args.window_size,
        overlap=args.overlap,
        label_mode="sequence",
        downsample_factor=1,
    )
    subjects = dataset.subject_ids()

    for test_subject in subjects:
        train_idx, test_idx = dataset.loso_split(test_subject)

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = TorchDataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = TorchDataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

        model = LSTMModel(in_channels=dataset.in_channels, num_classes=dataset.num_classes, device=str(device))

        sample_indices = np.random.choice(train_idx, min(len(train_idx), 1000), replace=False)
        sample_labels = torch.stack([dataset[i][1] for i in sample_indices])
        sample_labels = sample_labels.reshape(-1, sample_labels.shape[-1])
        pos_weight = compute_pos_weight(sample_labels).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_loss = float("inf")
        epochs_no_improve = 0
        best_path = output_dir / f"best_lstm_subject_{test_subject}.pth"

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            print(
                f"[{test_subject}] Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {test_loss:.4f} Acc: {test_acc:.3f}"
            )

            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), best_path)
                epochs_no_improve = 0
                print(f"[{test_subject}] Best model saved: {best_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[{test_subject}] Early stopping at epoch {epoch}")
                    break


if __name__ == "__main__":
    main()
