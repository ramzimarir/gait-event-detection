"""Shared training utilities for LOSO training scripts."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum(dim=0)
    total = labels.shape[0]
    negatives = total - positives
    pos_weight = negatives / torch.clamp(positives, min=1.0)
    return pos_weight


def align_logits_labels(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure labels include a class dimension.
    if labels.dim() == 2:
        labels = labels.unsqueeze(-1)

    # Convert logits from (B, C, T) to (B, T, C) when needed.
    if logits.dim() == 3 and logits.shape[1] < logits.shape[2]:
        logits = logits.permute(0, 2, 1)

    # Align label shape with logits.
    if labels.shape != logits.shape:
        labels = labels.permute(0, 2, 1)

    return logits, labels


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits, labels = align_logits_labels(logits, labels)
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).float().mean().item()
    return float(correct)


def train_one_epoch(
    model: nn.Module,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    accuracies: List[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        logits, y = align_logits_labels(logits, y)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(batch_accuracy(logits, y))

    return float(np.mean(losses)), float(np.mean(accuracies))


def evaluate(
    model: nn.Module,
    loader: TorchDataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    accuracies: List[float] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            logits, y = align_logits_labels(logits, y)
            loss = criterion(logits, y)
            losses.append(loss.item())
            accuracies.append(batch_accuracy(logits, y))

    return float(np.mean(losses)), float(np.mean(accuracies))
