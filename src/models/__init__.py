"""Model package exports."""

from src.models.base import GaitModel
from src.models.cnn import CNNModel, SimpleCNN

__all__ = ["GaitModel", "CNNModel", "SimpleCNN"]
