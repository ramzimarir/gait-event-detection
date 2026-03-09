"""Data loading and preprocessing utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from config import FEATURE_COLUMNS, INPUT_DATA_DIR, LABEL_COLUMNS, TIME_COLUMN

logger = logging.getLogger(__name__)


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)

    return logger_instance


def validate_directory_structure(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Directory {input_dir} does not exist. "
            "Place your CSV files in the data/ folder."
        )

    output_dir.mkdir(parents=True, exist_ok=True)


def get_input_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.csv"))


def validate_csv_columns(df_columns: List[str], required_columns: List[str], file_path: Path) -> None:
    missing = set(required_columns) - set(df_columns)
    if missing:
        raise ValueError(
            f"Missing columns in {file_path.name}: {missing}\n"
            f"Available columns: {df_columns}"
        )


class DataLoader:
    """Load and preprocess IMU data."""

    ACC_COLUMNS = ["x_acc", "y_acc", "z_acc"]
    GYRO_COLUMNS = ["x_gyro", "y_gyro", "z_gyro"]
    QUAT_COLUMNS = ["quat_1", "quat_2", "quat_3", "quat_4"]

    def load_data(self, file_path: Path) -> pd.DataFrame:
        logger.info(f"Loading {file_path.name}")
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            raise IOError(f"Error reading {file_path}: {exc}")

        validate_csv_columns(
            df.columns.tolist(),
            [TIME_COLUMN] + FEATURE_COLUMNS + LABEL_COLUMNS,
            file_path,
        )
        return df

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[FEATURE_COLUMNS].copy()

    def get_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[LABEL_COLUMNS].copy()

    def _required_raw_columns(self, side: str, require_gyro: bool) -> List[str]:
        columns = [TIME_COLUMN]
        columns += [f"{name}_{side}" for name in self.ACC_COLUMNS]
        columns += [f"{name}_{side}" for name in self.QUAT_COLUMNS]
        if require_gyro:
            columns += [f"{name}_{side}" for name in self.GYRO_COLUMNS]
        return columns

    def extract_imu_data(self, df: pd.DataFrame, side: str) -> Tuple[np.ndarray, np.ndarray]:
        acc_body = df[[f"{name}_{side}" for name in self.ACC_COLUMNS]].to_numpy()
        quaternions = df[[f"{name}_{side}" for name in self.QUAT_COLUMNS]].to_numpy()
        quaternions = np.roll(quaternions, -1, axis=1)
        return acc_body, quaternions

    def extract_gyro_data(self, df: pd.DataFrame, side: str) -> np.ndarray:
        return df[[f"{name}_{side}" for name in self.GYRO_COLUMNS]].to_numpy()

    @staticmethod
    def compensate_gravity(acc_body: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_quat(quaternions).as_matrix()
        acc_global = np.einsum("nij,nj->ni", rotation, acc_body)
        gravity = np.array([0.0, 0.0, 9.81])
        return acc_global - gravity

    def preprocess_side(
        self,
        df_raw: pd.DataFrame,
        side: str,
        require_gyro: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        required = self._required_raw_columns(side, require_gyro=require_gyro)
        validate_csv_columns(df_raw.columns.tolist(), required, Path("<dataframe>"))

        acc_body, quaternions = self.extract_imu_data(df_raw, side)
        acc_compensated = self.compensate_gravity(acc_body, quaternions)

        gyro_data = None
        if require_gyro:
            gyro_data = self.extract_gyro_data(df_raw, side)

        return acc_compensated, gyro_data


def load_subject_files(input_dir: Path = INPUT_DATA_DIR) -> List[Path]:
    return get_input_files(input_dir)
